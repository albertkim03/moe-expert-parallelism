"""THE expert-parallel MoE layer.

This is the file the whole take-home is about. Budget 90 minutes and do not
start it until `reference_moe.py` works and you have something to compare
against.

The seven-step algorithm, which you should be able to draw from memory:

    1. route        every rank runs the router on its OWN tokens (no comms)
    2. permute      sort local tokens by which rank owns their expert
    3. counts       tiny all-to-all so everyone can size their receive buffer
    4. dispatch     all-to-all — tokens fly to the rank owning their expert
    5. compute      each rank runs only ITS experts, purely locally
    6. combine      all-to-all — outputs fly home (step 4 with splits swapped)
    7. un-permute   invert step 2, then scale by the gate

Tokens move to the weights. The weights never move.

STEP 7 of the build plan.
"""

import torch
import torch.nn as nn

from .config import MoEConfig
from .distributed import all_to_all, exchange_counts
from .expert import Expert
from .router import Router


class ExpertParallelMoE(nn.Module):
    """An MoE layer whose experts are sharded across ranks.

    Every rank constructs this. Each ends up with:
      - an identical copy of the router          (replicated)
      - only the experts it owns, E // P of them (sharded)

    Shapes
    ------
    input   (T, d_model)   this rank's own tokens
    output  (T, d_model)   this rank's own tokens, transformed

    The output is the same shape as the input even though the middle of the
    computation happened somewhere else entirely. That is the whole trick.
    """

    def __init__(self, cfg: MoEConfig, rank: int, world_size: int, group=None) -> None:
        super().__init__()
        cfg.validate(world_size)
        self.cfg, self.rank, self.world_size, self.group = cfg, rank, world_size, group
        self.router = Router(cfg.d_model, cfg.num_experts, cfg.top_k)
        self.local_expert_ids = cfg.local_expert_ids(rank, world_size)
        all_experts = [Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.num_experts)]
        self.experts = nn.ModuleList([all_experts[i] for i in self.local_expert_ids])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- 1. route (local, no communication) ---------------------------
        topk_idx, topk_gate, probs = self.router(x)
        assert self.cfg.top_k == 1, "expert-parallel forward only supports top_k == 1 for now"

        # --- 2. permute ------------------------------------------------------
        experts_per_rank = self.cfg.experts_per_rank(self.world_size)
        dest = topk_idx[:, 0] // experts_per_rank      # which RANK owns each token's expert
        order = torch.argsort(dest, stable=True)
        inv = torch.argsort(order)                      # save: undoes this permutation in step 7
        x_sorted = x[order]
        expert_id_sorted = topk_idx[:, 0][order]         # carried along so receivers know which local expert to run

        # --- 3. counts ---------------------------------------------------------
        send_counts = torch.bincount(dest, minlength=self.world_size)
        recv_counts = exchange_counts(send_counts, self.group)

        # --- 4. dispatch ------------------------------------------------------
        recv = all_to_all(x_sorted, recv_counts.tolist(), send_counts.tolist(), self.group)
        # Option (a): a second all-to-all just for expert ids, alongside the tokens.
        # Simpler to get right than deriving ids from sort order within each block (b).
        recv_expert_id = all_to_all(expert_id_sorted, recv_counts.tolist(), send_counts.tolist(), self.group)

        # --- 5. local expert compute -------------------------------------------
        out = torch.zeros_like(recv)
        for local_i, global_id in enumerate(self.local_expert_ids):
            mask = recv_expert_id == global_id
            out[mask] = self.experts[local_i](recv[mask])   # empty mask -> 0-row matmul, handled fine

        # --- 6. combine ---------------------------------------------------------
        back = all_to_all(out, send_counts.tolist(), recv_counts.tolist(), self.group)  # splits swapped

        # --- 7. un-permute and scale ---------------------------------------------
        y = back[inv]
        y = y * topk_gate[:, 0].unsqueeze(-1)

        # TRADEOFF no aux loss
        return y, x.new_zeros(())
        

