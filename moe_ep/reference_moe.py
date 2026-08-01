"""The MoE layer in a single process. Your source of truth.

Write this BEFORE the distributed version. Everything you do afterwards is
checked against it, so if this is wrong you will spend the afternoon debugging
the wrong file.

It is deliberately the dumbest possible implementation: hold all E experts, loop
over them, gather the tokens each one wants. Slow, obvious, and easy to believe.

STEP 5 of the build plan.
"""

import torch
import torch.nn as nn

from .config import MoEConfig
from .expert import Expert
from .router import Router


class ReferenceMoE(nn.Module):
    """All E experts in one process. No communication anywhere.

    Shapes
    ------
    input   (T, d_model)
    output  (T, d_model)
    """
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg.d_model, cfg.num_experts, cfg.top_k)
        self.experts = nn.ModuleList([Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_gate, probs = self.router(x)
        y = torch.zeros_like(x)
        for e in range(self.cfg.num_experts):
            for slot in range(self.cfg.top_k):
                mask = topk_idx[:, slot] == e        # which tokens chose e
                if mask.any():
                    h = self.experts[e](x[mask])     # (n_e, d_model)
                    y[mask] += topk_gate[mask, slot].unsqueeze(-1) * h

        # TRADEOFF
        return y, x.new_zeros(())
