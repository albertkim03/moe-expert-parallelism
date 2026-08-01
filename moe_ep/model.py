"""A tiny model wrapping the MoE layer, so there is something to actually train.

The brief explicitly permits this:

    "You don't need to build a full transformer. Training a few experts on
     something trivial (like predicting a fixed number) is completely fine."

So: MoE layer -> linear head -> one number per token, regressed against 1.0.
No attention, no embeddings, no depth. None of that would demonstrate anything
about expert parallelism.
"""

import torch
import torch.nn as nn

from .config import MoEConfig


class TinyMoEModel(nn.Module):
    """MoE layer + linear head.

    Shapes
    ------
    input   (T, d_model)
    output  (T,)   one predicted number per token

    Takes the MoE layer as a constructor argument so the SAME model class can
    wrap either ReferenceMoE or ExpertParallelMoE. That is what makes the
    correctness comparison a one-line change instead of a fork of the codebase.
    """

    def __init__(self, cfg: MoEConfig, moe_layer: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe = moe_layer
        # Untagged, so sync.py all-reduces it. Having a non-expert parameter
        # besides the router matters: the brief says "expert and non-expert
        # parameters (e.g. routers)" — the "e.g." implies routers are one
        # example of a class, and the head shows the rule applies to the class.
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(T, d_model) -> (prediction (T,), aux_loss scalar)"""
        h, aux = self.moe(x)
        return self.head(h).squeeze(-1), aux


def make_batch(cfg: MoEConfig, rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-rank data.

    Returns x (T, d_model) and target (T,).

    Note the two different kinds of seeding in this project:

      - the MODEL is seeded identically on every rank, so all copies of the
        router start equal and stay comparable
      - the DATA is seeded per (rank, step), so every rank sees something
        different — that is what makes it data-parallel at all

    Getting those backwards is a classic bug: seed the model per-rank and the
    routers diverge immediately, so "router identical across ranks" fails and
    the correctness comparison is meaningless.

    A local Generator is used so this never disturbs the global RNG stream that
    the model construction depends on.
    """
    g = torch.Generator().manual_seed(cfg.seed * 100_003 + rank * 1_009 + step)
    x = torch.randn(cfg.tokens_per_rank, cfg.d_model, generator=g)
    target = torch.ones(cfg.tokens_per_rank)  # "predict a fixed number"
    return x, target
