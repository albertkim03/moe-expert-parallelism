"""Configuration for the expert-parallel MoE demo.

This file is PROVIDED COMPLETE — it is boilerplate, not the exercise. Everything
else in `moe_ep/` is yours to write.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    """Every knob the demo has, in one place.

    Keeping this tiny and frozen means a rank can construct it independently and
    every rank is guaranteed to agree — which matters, because the routing
    decision must be reproducible across processes.
    """

    # --- model shape -------------------------------------------------------
    num_experts: int = 8          # E — total experts across the whole job
    d_model: int = 16             # d — hidden size of a token vector
    d_ff: int = 32                # inner width of each expert's FFN
    top_k: int = 1                # k — experts each token is sent to

    # --- batch -------------------------------------------------------------
    tokens_per_rank: int = 12     # T — tokens each rank starts with

    # --- routing behaviour -------------------------------------------------
    capacity_factor: float | None = None
    """None  -> "dropless": buffers are sized from the real counts.
    A float -> each expert accepts at most
               ceil(capacity_factor * total_tokens * k / E) tokens and drops
               the rest. Static shapes, but tokens get discarded.

    Pick one and be ready to defend it. See the study app's
    "Expert Parallelism -> Two ways to size the buffers".
    """

    aux_loss_weight: float = 0.01  # alpha in the Switch load-balancing loss

    # --- reproducibility ---------------------------------------------------
    seed: int = 0
    """MUST be identical on every rank.

    The model (router + experts) is built from this seed so that all ranks agree
    on the initial parameters. Vary the DATA per rank, never the init — see
    Pitfalls -> "different seeds per rank".
    """

    def experts_per_rank(self, world_size: int) -> int:
        return self.num_experts // world_size

    def local_expert_ids(self, rank: int, world_size: int) -> list[int]:
        """Which global expert ids this rank owns (contiguous blocks)."""
        per = self.experts_per_rank(world_size)
        return list(range(rank * per, (rank + 1) * per))

    def owner_of(self, expert_id: int, world_size: int) -> int:
        """Which rank owns a given global expert id."""
        return expert_id // self.experts_per_rank(world_size)
