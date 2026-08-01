"""moe_ep — a minimal, readable demonstration of expert parallelism.

Read the modules in this order; each one only depends on the ones above it.

    config.py          knobs (provided complete)
    expert.py          one expert = one small feed-forward network
    router.py          the gating network that picks experts for tokens
    reference_moe.py   the whole MoE layer in ONE process — your source of truth
    distributed.py     process groups and the all-to-all helpers
    ep_moe.py          the same layer, experts sharded across ranks  <- the point
    sync.py            which gradients get all-reduced, and which do not
    model.py           a tiny model wrapping the MoE layer so it can be trained

Everything except config.py is a stub for you to implement.
"""

from .config import MoEConfig

__all__ = ["MoEConfig"]
