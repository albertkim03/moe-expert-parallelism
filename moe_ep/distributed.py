"""Process-group setup and the all-to-all plumbing.

Nothing here is MoE-specific — it is the distributed toolbox the MoE layer sits
on top of. Getting it boring and reliable early makes the rest of the day much
calmer.

STEP 6 of the build plan.

A NOTE ON GLOO
--------------
You may read that the gloo (CPU) backend does not support all-to-all. On
PyTorch 2.x that is out of date: `dist.all_to_all_single` works on gloo,
including uneven `input_split_sizes` / `output_split_sizes`. Verify it yourself
in about two seconds before designing around a workaround — the study app's
Local Torch Lab has a script for exactly this.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


# ---------------------------------------------------------------- lifecycle


def setup(rank: int, world_size: int, port: int = 29500, backend: str = "gloo") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1" # machine
    os.environ["MASTER_PORT"] = str(port)   # network port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.set_num_threads(1)

def cleanup() -> None:
    dist.destroy_process_group()

# ------------------------------------------------------------ count exchange

# Reconcile how many tokens are sent individually from every rank to every other rank
def exchange_counts(send_counts: torch.Tensor, group=None) -> torch.Tensor:
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts

# TRADEOFF
def all_to_all(x, out_splits, in_splits, group=None):
    """Send/receive token blocks between ranks. Uses the built-in differentiable
    all-to-all rather than a hand-rolled autograd.Function — see README for why.
    """
    out = torch.empty(sum(out_splits), *x.shape[1:], dtype=x.dtype)
    return dist_nn.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)

