"""Gradient synchronisation — the deliverable the brief names explicitly.

    "show that expert and non-expert parameters (e.g. routers) are correctly
     updated, and explain what gets all-reduced and what doesn't"

THE RULE, in one sentence:

    A parameter needs an all-reduce if and only if more than one rank holds a
    copy of it.

  - The router and the head are copied on every rank    -> all-reduce
  - Each expert lives on exactly one rank               -> nothing to reduce with

Why the second one is not hand-waving: the dispatch all-to-all already delivered
every token in the entire world that chose expert e onto e's owner. So the
gradient that owner accumulates is already the sum over the global batch.

In plain data parallelism, tokens stay put and gradients move.
In expert parallelism, tokens move and gradients stay put.
The sum over the global batch happens either way — just somewhere else.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def sync_replicated_grads(model: nn.Module, group=None) -> None:
    """All-reduce the gradients of REPLICATED parameters only.

    Expert parameters are tagged `is_expert=True` at construction (see
    expert.py), so this function is the single place the sharding rule is
    expressed in executable form.

    NOTE ON THE MISSING DIVISION: there is deliberately no `/= world_size` here.
    train.py scales each rank's loss by the GLOBAL token count, so the per-rank
    losses already SUM to the global mean loss. A plain SUM all-reduce therefore
    produces exactly the gradient a single-process run over the whole batch
    would produce — which is what makes the expert gradients match the reference
    exactly rather than being off by a factor of P.

    DO NOT replace this with DistributedDataParallel. DDP assumes every
    parameter is replicated: it broadcasts rank 0's weights at construction
    (destroying the shard before step 1) and then all-reduces every gradient,
    averaging experts that are completely different parameters. Neither failure
    raises an error and the loss still goes down a little.
    """
    for p in model.parameters():
        if getattr(p, "is_expert", False):
            continue  # sharded: unique owner, gradient is already global
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)


@torch.no_grad()
def check_parameter_health(model: nn.Module, rank: int, world_size: int, group=None) -> dict:
    """Produce the EVIDENCE the brief asks for. Print this; paste it in the README.

    Two claims, both checked here:

      1. Replicated parameters are IDENTICAL on every rank. We all_gather the
         router weight and measure the spread. Should be ~0. If it is not, the
         sync is wrong.

      2. Expert parameters DIFFER across ranks — and that is CORRECT, not a bug.
         Rank 0 holds E0/E1 and rank 1 holds E2/E3; they are different
         parameters, not disagreeing copies of one parameter. The fingerprint
         below is just a cheap scalar summary so the difference is visible in
         printed output.
    """
    w = model.moe.router.gate.weight
    gathered = torch.empty(world_size, *w.shape)
    dist.all_gather_into_tensor(gathered, w.unsqueeze(0).contiguous(), group=group)
    router_spread = (gathered - gathered[0]).abs().max().item()

    return {
        "router_spread_across_ranks": router_spread,
        "n_replicated_params": sum(
            p.numel() for p in model.parameters() if not getattr(p, "is_expert", False)
        ),
        "n_expert_params": sum(
            p.numel() for p in model.parameters() if getattr(p, "is_expert", False)
        ),
        "local_expert_ids": list(model.moe.local_expert_ids),
        # a scalar summary of this rank's own experts — differs per rank, by design
        "expert_fingerprint": sum(p.sum().item() for p in model.moe.experts.parameters()),
    }
