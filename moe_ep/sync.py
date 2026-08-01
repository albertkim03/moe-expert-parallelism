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
         parameters, not disagreeing copies of one parameter. Adding up each
         rank's expert numbers gives one value per rank that is easy to eyeball;
         they differ, and that is the intended result.
    """
    # Collect every rank's copy of the router weight, then measure the biggest
    # disagreement between them, element by element. Should be 0.
    w = model.moe.router.gate.weight
    gathered = torch.empty(world_size, *w.shape)
    dist.all_gather_into_tensor(gathered, w.unsqueeze(0).contiguous(), group=group)
    biggest_disagreement = (gathered.max(dim=0).values - gathered.min(dim=0).values).max().item()

    return {
        # how far apart the replicated copies have drifted — want exactly 0
        "replicated_disagreement": biggest_disagreement,
        "n_replicated": sum(
            p.numel() for p in model.parameters() if not getattr(p, "is_expert", False)
        ),
        "n_expert": sum(
            p.numel() for p in model.parameters() if getattr(p, "is_expert", False)
        ),
        "local_expert_ids": list(model.moe.local_expert_ids),
        # Adding up this rank's expert numbers gives a single value that is easy
        # to eyeball. It differs per rank BY DESIGN — these are different
        # experts, not copies of the same one.
        "expert_weight_sum": sum(p.sum().item() for p in model.moe.experts.parameters()),
    }
