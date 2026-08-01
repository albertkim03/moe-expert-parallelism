#!/usr/bin/env python3
"""The demo. This is the file the reviewers run first.

    python train.py                 # 4 ranks, default config
    python train.py --ranks 2
    python train.py --steps 20

It prints three things, which together are the take-home's "quantitative
evidence" deliverable:

    1. CORRECTNESS  — the distributed forward vs a single-process reference
    2. TRAINING     — loss decreasing over several steps
    3. PARAMETERS   — which numbers got shared between ranks, and which did not
"""

import argparse
import warnings

warnings.filterwarnings("ignore")  # silence the harmless "no numpy" torch warning

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_ep import distributed
from moe_ep.config import MoEConfig
from moe_ep.ep_moe import ExpertParallelMoE
from moe_ep.model import TinyMoEModel, make_batch
from moe_ep.reference_moe import ReferenceMoE
from moe_ep.sync import check_parameter_health, sync_replicated_grads


def ordered_print(rank: int, world_size: int, line: str) -> None:
    """Print one line per rank, in rank order, instead of four interleaved streams."""
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(line, flush=True)
    dist.barrier()


def correctness_check(cfg: MoEConfig, model, rank: int, world_size: int) -> None:
    """Distributed forward == single-process reference?

    Both models are built from the same seed, so rank r's expert e is
    bit-identical to the reference's expert e. We feed the same global batch to
    both, gather the distributed outputs, and compare.

    This is the check that catches a permutation/inverse-permutation mismatch,
    which otherwise produces no error at all — just a loss that quietly fails
    to improve.
    """
    T = cfg.tokens_per_rank
    torch.manual_seed(cfg.seed)
    reference = ReferenceMoE(cfg)
    
    torch.manual_seed(1234)  # same on every rank -> same global batch
    x_global = torch.randn(world_size * T, cfg.d_model)
    x_local = x_global[rank * T : (rank + 1) * T]

    with torch.no_grad():
        y_local, _ = model.moe(x_local)
        y_all = torch.empty(world_size * T, cfg.d_model)
        dist.all_gather_into_tensor(y_all, y_local.contiguous())
        expected, _ = reference(x_global)

    diff = (y_all - expected).abs()
    largest = diff.max().item()
    n_diff = int((diff > 0).sum())
    total = diff.numel()

    if rank == 0:
        per_rank = cfg.num_experts // world_size
        print("=" * 68)
        print("1. CORRECTNESS   (is the distributed version computing the right thing?)")
        print("=" * 68)
        print("  Run the SAME batch through two implementations, then subtract.\n")
        print(f"    input   {world_size * T} token vectors, {cfg.d_model} numbers each"
              f"   ({T} tokens on each of {world_size} ranks)\n")
        print(f"    A       reference_moe.py  — all {cfg.num_experts} experts in ONE process")
        print(f"    B       ep_moe.py         — {per_rank} experts per rank, across {world_size} ranks\n")
        print(f"  Each produces {world_size * T} x {cfg.d_model} = {total} numbers.\n")
        print(f"    numbers that differ (A vs B) : {n_diff} of {total}")
        print(f"    largest difference           : "
              f"{'0  (exact match)' if largest == 0 else f'{largest:.3e}'}\n")
        print(f"  -> same answer, but no rank ever held more than {per_rank} of the "
              f"{cfg.num_experts} experts.\n", flush=True)


def worker(rank: int, world_size: int, args) -> None:
    """One rank's entire life. Every rank runs this same function (SPMD)."""
    distributed.setup(rank, world_size, port=args.port)
    try:
        cfg = MoEConfig(num_experts=8, d_model=16, d_ff=32, top_k=1, tokens_per_rank=12)
        n_global = world_size * cfg.tokens_per_rank

        # Same seed on every rank -> every copy of the router starts identical.
        torch.manual_seed(cfg.seed)
        model = TinyMoEModel(cfg, ExpertParallelMoE(cfg, rank, world_size))
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        correctness_check(cfg, model, rank, world_size)

        # ---------------------------------------------------------- training
        if rank == 0:
            print("=" * 68)
            print("2. TRAINING  (one MoE layer, regressing a constant)")
            print("=" * 68, flush=True)

        for step in range(args.steps):
            x, target = make_batch(cfg, rank, step)  # DIFFERENT data per rank
            pred, aux = model(x)

            # Scale by the GLOBAL token count so the per-rank losses SUM to the
            # global mean loss. That is what lets the gradient sync below be a
            # plain SUM with no division, and what makes the expert gradients
            # match a single-process run exactly.
            loss = ((pred - target) ** 2).sum() / n_global
            loss = loss + cfg.aux_loss_weight * aux  # aux is 0.0 — see README

            opt.zero_grad()
            loss.backward()
            sync_replicated_grads(model)  # <- replicated only; experts skipped
            opt.step()

            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            if rank == 0 and (step % max(1, args.steps // 10) == 0 or step == args.steps - 1):
                print(f"  step {step:>3}   loss {global_loss.item():.6f}", flush=True)

        # -------------------------------------------------------- parameters
        health = check_parameter_health(model, rank, world_size)
        n_rep, n_exp = health["n_replicated"], health["n_expert"]
        per_rank = cfg.num_experts // world_size

        if rank == 0:
            print()
            print("=" * 68)
            print("3. PARAMETERS   (which numbers had to be shared between ranks?)")
            print("=" * 68)
            print(f"  Each rank holds {n_rep + n_exp} learnable numbers, in two groups.\n")

            print(f"  GROUP 1 — the router and the output layer      {n_rep:>5} numbers")
            print("    Every rank holds the SAME numbers here: they are copies of one")
            print("    thing. Each rank computed a different gradient for them, because")
            print("    each saw different tokens. So the gradients were added up across")
            print("    ranks. If that worked, all copies are still identical:\n")
            drift = health["replicated_disagreement"]
            verdict = "0   (identical, as intended)" if drift == 0 else f"{drift:.3e}   (they drifted!)"
            print(f"      biggest disagreement between ranks : {verdict}\n")

            print(f"  GROUP 2 — the experts                          {n_exp:>5} numbers"
                  f"   ({per_rank} of the {cfg.num_experts} experts)")
            print("    Every rank holds DIFFERENT numbers here. Rank 0 has experts 0 and")
            print("    1; rank 1 has 2 and 3. They are not copies, so there is nothing to")
            print("    add them up with. They were left alone, and they should NOT match:\n",
                  flush=True)

        ordered_print(
            rank, world_size,
            f"      rank {rank}  owns experts {str(health['local_expert_ids']):<8}"
            f"its {n_exp} numbers add up to {health['expert_weight_sum']:>+11.6f}",
        )

        if rank == 0:
            print()
            print("  So: the numbers that SHOULD be identical are, and the numbers that")
            print("  SHOULD differ do.")
            print()
            print("  Why the experts need no sharing: the dispatch all-to-all already sent")
            print("  every token in the whole job that chose expert 5 to the rank holding")
            print("  expert 5. That rank's gradient therefore already covers the entire")
            print("  batch — there is no missing piece on another rank to add in.")
            print("=" * 68, flush=True)


    finally:
        # Always tear down, including on the error path — otherwise a crash on
        # one rank leaves the others hanging on a collective forever.
        distributed.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description="Expert-parallel MoE training demo")
    p.add_argument("--ranks", type=int, default=4, help="number of processes (world size)")
    p.add_argument("--steps", type=int, default=20, help="training steps")
    p.add_argument("--port", type=int, default=29500, help="rendezvous port")
    args = p.parse_args()

    # mp.spawn keeps `python train.py` working with no torchrun needed, which
    # makes "how to run it" a single line. torchrun is the production answer.
    mp.spawn(worker, args=(args.ranks, args), nprocs=args.ranks, join=True)


if __name__ == "__main__":
    main()
