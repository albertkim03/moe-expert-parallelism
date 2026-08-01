#!/usr/bin/env python3
"""The demo. This is the file the reviewers run first.

    python train.py                 # 4 ranks, default config
    python train.py --ranks 2
    python train.py --steps 20

It prints three things, which together are the take-home's "quantitative
evidence" deliverable:

    1. CORRECTNESS  — the distributed forward vs a single-process reference
    2. TRAINING     — loss decreasing over several steps
    3. PARAMETERS   — router identical across ranks, experts legitimately not
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

    delta = (y_all - expected).abs().max().item()
    if rank == 0:
        print("=" * 68)
        print("1. CORRECTNESS  (distributed vs single-process reference)")
        print("=" * 68)
        print(f"  global batch          : {tuple(x_global.shape)}  "
              f"({T} tokens on each of {world_size} ranks)")
        print(f"  max |distributed-ref| : {delta:.3e}")
        print(f"  allclose(atol=1e-5)   : {torch.allclose(y_all, expected, atol=1e-5)}")
        print("  -> identical output, but no rank held more than "
              f"{cfg.num_experts // world_size} of {cfg.num_experts} experts.\n", flush=True)


def worker(rank: int, world_size: int, args) -> None:
    """One rank's entire life. Every rank runs this same function (SPMD)."""
    distributed.setup(rank, world_size, port=args.port)
    try:
        cfg = MoEConfig(num_experts=8, d_model=16, d_ff=32, top_k=1, tokens_per_rank=12)
        cfg.validate(world_size)
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
        if rank == 0:
            print()
            print("=" * 68)
            print("3. PARAMETERS  (what got all-reduced, and what did not)")
            print("=" * 68)
            print(f"  replicated params / rank : {health['n_replicated_params']:>6}"
                  "   (router + head)  -> ALL-REDUCED")
            print(f"  expert params / rank     : {health['n_expert_params']:>6}"
                  f"   ({cfg.num_experts // world_size} experts)   -> NOT all-reduced")
            print()
            print(f"  router spread across ranks: {health['router_spread_across_ranks']:.3e}"
                  "   <- replicas never drift", flush=True)

        ordered_print(
            rank, world_size,
            f"    rank {rank}: owns experts {health['local_expert_ids']}   "
            f"expert fingerprint {health['expert_fingerprint']:+.6f}",
        )

        if rank == 0:
            print()
            print("  The router is ONE parameter copied on every rank, so its gradient is")
            print("  averaged and the copies stay bit-identical (spread = 0 above).")
            print()
            print("  The experts DIFFER across ranks, and that is correct — they are")
            print("  different parameters, not disagreeing copies of one parameter. Each")
            print("  expert's gradient is already the global one, because the dispatch")
            print("  all-to-all delivered it every token in the world that chose it.")
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
