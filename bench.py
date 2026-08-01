#!/usr/bin/env python3
"""Scaling analysis for expert parallelism.

The brief pre-excuses wall-clock speedups:

    "since you won't necessarily have access to multiple GPUs, wall-clock
     improvements may not be possible. We care a lot about your analysis of how
     and why this scales"

So this is an ARGUMENT supported by arithmetic, not a benchmark. No torch, no
processes — everything here is closed-form and runs in milliseconds.

    python bench.py
    python bench.py --experts 64 --d-ff 14336 --top-k 2     # Mixtral-ish

SYMBOLS (used throughout)
    P         number of processes, a.k.a. ranks or world_size
    E         number of experts, total across all processes
    E/P       experts held by each process
    T         tokens each process handles per step
    k         experts each token is routed to (1 = Switch, 2 = Mixtral)
    d_model   width of one token vector
    d_ff      width of the middle layer inside an expert
    b         bytes per number (4 = fp32, 2 = bf16)
"""

import argparse

LINE = "-" * 72


def analytic(E: int, P: int, d_model: int, d_ff: int, T: int, k: int, b: int = 4) -> dict:
    """Closed-form scaling quantities. Pure arithmetic — derive these in the README.

    An expert is two weight matrices, each d_model x d_ff, hence the 2 in
    `2 * d_model * d_ff`.

    The (P-1)/P factor appears in every collective: with P processes, each one
    ships (P-1)/P of its data off-box and keeps the rest. It rises from 0 at
    P=1 towards 1 as P grows, so it can at most double the traffic.
    """
    per_expert = 2 * d_model * d_ff
    ring = (P - 1) / P if P > 1 else 0.0

    return {
        # --- parameters -----------------------------------------------------
        "per_expert": per_expert,
        "all_experts": E * per_expert,
        "experts_per_proc": E // P,
        "expert_params_per_proc": (E // P) * per_expert,
        "router_params": E * d_model,
        # --- compute --------------------------------------------------------
        # two matmuls per expert, k experts per token. NOTE: no E in this.
        "flops_per_token_fwd": 2 * k * 2 * d_model * d_ff,
        # --- communication, per process per step ----------------------------
        # dispatch + combine, each moving T*k token vectors of width d_model
        "a2a_bytes": 2 * T * k * d_model * b * ring,
        "a2a_ceiling": 2 * T * k * d_model * b,
        # what you would pay instead if every process held a copy of every
        # expert and you all-reduced their gradients
        "allreduce_bytes": 2 * ring * E * per_expert * b,
        # --- the crossover ---------------------------------------------------
        # set a2a_bytes == allreduce_bytes; the ring factor and d_model cancel
        "crossover_T": 2 * E * d_ff / k,
    }


def fmt(n: float) -> str:
    return f"{n:,.0f}"


def report(args) -> None:
    E, P, d, ff, T, k, b = (
        args.experts, args.ranks, args.d_model, args.d_ff, args.tokens, args.top_k, args.bytes
    )
    r = analytic(E, P, d, ff, T, k, b)

    print("=" * 72)
    print("EXPERT PARALLELISM — SCALING ANALYSIS")
    print("=" * 72)
    print("\nConfiguration")
    for sym, desc, val in [
        ("P", "processes (ranks)", P),
        ("E", "experts, total", E),
        ("E/P", "experts per process", r["experts_per_proc"]),
        ("T", "tokens per process per step", T),
        ("k", "experts each token is routed to", k),
        ("d_model", "width of a token vector", d),
        ("d_ff", "width of an expert's middle layer", ff),
        ("b", "bytes per number", b),
    ]:
        print(f"  {sym:<9} {desc:<36} {val}")

    if E % P:
        print(f"\n  WARNING: E={E} is not divisible by P={P} — processes would own "
              "unequal numbers of experts.")

    # ------------------------------------------------------------------ 1
    print(f"\n{LINE}\n1. PARAMETERS\n{LINE}")
    print(f"  per expert              2*d_model*d_ff           = {fmt(r['per_expert'])}")
    print(f"  all experts             E*2*d_model*d_ff         = {fmt(r['all_experts'])}")
    print(f"  expert params/process   (E/P)*2*d_model*d_ff     = {fmt(r['expert_params_per_proc'])}"
          "    <- falls as 1/P")
    print(f"  router (replicated)     E*d_model                = {fmt(r['router_params'])}"
          "    <- same on every process")
    print("\n  Sharding the experts is the entire memory argument: each process")
    print("  stores E/P of them instead of all E.")

    # ------------------------------------------------------------------ 2
    print(f"\n{LINE}\n2. COMPUTE\n{LINE}")
    print(f"  FLOPs/token (forward)   2*k*2*d_model*d_ff       = {fmt(r['flops_per_token_fwd'])}")
    print(f"  FLOPs/token (fwd+bwd)   ~3x forward              = {fmt(r['flops_per_token_fwd']*3)}")
    print("\n  There is no E in that formula. Doubling the number of experts doubles")
    print("  the parameters and costs nothing extra per token, because each token")
    print("  still visits only k of them. That is why MoE exists.")

    # ------------------------------------------------------------------ 3
    print(f"\n{LINE}\n3. COMMUNICATION  (per process, per step, per MoE layer)\n{LINE}")
    print("  What expert parallelism COSTS:")
    print(f"    all-to-all            2*T*k*d_model*b*(P-1)/P  = {fmt(r['a2a_bytes'])} bytes")
    print(f"                          (dispatch + combine; ceiling {fmt(r['a2a_ceiling'])})")
    print("  What expert parallelism SAVES:")
    print(f"    all-reduce avoided    2*(P-1)/P*E*2*d_model*d_ff*b")
    print(f"                                                   = {fmt(r['allreduce_bytes'])} bytes")
    print("                          (if every process held a copy of every expert)")
    if r["a2a_bytes"] > 0:
        ratio = r["allreduce_bytes"] / r["a2a_bytes"]
        verdict = (f"EP moves {ratio:.1f}x FEWER bytes" if ratio >= 1
                   else f"replication would move {1/ratio:.1f}x fewer bytes")
        print(f"\n  ratio: {verdict} at this configuration.")
    print("\n  The asymmetry that matters: the all-to-all carries TOKENS, so it does")
    print("  not depend on E at all. The all-reduce carries EXPERT GRADIENTS, so it")
    print("  grows linearly with E. Add experts and the cost you pay stays put while")
    print("  the cost you avoided climbs.")

    # ------------------------------------------------------------------ 4
    print(f"\n{LINE}\n4. THE CROSSOVER\n{LINE}")
    print("  Set the two equal. The (P-1)/P factor and d_model cancel from both")
    print("  sides, leaving:")
    print("\n      T* = 2 * E * d_ff / k")
    print(f"\n  T* = {fmt(r['crossover_T'])} tokens per process.   You run T = {T}.")
    if T < r["crossover_T"]:
        print("  You are below the crossover, so expert parallelism moves fewer bytes.")
    else:
        print("  You are ABOVE the crossover — the batch is large enough that shipping")
        print("  activations costs more than shipping expert gradients would.")
    print("\n  For Mixtral-like numbers (E=8, d_ff=14336, k=2) T* is about 114,688")
    print("  tokens per process — far more than anyone runs. So in practice expert")
    print("  parallelism essentially always wins on byte volume, and the real limits")
    print("  are latency and load balance rather than bandwidth.")

    # ------------------------------------------------------------------ 5
    print(f"\n{LINE}\n5. SWEEP OVER P   (E, T, k, widths held fixed)\n{LINE}")
    print(f"  {'P':>4} {'E/P':>6} {'params/proc':>13} {'a2a bytes':>12} {'all-reduce if repl.':>21}")
    for p in [x for x in (1, 2, 4, 8, 16, 32) if E % x == 0]:
        s = analytic(E, p, d, ff, T, k, b)
        print(f"  {p:>4} {s['experts_per_proc']:>6} {fmt(s['expert_params_per_proc']):>13}"
              f" {fmt(s['a2a_bytes']):>12} {fmt(s['allreduce_bytes']):>21}")
    print("\n  params/process HALVES every time P doubles          <- the memory win")
    print(f"  a2a bytes rise but are bounded by {fmt(r['a2a_ceiling'])}         <- bounded, not linear")
    print("  the avoided all-reduce climbs towards its own ceiling too")

    # ------------------------------------------------------------------ 6
    print(f"\n{LINE}\n6. SWEEP OVER E   (P, T, k, widths held fixed)\n{LINE}")
    print(f"  {'E':>4} {'E/P':>6} {'params/proc':>13} {'FLOPs/token':>13} {'a2a bytes':>12}")
    for e in (P, P * 2, P * 4, P * 8, P * 16):
        s = analytic(e, P, d, ff, T, k, b)
        print(f"  {e:>4} {s['experts_per_proc']:>6} {fmt(s['expert_params_per_proc']):>13}"
              f" {fmt(s['flops_per_token_fwd']):>13} {fmt(s['a2a_bytes']):>12}")
    print("\n  This is the table that makes the case. As E grows 16x:")
    print("    - FLOPs per token:  UNCHANGED")
    print("    - all-to-all bytes: UNCHANGED")
    print("    - params per process: grows, but only as E/P — add processes to hold it")
    print("  Model capacity scales while per-token cost and communication do not.")

    # ------------------------------------------------------------------ 7
    print(f"\n{LINE}\n7. ON WALL-CLOCK, HONESTLY\n{LINE}")
    print("""  Measured on one laptop, more ranks make this SLOWER, and that is expected:

    - The P "processes" time-slice the same few CPU cores. You are dividing
      one machine, not adding machines. There is no extra hardware to exploit.
    - gloo over loopback has a real fixed cost per message, and these tensors
      are tiny, so you measure latency and never reach bandwidth.
    - The experts are small (d_ff=32), so there is almost no arithmetic to
      overlap the transfer with.

  On P real GPUs the same code moves the same NUMBER OF BYTES over a link that
  is orders of magnitude faster, and the expert matmuls become large enough to
  hide the transfer behind. The byte counts above are hardware-independent;
  only the time to move them changes.

  Reporting these numbers with the reason is more useful than omitting them.""")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Scaling analysis for expert parallelism")
    p.add_argument("--experts", type=int, default=8, help="E — total experts")
    p.add_argument("--ranks", type=int, default=4, help="P — processes")
    p.add_argument("--d-model", type=int, default=16, help="width of a token vector")
    p.add_argument("--d-ff", type=int, default=32, help="width of an expert's middle layer")
    p.add_argument("--tokens", type=int, default=12, help="T — tokens per process per step")
    p.add_argument("--top-k", type=int, default=1, help="k — experts per token")
    p.add_argument("--bytes", type=int, default=4, help="b — bytes per number (4=fp32, 2=bf16)")
    report(p.parse_args())


if __name__ == "__main__":
    main()
