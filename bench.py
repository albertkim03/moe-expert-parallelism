#!/usr/bin/env python3
"""Scaling evidence.

The brief pre-excuses you from wall-clock speedups:

    "since you won't necessarily have access to multiple GPUs, wall-clock
     improvements may not be possible. We care a lot about your analysis of how
     and why this scales"

So the deliverable here is an ARGUMENT supported by numbers, not a benchmark.
Two things, side by side:

  (a) an ANALYTICAL model — closed-form formulas for memory, FLOPs and bytes
  (b) MEASURED numbers where measuring is meaningful (parameter counts, tensor
      sizes, token counts per rank), plus honest timings with a paragraph
      explaining why they look the way they do on a laptop

STEP 11 of the build plan.
"""

import argparse


def analytic_table(E: int, P: int, d_model: int, d_ff: int, T: int, k: int, bytes_per_elem: int = 4):
    """Closed-form scaling numbers. No torch needed — this is arithmetic.

    The formulas you are implementing (derive these in the README, do not just
    assert them):

      params per expert        2 * d_model * d_ff
      total expert params      E * 2 * d_model * d_ff
      expert params per rank   (E / P) * 2 * d_model * d_ff      <- falls as 1/P
      router params            E * d_model                        <- replicated, tiny

      FLOPs per token (fwd)    2 * k * 2 * d_model * d_ff         <- INDEPENDENT of E
                                                                     this is the whole point

      all-to-all bytes per rank per layer per step
                               ~ 2 * T * k * d_model * bytes_per_elem
                               (dispatch + combine; the (P-1)/P factor is ~1)
                                                                  <- ~flat in P and E

      if you REPLICATED experts instead and all-reduced their grads:
                               2 * (P-1)/P * E * 2 * d_model * d_ff * bytes_per_elem
                                                                  <- GROWS with E

    The crossover — the point of the whole analysis — is where those last two are
    equal. Solve for T and you get the batch size beyond which expert
    parallelism stops paying. Report that number for your configuration.

    TODO(step 11): implement and return rows for printing.
    """
    raise NotImplementedError("TODO(step 11): implement the analytical model")


def sweep(**kwargs) -> None:
    """Print how the numbers move as you sweep P (and separately E).

    TODO(step 11): produce a small table, e.g.

        P    experts/rank   params/rank   a2a bytes/rank   all-reduce-if-replicated
        1        8            262,144         0                    0
        2        4            131,072      12,288              524,288
        4        2             65,536      12,288              786,432
        8        1             32,768      12,288              917,504

    Then say in words what the shape of each column means: params/rank halves
    with every doubling of P, the all-to-all traffic is flat, and the
    replication alternative grows. Plain text is fine — no plotting library.
    """
    raise NotImplementedError("TODO(step 11): implement the sweep")


def timing_note() -> str:
    """Be honest about laptop timings, in the output as well as the README.

    What you will actually observe on CPU + gloo: the all-to-all dominates, and
    adding ranks makes things SLOWER. All of that is expected. Say why:

      - every "rank" is a process on the same few cores, so you are
        time-slicing one machine, not adding hardware
      - gloo over loopback has real per-message latency, and the tensors here
        are tiny, so you are measuring latency and never reach bandwidth
      - the experts are small, so there is almost no compute to hide the
        communication behind
      - on 8 real GPUs with NVLink, the same code moves the same number of BYTES
        but over a link that is orders of magnitude faster, and the expert
        matmuls become large enough to overlap with the transfer

    Presenting bad timings WITH this explanation is much stronger than not
    reporting timings at all. It shows you know what you are measuring.

    TODO(step 11): return the text you want printed.
    """
    raise NotImplementedError("TODO(step 11): write the timing caveat")


def main() -> None:
    p = argparse.ArgumentParser(description="Scaling analysis for expert parallelism")
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=32)
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--top-k", type=int, default=1)
    args = p.parse_args()
    raise NotImplementedError("TODO(step 11): wire up the sweep and print it")


if __name__ == "__main__":
    main()
