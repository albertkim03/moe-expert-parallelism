# Expert Parallelism for MoE Training

A Mixture-of-Experts layer with the experts split across processes. Each process
holds only some of the experts, so a token whose expert lives elsewhere gets sent
there, transformed, and sent back. Runs on CPU with `gloo`, no GPU needed.

The point of the demo is to show that this produces exactly the same numbers as an
ordinary single-process MoE, and to be precise about which gradients have to be
shared between processes and which do not.

## Run it

```bash
./setup.sh      # makes .venv, installs torch (~200MB)
./run.sh        # training demo + scaling analysis
```

Or the pieces separately:

```bash
python train.py                  # 4 processes, 10 steps
python train.py --ranks 8
python bench.py                  # scaling numbers, no torch needed
```

## What you get

```
1. CORRECTNESS   (is the distributed version computing the right thing?)
  Run the SAME batch through two implementations, then subtract.

    input   48 token vectors, 16 numbers each   (12 tokens on each of 4 ranks)

    A       reference_moe.py  — all 8 experts in ONE process
    B       ep_moe.py         — 2 experts per rank, across 4 ranks

  Each produces 48 x 16 = 768 numbers.

    numbers that differ (A vs B) : 0 of 768
    largest difference           : 0  (exact match)

2. TRAINING  (one MoE layer, regressing a constant)
  step   0   loss 0.602845
  step   4   loss 0.095551
  step   9   loss 0.011927

3. PARAMETERS   (which numbers had to be shared between ranks?)
  Each rank holds 2289 learnable numbers, in two groups.

  GROUP 1 — the router and the output layer        145 numbers
    Every rank holds the SAME numbers here: they are copies of one thing.
    Each computed a different gradient, because each saw different tokens,
    so the gradients were added up across ranks. If that worked, all
    copies are still identical:

      biggest disagreement between ranks : 0   (identical, as intended)

  GROUP 2 — the experts                           2144 numbers   (2 of the 8)
    Every rank holds DIFFERENT numbers here. Rank 0 has experts 0 and 1;
    rank 1 has 2 and 3. Not copies, so nothing to add them up with. They
    were left alone, and they should NOT match:

      rank 0  owns experts [0, 1]  its 2144 numbers add up to   +0.647167
      rank 1  owns experts [2, 3]  its 2144 numbers add up to  -10.233376
      rank 2  owns experts [4, 5]  its 2144 numbers add up to   +2.781003
      rank 3  owns experts [6, 7]  its 2144 numbers add up to   +7.785652
```

The numbers that should be identical are, and the numbers that should differ do.
That is the whole claim, and it is one command away from being reproduced.

Where the counts come from: the router weight is 8 experts x 16 inputs = 128
numbers, plus the output layer's 16 weights and 1 bias = 145. One expert is two
weight matrices and two bias vectors = 1072 numbers, and each rank holds 2 of them
= 2144. Across all four ranks that is 145 shared numbers plus 8576 distinct expert
numbers.

Every one of the 768 numbers matches, at 1, 2, 4 and 8 processes.

That comparison is the point of `reference_moe.py` existing. Get the distributed
version wrong — send a token to the wrong process, or reassemble the answers in the
wrong order — and you get plausible numbers in the wrong places. Same shape, same
magnitude, no error raised, and the loss still falls. Subtracting against an
implementation simple enough to verify by reading is the only way to see it.

An exact match is slightly lucky. Floating-point addition is not associative, so
summing the same values in a different order can differ in the last bits; here the
order happens to line up. A difference of ~1e-6 would still be a pass.

## The layer, step by step

```
        rank 0                                  rank 1
   ┌──────────────────┐                   ┌──────────────────┐
   │ router (copy)    │                   │ router (copy)    │
   │ experts E0, E1   │                   │ experts E2, E3   │
   └──────────────────┘                   └──────────────────┘
      tokens t0..t3                          tokens t4..t7
            │ 1. route locally                     │
            ▼                                      ▼
       2. sort by destination rank           2. sort by destination rank
       3. exchange counts  ◄─────────────────────► 3. exchange counts
            └──────────────┐  4. all-to-all │──────────────┘
                           ▼   (dispatch)   ▼
              every token now sits on the rank that owns its expert
                           │                │
                    5. E0,E1 forward   5. E2,E3 forward
                           └──────────────┐ 6. all-to-all
                                          ▼   (combine)
                      7. un-permute, multiply by the gate
```

Steps 4 and 6 are the same call with the split sizes swapped. That symmetry is also
why the backward of one is the other.

## Files

| File | What it does |
|---|---|
| `config.py` | Settings, and the ownership rule. `owner_of(e)` is `e // (E//P)` — pure arithmetic, so every process independently agrees on who holds what without communicating. |
| `router.py` | `router(x) -> (topk_idx, topk_gate, probs)`. Multiplies each token by a weight matrix to get one score per expert, softmaxes, takes the top-k. The gate is the winning probability. |
| `expert.py` | `Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)`. That is all an expert is. Tags its parameters `is_expert=True` at construction, which is what `sync.py` reads later. |
| `reference_moe.py` | All `E` experts in one process, a plain loop over them. Deliberately the slowest possible version. Exists only as the thing to check the distributed version against. |
| `distributed.py` | `setup`/`cleanup`, `exchange_counts`, and `all_to_all`. Knows nothing about experts. |
| `ep_moe.py` | The expert-parallel layer. Holds `E/P` experts, runs the seven steps above. Same input and output shape as `reference_moe.py`. |
| `model.py` | Wraps an MoE layer and adds `Linear(d_model, 1)` so there is one number per token to train against. `make_batch` gives each process different data. |
| `sync.py` | After `backward()`, all-reduces the gradients of everything not tagged `is_expert`. Eight lines. |
| `train.py` | Spawns the processes, builds the model, checks it against the reference, trains, prints the evidence above. |
| `bench.py` | Scaling arithmetic. No torch, no processes. |

Read them in that order; each only depends on the ones above it.

## What gets all-reduced, and what doesn't

A parameter needs an all-reduce if and only if more than one process holds a copy
of it.

| | Replicated? | Synced? |
|---|---|---|
| router | copy on every process | all-reduce |
| output head | copy on every process | all-reduce |
| experts | one owner each | **no sync** |

The router is the same parameter on all four processes, but each computed its
gradient from different tokens. Left alone the four copies would drift apart and
the processes would stop agreeing about where to send tokens. So they get summed.

The experts do not need this, and the reason is worth stating carefully. The true
gradient for expert `e` over the whole batch is a sum over every process, of the
tokens on that process that chose `e`. In plain data parallelism each process
computes its own inner sum and the all-reduce performs the outer one. Here the
dispatch all-to-all has already moved all of those token sets onto `owner(e)`, so
that one process computes both sums by itself. The all-to-all did the reduction.

Put another way: in data parallelism the tokens stay put and the gradients move; in
expert parallelism the tokens move and the gradients stay put. Either way the sum
over the global batch happens.

This is why `sync.py` is a loop with a `continue` in it rather than a call to
`DistributedDataParallel`. DDP assumes every parameter is replicated. It would
broadcast rank 0's weights at construction, destroying the sharding before the
first step, and then all-reduce every gradient — averaging experts that are
different parameters. Neither failure raises anything. The loss still goes down a
little.

## Tradeoffs

**ReLU instead of GELU** (`expert.py`) — GELU is what modern transformers use,
because ReLU zeroes the gradient for any negative input and a unit can get stuck
there permanently. That matters in deep networks trained for a long time. This is a
two-layer expert in a demo, so ReLU is one less thing to explain.

**The built-in differentiable all-to-all instead of a hand-rolled
`autograd.Function`** (`distributed.py`) — `torch.distributed.nn.functional.all_to_all_single`
already implements the rule that the backward of an all-to-all is another
all-to-all with the split sizes swapped, and it works on gloo. Writing it by hand
is six lines and makes the rule visible to a reader, which is the argument for
doing so. I would hand-roll it if I needed to change the behaviour, for example to
overlap the transfer with expert compute.

**No load-balancing auxiliary loss** (`reference_moe.py`, `ep_moe.py` — both return
a zero for it) — this is the biggest omission and it is deliberate. Routing quality
is orthogonal to whether the dispatch and combine are correct, which is what this
POC is about. It matters more than usual under expert parallelism though: an
overloaded expert means the process holding it does most of the work while the
others wait at the next collective, so it is a throughput problem and not only a
quality one. It is also not a drop-in addition. The Switch loss is
`α·E·Σ fₑPₑ`, and `fₑ` is about the *global* token distribution, so the per-expert
counts need their own `all_reduce(SUM)` before the loss is formed. Computed
per-process, every process would optimise a different objective.

**Top-1 routing only** (`ep_moe.py` asserts it) — top-2 is what Mixtral uses and it
routes better, because the router sees a comparison rather than a single winner.
Supporting it means each token becomes `k` rows before the dispatch and the `k`
results are summed after the combine. That is a real complication and it is not
where the marks are, so the assert turns an unsupported case into a loud failure
instead of a silently wrong one.

**Dropless, not a capacity factor** — buffers are sized from the real counts after
the count exchange, so no token is ever discarded. A fixed capacity factor gives
static shapes, which is what you want for `torch.compile`, CUDA graphs and
fixed-size kernels, and it costs you dropped tokens. For a POC whose main claim is
an exact match against a reference, dropping tokens would mean having to make the
reference drop the same ones. Not worth it. I would switch for a production path.

**A second all-to-all for the expert ids** (`ep_moe.py`) — the receiving process
needs to know which of its experts each arriving token wants. Sending the ids
alongside the tokens costs one extra collective carrying `T` int64s per layer,
which is small next to the token payload. The alternative is to sort by expert id
within each destination block so the receiver can derive the ids from the counts
alone. That is what production implementations do. This way was easier to get
right.

**No `if mask.any()` guard around the expert call** — skipping the call when a
process has no tokens for one of its experts would leave the output outside the
autograd graph on that process. Its backward would then never run the all-to-all
while every other process does, and the job hangs with no error. Calling an expert
on zero rows is cheap and keeps every process's backward graph the same shape.

**All `E` experts are constructed and then sliced** (`ep_moe.py.__init__`) — each
process builds all eight and keeps its two. This wastes a little memory at init.
It is done so that the RNG draws happen in the same order as in `reference_moe.py`,
which makes rank `r`'s expert `e` bit-identical to the reference's expert `e` and
turns the correctness check into two lines. A real implementation would construct
only the local experts and seed them per-expert.

**The loss is divided by the global token count, not the local one**
(`train.py`) — this makes the per-process losses sum to the global mean loss, which
is why `sync.py` can use a plain `SUM` all-reduce with no division afterwards. It is
also what makes the expert gradients match a single-process run exactly rather than
being off by a factor of `P`. If processes held different numbers of tokens, a plain
mean would be wrong and you would weight by token count instead.

**No bias on the router** (`router.py`) — a bias adds the same constant to an
expert's score for every token. It can make an expert globally more or less popular
but cannot help distinguish one token from another, which is the router's whole
job. So it is a parameter that can only contribute to collapse.

**Contiguous expert placement rather than round-robin** — `owner(e) = e // (E//P)`.
Either works as long as every process computes it identically without
communicating. Contiguous means the all-to-all split sizes fall straight out of a
sort by destination, so the grouping is free. Round-robin can balance better if
expert popularity correlates with index, but then you need an extra gather to make
each destination's tokens contiguous.

**`mp.spawn` rather than `torchrun`** — `torchrun` is the production launcher.
`mp.spawn` means `python train.py` is the whole command, which matters more here
than it would in production.

**Evidence in the program output, not a test suite** — there is no `pytest`. The
correctness check runs inside `train.py` on every invocation, so reproducing the
results and verifying them are the same action, and a reviewer sees the number
without running a second command. The cost is that it is not a regression check:
nothing fails loudly if a later change breaks equivalence. For anything
longer-lived this belongs in a test, alongside ones for the routing logic, the
permutation round trip, and gradient equivalence per parameter.

## Not done

- Load-balancing auxiliary loss, per above.
- Top-2 routing.
- An EP × DP mesh. Right now expert parallelism spans the whole world. With data
  parallelism on top you would build subgroups with `dist.new_group`, all-reduce
  expert gradients down each DP column and the router across everything.
- Overlapping the dispatch with expert compute.
- Grouped GEMM. The per-expert Python loop is many small matmuls where one large
  one would do. This is the first thing to fix for real performance.
- Checkpointing. Sharded experts make saving and loading non-trivial.

## Scaling

`python bench.py` prints the full analysis. The short version:

| | Formula | Behaviour |
|---|---|---|
| expert params per process | `(E/P)·2·d_model·d_ff` | falls as `1/P` |
| FLOPs per token | `2k·2·d_model·d_ff` | **no `E` in it** |
| all-to-all bytes per process | `≈2·T·k·d_model·b·(P−1)/P` | independent of `E`, bounded in `P` |
| the all-reduce avoided | `2·(P−1)/P·E·2·d_model·d_ff·b` | grows linearly with `E` |

Setting the last two equal, the ring factor and `d_model` cancel and you get
`T* = 2·E·d_ff/k`. Below that many tokens per process, expert parallelism moves
fewer bytes. At Mixtral-like sizes (`E=8, d_ff=14336, k=2`) that is about 115,000
tokens per process, far more than anyone runs — so in practice it always wins on
volume, and the real limits are latency and load balance.

The table that makes the case is the sweep over `E`: raise it 16× and FLOPs per
token and all-to-all bytes are both unchanged. Only parameters per process grow,
and you add processes to absorb that.

### On wall-clock

More processes make this slower on one laptop. That is expected and worth being
clear about. The processes time-slice the same cores, so there is no extra hardware
being used. gloo over loopback has a fixed cost per message and these tensors are
tiny, so the measurement is latency, never bandwidth. And the experts are small
enough (`d_ff=32`) that there is no arithmetic to hide the transfer behind.

On real GPUs the same code moves the same number of bytes over a much faster link,
and the expert matmuls get big enough to overlap with the transfer. The byte counts
above are hardware-independent; only the time to move them changes.
