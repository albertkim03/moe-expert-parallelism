# Expert Parallelism for MoE Training

> **This README is a SKELETON.** The headings and prompts below are the structure
> your submission should have — every `> _prompt_` line tells you what to write
> there, then delete the prompt. The brief says *"We care about how you present
> the idea as much as the code itself"*, so budget a full hour for this file.
>
> Delete this blockquote before submitting.

---

## What this is

> _One paragraph, no jargon, for a reader who has never heard of Mixture of
> Experts. Roughly: "an MoE layer has many small feed-forward networks and a
> router that sends each token to just one or two of them. Expert parallelism
> puts different experts on different machines and ships each token to the
> machine holding its expert. This repo demonstrates that on CPU with N
> processes."_

## Quick start

```bash
./setup.sh      # creates .venv, installs pinned deps
./run.sh        # tests + training demo + scaling analysis
```

> _Verify this works in a **fresh clone on a clean machine** before you submit.
> "We will just want to reproduce your results on our own devices" — if the
> first command fails, nothing else gets read._

Individual pieces:

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python train.py --ranks 4 --steps 10
.venv/bin/python bench.py
```

## The idea in one picture

> _An ASCII diagram of dispatch → compute → combine. Something like the sketch
> below, but make it yours and make it match your actual code._

```
        rank 0                                  rank 1
   ┌──────────────────┐                   ┌──────────────────┐
   │ router (replica) │                   │ router (replica) │
   │ experts E0, E1   │                   │ experts E2, E3   │
   └──────────────────┘                   └──────────────────┘
      tokens t0..t3                          tokens t4..t7
            │  route locally                       │
            ▼                                      ▼
       sort by destination rank              sort by destination rank
            └──────────────┐   all-to-all   ┌──────────────┘
                           ▼   (dispatch)   ▼
              every token now sits on the rank that owns its expert
                           │                │
                     E0,E1 forward     E2,E3 forward
                           └──────────────┐ all-to-all
                                          ▼ (combine)
                        outputs return home, un-permute, × gate
```

## Design

> _What you built and why it is shaped that way. Cover at least:_
>
> - _Module layout and why the reference implementation exists separately_
> - _Expert placement rule (`owner(e) = e // (E//P)`) and why contiguous_
> - _Dropless vs capacity factor — which you chose and why_
> - _How the token→expert-id mapping reaches the receiving rank_
> - _Whether you wrote your own `autograd.Function` for the all-to-all or used
>   `torch.distributed.nn.functional`, and why_
> - _Why `mp.spawn` rather than `torchrun` (or the reverse)_

## What gets all-reduced, and what doesn't

> _The brief names this explicitly, so give it its own section and be precise._
>
> _The rule: **a parameter needs an all-reduce if and only if more than one rank
> holds a copy of it.** Then work through it:_
>
> | Parameter | Replicated or sharded | Synced? |
> |---|---|---|
> | router | replicated on every rank | all-reduce, ÷ world_size |
> | experts | sharded, one owner each | **no sync** |
> | head / other non-expert | replicated | all-reduce, ÷ world_size |
>
> _Then the argument for why the expert gradient is already correct: the
> dispatch all-to-all delivered every token in the world that chose expert `e`
> onto `e`'s owner, so that rank's local gradient is already the sum over the
> global batch. In data parallelism tokens stay put and gradients move; in
> expert parallelism tokens move and gradients stay put._
>
> _Mention the DDP trap — wrapping this model in `DistributedDataParallel`
> silently averages the gradients of **different** experts across ranks._

## Correctness

> _Paste the actual output. Numbers, not adjectives._
>
> - _Forward: distributed vs single-process reference, max |Δ| = ..._
> - _Gradients: per-parameter max |Δ| = ..._
> - _Router weights identical across ranks after N steps (max |Δ| = 0.0)_
> - _Expert weights differ across ranks — **and why that is correct**_
> - _Your tolerance choice and why_

## Scaling

> _The brief pre-excuses wall-clock, so lead with the analysis._
>
> _Give the closed-form model:_
>
> - _expert params per rank = `(E/P) · 2 · d_model · d_ff` → falls as 1/P_
> - _FLOPs per token = `2 · k · 2 · d_model · d_ff` → **independent of E**_
> - _all-to-all bytes per rank ≈ `2 · T · k · d_model · bytes` → flat in P and E_
> - _the all-reduce you avoided = `2(P−1)/P · E · 2 · d_model · d_ff · bytes` → **grows with E**_
>
> _Then the crossover: set the last two equal and solve for T. Report the number
> for your config — that is the actual insight._
>
> _Then the measured table from `bench.py`, and an honest paragraph on why
> laptop timings get worse with more ranks (P processes time-slicing the same
> cores; gloo over loopback is latency-bound at these tensor sizes; tiny experts
> mean no compute to hide the transfer behind). Explaining bad numbers well beats
> omitting them._

## Tradeoffs and limitations

> _The section people skip. Do not skip it — naming your own limits reads as
> senior. Be specific about:_
>
> - _What you did not implement (top-k > 1? EP×DP mesh? capacity factor?)_
> - _Where this breaks: load imbalance and stragglers; all-to-all across slow
>   inter-node links; batch sizes large enough that activation traffic exceeds
>   the gradient traffic you avoided_
> - _What you would do with another day_

## Extending at scale

> _They said the follow-up digs into "how the approach would extend at scale".
> Have written answers ready for:_
>
> - _EP × DP meshes, and which group each collective runs in_
> - _Composing with FSDP/ZeRO and tensor parallelism, and the ordering
>   convention (EP nearest the fast interconnect)_
> - _Overlapping the all-to-all with expert compute_
> - _Grouped GEMM / MegaBlocks-style block-sparse kernels instead of a Python
>   loop over experts_
> - _What changes for inference rather than training_

## Repo layout

```
moe_ep/
  config.py          knobs (provided complete)
  expert.py          one expert = one small feed-forward network
  router.py          gating network: logits → softmax → top-k, + aux loss
  reference_moe.py   the whole layer in ONE process — the source of truth
  distributed.py     process groups, count exchange, differentiable all-to-all
  ep_moe.py          the same layer with experts sharded across ranks
  sync.py            which gradients get all-reduced, and which do not
  model.py           tiny model wrapping the MoE layer so it can be trained
train.py             the demo entry point
bench.py             scaling numbers
tests/
  conftest.py        multi-process test harness (provided complete)
  test_routing.py    single-process routing logic
  test_equivalence.py distributed forward == reference
  test_gradients.py  the all-reduce rules, demonstrated
```

## Notes for whoever reads this next

> _Optional, but a nice touch given they said they may share the codebase
> internally. A short "if you want to understand this in 10 minutes, read these
> three files in this order" pointer._
