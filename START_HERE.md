## What is already done for you

These are boilerplate, not the exercise:

- `moe_ep/config.py` — the config dataclass and the expert-ownership helpers
- `tests/conftest.py` — the multi-process test harness (`run_distributed`)
- `requirements.txt`, `setup.sh`, `run.sh`, `.gitignore`
- `README.md` — a skeleton with a prompt under every heading

## What you write

Everything else. Follow the `TODO(step N)` markers in this order:

| Step | File | What |
|-----:|------|------|
| 1–2 | — | `./setup.sh`, then `pytest tests/ -q` runs without import errors |
| 3 | `moe_ep/router.py` | routing + the load-balancing loss |
| 4 | `moe_ep/expert.py` | one expert FFN |
| 5 | `moe_ep/reference_moe.py` | the single-process layer — **your source of truth** |
| 6 | `moe_ep/distributed.py` | process groups, count exchange, differentiable all-to-all |
| 7 | `moe_ep/ep_moe.py` | **the expert-parallel layer** — the hard 90 minutes |
| 8 | `tests/test_equivalence.py` | distributed == reference |
| 9 | `moe_ep/model.py`, `train.py` | the training loop |
| 10 | `moe_ep/sync.py`, `tests/test_gradients.py` | what gets all-reduced |
| 11 | `bench.py` | scaling analysis |
| 12 | `README.md` | the write-up — budget a full hour |

Write step 5 **before** step 7. The reference implementation is what makes the
distributed one debuggable; without it you are guessing.

You should see `2 passed, 13 skipped` on a fresh scaffold. The 2 that pass are the
expert-ownership checks, which are already written. The 13 skips are your work —
each one's docstring is its specification, so read them as you go.


## Sanity check before you submit

- [ ] Fresh clone, `./setup.sh && ./run.sh` — works first try, exits 0
- [ ] No GPU assumed; runs on Linux and macOS
- [ ] The printed output alone tells the story (correctness, loss, param proof)
- [ ] README has design, tradeoffs, how-to-run, and the all-reduce explanation
- [ ] You can explain every line without notes
