#!/usr/bin/env bash
# The single command reviewers should need.
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
[ -x "$PY" ] || { echo "No .venv found — run ./setup.sh first."; exit 1; }

echo
echo "############ training demo (4 ranks) ############"
"$PY" train.py --ranks 4 --steps 10

echo
echo "############ scaling analysis ############"
"$PY" bench.py
