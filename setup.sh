#!/usr/bin/env bash
# One-time setup. Creates a local virtualenv and installs pinned deps.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> creating .venv with $($PY -V)"
"$PY" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
echo "==> installing requirements (torch is a large download, be patient)"
./.venv/bin/pip install --quiet -r requirements.txt
echo "==> done. Run the demo with:  ./run.sh"
