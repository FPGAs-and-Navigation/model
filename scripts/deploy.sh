#!/usr/bin/env bash
# Push the code (not data/runs/venv) from this checkout to ~/BTP_impl on the training host.
set -euo pipefail
HOST="${HOST:-kratos}"
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .git --exclude .venv --exclude data --exclude runs \
  --exclude __pycache__ ./ "$HOST:BTP_impl/"
echo "deployed to $HOST:~/BTP_impl"
