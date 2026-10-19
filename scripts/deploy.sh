#!/usr/bin/env bash
# Push the code (not data/runs/venv) from this checkout to ~/BTP_impl on the training host.
set -euo pipefail
HOST="${HOST:-kratos}"
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .git --exclude .venv --exclude data --exclude runs \
  --exclude __pycache__ \
  --exclude report/generated --exclude report/figures/generated --exclude report/main.pdf \
  --exclude 'report/*.aux' --exclude 'report/*.log' --exclude 'report/*.fls' --exclude 'report/*.fdb_latexmk' \
  --exclude 'report/*.out' --exclude 'report/*.toc' --exclude 'report/*.bbl' --exclude 'report/*.blg' ./ "$HOST:BTP_impl/"
echo "deployed to $HOST:~/BTP_impl"
