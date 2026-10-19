#!/usr/bin/env bash
# The midterm timeline as three runs: vision only -> + magnetometry -> + gravimetry.
# Each run is evaluated on the held-out test lines and flown in simulation.
# Usage (on kratos, inside tmux):  scripts/train_staged.sh [extra --set overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs
for stage in "vision" "vision,mag" "vision,mag,grav"; do
  name="${stage//,/+}"
  echo "=== $name ===" | tee -a runs/staged.log
  $PY -m btp.train --name "$name" --set "modalities=[$stage]" "$@" 2>&1 | grep --line-buffered -v Warning | tee "runs/$name.log"
  $PY -m btp.evaluate "runs/$name/best.pt" 2>&1 | grep -v Warning | tee -a "runs/$name.log" | tee -a runs/staged.log
  $PY -m btp.simulate "runs/$name/best.pt" 2>&1 | tail -2 | tee -a runs/staged.log
done
