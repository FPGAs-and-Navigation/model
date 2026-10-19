#!/usr/bin/env bash
# The midterm timeline as three runs: vision only -> + magnetometry -> + gravimetry.
# Each run is evaluated on the held-out test lines and flown in simulation.
# Re-runnable: finished stages are skipped and an interrupted one resumes from last.pt.
# Extra arguments are passed to btp.train as --set overrides (e.g. train.epochs=10).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs
for stage in "vision" "vision,mag" "vision,mag,grav"; do
  name="${stage//,/+}"
  if [ -f "runs/$name/DONE" ]; then continue; fi
  # Pick up imagery years whose download finished since the last stage.
  scripts/sync_data.sh > /dev/null
  $PY -m btp.prepare_imagery 2>&1 | grep -v Warn | grep -v exists >> runs/staged.log || true
  echo "=== $name $(date '+%F %T') ===" | tee -a runs/staged.log
  $PY -m btp.train --name "$name" --resume --set "modalities=[$stage]" "$@" 2>&1 \
    | grep --line-buffered -v Warning | tee -a "runs/$name.log"
  $PY -m btp.evaluate "runs/$name/best.pt" 2>&1 | grep -v Warning | tee -a "runs/$name.log" | tee -a runs/staged.log
  $PY -m btp.simulate "runs/$name/best.pt" 2>&1 | tail -2 | tee -a runs/staged.log
  date '+%F %T' > "runs/$name/DONE"
done
