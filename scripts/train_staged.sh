#!/usr/bin/env bash
# The midterm timeline as runs: vision only -> + magnetometry -> + gravimetry (lat/lon regression),
# then the coarse cell classifier that feeds the coarse-to-fine pipeline (btp/c2f.py),
# then phase 2: retrieval fine-tuning and the full evaluation (scripts/phase2.sh).
# Each run is evaluated on the held-out test lines, flown in simulation and run through c2f.
# Re-runnable: finished stages are skipped and an interrupted one resumes from last.pt.
# Extra arguments are passed to btp.train as --set overrides (e.g. train.epochs=10).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs

# name | --set overrides | c2f priors
STAGES=(
  "vision|modalities=[vision]|vision"
  "vision+mag|modalities=[vision,mag]|fused"
  "vision+mag+grav|modalities=[vision,mag,grav]|fused"
  "coarse|modalities=[vision,mag,grav] model.head=classify|fused vision"
)

for spec in "${STAGES[@]}"; do
  IFS='|' read -r name sets priors <<< "$spec"
  if [ -f "runs/$name/DONE" ]; then continue; fi
  # Pick up imagery years whose download finished since the last stage.
  scripts/sync_data.sh > /dev/null
  $PY -m btp.prepare_imagery 2>&1 | grep -v Warn | grep -v exists >> runs/staged.log || true
  echo "=== $name $(date '+%F %T') ===" | tee -a runs/staged.log
  # shellcheck disable=SC2086
  $PY -m btp.train --name "$name" --resume --set $sets "$@" 2>&1 \
    | grep --line-buffered -v Warning | tee -a "runs/$name.log"
  $PY -m btp.evaluate "runs/$name/best.pt" 2>&1 | grep -v Warning | tee -a "runs/$name.log" | tee -a runs/staged.log
  $PY -m btp.simulate "runs/$name/best.pt" 2>&1 | tail -2 | tee -a runs/staged.log
  for p in $priors; do
    $PY -m btp.c2f "runs/$name/best.pt" --prior "$p" 2>&1 | grep -v Warning | tr '\r' '\n' | grep -v "^  [0-9]*/" \
      | tee -a "runs/$name.log" | tee -a runs/staged.log
  done
  date '+%F %T' > "runs/$name/DONE"
done

# Phase 2: retrieval fine-tuning (starts from the classifier's ViT), then all phase-2 evaluations.
if [ ! -f runs/retrieval/DONE ]; then
  echo "=== retrieval $(date '+%F %T') ===" | tee -a runs/staged.log
  $PY -m btp.train_retrieval --init runs/coarse/best.pt 2>&1 | grep --line-buffered -v Warning | tee -a runs/retrieval.log
  date '+%F %T' > runs/retrieval/DONE
fi
if [ ! -f runs/phase2/DONE ]; then
  echo "=== phase2 $(date '+%F %T') ===" | tee -a runs/staged.log
  scripts/phase2.sh 2>&1 | tee -a runs/phase2.log
  date '+%F %T' > runs/phase2/DONE
  grep -A200 "^# Phase 2 results" runs/phase2/report.md >> runs/staged.log || true
fi
