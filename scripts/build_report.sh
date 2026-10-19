#!/usr/bin/env bash
# Build report/main.pdf from the current results.
#   on kratos (where runs/ lives):  scripts/build_report.sh
#   from a laptop checkout:         scripts/build_report.sh --remote   (builds on kratos, copies back)
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-kratos}"
if [ "${1:-}" = "--remote" ]; then
  scripts/deploy.sh >/dev/null
  ssh "$HOST" 'cd ~/BTP_impl && scripts/build_report.sh'
  rsync -az "$HOST:BTP_impl/report/generated/" report/generated/
  rsync -az "$HOST:BTP_impl/report/figures/generated/" report/figures/generated/
  scp -q "$HOST:BTP_impl/report/main.pdf" report/main.pdf
  echo "report/main.pdf updated"
  exit 0
fi
.venv/bin/python -m btp.latex_tables
G=report/figures/generated
mkdir -p "$G"
cp_if() { [ -f "$1" ] && cp "$1" "$G/$2" || true; }
cp_if data/processed/qc/geophysics.png geophysics.png
cp_if data/processed/qc/naip_2016.jpg naip_2016.jpg
cp_if "$(ls runs/vision+mag+grav/sim_line*.png 2>/dev/null | head -1)" sim_regression.png
cp_if "$(ls runs/phase2/track_full_flight*.png 2>/dev/null | head -1)" track_full.png
cp_if "$(ls runs/phase3/track_best_flight*.png 2>/dev/null | head -1)" track_best.png
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/main.pdf"
