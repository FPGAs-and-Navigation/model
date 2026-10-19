#!/usr/bin/env bash
# Phase 3: literature-driven fixes for the weaknesses measured in phases 1-2, each tested on its own
# and then combined. Writes runs/phase3/report.md.
#   - OGM balanced training + heavier vision dropout (Peng et al. CVPR'22): does UDML stop collapsing
#     onto vision, and do its weights now react correctly to a noisy sensor?
#   - SeqSLAM-style motion-compensated sequence prior: fewer no-fix frames in single-shot mode
#   - instance colour alignment to the reference map: smaller imagery-year gap
#   - more candidate cells (top-30)
#   - particle filter that estimates the dead-reckoning bias (marginalised-PF idea), incl. sparse fixes
# Steps are skipped when their output exists, so the script can be re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=runs/phase3
mkdir -p "$OUT"
COARSE=runs/coarse/best.pt
BAL=runs/coarse_balanced/best.pt
RETR=runs/retrieval/best.pt
run() {  # run <output file> <command...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "have $out"; return; fi
  echo "=== $* ($(date '+%F %T'))"
  "$@" 2>&1 | grep -v Warning | tr '\r' '\n' | grep -v '^  [0-9]*/' || true
  [ -f "$out" ] || echo "MISSING $out after: $* (continuing; the report will omit it)"
}

# 1. Balanced classifier: OGM + vision dropped 40% of the time.
if [ ! -f runs/coarse_balanced/DONE ]; then
  $PY -m btp.train --name coarse_balanced --resume --set "modalities=[vision,mag,grav]" model.head=classify \
    train.ogm_alpha=0.1 "train.mod_dropout={vision: 0.4, mag: 0.15, grav: 0.15}" 2>&1 \
    | grep --line-buffered -v Warning | tee -a runs/coarse_balanced.log
  date '+%F %T' > runs/coarse_balanced/DONE
fi
run runs/coarse_balanced/test_metrics.json $PY -m btp.evaluate $BAL

# 2. Single-shot pipeline variants (outputs next to the checkpoint).
C="$PY -m btp.c2f"
run runs/coarse/c2f_p3_base_metrics.json      $C $COARSE --retrieval $RETR --tag p3_base
run runs/coarse/c2f_p3_seq5_metrics.json      $C $COARSE --retrieval $RETR --tag p3_seq5 --set coarse.seq_len=5
run runs/coarse/c2f_p3_color_metrics.json     $C $COARSE --retrieval $RETR --tag p3_color --set img.color_align=instance
run runs/coarse/c2f_p3_top30_metrics.json     $C $COARSE --retrieval $RETR --tag p3_top30 --set coarse.topk=30
run runs/coarse/c2f_p3_all_metrics.json       $C $COARSE --retrieval $RETR --tag p3_all \
    --set coarse.seq_len=5 img.color_align=instance coarse.topk=30
run runs/coarse_balanced/c2f_p3_all_metrics.json $C $BAL --retrieval $RETR --tag p3_all \
    --set coarse.seq_len=5 img.color_align=instance coarse.topk=30

# 3. Particle filter variants.
T="$PY -m btp.track"
O="--outdir phase3"
run "$OUT/track_base.json"            $T $COARSE --retrieval $RETR $O --tag base
run "$OUT/track_bias.json"            $T $COARSE --retrieval $RETR $O --tag bias --set track.estimate_bias=true
run "$OUT/track_sparse.json"          $T $COARSE --retrieval $RETR $O --tag sparse --set track.image_every=4 track.dr_heading_bias_deg=3
run "$OUT/track_sparse_bias.json"     $T $COARSE --retrieval $RETR $O --tag sparse_bias \
    --set track.image_every=4 track.dr_heading_bias_deg=3 track.estimate_bias=true
run "$OUT/track_balanced.json"        $T $BAL --retrieval $RETR $O --tag balanced --set track.estimate_bias=true
run "$OUT/track_balanced_equal.json"  $T $BAL --retrieval $RETR $O --tag balanced_equal --weights equal --set track.estimate_bias=true
run "$OUT/track_balanced_magnoise_udml.json"  $T $BAL --retrieval $RETR $O --tag balanced_magnoise_udml --mag-noise 2 \
    --set track.estimate_bias=true
run "$OUT/track_balanced_magnoise_equal.json" $T $BAL --retrieval $RETR $O --tag balanced_magnoise_equal --mag-noise 2 \
    --weights equal --set track.estimate_bias=true
run "$OUT/track_best.json"            $T $BAL --retrieval $RETR $O --tag best \
    --set track.estimate_bias=true img.color_align=instance

$PY -m btp.phase2_report --phase3
