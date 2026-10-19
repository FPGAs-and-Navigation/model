#!/usr/bin/env bash
# Phase 2 evaluation: matchers, retrieval, single-shot c2f and the particle-filter tracker,
# with ablations, then runs/phase2/report.md. Needs runs/coarse/best.pt and runs/retrieval/best.pt.
# Each step is skipped if its output already exists, so the script can be re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=runs/phase2
mkdir -p "$OUT"
EASY="dsmac.rot_sigma=2.0 dsmac.scale_sigma=0.03 dsmac.tilt_deg=0.0"
COARSE=runs/coarse/best.pt
RETR=runs/retrieval/best.pt
run() {  # run <output file> <command...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "have $out"; return; fi
  echo "=== $* ($(date '+%F %T'))"
  "$@" 2>&1 | grep -v Warning | tr '\r' '\n' | grep -v '^  [0-9]*/' || true
  [ -f "$out" ] || echo "MISSING $out after: $* (continuing; the report will omit it)"
}

# Matchers at the true position (val lines), easy and realistic views.
run "$OUT/matchers_tilt0.0_rot2.0.json" $PY -m btp.matchers --n 300 --set $EASY
run "$OUT/matchers_tilt10.0_rot8.0.json" $PY -m btp.matchers --n 300

# Retrieval: zero-shot ViT-S, zero-shot ViT-B (capacity check), fine-tuned.
run "$OUT/retrieval_zeroshot_vit_small_patch14_dinov2.json" $PY -m btp.retrieval --zero-shot
run "$OUT/retrieval_zeroshot_vit_base_patch14_dinov2.json" $PY -m btp.retrieval --zero-shot --vit vit_base_patch14_dinov2.lvd142m
run "$OUT/retrieval_finetuned.json" $PY -m btp.retrieval "$RETR"

# Single-shot coarse -> TERCOM -> DSMAC (results land next to the checkpoint).
run runs/coarse/c2f_fused_ncc_real_metrics.json $PY -m btp.c2f $COARSE --tag fused_ncc_real --set dsmac.method=ncc
run runs/coarse/c2f_fused_loftr_real_metrics.json $PY -m btp.c2f $COARSE --tag fused_loftr_real
run runs/coarse/c2f_fused+retrieval_loftr_real_metrics.json $PY -m btp.c2f $COARSE --retrieval $RETR --tag fused+retrieval_loftr_real
run runs/coarse/c2f_fused+retrieval_loftr_easy_metrics.json $PY -m btp.c2f $COARSE --retrieval $RETR --tag fused+retrieval_loftr_easy --set $EASY

# Particle filter: the full combination, then one change at a time.
T="$PY -m btp.track $COARSE"
run "$OUT/track_full.json"            $T --retrieval $RETR --tag full
run "$OUT/track_full_easy.json"       $T --retrieval $RETR --tag full_easy --set $EASY
run "$OUT/track_no_retrieval.json"    $T --tag no_retrieval
run "$OUT/track_ncc.json"             $T --retrieval $RETR --tag ncc --set dsmac.method=ncc
run "$OUT/track_no_image.json"        $T --retrieval $RETR --no-image --tag no_image
run "$OUT/track_no_mag.json"          $T --retrieval $RETR --no-mag --tag no_mag
run "$OUT/track_no_grav.json"         $T --retrieval $RETR --no-grav --tag no_grav
run "$OUT/track_equal_weights.json"   $T --retrieval $RETR --weights equal --tag equal_weights
run "$OUT/track_magnoise_udml.json"   $T --retrieval $RETR --mag-noise 2 --tag magnoise_udml
run "$OUT/track_magnoise_equal.json"  $T --retrieval $RETR --mag-noise 2 --weights equal --tag magnoise_equal
run "$OUT/track_vision_regressor.json" $PY -m btp.track runs/vision/best.pt --prior vision --retrieval $RETR --tag vision_regressor

$PY -m btp.phase2_report
