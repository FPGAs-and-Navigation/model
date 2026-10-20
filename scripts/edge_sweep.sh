#!/usr/bin/env bash
# Edge tolerance sweep: particle-filter accuracy for each image matcher at several fix spacings
# (a fix every 1/2/4/8 steps of 50 m), with the edge settings. Results: runs/edge/track_*.json
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
EDGE="track.kernel=numba track.particles=1000 track.model_every=10 track.estimate_bias=true"
run() {  # run <tag> <extra --set args...>
  local tag="$1"; shift
  [ -f "runs/edge/track_$tag.json" ] && { echo "have $tag"; return; }
  $PY -m btp.track runs/coarse/best.pt --retrieval runs/retrieval/best.pt --outdir edge --plots 1 --tag "$tag" \
    --set $EDGE "$@" 2>&1 | grep -E "^$tag:" || echo "FAILED $tag"
}
for every in 1 2 4 8; do
  run "loftr_s1.0_e$every"  dsmac.method=loftr track.image_every=$every
  run "loftr_s0.75_e$every" dsmac.method=loftr dsmac.scale=0.75 track.image_every=$every
  run "loftr_s0.5_e$every"  dsmac.method=loftr dsmac.scale=0.5 track.image_every=$every
  run "ldsmac_e$every"      dsmac.method=ldsmac track.image_every=$every
done
[ -f runs/edge/track_noimage.json ] || $PY -m btp.track runs/coarse/best.pt --retrieval runs/retrieval/best.pt \
  --outdir edge --plots 0 --tag noimage --no-image --set $EDGE 2>&1 | grep -E "^noimage:"
echo SWEEP DONE
