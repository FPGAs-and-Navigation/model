#!/usr/bin/env bash
# Self-healing driver for kratos, run by cron every 10 min and at boot (entries tagged # btp-train).
# Install:  scripts/autorun.sh --install     Remove:  scripts/autorun.sh --uninstall
#  - copies newly verified NAIP years (CoW) and builds their mosaics,
#  - runs/resumes the staged training (scripts/train_staged.sh lists the stages),
#  - re-evaluates finished stages once the held-out 2022 imagery is ready,
#  - removes its cron entries when everything is done.
# Progress: runs/autorun.log, runs/staged.log, runs/<stage>.log
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TAG="# btp-train"

uninstall() { crontab -l 2>/dev/null | grep -v "# btp-" | crontab -; }   # also the phase-2 entry
case "${1:-}" in
  --install)
    uninstall
    (crontab -l 2>/dev/null
     echo "*/10 * * * * $ROOT/scripts/autorun.sh $TAG"
     echo "@reboot sleep 120 && $ROOT/scripts/autorun.sh $TAG") | crontab -
    echo "installed:"; crontab -l | grep "$TAG"; exit 0 ;;
  --uninstall) uninstall; echo "removed"; exit 0 ;;
esac

mkdir -p runs
exec 9>runs/.autorun.lock
flock -n 9 || exit 0
log() { echo "$(date '+%F %T') $*" >> runs/autorun.log; }
PY=.venv/bin/python
TEST_YEAR=$($PY -c "from btp import config; print(config.load()['test_year'])")

log "start"
scripts/sync_data.sh >> runs/autorun.log 2>&1
$PY -m btp.prepare_imagery 2>&1 | grep -v Warn >> runs/autorun.log

scripts/train_staged.sh > /dev/null 2>> runs/autorun.log
rc=$?
if [ "$rc" != 0 ]; then
  log "training failed (exit $rc); retrying on the next cron tick"; exit 1
fi

# Stages finished before the held-out year was ready were tested on a training year; redo them.
pending=0
for d in runs/*/; do
  d="${d%/}"
  [ -f "$d/DONE" ] && [ -f "$d/test_metrics.json" ] || continue    # localizer runs only
  if [ -f "data/processed/naip_${TEST_YEAR}.npy" ]; then
    if ! grep -q "\"imagery_year\": ${TEST_YEAR}" "$d/test_metrics.json" 2>/dev/null; then
      log "re-evaluating $d on $TEST_YEAR imagery"
      $PY -m btp.evaluate "$d/best.pt" 2>&1 | grep -v Warning | tee -a "$d.log" >> runs/staged.log
      $PY -m btp.simulate "$d/best.pt" 2>&1 | tail -2 >> runs/staged.log
    fi
  else
    pending=1
  fi
done

if [ "$pending" = 0 ]; then
  log "ALL STAGES TRAINED AND TESTED ON ${TEST_YEAR} IMAGERY"
  uninstall
else
  log "training done; waiting for ${TEST_YEAR} imagery to re-test"
fi
