#!/usr/bin/env bash
# Copy-on-write snapshot of ~/BTP/data into data/raw (btrfs reflinks: no extra disk space).
# The originals are never touched; the copies are made read-only.
# Re-run any time: NAIP years are copied once their download is verified.
set -euo pipefail
SRC="$HOME/BTP/data"
DST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DST"

copy() {
  if [ -e "$DST/$1" ]; then echo "have   $1"; return; fi
  cp -a --reflink=always "$SRC/$1" "$DST/$1.partial"
  mv "$DST/$1.partial" "$DST/$1"
  chmod -R a-w "$DST/$1"
  echo "copied $1"
}

copy geophysics
for y in 2016 2018 2020 2022; do
  if grep -q " $y verified$" "$SRC/download_cron.log" 2>/dev/null; then
    copy "naip_$y"
  else
    echo "skip   naip_$y (download not verified yet)"
  fi
done
