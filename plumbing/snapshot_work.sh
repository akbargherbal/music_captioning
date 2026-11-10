#!/usr/bin/env bash
# Independent 2-minute snapshot of the caption album to GCS.
#
# The GPU worker checkpoints the album locally every few tracks; this loop is a
# separate process so a slow upload can never stall captioning and a GPU crash
# can never lose what is already on disk. Versioned copies only: the publisher
# owns `work/caption_album.json`, this never writes that key.
#
#   setsid nohup ./plumbing/snapshot_work.sh > logs/snapshot.out 2>&1 < /dev/null &
#
# Restore after a VM loss:
#   gcloud storage cp "$DEST/latest.json" work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DEST="${SNAPSHOT_DEST:-gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/backup}"
SRC="work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json"
INTERVAL="${SNAPSHOT_INTERVAL:-120}"
last=""

echo "snapshot: $SRC -> $DEST every ${INTERVAL}s"
while true; do
  if [ -f "$SRC" ]; then
    sum=$(md5sum "$SRC" | awk '{print $1}')
    if [ "$sum" != "$last" ]; then
      ts=$(date -u +%Y%m%dT%H%M%SZ)
      if gcloud storage cp -q "$SRC" "$DEST/caption_album_$ts.json" \
         && gcloud storage cp -q "$SRC" "$DEST/latest.json"; then
        last="$sum"
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot $ts ($(stat -c %s "$SRC") B)"
      else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) upload failed, retrying" >&2
      fi
    fi
  fi
  sleep "$INTERVAL"
done
