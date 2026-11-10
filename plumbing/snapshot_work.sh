#!/usr/bin/env bash
# Independent 2-minute snapshot of the caption album: GCS + optionally git.
#
# The GPU worker checkpoints the album locally every few tracks; this loop is a
# separate process so a slow upload can never stall captioning and a GPU crash
# can never lose what is already on disk. It never writes the key the publisher
# owns, and it never touches audio.
#
#   setsid nohup ./plumbing/snapshot_work.sh > logs/snapshot.out 2>&1 < /dev/null &
#
# Restore after a VM loss:
#   gcloud storage cp "$DEST/latest.json" work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json
#
# The album carries chunk_captions and caption_raw, so a different (better)
# polish model can be applied without re-running the audio captioner.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DEST="${SNAPSHOT_DEST:-gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/backup}"
SRC="work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json"
INTERVAL="${SNAPSHOT_INTERVAL:-120}"
COMMIT_EVERY="${SNAPSHOT_COMMIT_EVERY:-900}"
GIT_ID=(-c user.name=yt500-snapshot -c user.email=yt500@local)
last=""
last_commit=0

echo "snapshot: $SRC -> $DEST every ${INTERVAL}s, git commit every ${COMMIT_EVERY}s"
while true; do
  if [ -f "$SRC" ]; then
    sum=$(md5sum "$SRC" | awk '{print $1}')
    if [ "$sum" != "$last" ]; then
      ts=$(date -u +%Y%m%dT%H%M%SZ)
      if gcloud storage cp -q "$SRC" "$DEST/caption_album_$ts.json" \
         && gcloud storage cp -q "$SRC" "$DEST/latest.json"; then
        last="$sum"
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot $ts ($(stat -c %s "$SRC") B)"

        stage=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('caption_stage',''))" "$SRC" 2>/dev/null || echo "")
        now=$(date +%s)
        if [ "$stage" = "polished" ] || [ $((now - last_commit)) -ge "$COMMIT_EVERY" ]; then
          tracks=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tracks']))" "$SRC" 2>/dev/null || echo "?")
          git add work/yt500/captions work/yt500/caption_album.json work/yt500/captions.jsonl 2>/dev/null
          if ! git diff --cached --quiet -- work/yt500; then
            if GIT_TERMINAL_PROMPT=0 git "${GIT_ID[@]}" commit -q -m \
                 "yt500: captions snapshot ($tracks tracks, $stage) $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                 -- work/yt500/captions work/yt500/caption_album.json work/yt500/captions.jsonl \
               && GIT_TERMINAL_PROMPT=0 git push -q origin HEAD; then
              last_commit=$now
              echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) git commit $tracks tracks ($stage)"
            else
              echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) git commit/push failed, will retry" >&2
            fi
          else
            last_commit=$now
          fi
        fi
      else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) upload failed, retrying" >&2
      fi
    fi
  fi
  sleep "$INTERVAL"
done
