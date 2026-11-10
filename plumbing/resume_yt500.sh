#!/usr/bin/env bash
# Bring this VM back to a captioning state after a session loss, or start the
# run on a fresh one. Idempotent: safe to run while everything is healthy
# (it detects running workers and leaves them alone), and safe to run twice.
#
#   ./plumbing/resume_yt500.sh
#
# What is durable and where:
#   GCS   gs://.../GENERATION/YT500_INSTRUMENTAL/work/   album, manifest, ledger
#   git   work/yt500/caption_album.json + captions/     snapshots every 15 min
#   disk  work/yt500/audio/                              re-downloadable by sync
#
# A caption pass checkpoints every 5 tracks and each pass is 40 tracks, so a
# hard kill costs at most a few minutes of GPU, never the corpus.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
say() { echo "[resume $(date -u +%H:%M:%SZ)] $*"; }
running() { pgrep -f "$1" >/dev/null 2>&1; }

say "repo $REPO ($(git rev-parse --short HEAD))"
git pull -q --ff-only 2>/dev/null && say "pulled" || say "pull skipped"

# --- 1. generation corpus index (the length gate reads it) -------------------
INDEX="$REPO/work/yt500/corpus/index.json"
if [ ! -f "$INDEX" ] && [ ! -f ../arabmaqamrock/outputs/YT500/generation/index.json ] \
   && [ ! -f /content/arabmaqamrock/outputs/YT500/generation/index.json ]; then
  say "corpus index missing, fetching (needs ../arabmaqamrock)"
  if [ ! -d ../arabmaqamrock ]; then
    git clone -q https://github.com/akbargherbal/arabmaqamrock.git ../arabmaqamrock \
      || { say "FATAL: cannot clone arabmaqamrock"; exit 1; }
  fi
  ( cd ../arabmaqamrock && python3 plumbing/yt500_generation_corpus.py --fetch ) \
    || { say "FATAL: corpus fetch failed"; exit 1; }
fi

# --- 2. weights, rewriter GGUF, llama-server --------------------------------
if [ ! -f models/Ace-Step-Captioner-4bit/spk_dict.pt ]; then
  say "captioner weights missing: downloading (6.4 GB)"
  python3 -u - > "$LOG_DIR/hf_download.out" 2>&1 <<'PY' \
    || { say "FATAL: weight download failed, see logs/hf_download.out"; exit 1; }
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit",
                  local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                  "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt",
                local_dir="models/Ace-Step-Captioner-4bit")
print("WEIGHTS_OK")
PY
fi

GGUF=Qwen3-4B-Instruct-2507-Q8_0.gguf
if [ ! -f "models/$GGUF" ]; then
  say "rewriter GGUF missing: trying GCS, else Hugging Face"
  gcloud storage cp -q \
    "gs://akbar-december-2024-backup/music_captioning/gguf/$GGUF" models/ 2>/dev/null \
  || python3 - "$GGUF" <<'PY' || { say "FATAL: rewriter GGUF unavailable"; exit 1; }
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("unsloth/Qwen3-4B-Instruct-2507-GGUF", sys.argv[1],
                local_dir="models")
PY
fi
[ -x llama.cpp-text/build/bin/llama-server ] \
  || ./plumbing/fetch_llamacpp_text.sh > "$LOG_DIR/llamacpp_build.out" 2>&1 \
  || { say "FATAL: llama-server unavailable"; exit 1; }
say "weights + backend ok"

# --- 3. workers (never start a second copy of a role) -----------------------
start() {  # start <name> <pgrep pattern> <command...>
  local name="$1" pattern="$2"; shift 2
  if running "$pattern"; then say "$name already running"; return; fi
  setsid nohup "$@" > "$LOG_DIR/$name.sup.out" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  say "$name started"
}

start yt500_publish "scripts/yt500_publish.py --watch" \
  python3 -u scripts/supervise.py --script scripts/yt500_publish.py \
  --log "$LOG_DIR/yt500_publish.out" -- --watch --interval 60
start yt500_sync "scripts/yt500_sync.py --watch" \
  python3 -u scripts/supervise.py --script scripts/yt500_sync.py \
  --log "$LOG_DIR/yt500_sync.out" -- --watch --interval 60 \
  --min-ratio 0.85 --max-ratio 1.25
start yt500_learn "scripts/yt500_learn.py --watch" \
  python3 -u scripts/supervise.py --script scripts/yt500_learn.py \
  --log "$LOG_DIR/yt500_learn.out" -- --watch --phase caption \
  --min-batch 12 --flush-minutes 20 --interval 60 --caption-batch 40

if ! running "snapshot_wor[k].sh"; then
  setsid nohup ./plumbing/snapshot_work.sh > "$LOG_DIR/snapshot.out" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  say "snapshot loop started"
fi

sleep 2
say "--- state ---"
python3 scripts/yt500_dataset.py --status 2>&1 | sed -n '1,9p'
say "logs: $LOG_DIR/yt500_{sync,learn,publish}.out"
