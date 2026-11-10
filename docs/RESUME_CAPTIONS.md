# Resuming the YT500 instrumental caption run

**Scope reminder:** captions and polish only. No SheetSage2, no `transcribe.sh`,
no `work/transcripts/`. `docs/YT500_INSTRUMENTAL_OPS.md` is the interface,
`docs/YT500_INSTRUMENTAL_DATASET.md` the rationale.

This page exists so a fresh Colab runtime, or a fresh agent session, can pick the
run up without re-deriving anything.

## One command

```bash
git clone https://github.com/akbargherbal/music_captioning.git /content/music_captioning
cd /content/music_captioning
pip install -q -r requirements.txt
./plumbing/resume_yt500.sh
```

The script is idempotent and safe to run while the pipeline is healthy — it
starts only the roles that are not already running. It:

1. `git pull`s the repo, which also restores `work/yt500/caption_album.json`
   (committed by the snapshot loop);
2. clones `../arabmaqamrock` and runs
   `plumbing/yt500_generation_corpus.py --fetch` **only** when the corpus index
   is missing (that is the only thing this run uses that repo for);
3. downloads the captioner weights only if absent (~6.4 GB);
4. restores `models/Qwen3-4B-Instruct-2507-Q8_0.gguf` from GCS
   (`music_captioning/gguf/`), falling back to Hugging Face;
5. stages `llama-server` from `music_captioning/binaries/sm89-l4/` (L4) or
   `sm75-t4/` (T4) — a ~1 min download, not a build;
6. starts `yt500_publish`, `yt500_sync`, `yt500_learn` (each under
   `supervise.py`) plus the snapshot loop, and prints `--status`.

Time to the first new caption on a cold VM: ~5–8 min, dominated by the weight
download.

## Where the work lives

| durable artefact | location | written |
|---|---|---|
| album: `chunk_captions`, `caption_raw`, `caption`, tags | GCS `GENERATION/YT500_INSTRUMENTAL/work/caption_album.json` | each pass, and every 5 tracks mid-pass |
| same album, restorable copy | GCS `.../work/backup/latest.json` | every 2 min (snapshot loop) |
| same album, versioned in git | `work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json` (+ `caption_album.json`, `captions.jsonl`) | every 15 min, and on each `polished` flip |
| planner view | `.../work/captions.jsonl` | with the album |
| accepted/rejected takes, `abc_sha256` | `.../work/manifest.json`, `rejected.json` | every sync pass |
| upload ledger, MD5-verified | `.../work/publish/ledger.jsonl` (gitignored) | every publish pass |
| MP3s | local `work/yt500/audio/` only | re-downloaded by sync |

Nothing is ever deleted remotely. `yt500_publish.py` has no `--delete`.

## Knowing where the run stopped

```bash
python3 scripts/yt500_dataset.py --status      # accepted / captioned / pending
git log --oneline -5 -- work/yt500             # last caption snapshot in git
gcloud storage ls -l "gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/"
tail -3 logs/yt500_learn.out logs/yt500_sync.out logs/yt500_publish.out
```

Read `captioned` as **tracks with a final polished caption**. The album file can
hold more tracks than that: entries written mid-pass have `caption: ""` until
the polish phase of that pass runs. `caption_stage` says which side you are on
(`captioned` vs `polished`).

Ignore `transcribed` / `pending trs` lines — this run produces no transcripts.

## Invariants a resume relies on

- A caption pass is **bounded** (`--caption-batch 40`): caption → polish →
  write → publish, roughly every 26 min. Resuming mid-pass is normal.
- The album checkpoints **every 5 tracks**, and every write carries forward
  tracks stored by earlier passes. A killed VM costs minutes, never the corpus.
- `run_tracks` skips anything already in the album, so **finished captions are
  never recomputed**. Polishing only touches entries with an empty `caption`
  unless `--repolish` is passed.
- The captioner and the rewriter never run at the same time.
- Timestamps never reach the final `caption`: the polish prompt forbids them,
  `check_rewrite` rejects a timed output (2 retries, then
  `rewrite_status: failed`), and `polish_album` scrubs every path
  (`single_chunk`, `fallback`, `failed`) deterministically, keeping the original
  in `caption_pre_cleanup`.

## If something is off

| symptom | action |
|---|---|
| `another … holds the lock` | normal: one role, one process. Do not delete the lock while it is alive. |
| sync exits on `abc_mismatch` | the frozen corpus index and the generation run disagree. Stop and report; `--allow-abc-mismatch` keeps polling. |
| `gcloud` auth error | `gcloud auth login --no-launch-browser` is interactive; ask the user. |
| no `llama-server` | `./plumbing/fetch_llamacpp_text.sh` (uses the published tag for the local compute capability). Do not hand-roll a build. |
| `rewrite_status: failed` | guard or verifier rejected the output; the raw caption is kept. Listed for review, retried on the next polish pass. |
| disk < ~15 GB | `work/yt500/audio/` is re-downloadable; the album is already in GCS and git. Report before deleting. |

## Not part of this run

`yt500_assemble.py` needs 489 **transcribed** takes and will never run here. The
final dataset package is a separate, later decision.
