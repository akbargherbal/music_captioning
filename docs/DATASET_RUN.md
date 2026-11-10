# Dataset run — `fadhl_shaker_instrumental_18`

Captions and ABC scores for one generated-instrumental run of the
`arabmaqamrock` project, built on 2026-11-10 (NVIDIA L4, 23 GB).

| Item | Value |
|---|---|
| Source audio | `gs://akbar-december-2024-backup/arabmaqamrock/runs/20261110-050958_fadhl_shaker_instrumental_18/*.mp3` |
| Tracks | 24, 6,550 s (1.82 h), 149.9 MiB, 192 kbps |
| Output | `gs://akbar-december-2024-backup/DATASET/fadhl_shaker_instrumental_18/` |
| Package size | 664 KiB, 77 files |

The run is the release of 24 instrumentals whose audio scores live in
`arabmaqamrock/scores/fadhl_shaker_instrumental/`. Every MP3 carries its
generation seed in the filename, so the reference score is unambiguous; all 24
reference ABC files were verified against the `abc_sha256` recorded in the run's
`batch_manifest.json`.

## What the dataset contains

```
tracks.jsonl                  one row per track (flat, JSONL)
tracks.parquet                same rows, flat columns
captions/<track>.json         full caption record incl. per-chunk text
abc/<track>.abc               reference score used to generate the MP3
abc_transcription/<track>.abc SheetSage2 transcription of the rendered MP3
manifest.json                 counts, provenance, SHA-256 of every file
DATASET_CARD.md               human-readable summary
_SUCCESS
```

Audio is referenced by `gs://` URI + SHA-256; no audio bytes are copied.

## Measured results

### ABC transcription (SheetSage2, full = melody + chords)

| Metric | Value |
|---|---|
| Tracks | 24/24 `status: ok` |
| Wall time | 1,039.8 s (17.3 min) |
| Audio | 6,550 s |
| RTF | 6.3x |
| `abc2midi` parse | 24/24 ok |
| Coverage | 21 tracks 1.00; one 0.98; two below 0.85 (0.838, 0.807) |
| QC flags | `vocal_sparse` on 23/24 (expected for instrumentals) |

### Captioning + polish (ACE-Step 4-bit, then Qwen3-4B Q8)

| Metric | Value |
|---|---|
| Tracks | 24/24, status `ok`, stage `polished` |
| Caption compute | 692.5 s (11.5 min) |
| Audio | 6,550 s |
| RTF | 9.5x |
| Per-track | 19-69 s; mean 29 s |
| Peak VRAM | ~9.8 GiB (captioner loaded, observed) |
| Rewrite | 24/24 `ok`, 0 fallback, 0 error |

Wall time from model load to the final album JSON was 13.8 min
(07:55:01 -> 08:08:49 UTC), including the phase break and ~2 min of polish.

## How it was built

```bash
# 1. stage models (~11 GB) and build llama-server
python3 -c "from huggingface_hub import snapshot_download; ..."
./plumbing/fetch_llamacpp_text.sh

# 2. fetch audio + reference ABC, verify hashes
gcloud storage cp ".../runs/20261110-050958_.../*.mp3" work/dataset/audio/

# 3. transcribe the rendered audio (separate env, arabmaqamrock repo)
/content/arabmaqamrock/transcribe.sh work/dataset/audio --full \
    --device cuda --dtype bf16 --out-dir work/dataset/transcripts

# 4. caption + polish (one captioner load, one album JSON)
python3 scripts/build_dataset.py caption \
    --audio-dir work/dataset/audio --run-manifest work/dataset/run_manifest.json

# 5. assemble, then upload and verify
python3 scripts/build_dataset.py assemble \
    --run-manifest work/dataset/run_manifest.json --audio-dir work/dataset/audio \
    --captions work/dataset/captions --abc-ref work/dataset/abc_ref \
    --transcripts work/dataset/transcripts --out work/dataset/package \
    --source-uri gs://.../runs/20261110-050958_fadhl_shaker_instrumental_18
gcloud storage rsync -r --checksums-only work/dataset/package \
    gs://akbar-december-2024-backup/DATASET/fadhl_shaker_instrumental_18
```

Upload verification compared all 77 files by size and MD5 (`gcloud storage ls
--json`): 0 missing, 0 mismatched, 0 extra.

## Caveats

- Captions are machine output; the term lists are deterministic lexicon matches
  and stay marked `tags_experimental: true` (`docs/TAG_AUDIT.md`).
- The reference ABC is the generation input. Only `abc_transcription/*.abc`
  describes what was actually rendered; it is not expected to match the input
  note-for-note.
- `vocal_sparse` QC flags on an instrumental run are expected, not a failure.
