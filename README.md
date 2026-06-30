# AI Music Captioning

A small, crash-resumable pipeline for analyzing a directory of audio tracks with Nvidia's `music-flamingo` model. For every track, it asks a fixed set of questions (genre, vocals, instrumentation, production, mastering) and writes one JSON file per track containing the answers.

## 🚀 Key Features

- **Full-track analysis.** Each track is analyzed as a whole (trimmed to a configurable max duration as a VRAM-safety cap), not split into fixed-length chunks.
- **Prompt set lives in one file.** `prompts.py` holds a `CAPTIONING_PROMPTS` dict — one entry per question asked of every track. Add, remove, or edit a category there; `caption.py` needs no changes.
- **Crash-resumable.** Each track's JSON is rewritten to disk immediately after *every* category finishes, not after the whole track or the whole run. If the process dies partway through (e.g. a Colab disconnect), at most the one in-flight category is lost.
- **Resumes automatically on re-run.** Re-running the same command skips any track that's already fully done and, for a partially-done track, only re-runs the categories still missing. `--overwrite` forces everything to be redone.
- **Recursive directory scan.** Pass any folder via `--dir`; subdirectories at any depth are scanned. Non-audio files (scripts, text, logs, partial-download artifacts) are simply ignored.
- **Degrades gracefully on bad input.** Missing files, corrupt/empty audio, and CUDA failures are caught per-track and recorded as `status: "error"` rather than crashing the whole batch.
- **`--dry-run` mode.** Exercises the full file I/O / resume logic with placeholder text and no GPU or model — useful for validating a run before committing GPU time.

## 📋 Prerequisites & Installation

Python 3.10+ (uses `tuple[bool, str | None]`-style type hints).

```bash
pip install --upgrade pip
pip install torch torchaudio
pip install "git+https://github.com/huggingface/transformers" accelerate bitsandbytes
pip install librosa soundfile numpy
```

`--dry-run` only needs `numpy` — torch and transformers are imported lazily so you can sanity-check the pipeline without the GPU stack installed.

If your audio lives in a GCS bucket, download it locally first (e.g. `gsutil -m cp -r gs://your-bucket/tracks ./tracks`) so the script just sees a normal directory — it does not talk to GCS itself.

## 📁 Project Structure

```
caption.py    # CLI entry point: discovery, audio prep, inference, resume, summary
prompts.py    # CAPTIONING_PROMPTS — the set of questions asked per track
```

Both files should sit in the same directory. Your audio directory can be anywhere — pass its path via `--dir`.

## 🛠️ Usage

```bash
# Scan a directory, run every prompt in prompts.py against every track
python caption.py --dir ./tracks --output-dir ./captions

# No args: scans the current directory, writes to ./captions/
python caption.py

# Only run a subset of categories
python caption.py --dir ./tracks --categories genre,vocals

# Force re-run of every category, even ones already marked done
python caption.py --dir ./tracks --overwrite

# Validate the pipeline without a GPU/model
python caption.py --dir ./tracks --dry-run
```

This produces one `<track_stem>.caption.json` per audio file in `--output-dir`, plus two run-level summaries once the batch finishes: `results_summary.json` and `results_summary.csv` (one row per track, one column per category).

### Folder layout

The scan is recursive and extension-based (`.mp3 .wav .flac .m4a .ogg`), so flat or nested folders both work:

```
tracks/
├── 01.mp3
├── 02.mp3
└── album2/
    └── 01.mp3
```

One caveat: output filenames are derived from the track's filename stem only, not its folder path. `tracks/01.mp3` and `tracks/album2/01.mp3` would both write to `01.caption.json` and overwrite each other — keep filenames unique across subfolders, or rename before running at scale.

## 🎛️ CLI Reference (`caption.py`)

| Option           | Default        | Description                                                                 |
| :---------------- | :------------- | :---------------------------------------------------------------------------- |
| `--dir`           | `.`            | Directory to scan recursively.                                                |
| `--file`          | `None`         | Process a single file instead of `--dir`.                                     |
| `--output-dir`    | `./captions/`  | Where per-track JSON and summary files go.                                    |
| `--categories`    | `all`          | Comma-separated subset of `prompts.py` keys to run.                           |
| `--max-tokens`    | `1024`         | Max new tokens per prompt (ceiling: 4096).                                    |
| `--precision`     | `bf16`         | `bf16` or `4bit`. 4-bit trades quality for VRAM headroom.                     |
| `--overwrite`     | `False`        | Re-run categories even if already marked `"ok"` in existing output.           |
| `--dry-run`       | `False`        | Skip model loading/inference; write placeholder text to test the pipeline.    |

## 📂 Output JSON Schema

One file per track, updated incrementally as each category completes:

```json
{
  "file": "01.mp3",
  "path": "/absolute/path/to/01.mp3",
  "model_id": "nvidia/music-flamingo-2601-hf",
  "last_updated": "2026-06-30T09:46:37+00:00",
  "categories": {
    "genre": {
      "status": "ok",
      "output": "Middle Eastern folk fusion blended with...",
      "error": null,
      "processing_time_sec": 4.2,
      "audio_windows": 9,
      "audio_covered_sec": 270,
      "output_truncated": false,
      "prompt_used": "Listen to this track and identify its genre..."
    },
    "vocals": { "status": "ok", "output": "...", "..." : "..." },
    "instrumentation": { "status": "ok", "output": "...", "...": "..." },
    "production": { "status": "ok", "output": "...", "...": "..." },
    "mastering": { "status": "ok", "output": "...", "...": "..." }
  }
}
```

A category that failed (bad audio, CUDA error, etc.) records `"status": "error"` with an `"error"` message instead of `"output"`, so partial results for a track are always inspectable rather than silently missing.

## 📊 Summary Outputs

After the run, `results_summary.csv` gives a flat, spreadsheet-friendly view:

```
file,path,genre,vocals,instrumentation,production,mastering
01.mp3,/path/to/01.mp3,"Middle Eastern folk fusion...","male baritone...","oud, acoustic guitar...","...","..."
```

`results_summary.json` contains the same data in full structured form (every track's complete JSON record).
