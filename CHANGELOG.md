# Changelog

## [Unreleased]

### Added

- `scripts/yt500_sync.py`: watcher for the YT500 instrumental generation
  bucket. It is the only process that reads `runs/*_yt500inst*` and downloads
  audio; it never loads a model. Every take is gated (§3.1 of
  `docs/YT500_INSTRUMENTAL_DATASET.md`): `abc_sha256` equals the frozen corpus
  hash, `_time.txt` says `Exit status: 0`, the MP3 and sidecar exist, and
  `0.85 <= wav_duration_s / nominal_seconds <= 1.25`. One accepted take per
  name (latest `end_utc`), so a regenerated take with a new seed is picked up
  on the next poll without waiting for the full corpus. Writes `manifest.json`,
  `rejected.json` and `status_sync.json`; restores `caption_album.json` and
  `work/transcripts/` after a VM loss. A take inside a run that has not
  finished is *pending*, not rejected. One `gcloud storage cp` per run copies
  the sidecars and markers (the duration gate never downloads audio), and a
  finalised run is copied once.
- `scripts/yt500_gcs.py`: the shared GCS layer for the watcher and the
  publisher — metadata listing, atomic downloads (`part` + fsync + rename),
  size+MD5 verified uploads, `rsync` without `--delete`, a directory
  comparison for the final package, and a per-role `flock`.
- `tests/test_yt500_sync.py` and `tests/test_yt500_incremental.py`: CPU-only
  coverage of the gate (0.84 and 1.26 rejected, 0.85/1.25 accepted), marker and
  sidecar requirements, pending-vs-rejected for an unfinished run, latest-take
  selection, one sync pass against an in-memory bucket, and an incremental
  polish pass that re-polishes zero stems.

### Changed

- `scripts/build_dataset.py caption`: an album with `failed_tracks` is no
  longer treated as done, so a failed track is retried instead of being
  skipped forever (`already_polished()`).
- `scripts/build_dataset.py caption`: on an incremental pass only pending
  tracks are polished (`--repolish` restores the old rewrite-everything
  behaviour), and the captioner is not loaded at all when no track needs
  captioning. `rewrite.reused` counts the tracks kept from the previous pass.
- `scripts/build_dataset.py caption`: `--max-track-seconds` now defaults to `0`
  (no cap). A growing corpus must not leave a track uncounted, which would stop
  the track count from ever settling.
- `requirements.txt`: add `pyarrow` (needed by `build_dataset.py assemble`,
  previously missing).
- `scripts/build_dataset.py assemble`: a package can now span several
  generation runs. Each row takes its own `run_id` and `gs_uri` from the take
  (falling back to `--run-id` / `--source-uri`), `track_id` is
  `<run_id>/<stem>`, and the corpus columns (`title`, `source_artist`,
  `country`, `tier`, `source_rank`, `source_duration_s`, `source_abc_sha256`,
  `generation_status`, `take_ratio`, `take`) are added when the manifest
  carries them — so the single-run `fadhl_shaker_instrumental_18` schema is
  unchanged. `manifest.json` gains `source_runs`, `run_ids`, `rejected`,
  `take_selection` and `corpus`. The dataset card says "one or more runs".
  `--instrumental/--no-instrumental` overrides the run-id inference.
- `load_run_manifest` accepts the watcher's `{accepted: [...]}` queue as well as
  a single run's `{tracks: [...]}`.


- **Deterministic term extraction is now explicitly experimental.** The
  comma-joined `suno_prompt` is no longer a default output; it is opt-in via
  `experimental_suno=True` / `--experimental-suno`, and is marked
  `suno_experimental`. Outputs now carry `tags_experimental: true`.
- Removed `suno_prompt` from all existing files under `outputs/` and from the
  demo, batch, benchmark and ABC-dataset paths.
- Added `docs/TAG_AUDIT.md`: 5-track audit plus corpus numbers for the known
  gaps (no genre/mood, plural and phrase misses, chunk contradictions, no
  negation). The feature stays experimental until that bar is met.
- README/AGENTS now describe the output as experimental term lists rather than
  a finished Suno prompt.

### Added

- `scripts/build_dataset.py`: build a caption + ABC dataset for one generated
  run. `caption` runs the captioner and then the polish model into one album
  JSON (no git, no manifest); `assemble` joins that JSON with the reference ABC
  (SHA-256-checked against the run manifest) and the SheetSage2 transcription
  into JSONL + Parquet + per-track files + a dataset card. See
  `docs/DATASET_RUN.md` for the first run's measured numbers.
- `tests/test_build_dataset.py`: CPU-only coverage for `assemble` (package
  layout, manifest hashes, reference-ABC mismatch, missing transcript).
- `docs/QUICKSTART.md`: a beginner-friendly, end-to-end guide (Colab setup,
  first run, output fields, common tasks, troubleshooting, glossary) aimed at
  people who are new to the project.
- `docs/README.md`: an index of the project documentation, linked from the root
  README.

- `scripts/batch.py`: resumable discography runner. After every album it writes
  `outputs/<ARTIST>/<album>.json`, updates `outputs/index.json`, and commits +
  pushes — so a fresh (ephemeral) Colab runtime resumes where it stopped.
- `src/resume.py`: manifest + resume helpers (atomic writes, done/pending
  detection), with CPU-only tests.
- `--max-minutes`, `--max-albums`, `--list`, `--force`, `--no-push` controls.
- Per-track failure isolation: a bad track is logged in `failed_tracks` and the
  album is still saved (manifest status `partial`); `--retry-failed` re-captions
  only the failed tracks. Only an all-tracks-failed album stays pending.
- `--max-track-seconds` (default 480 = 8 min): tracks longer than this are
  recorded under `skipped_tracks` and not captioned (guards against concerts);
  skips are not failures and don't affect album status.
- `--min-bitrate-kbps` (default 128; `0` disables): low-quality tracks are
  skipped the same way. `--artist-name` overrides the artist label for a local
  `--root` (default: the root directory name).
- `scripts/supervise.py`: restarts the batch after any non-zero exit (including
  SIGKILL) so a killed process resumes instead of stopping. Run it detached.

## [1.0.0] - 2026-10-09

First stable release.

### Added

- Pipeline: audio → ACE-Step captioner → prose caption → deterministic lexicon
  tag extractor → Suno prompt string.
- 4-bit `transformers` backend (nf4, fp16 compute, SDPA) for the Colab T4
  (`src/captioner_hf.py`) — no llama.cpp build required.
- llama.cpp GGUF backend kept as an alternative (`src/captioner.py`,
  `scripts/build_llamacpp.sh`).
- `scripts/demo.py` for a single track, with `--out` to save JSON.
- `scripts/benchmark.py` for a file or a whole album: per-track chunk timings,
  peak VRAM, totals, and `realtime_factor`.
- 27 CPU-only tests (`pytest tests/`).
- `notebooks/colab_setup.ipynb`, `README.md`, and measured numbers in
  `docs/RESULTS.md`.

### Measured on a Colab T4

- Single track, 317 s: 68.5 s captioning, 8.0 GiB peak VRAM.
- Album `Elissa - 2008 Ayami Beek`, 11 tracks, 3034 s: 617 s captioning
  (RTF 0.203, ≈4.9× faster than real time), 8.0 GiB peak VRAM.
