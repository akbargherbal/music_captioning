# Changelog

## [Unreleased]

### Changed

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
