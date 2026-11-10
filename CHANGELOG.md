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
- `tests/test_yt500_learn.py` and `tests/test_yt500_publish.py`: nothing
  pending starts no subprocess, a stale/`error`/hash-mismatched transcript is
  pending again, `--verify-schema` catches an empty caption, only `ok`
  transcripts upload, a second publish re-uploads no artifact, and no publish
  path deletes a remote object.
- `scripts/yt500_learn.py`: the only GPU process. It reads `manifest.json`,
  captions only the new tracks (via `build_dataset.py caption`, which now also
  skips the captioner when nothing needs captioning), then transcribes the
  accepted audio that has no `ok` transcript (via
  `transcribe.sh --full --dtype auto`). `--min-batch`/`--flush-minutes` batch
  small arrivals, `--max-batch` bounds one invocation, `--verify-schema` checks
  the album after captioning. A phase with nothing pending starts no
  subprocess, and a phase that makes no progress is not retried in the same
  session.
- `scripts/yt500_publish.py`: uploads the work state and the final package to
  GCS with size+MD5 verification and a ledger, so a second pass uploads
  nothing. Only transcripts whose `meta.json.status` is `ok` and whose
  `score.abc` matches `meta.abc_sha256` are uploaded. It never deletes
  anything.
- `scripts/yt500_assemble.py`: refuses to build the final package unless the
  corpus is complete (489 accepted, 489 captions, 489 `ok` transcripts, no
  rejects), then runs `build_dataset.py assemble`. `--dry-run` prints the
  readiness report.
- `scripts/yt500_dataset.py`: read-only `--status` view that merges the four
  `status_<role>.json` files with the manifest. It also holds the shared
  `work/` layout and the "is this track done?" predicates, so the watcher, GPU
  worker, publisher and assembler cannot disagree.
- `docs/YT500_INSTRUMENTAL_OPS.md`: the implemented interface (flags, worker
  contracts, the behaviours the plan left open, and the first live numbers).

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
