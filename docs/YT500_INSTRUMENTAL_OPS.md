# YT500 instrumental — streaming operators

The **plan and rationale** are in
[YT500_INSTRUMENTAL_DATASET.md](YT500_INSTRUMENTAL_DATASET.md). This page is
the implemented interface: what each script does, its flags, and the few
behaviours the plan left open.

Four processes, one shared `work/` directory, no shared writable state. Only
the watcher reads the generation bucket; only the publisher writes to GCS.

| Process | Runs on | Owns | Never does |
|---|---|---|---|
| `scripts/yt500_sync.py` | CPU/net | `work/audio/`, `manifest.json`, `rejected.json`, `rejected_history.json`, `status_sync.json` | load a model, caption, transcribe |
| `scripts/yt500_learn.py` | GPU | `work/captions/`, `caption_album.json`, `captions.jsonl`, `work/transcripts/`, `status_learn.json` | read/write GCS |
| `scripts/yt500_publish.py` | CPU/net | GCS `GENERATION/YT500_INSTRUMENTAL/work/`, the final package, `status_publish.json` | caption, transcribe, download audio |
| `scripts/yt500_assemble.py` | CPU | `work/package/`, `status_assemble.json` | load a model |

`scripts/yt500_gcs.py` is the shared GCS layer; `scripts/yt500_dataset.py` is
the shared read-only state view (`--status`).

## Decisions taken (2026-11-10)

The §0.1 answers: **L4 24 GB**, **full parity**, **`0.85` lower bound**
(`1.25` upper), **never wait for the bulk to finish**, final prefix
`DATASET/YT500_INSTRUMENTAL`, final package only at 489/489, and **§6.4
(`plumbing/regen_short_takes.py`) is not implemented in this repo** — it lives
on the A100 side. This repo only watches, captions and polishes.

"Never wait" is the operational consequence: the watcher processes accepted
takes as they land and a regenerated take (new seed, new `end_utc`) simply
wins the selection on the next poll. Nothing has to be restarted, and no name
is blocked by an unfinished sibling.

## Commands

```bash
cd /content/music_captioning

# pilot (plan §8) — 4 accepted takes: short / median / long / newest
python3 scripts/yt500_sync.py --once --limit 4
python3 scripts/yt500_learn.py --once --phase both --verify-schema
python3 scripts/yt500_assemble.py --dry-run

# bulk
python3 scripts/yt500_sync.py --once
python3 scripts/yt500_learn.py --once --phase both
python3 scripts/yt500_publish.py --once

# the three workers, detached (plan §9.2)
setsid nohup python3 -u scripts/yt500_publish.py --watch --interval 60 \
  > logs/yt500_publish.out 2>&1 < /dev/null &
setsid nohup python3 -u scripts/yt500_sync.py --watch --interval 60 \
  > logs/yt500_sync.out 2>&1 < /dev/null &
setsid nohup python3 -u scripts/yt500_learn.py --watch --phase both \
  --min-batch 12 --flush-minutes 20 --interval 60 \
  > logs/yt500_learn.out 2>&1 < /dev/null &

python3 scripts/yt500_dataset.py --status
```

## Flags worth knowing

### `yt500_sync.py`

- `--min-ratio 0.85 --max-ratio 1.25` — the §3.1 band.
- `--limit N` — queue only N accepted takes. The pick is deterministic and
  spread over length (shortest, median, longest) plus the newest take, which is
  exactly the pilot's request. `counts.accepted` still reports the full number,
  `counts.accepted_in_manifest` the queued slice.
- `--download-limit N` — at most N new MP3s per pass (0 = all).
- Superseded local takes are deleted from `work/audio` (never from GCS) so one
  name never yields two captions.
- `--allow-abc-mismatch` — see below.
- `--idle-stop-minutes N` — exit `--watch` after N minutes without a new
  accepted take. Only meaningful once every name is accepted.
- `--corpus-index PATH` — defaults to a sibling
  `../arabmaqamrock/outputs/YT500/generation/index.json`, then
  `/content/arabmaqamrock/...`. A byte-identical copy is kept at
  `work/corpus/index.json` and its SHA-256 is recorded in `manifest.json`, so
  the gate is reproducible.

Cost: one `gcloud storage ls -r` plus one `gcloud storage cp` per run whose
metadata changed. A finalised run is copied once (append-only), so the steady
state is one listing and one copy for the run in flight — 35–55 s over 4 runs /
1,425 takes on 2026-11-10.

### `yt500_learn.py`

- `--phase caption|transcribe|both` (default `both`), always serial.
- `--min-batch 12 --flush-minutes 20` — a phase starts when 12 tracks are
  pending, or 20 minutes after the last pass whatever the count.
- `--max-batch 48` — transcription tracks per invocation (0 = all).
  Captioning always drains the whole album: one captioner load for every new
  take is cheaper than several, and `build_dataset.py caption` checkpoints the
  raw captions before it starts polishing.
- `--verify-schema` — after captioning, check `caption_pipeline_version`,
  `caption_stage == polished`, the required per-track keys, no empty captions,
  no `failed_tracks`. Non-zero exit on a problem.
- `--dtype auto` — SheetSage2 precision; use `fp32` or `auto` on a T4.
- `--transcribe-sh PATH` — defaults to a sibling
  `../arabmaqamrock/transcribe.sh`.
- A phase with nothing pending starts no subprocess, so the captioner is not
  loaded and VRAM stays free.
- If a pass makes no progress (`pending` did not shrink), that phase is not
  retried in this session; the reason is in `work/logs/learn_<phase>.log`.

### `yt500_publish.py`

- Uploads `manifest.json`, `rejected.json`, `rejected_history.json`, every
  `status_*.json`, `caption_album.json` and `captions.jsonl`.
- Only transcripts with `meta.json.status == "ok"` **and** a `score.abc`
  matching `meta.abc_sha256` are uploaded.
- `--settle-seconds 30` — the album is uploaded only when it has been quiet
  that long, so a mid-write album is never copied.
- Every upload is verified by size + MD5 and recorded in
  `work/publish/ledger.jsonl`; a second pass re-uploads nothing but the status
  files.
- **Nothing is deleted.** There is no `--delete` and no remote `rm` in the
  script; a remote object with no local counterpart is reported by
  `compare_dir` and left in place.

### `yt500_assemble.py`

- `--dry-run` prints the readiness report (§12 gate) and writes nothing. It
  exits 0 even when the corpus is incomplete, so it is usable during the pilot.
- A real run refuses unless `accepted == corpus_total`, `rejected` is empty,
  and every accepted take has a caption and an `ok` transcript. `--force` is a
  debug escape hatch only.
- Runs `build_dataset.py assemble` over the whole queue; the publisher uploads
  `work/package`.

## Behaviours the plan left open

1. **`abc_mismatch` is a hard stop only when it leaves a name without any
   usable take.** The score hash of a take that disagrees with the frozen
   corpus index means the run and the corpus disagree; that name is never
   captioned. If *every* take of a name mismatches, the watcher stops with exit
   2 so someone fixes the corpus index (`--allow-abc-mismatch` keeps polling).
   A stray mismatched take next to a good take is just rejected.
2. **An unfinished run produces `pending`, not `rejected`.** A take with no
   MP3/marker/sidecar in a run that has not written `batch_run_summary.json` is
   still rendering. Only a finalised run can produce `missing_mp3`,
   `no_marker` or `missing_sidecar`. Without this, a live run would fill
   `rejected.json` with names that are merely not done yet.
3. **`rejected_history.json`** remembers every name ever rejected, so
   `--status` can show `regen N / M recovered`: names a regen take has since
   rescued.
4. **`--max-track-seconds` now defaults to 0** in `build_dataset.py caption`.
   With a cap, a long take would be skipped and the track count would never
   settle, which breaks the incremental pass. The gate's `1.25` upper bound
   already prevents a runaway take from being queued.
5. **A no-op caption pass loads no weights.** If every track is already in the
   album, `build_dataset.py caption` prints that and does not construct the
   captioner; it only polishes what is pending.
6. **A superseded local take is pruned.** When a regen take replaces an
   accepted take, the watcher deletes the old MP3 from `work/audio` (never from
   GCS) so the album cannot caption two takes of the same name. The album is
   manifest-independent here only in the sense that `assemble` iterates the
   manifest, so a stale album entry is never emitted.
7. **A caption crash re-captions the whole in-flight pass.** The album is
   written after the caption phase completes (that is where the raw-caption
   checkpoint lives), so a native crash at track 180 of 185 loses that pass's
   captioning; the next pass redoes it. The streaming run is not affected
   (a pass is a handful of new takes), but for a large backlog run it in
   tranches with `yt500_sync.py --limit` rather than all at once, so a crash
   costs minutes, not hours.

## Reference numbers (2026-11-10, live bucket)

```
corpus         489 ready names
runs             4 (plus later *_yt500inst-regen)
takes scanned   1425
accepted         185
rejected           8   (0.32–0.83 of nominal length, all short_take)
pending          296
```

The eight rejected ratios: 0.315, 0.332, 0.558, 0.604, 0.644, 0.740, 0.759,
0.829 — the plan's predicted 0.32–0.83 band.
