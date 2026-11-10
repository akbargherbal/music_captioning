# YT500 instrumental — streaming captions + ABC dataset

**Status:** implemented, captions only (see the scope amendment) · **Date:** 2026-11-10 · **Owner:** akbar
**Generation side:** `arabmaqamrock/docs/plans/yt500-instrumental-generation.md`
**Reference package:** `gs://akbar-december-2024-backup/DATASET/fadhl_shaker_instrumental_18/`

## Scope amendment (2026-11-10, supersedes every transcription statement below)

**This pipeline produces captions and polish only. It never runs SheetSage2 and
never transcribes anything.**

The ABC scores already exist on GCS
(`gs://akbar-december-2024-backup/TRANSCRIPTS/YT500/`) and are the **input** to
generation. They are not an output of this pipeline, and the instrumentals on
the A100 are rendered from them.

What that means section by section:

| Section | Status |
|---|---|
| §0.1 #1, "L4 24 GB" | No longer load-bearing. The 24 GB existed for caption **+** SheetSage2 on one box; the captioner peaks at 8.0 GiB and the rewriter runs in a separate phase. **A T4 is sufficient** (~56 s/track, ~7.6 h for 489) and needs no compile for `llama-server`, since `sm75-t4` is published. An L4 stays ~2× faster (~29 s/track, ~3.9 h). |
| §1, §2, §5 ("full parity" / reference ABC / transcription) | Transcription is dropped. The deliverable is the caption album. |
| §6.5 `--phase transcribe` | Implemented, but **never invoked**. Everything runs with `--phase caption`. |
| §6.7, §8 transcript gates | Not applicable. The pilot gates on captions only. |
| §7.4 transcription environment | **Skip entirely** — no `setup_transcribe.sh`, no abcmidi. |
| §9.2 `--phase both` | Use `--phase caption`. |
| §9.4 captions-only profile | This is now the *only* profile, not a T4 fallback. |
| §10 ETA table | The caption row applies: ~29 s/track, ~4 h for 489 on an L4. |
| §11 `--status` example | `pending trs` counts every local MP3 (no transcripts exist), so it equals the local audio count. Ignore the `transcribed` / `pending trs` lines. |
| §12 final assembly, §16 checklist | Deferred. `yt500_assemble.py` needs 489 `ok` transcripts, so it is not part of this run. |

The implemented interface (flags, contracts, the behaviours this plan left open)
is in [YT500_INSTRUMENTAL_OPS.md](YT500_INSTRUMENTAL_OPS.md).

## 0. What this is (read this first)

The `arabmaqamrock` A100 is rendering the **489 usable tracks of the YT500 ABC
corpus** as instrumentals with the exact `fadhl_shaker_instrumental_18` recipe
(Yue2 3B, `instrumental_v9` LoRA, style + full CoT, AR 0.85 / NAR 1.0), uploading
each finished MP3 to `gs://akbar-december-2024-backup/arabmaqamrock/runs/`. This
project turns those MP3s into a published dataset while generation is still
running: it watches the run bucket, rejects takes whose duration is far below the
score (they will be regenerated), captions and transcribes only the accepted
takes, and assembles the same package layout as the 24-track Fadl Shaker
reference dataset — caption + reference ABC + independent SheetSage2
transcription, with audio referenced by GCS URI and SHA-256.

At the time of writing: **122 / 489 generated**, 367 pending, A100 ETA ~19:40
UTC. The dataset can start immediately on the finished takes and follow the rest.

### 0.1 Decisions to make in this chat (answer before implementing)

Recommended answers are first. A short reply such as "1A, 2A, 3A, 4A" is enough.

| # | Question | Options |
|---|---|---|
| 1 | Which VM? | **A. L4 24 GB** — caption + transcription on one box, ~10–11 h for 489. **B. T4** — captions only (~6.8 h); transcription needs a second VM later. |
| 2 | Scope now? | **A. Full parity** — caption + reference ABC + SheetSage2 transcription. **B. Captions first**, transcription as a separate later pass. |
| 3 | Rejection threshold? | **A. `0.85`** of the score's nominal length (catches the measured 0.32–0.83 takes; 7/147 at the last scan). **B. `0.90`** (stricter; 9/147). Upper bound `1.25` either way. |
| 4 | When to regenerate short takes? | **A. After 489/489 on the A100** — one small regen run, no interference. **B. Restart the live supervisor with the gate now** — no waiting, but interrupts the bulk and mixes regen takes into the main run. |
| 5 | Final prefix? | **A. `DATASET/YT500_INSTRUMENTAL`** (version inside `manifest.json`). **B. `DATASET/YT500_INSTRUMENTAL_489`** (count in the name). |
| 6 | Publish policy? | **A. Work state under `GENERATION/.../work/`; final package only at 489/489** (no partial dataset). **B. Assemble/upload a preview package periodically** (more visibility, more hashing/upload work). |

**Answered 2026-11-10: 1A, 2 → captions only (amended, see the scope amendment
above; option 2B without the "later pass" commitment), 3A, 4A-with-a-caveat,
5A, 6A, §6.4 deferred.**

- **4** was answered as "never wait": the watcher queues accepted takes as they
  land and a regenerated take simply wins the `end_utc` selection on the next
  poll, so nothing is blocked by an unfinished sibling and nothing has to be
  restarted for a regen. The regen *render* still happens after 489/489 (4A).
- **§6.4** (`plumbing/regen_short_takes.py`) is **not** part of this repo's
  implementation: the A100 side owns it. This repo only watches, captions and
  polishes. Implemented behaviour is in
  [YT500_INSTRUMENTAL_OPS.md](YT500_INSTRUMENTAL_OPS.md).

### 0.2 How to use this document in this chat

1. Read this file fully; it is self-contained.
2. Answer §0.1, and say whether the A100 regen script below is implemented in
   this chat (with `arabmaqamrock` cloned) or left to the A100 chat.
3. Implement §6 in the stated order, run the CPU tests, push.
4. Do not start GPU work until the §8 pilot gates pass.
5. The A100 generation run is read-only from this repo; never write to
   `runs/`, never delete a take.

## 1. Goal

While the A100 generates the 489 YT500 instrumentals, a second Colab VM
(T4 or L4) watches the run bucket and, for every **accepted** MP3, produces:

1. a prose **caption** (ACE-Step 4-bit + Qwen3 polish), and
2. a **SheetSage2 ABC transcription** of the rendered audio,

plus the **reference ABC** that generated it. The result is one package with the
same schema as `fadhl_shaker_instrumental_18`, published at

```
gs://akbar-december-2024-backup/DATASET/YT500_INSTRUMENTAL/
```

**Definition of done:** 489 names, each with an **accepted** take, a caption, a
reference ABC and an ABC transcription; `manifest.json` verifies; `_SUCCESS`
exists; the upload comparison reports 0 missing / 0 mismatched / 0 extra.

Audio is **never copied into the dataset**. Rows reference the MP3 by
`gs://` URI + SHA-256.

## 2. State at the start

| Item | Value |
|---|---|
| A100 generation | **122 / 489 done**, 367 pending at 15:01 Bahrain |
| A100 rate | ~48 tracks/h, ETA ~19:40 UTC (~22:40 Bahrain) |
| Completed MP3s | `gs://.../arabmaqamrock/runs/*_yt500inst/<name>_<seed>.mp3` |
| Per-take proof | `<name>_<seed>_time.txt` contains `Exit status: 0` |
| Per-take metadata | `<name>_<seed>.json` sidecar + `batch_manifest.json` per run |
| Corpus | 489 ready scores; 4 `no-abc` + 2 inline-key excluded |
| Reference ABC | `scores/yt500_instrumental/<name>.abc` (fetchable, hashed) |
| Reference dataset | 24 tracks, 6,550 s; caption RTF 9.5×, transcription RTF 6.3× |

**Measured truncation, 147 finished takes scanned on 2026-11-10:**

| Ratio (`wav_duration_s / nominal_seconds`) | Takes |
|---|---:|
| `< 0.85` | **7** |
| `< 0.80` | **6** |
| `< 0.75` | **6** |
| 0.85–1.25 (accepted band) | 140 |

The two worst are ~0.32× (98 s and 93 s vs ~300 s scores). The batch sidecars
carry `cap` and `wav_duration_s` but **no `truncated` field**, and the A100
done-set only checks the score hash and `Exit status: 0`. These takes are
therefore "done" today and would never be regenerated. §3 and §6.4 fix that.

## 3. Acceptance gate and take selection

This is the rule the watcher and the final audit use.

### 3.1 One accepted take per name

For every corpus name, consider all takes in all `*_yt500inst*` runs:

```
ratio = wav_duration_s / nominal_seconds        (from the corpus index)
```

A take is **accepted** only when all of these hold:

- `batch_manifest.json` has the name with `abc_sha256` == frozen corpus hash;
- `<name>_<seed>_time.txt` says `Exit status: 0`;
- `<name>_<seed>.mp3` exists;
- `<name>_<seed>.json` sidecar has `wav_duration_s`;
- `0.85 <= ratio <= 1.25`.

Defaults are configurable: `--min-ratio 0.85 --max-ratio 1.25`. The lower bound
is the plan's ±15 % length criterion; the upper bound catches runaway loops.
A scan of the 147 finished takes found a maximum of 1.184, so the upper bound is
a runaway guard, not a routine filter.

**Selection:** among accepted takes for a name, choose the one with the latest
`end_utc` (a regen take replaces the original). Record `run_id`, `seed`, `take`,
`ratio`, `wav_duration_s`, `nominal_seconds`.

**Rejection:** a name with no accepted take goes to `work/rejected.json` with
every take's reason (`short_take`, `long_take`, `no_marker`, `abc_mismatch`,
`missing_mp3`, `missing_sidecar`) and is **not** downloaded or captioned. It
stays pending until a regen take arrives.

### 3.2 Regeneration on the A100

Skipping alone would leave permanent holes. After the bulk generation reaches
489/489, run a **regen phase** on the A100 (new script, §6.4):

- read the rejected list (or rescan GCS with the same ratio rule);
- for each rejected name, render a new take with a **new seed**, same score and
  recipe, into a new `*_yt500inst-regen` run;
- accept when the new ratio is in band; retry up to 3 seeds;
- never delete or overwrite the short take (append-only);
- write `regen_summary.json` and stop.

The watcher sees the new run normally, reselects the take by `end_utc`, and the
GPU worker captions/transcribes only the replacement. A name still out of band
after 3 seeds is reported as `needs_human`; the final gate does not silently
accept it.

Why after the bulk: the live supervisor is serial and GPU-saturated. Changing
its done-set mid-run would require a restart and would mix regen takes into the
main run. The regen set is tiny (~7 per 147, ~23 projected for 489) and costs
~30–45 min.

## 4. Separation of concerns

Four independent processes, one shared local filesystem, **no shared writable
state**. The contract is the files in `work/`, written atomically. GCS I/O is
confined to the watcher (read/restore) and the publisher (write/verify); the GPU
worker never talks to GCS.

```
GCS runs/*_yt500inst*/ ──▶ yt500_sync.py ──▶ work/audio/, work/manifest.json,
   (watcher, CPU/net)                        work/rejected.json
                                                    │
                                                    ▼
                                          yt500_learn.py (GPU)
                                          caption album + transcripts
                                                    │
                                                    ▼
                    work/caption_album.json, work/captions.jsonl, work/transcripts/
                                                    │
                                                    ▼
                                          yt500_publish.py (CPU/net)
                                          upload + verify work artifacts
                                                    │
                                                    ▼
                                          yt500_assemble.py (CPU)
                                          final package + _SUCCESS
```

| Process | Runs on | Owns (single writer) | Never does |
|---|---|---|---|
| `yt500_sync.py` | CPU/network | `work/audio/`, `work/manifest.json`, `work/rejected.json`, `work/status_sync.json` | load a model, caption, transcribe |
| `yt500_learn.py` | GPU | `work/caption_album.json`, `work/captions.jsonl`, `work/transcripts/`, `work/status_learn.json` | read/write GCS |
| `yt500_publish.py` | CPU/network | GCS `work/` prefix, `work/status_publish.json`, upload ledger | caption, transcribe, download audio |
| `yt500_assemble.py` | CPU | final package, `work/status_assemble.json` | load a model |

Why this matters:

- A slow upload never stalls the GPU; a GPU crash never loses downloaded audio.
- The watcher can prefetch while the GPU worker is captioning.
- A fresh VM restores `caption_album.json` and `transcripts/` through the
  watcher; the GPU worker only reads local files.
- Each role has its own lock, log and done marker; restart one without touching
  the others.

### 4.1 Contracts

- **Downloads:** write `*.mp3.part`, `fsync`, rename to `*.mp3`, then update the
  manifest. The GPU worker reads only paths listed in the manifest.
- **Caption album:** written to `.tmp` then `os.replace` (already how
  `build_dataset.py` works). The publisher uploads only after the mtime is stable.
- **Transcripts:** upload only stems whose `meta.json.status == "ok"` and whose
  `score.abc` matches `meta.abc_sha256`.
- **Status:** each process writes its own `status_<role>.json`; `--status`
  merges them. No shared status file.
- **Locks:** `work/.sync.lock`, `.learn.lock`, `.publish.lock`, `.assemble.lock`.
  A second instance exits instead of double-writing.
- **Manifest:** `{corpus_total, counts, accepted: [...], rejected: [...]}`,
  rewritten atomically by the watcher only.

## 5. Design decisions (detailed rationale)

These are the defaults behind §0.1. Change them only if the answers there differ.

| Decision | Recommendation | Why |
|---|---|---|
| VM | **L4 24 GB**, one VM | Runs caption + transcription in ~10 h; a T4 cannot do both in real time |
| Scope | **Full parity**: caption + reference ABC + transcription | Matches `fadhl_shaker_instrumental_18` |
| Truncation | **Skip** `ratio < 0.85` or `> 1.25`; regen on A100 | Prevents captioning bad takes |
| Deployment | **Three detached workers** (sync / learn / publish) + assembler | Isolation; matches the YT500 transcription precedent |
| Order | Captions first, then transcription, in passes | Captions are the immediate deliverable |
| Publish | Work state under `GENERATION/.../work/`; final package only at 489/489 | No partial dataset published as final |
| Dataset prefix | `DATASET/YT500_INSTRUMENTAL` | One corpus package; version in `manifest.json` |
| Batch size | `--min-batch 12`, `--flush-minutes 20` | Amortizes model load without lagging |
| Dedupe | One accepted take per `name`, latest `end_utc` | Regen take replaces the short one |
| Audio | GCS URI + SHA-256, no bytes | Same as reference |

If only a **T4** is available, run `yt500_learn.py --phase caption` now
(~6.8 h for 489); add transcription later on a second VM.

## 6. Code changes

§6.1–§6.3 and §6.5–§6.6 are changes in **this repo** (`music_captioning`);
§6.4 is a change in **`arabmaqamrock`** (A100 side) and only listed here because
the dataset depends on it. The reference package was built from
`music_captioning@2368821`; the changes keep schema and pipeline version 3.

### 6.1 New `scripts/yt500_sync.py` (watcher)

- list `runs/*_yt500inst*`, read `batch_manifest.json`, sidecars, `_time.txt`
  and MP3 metadata (no audio download for metadata);
- apply the §3 gate, group by name, select one accepted take;
- download accepted MP3s atomically; optionally prefetch a bounded window;
- restore `caption_album.json` and `transcripts/` from GCS on a fresh VM;
- write `manifest.json`, `rejected.json`, `status_sync.json`;
- `--watch/--once`, `--min-ratio/--max-ratio`, `--limit`, `--interval`.

`--limit N` truncates the queue in `manifest.json` to N accepted takes (the
full count stays in `counts.accepted`). It exists for the 4-track pilot and is
**pilot-only**: a later pass without `--limit` rewrites the manifest, but until
then the caption album cannot exceed N tracks.

It reads the corpus index (`outputs/YT500/generation/index.json`) for
`nominal_seconds` and metadata.

### 6.2 `scripts/build_dataset.py` — incremental caption

1. **Skip the short-circuit when a caption failed.**
   `caption_phase` returns "already polished" when the track count and stage
   match, even with `failed_tracks`. Add `and not data.get("failed_tracks")`.
2. **Polish only pending tracks on an incremental pass.**
   `polish_album` re-runs the rewriter for every track each invocation; at 300+
   tracks that is 15–25 min wasted per pass. Add `only_pending=True` (when
   `existing` was reused): skip tracks with
   `rewrite_status ∈ {ok, fallback, single_chunk}` and a non-empty `caption`.
   Keep `--repolish`. Count reused tracks in the summary.
3. **No skips.** Run with `--max-track-seconds 0 --min-bitrate-kbps 0`. The
   longest source is 416 s and generated takes stay under ~500 s; a skip would
   also make the track count never settle.

These two code edits do not change the model, prompt, chunking, merge or
pipeline version.

### 6.3 `scripts/build_dataset.py assemble` — multi-run corpus

- use `track["run_id"]` when present, else `--run-id`, for `row["run_id"]` and
  `track_id = f"{row_run_id}/{stem}"`;
- use `track["gs_uri"]` when present, else `f"{source_uri}/{audio_name}"`;
- add optional columns from the corpus index: `title`, `source_artist`,
  `country`, `tier`, `source_rank`, `source_duration_s`,
  `source_abc_sha256`, `generation_status`, `take_ratio`, `take`;
- add `source_runs`, `rejected`, and take-selection counts to `manifest.json`;
- adjust the card wording from "one run" to "one or more runs".

Keep every existing field and the `qc_track` path unchanged.

### 6.4 New `plumbing/regen_short_takes.py` (arabmaqamrock, A100)

- input: `--rejects work/.../rejected.json` or `--scan` (recompute the ratio
  from GCS);
- build a session manifest per rejected name with a new seed, `take = old + 1`,
  same `cap`, `cot=full`, `lora=instrumental_v9`, AR 0.85 / NAR 1.0;
- render one batch into `out/<utc>_yt500inst-regen`, reusing the driver's
  `run_batch` and `--request-sequence`;
- accept when the new ratio is in band; retry up to `--max-attempts 3`;
- write `regen_summary.json`; never delete the short take;
- refuse to run while the generation supervisor is alive (`run.lock`).

### 6.5 New `scripts/yt500_learn.py` (GPU loop) + phase wrappers

`yt500_learn.py` is the only GPU process. It reads `work/manifest.json`, picks
the pending phases, and calls two thin wrappers:

- `caption`: restores/validates the local album, calls
  `build_dataset.py caption` with the growing manifest, writes
  `caption_album.json` + `captions.jsonl`;
- `transcribe`: calls `transcribe.sh --full` on the accepted local audio,
  `--dtype auto`, `--out-dir work/transcripts`;
- both skip the model load when their pending count is zero.

It never reads or writes GCS; the watcher restores its inputs and the publisher
uploads its outputs.

### 6.6 New `scripts/yt500_publish.py`, `yt500_assemble.py`, `yt500_dataset.py`

- `publish`: rsync `caption_album.json`, `captions.jsonl`, `transcripts/`,
  `manifest.json`, `rejected.json`, status files to `work/`; verify size+MD5;
  write an upload ledger; never delete.
- `assemble`: check 489 accepted + 489 captions + 489 transcripts; run
  `build_dataset.py assemble`; write the package; the publisher uploads it.
- `yt500_dataset.py --status`: read-only aggregator that merges the four
  `status_<role>.json` files and the manifest for the progress view. It starts
  no model and writes no state.

### 6.7 Tests (CPU only)

- gate: `_time.txt` + ABC hash + MP3 + ratio band; rejects 0.84 and 1.26;
- selection: latest accepted take wins over an older short take;
- sync: atomic download; manifest excludes a rejected stem;
- learn: a second pass re-captions **zero** and re-polishes **zero** stems;
- publish: only `status: ok` transcripts upload; no delete;
- assemble: per-track `run_id`/`gs_uri`, reference schema, 489-row fixture;
- regen: a new seed is chosen; 3 failures produce `needs_human`; old take intact.

### 6.8 Delivery order

1. §6.1 + §6.2 + tests; push.
2. §6.3 + §6.4 + §6.7; push.
3. §6.5 + §6.6; push.
4. On the L4: §7 setup, §8 pilot.
5. §9 backlog + watch; §10 regen on the A100 when generation completes.
6. §12 final assembly.

## 7. Setup runbook (fresh Colab L4/T4)

### 7.1 Repos and scores

```bash
git clone https://github.com/akbargherbal/arabmaqamrock.git /content/arabmaqamrock
git clone https://github.com/akbargherbal/music_captioning.git /content/music_captioning

cd /content/arabmaqamrock
python3 plumbing/yt500_generation_corpus.py --fetch   # 489 scores + index + manifest
```

`--fetch` verifies the bundle checksums and populates
`scores/yt500_instrumental/*.abc` and `outputs/YT500/generation/index.json`
(including `nominal_seconds`).

### 7.2 Caption environment

```bash
cd /content/music_captioning
pip install -q -r requirements.txt

python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit",
                  local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                  "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt",
                local_dir="models/Ace-Step-Captioner-4bit")
PY
```

### 7.3 Polish backend

```bash
cd /content/music_captioning
GGUF=Qwen3-4B-Instruct-2507-Q8_0.gguf
PREFIX=gs://akbar-december-2024-backup/music_captioning

gcloud storage cp "$PREFIX/gguf/$GGUF" models/ 2>/dev/null || {
  python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q8_0.gguf", local_dir="models")
PY
  gcloud storage cp "models/$GGUF" "$PREFIX/gguf/$GGUF"
}

./plumbing/fetch_llamacpp_text.sh
```

`sm75-t4` is published. **There is no `sm89-l4` tarball yet**, so an L4 builds
`llama-server` from source (~5–10 min). If it fails (no `nvcc`), publish one
from a CPU VM:

```bash
./plumbing/fetch_llamacpp_text.sh --build --publish --cuda-arch 89
```

### 7.4 Transcription environment — **SKIPPED (see the scope amendment)**

Do not run any of this. No `setup_transcribe.sh`, no `abcmidi`, no
`transcribe.sh`, no transcribe venv. The ABC scores already exist on GCS and
this pipeline only captions.

### 7.5 Disk / auth

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h /content
gcloud storage ls gs://akbar-december-2024-backup/ >/dev/null && echo GCS_OK
```

Budget: ~11 GB models (no transcribe venv) + ~3 GB audio + temp ≈ 15–20 GB.

## 8. Pilot and gates

Pick 4 accepted tracks spanning runs and lengths (short < 220 s, median
~270 s, long > 350 s, one from the active run), plus **one known short take**:

```bash
cd /content/music_captioning
python3 scripts/yt500_sync.py --once --limit 4
python3 scripts/yt500_learn.py --once --phase caption --verify-schema
python3 scripts/yt500_publish.py --once --settle-seconds 0
```

Gates:

- [ ] 4/4 captions non-empty; `rewrite_status` `ok` (or `single_chunk`).
- [ ] `--verify-schema` prints `schema  : ok`.
- [ ] ~~4/4 transcripts~~ — not applicable, see the scope amendment.
- [ ] `manifest.json` accepted rows all carry `abc_sha256` from the frozen
      corpus (checked by the watcher's hard stop).
- [ ] **Truncation gate:** the known short take appears in `rejected.json` with
      `reason: short_take`, is **not** in `manifest.json`, and was **not**
      downloaded or captioned.
- [ ] Peak VRAM: caption < 11 GiB; no OOM.
- [ ] **Resume gate:** after the pilot publish has put the album in GCS,
      delete only the local album; re-run `--once`. The watcher restores it and
      the GPU worker does **zero** model work (no `caption:` line in the log).
- [ ] **Streaming gate:** wait for one new MP3; the next sync adds only that
      stem; the next learn pass processes only it.
- [ ] Listen to 2 tracks; confirm the description matches the audio and does
      not invent a lead vocal.

If a gate fails, do not start the bulk.

## 9. Bulk

### 9.1 Backlog first (122 accepted, 7 rejected at the 147-take snapshot)

```bash
cd /content/music_captioning
python3 scripts/yt500_sync.py --once            # downloads accepted audio
python3 scripts/yt500_learn.py --once --phase caption
python3 scripts/yt500_publish.py --once
```

Expect ~1 h of captioning on an L4 for 122 tracks (~29 s/track); the
transcription half of the original estimate does not apply.

### 9.2 Start the three independent workers

Start the publisher first (it waits), then the watcher, then the GPU worker:

```bash
cd /content/music_captioning
mkdir -p logs

setsid nohup python3 -u scripts/supervise.py \
  --script scripts/yt500_publish.py --log logs/yt500_publish.out -- \
  --watch --interval 60 > logs/yt500_publish.sup.out 2>&1 < /dev/null &

setsid nohup python3 -u scripts/supervise.py \
  --script scripts/yt500_sync.py --log logs/yt500_sync.out -- \
  --watch --interval 60 --min-ratio 0.85 --max-ratio 1.25 \
  > logs/yt500_sync.sup.out 2>&1 < /dev/null &

setsid nohup python3 -u scripts/supervise.py \
  --script scripts/yt500_learn.py --log logs/yt500_learn.out -- \
  --watch --phase caption --min-batch 12 --interval 60 --flush-minutes 20 \
  > logs/yt500_learn.sup.out 2>&1 < /dev/null &

disown || true
```

Each worker is independently restartable, and `supervise.py` restarts one that
is killed from outside Python (host OOM, native crash, session reaper). Do not
pass `--stop-when-complete`: a regen take can still arrive after 489/489. The watcher never blocks the GPU; the
publisher never blocks either. The GPU worker never calls GCS.

One immediate liveness check:

```bash
pgrep -af "yt500_sync|yt500_learn|yt500_publish|build_dataset|transcribe.py" | head
tail -20 logs/yt500_sync.out logs/yt500_learn.out logs/yt500_publish.out
python3 scripts/yt500_dataset.py --status
```

Then end the turn.

### 9.3 Regen phase (A100, after 489/489)

The A100 supervisor writes `logs/yt500_supervisor_done.txt` with
`exit=0 pending=0`. Then, with no generation process running:

```bash
cd /content/arabmaqamrock
python3 plumbing/regen_short_takes.py \
  --rejects gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/rejected.json \
  --out-root out --max-attempts 3
```

(Or `--scan` to recompute the list from GCS; the script downloads the GCS
rejects file if given a `gs://` URI.) The regen run is
`out/<utc>_yt500inst-regen`; the caption VM's watcher picks it up on the next
poll. No caption/transcript work happens for a rejected stem before this.

### 9.4 Captions-only profile (T4)

```bash
setsid nohup python3 -u scripts/yt500_learn.py --watch \
  --phase caption --min-batch 12 --interval 60 --flush-minutes 20 \
  > logs/yt500_learn.out 2>&1 < /dev/null &
disown || true
```

Transcripts are added later with `--phase transcribe`; the caption album is
reused.

## 10. Throughput and ETA

Measured on the reference L4 run (24 tracks, 6,550 s):

| Phase | Per track | 489 tracks | 122 backlog |
|---|---:|---:|---:|
| Caption + polish | 28.9 s | 3.9 h | 59 min |
| SheetSage2 full | 43.3 s | 5.9 h | 88 min |
| **Combined (serial GPU)** | **72.2 s** | **9.8 h** | **2.5 h** |

Add ~10 % for model loads/uploads → **~10.5–11 h** on an L4, plus the regen
phase (~30–45 min) and captioning the replacement takes (~10 min):
**~11.5–12 h**, i.e. ~3 h after the A100 finishes.

T4: caption ~50 s, transcribe ~70 s → **~2 min/track**, ~16 h combined;
captions alone ~6.8 h.

## 11. Monitoring

```bash
cd /content/music_captioning
python3 scripts/yt500_dataset.py --status
```

Expected shape:

```text
corpus         489
accepted       147
rejected         7
captioned      122
transcribed    118
pending cap     25
pending trs     29
last sync      2026-11-10T15:42:10Z
last learn     2026-11-10T15:40:02Z (caption, 25 tracks)
last publish   2026-11-10T15:41:16Z
regen          0 / 7 accepted
```

GCS work prefix:

```
GENERATION/YT500_INSTRUMENTAL/work/
  manifest.json            accepted takes (one per name) + counts
  rejected.json            names with no accepted take + reasons
  caption_album.json       resumable caption album (all captions + polish)
  captions.jsonl           one readable line per captioned track
  transcripts/<stem>/      score.abc + meta.json + side artifacts
  status_sync.json         watcher status
  status_learn.json        GPU worker status
  status_publish.json      publisher status
  status_assemble.json     final assembly status
  logs/                    per-role logs
```

Read captions as they land:

```bash
gcloud storage cat gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/captions.jsonl \
  | tail -5
```

Read the rejected list:

```bash
gcloud storage cat gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/rejected.json
```

Listen without disturbing either VM:

```bash
mkdir -p /tmp/yt500_listen
gcloud storage cp "gs://akbar-december-2024-backup/arabmaqamrock/runs/*_yt500inst*/*.mp3" /tmp/yt500_listen/
ls -lt /tmp/yt500_listen | head
```

Final prefix only after 489/489 accepted:

```
DATASET/YT500_INSTRUMENTAL/
  tracks.jsonl  tracks.parquet  manifest.json  DATASET_CARD.md  _SUCCESS
  captions/<stem>.json
  abc/<stem>.abc
  abc_transcription/<stem>.abc
```

## 12. Final assembly, upload and verification — **DEFERRED**

Not part of the captioning run: `yt500_assemble.py` requires 489 `ok`
transcripts and this pipeline produces none (see the scope amendment). The
commands below stay as the operator fallback for whenever the transcription
side of the package is resolved. Never use `--force` here — a debug build goes
to `work/package_debug/` so a partial package cannot reach the final prefix.

```bash
cd /content/music_captioning
python3 scripts/build_dataset.py assemble \
  --run-manifest  work/manifest.json \
  --audio-dir     work/audio \
  --captions      work/captions \
  --abc-ref       /content/arabmaqamrock/scores/yt500_instrumental \
  --transcripts   work/transcripts \
  --out           work/package \
  --run-id        YT500_INSTRUMENTAL \
  --source-uri    gs://akbar-december-2024-backup/arabmaqamrock/runs \
  --dataset-name  "YT500 Instrumental 489" \
  --dataset-version v0.1

gcloud storage rsync -r --checksums-only work/package \
  gs://akbar-december-2024-backup/DATASET/YT500_INSTRUMENTAL
```

Then verify:

- [ ] `wc -l tracks.jsonl` = **489**; parquet rows = 489.
- [ ] Every row: non-empty `caption`, `rewrite_status ∈ {ok,fallback,single_chunk}`,
      `abc_transcription_parse == "ok"`, `abc_reference_sha256` equals the
      corpus `instrumental_abc_sha256`, `0.85 <= take_ratio <= 1.25`.
- [ ] `manifest.json.counts` = 489 / 489 / 489 / 489; `rejected` is empty.
- [ ] `sha256sum -c` over the package's `files` list passes.
- [ ] Upload comparison by size + MD5: 0 missing, 0 mismatched, 0 extra.
- [ ] 5 spot-listens across different runs; captions match the audio.
- [ ] `_SUCCESS` contains the build timestamp.

## 13. Resume after a VM loss

Everything needed is in GCS. On a fresh VM:

```bash
cd /content/arabmaqamrock && git pull --ff-only
cd /content/music_captioning && git pull --ff-only
# redo §7.2–7.4 (models/venv are lost), then relaunch §9.2.
```

Each worker restores its own state:

- **sync** re-lists GCS, restores `caption_album.json` and `transcripts/`, and
  downloads only accepted audio that is missing locally;
- **learn** sees the restored album/transcripts and skips every valid stem;
- **publish** re-verifies the remote ledger and uploads only what changed
  (no duplicate uploads, no deletes).

`exit=143` (SIGTERM) is a VM teardown, not a pipeline failure. If a worker is
alive, leave it alone; do not start a second one — the per-role lock prevents it.

## 14. Failure handling

| Symptom | Worker | Action |
|---|---|---|
| Captioner OOM on a long take | learn | Keep `--max-track-seconds 0`; check `nvidia-smi`; serial phases stay < 10 GiB. Do not change the recipe |
| Rewrite error/OOM | learn | `rewrite_status=error`; album stays `captioned`; next pass retries only that track; `caption_raw` is preserved |
| SheetSage2 `no-abc` / `parse_fail` | learn | Row keeps its QC; retry once with a subset `--force`; never `--force` in bulk |
| Transcript `stale`/`invalid` | learn | Source MP3 changed; retry that stem; investigate why an immutable take changed |
| Short take (`ratio < 0.85`) | sync | `rejected.json`, no download/caption; A100 regen (§9.3) |
| Long take (`ratio > 1.25`) | sync | Same; inspect for a runaway loop |
| MP3 present, `_time.txt` absent | sync | In-flight or failed take; skip until the marker exists |
| ABC hash mismatch | sync | Corpus/run mismatch; stop and report; do not caption |
| GCS auth expired | sync/publish | `gcloud auth login` is interactive; ask the user, do not loop |
| Disk < 20 GB | sync | Publish state, remove `work/audio` (re-syncable) and old logs; resume |
| Same stem in two runs | sync | Group by name; latest accepted `end_utc` wins; log the duplicate |
| Publisher down | publish | GPU keeps captioning; uploads catch up on restart; no data loss (local state) |
| Watcher down | sync | GPU finishes the local backlog, then waits; no new downloads |
| Second worker started | any | Per-role lock exits; stop one |
| A100 stalls but caption VM is fine | sync | Keep polling; `rejected` stays pending; `--idle-stop-minutes` only after 489 accepted |

## 15. Risks and open questions

1. **Regen is new code on the A100.** It reuses `run_batch`, but it must be
   tested on one short take before the regen set. If it fails, the affected
   names remain rejected; the dataset is published at 489 − N, not silently
   with bad takes.
2. **The `0.85` threshold is a judgement.** It catches the measured 0.32–0.83
   cases. A stricter 0.90 rejects 9/147; it is configurable. A human can review
   `rejected.json` before regenerating.
3. **L4 `llama-server`:** no `sm89-l4` GCS tarball exists. Build locally (§7.3)
   or publish one from a CPU VM. Without it, the GGUF polish backend cannot run.
4. **One T4 cannot keep up** with both phases at 48 tracks/h. Captions only, or
   a second T4.
5. **Concurrent caption/transcribe on an L4:** captioner ~9.8 GiB + SheetSage2
   ~3.8 GiB fit in 24 GiB; the rewriter adds ~4.5 GiB. Could halve wall time,
   but needs a VRAM-watchdog pilot. Not in v1.
6. **Caption quality on instrumentals:** the model can mention a brief vocal
   because the generated audio occasionally contains vocal artifacts. The
   reference package shows the same (`vocal_sparse`); do not change the prompt.
7. **Rows reference audio, not ship it:** consumers need GCS read access. Same
   policy as `fadhl_shaker_instrumental_18`.
8. **Licensing:** source audio is YouTube-derived; the package contains
   captions/ABC and GCS URIs. Confirm distribution before making the bucket
   public.

## 16. Definition of done — checklist

- [ ] 489 / 489 generated MP3s accepted (`ratio` in band; `rejected` empty).
- [ ] 489 / 489 captions polished; `rewrite_status` distribution recorded.
- [ ] 489 / 489 SheetSage2 transcriptions `ok`; QC flags summarized.
- [ ] 489 reference ABCs match the corpus hashes.
- [ ] `DATASET/YT500_INSTRUMENTAL/` uploaded and checksum-verified.
- [ ] `_SUCCESS` + `DATASET_CARD.md` written.
- [ ] 5 spot-listens done.
- [ ] Work state (`caption_album.json`, transcripts, `manifest.json`,
      `rejected.json`) retained for future re-polish or schema upgrades.
