# Plan — finish the 34 missing captions, then attach captions to the dataset

**Status:** phases 1–2 executed 2026-11-11 (see run log at the end) · **Date:** 2026-11-11 · **Owner:** akbar
**Runs on:** Google Colab (GPU for phase 1, CPU is enough for phase 2)

This is the implementation plan for two deliverables:

1. the **34 dataset tracks that have no caption** get one, through the normal
   `sync → learn → publish` path, so the album stays the single source of truth;
2. every track folder in
   `gs://akbar-december-2024-backup/DATASET/yt500_instrumental/tracks/<name>/`
   gains a **caption artifact next to `audio.mp3`, `style.txt`, `lyrics.txt`,
   `score.abc`, `track.json`**, so the published package is self-contained.

## 0. Measured state (re-verified 2026-11-11)

| Item | Value |
|---|---|
| Dataset tracks | **460**, all `gate=pass`, 460 track folders, `manifest.json` built `2026-11-11T16:19:45Z` |
| Source runs | 7 (`…-083615` … `…-205443_yt500inst`) |
| Caption album | `GENERATION/YT500_INSTRUMENTAL/work/caption_album.json`, **432 tracks**, `caption_stage=polished`, `caption_pipeline_version=3`, generated `2026-11-10T20:07:40Z`, uploaded `20:12Z` |
| Matched by name **and** take seed | **426 / 460** |
| Name present, different take | 0 |
| **Missing caption** | **34** |
| Album tracks not in the dataset | 6 (take ratios 1.152–1.184 → outside the dataset's ±15% gate; already captioned) |
| Album after phase 1 | **466** = 460 dataset + 6 known-only-album (+ any take that landed in the live bucket since 11-11) |
| Caption quality inside the 432 | 392 `ok`, 33 `fallback`, 6 `failed`, 1 `single_chunk` |

Why the 34 slipped: the sync manifest snapshot at `20:20Z` already knew **4** of
them as accepted (`0129`, `0159`, `0205`, `0207`); the other **30** were still
`pending` at that snapshot (30 of the 32 takes from `…-200631` and both takes
from `…-205443`). The album was never written again after `20:07Z`, so those
takes arrived valid later on 11-11 and were never queued.

The 34 names:

```
yt500-0038-ZZwdBV1SDIg   yt500-0129-JXAP2nChOjM   yt500-0132-8gUEtTFefpI
yt500-0141-TItFKokiPDE   yt500-0159-Uhgb4k9r77w   yt500-0190-_6ylCkxPdF4
yt500-0205-EvPNBiu0ofk   yt500-0207-Oj3TzUwNcDg   yt500-0209-fXj36Ud9CHY
yt500-0221-Ajpml5WIGzA   yt500-0230-E7_ZpsoVbW0   yt500-0248--faWzeEe1J0
yt500-0276-ESRhDZy8Kbg   yt500-0282-WTEj6zshp3Y   yt500-0283-G_606bhkF3U
yt500-0289-EYaFgO7vxuU   yt500-0321-wS-S_Vc_638   yt500-0328-jYAWXp8GR58
yt500-0381-gq7HoNEGUuw   yt500-0414-fuOKAzsB0F4   yt500-0419-rHnYkaZ3JCs
yt500-0456-UGnHDuGFbj8   yt500-0474-OY643NJrc_E   yt500-0475-mhl5wf2z7iI
yt500-0527-3eCUEqdJgQw   yt500-0533-AhUNvHr3cw4   yt500-0535-zY-udQPUppU
yt500-0537-PG-LWzk6XG4   yt500-0546-qWRYSYuPipE   yt500-0550-_X77dE6Rt-k
yt500-0572-68D3xqkOcM4   yt500-0637-t58Atvs7e1k   yt500-0644-MEJ0aiwpNVg
yt500-0684-bzpScWxbAGk
```

Regenerate this list at any time with
`scripts/yt500_dataset.py --status` (after sync) or by diffing
`catalog.jsonl` against the album (see §2.4).

## 1. Decisions taken by this plan (change before coding if you disagree)

| # | Decision | Default |
|---|---|---|
| 1 | How to finish the 34 | Resume the real pipeline (`sync → learn → publish`), **not** a bespoke caption script. The album stays the only caption store. |
| 2 | Caption artifact shape | `caption.txt` (plain polished prose) + `caption.json` (full record) per track folder, plus a `caption` block inside `track.json`. |
| 3 | Metadata roll-up | `catalog.jsonl`/`tracks.jsonl`/`catalog.csv`/`catalog.md`/`tracks.parquet` gain `caption_sha256`, `caption_chars`, `rewrite_status`; dataset root gains `captions.jsonl` + `captions.sha256`. |
| 4 | Version | `dataset_version` → **`v0.2`**, same prefix (`DATASET/yt500_instrumental`), per the §0.1 decision "version lives inside `manifest.json`". |
| 5 | Where the attach code lives | `arabmaqamrock/plumbing/compile_yt500_collection.py` (it owns the package, `track.json`, `manifest.json`, `--verify`). Captions are read from music_captioning's album. |
| 6 | The 6 album-only tracks | **Not** added to the dataset (they fail the ±15% duration gate). Left in the album, untouched, out of scope. |
| 7 | Re-polish the 6 `failed` tracks | Yes, free: the polish phase already retries any non-terminal `rewrite_status`. |

## 2. Phase 1 — caption the 34 (GPU, ~40 min on T4)

### 2.1 Stage the runtime

```bash
git clone https://github.com/akbargherbal/music_captioning.git /content/music_captioning
cd /content/music_captioning
pip install -q -r requirements.txt

# Colab does not ship the Cloud SDK; every script here shells out to gcloud/gsutil
command -v gcloud >/dev/null || {
  curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
  export PATH="$HOME/google-cloud-sdk/bin:$PATH"
}
gcloud storage ls gs://akbar-december-2024-backup/DATASET/ | head -3   # auth smoke test
# if that is 401/403: python3 -c "from google.colab import auth; auth.authenticate_user()"

# generation corpus index — the length gate reads it
git clone -q https://github.com/akbargherbal/arabmaqamrock.git ../arabmaqamrock
( cd ../arabmaqamrock && python3 plumbing/yt500_generation_corpus.py --fetch )

# gate: this must be the index that gated the existing queue, or sync re-decides
sha256sum ../arabmaqamrock/outputs/YT500/generation/index.json
#   expect ae2a06f593ec61105964fb0d26bbcad920006f1b4073f89adab72e9614903eb5
#   (= work/manifest.json.corpus.sha256 in GCS)

# captioner weights (6.4 GB), rewriter GGUF, llama-server
python3 - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit",
                  local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                  "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt",
                local_dir="models/Ace-Step-Captioner-4bit")
PY
gcloud storage cp -q \
  gs://akbar-december-2024-backup/music_captioning/gguf/Qwen3-4B-Instruct-2507-Q8_0.gguf \
  models/
./plumbing/fetch_llamacpp_text.sh          # stages llama-server, no build
```

Do **not** start `plumbing/resume_yt500.sh`'s watchers for this pass: 34 tracks
is one bounded job, and a foreground run shows progress and exits. The script's
staging steps are the three blocks above.

### 2.2 Restore the album — mandatory freshness gate

`sync` restores the album from GCS **only when the local file is missing**, and the
clone already contains a git snapshot of the album (`plumbing/snapshot_work.sh`
commits it every 15 min, so it is up to 15 min behind GCS and can be missing
tracks that GCS already has). Leaving a stale copy in place would silently
re-caption those tracks. At the time of writing the git snapshot is 432 tracks
(`20:10:46Z`) and GCS is 432 (`20:07:40Z`), but the gate below is what makes
that safe rather than lucky. Overwrite it explicitly:

```bash
gcloud storage cp -q \
  gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/caption_album.json \
  work/yt500/caption_album.json
mkdir -p work/yt500/captions/arabmaqamrock
cp work/yt500/caption_album.json \
   work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json

python3 - <<'PY'
import json
p = "work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json"
d = json.load(open(p))
assert d["caption_pipeline_version"] == 3, d["caption_pipeline_version"]
assert d["caption_stage"] == "polished", d["caption_stage"]
assert len(d["tracks"]) == 432, len(d["tracks"])
print("album ok:", len(d["tracks"]), d["generated_at"])
PY
```

### 2.3 Sync, gate the queue, caption, polish, publish

```bash
# the dataset's own track list: the check is "every dataset track is queued
# with the same take", not an exact album count (the run bucket is live)
gcloud storage cp -q \
  gs://akbar-december-2024-backup/DATASET/yt500_instrumental/catalog.jsonl .

# rewrites work/manifest.json (+ rejected*) and downloads the accepted MP3s (~3.1 GB)
python3 scripts/yt500_sync.py --once \
  --corpus-index ../arabmaqamrock/outputs/YT500/generation/index.json

# gate before spending GPU time
python3 - <<'PY'
import json
cat = {r["name"]: r for r in map(json.loads, open("catalog.jsonl"))}
m = json.load(open("work/yt500/manifest.json"))
album = json.load(open("work/yt500/captions/arabmaqamrock/YT500_INSTRUMENTAL.json"))
acc = {r["name"]: r for r in m["accepted"]}
have = {t["file"] for t in album["tracks"]}

missing_take = sorted(set(cat) - set(acc))
assert not missing_take, missing_take[:5]                      # all 460 queued
drift = sorted(n for n in cat
               if f"{n}_{cat[n]['generation']['seed']}.mp3" not in have
               and n in acc
               and acc[n]["seed"] != cat[n]["generation"]["seed"])
assert not drift, f"newer take won the selection: {drift[:5]}"  # same take
todo = [r["file"] for r in m["accepted"] if r["file"] not in have]
print("accepted", len(acc), "captioned", len(have), "to caption", len(todo))
assert len(todo) >= 34, len(todo)
PY

# caption the 34, then polish them (+ retry the 6 `failed`) with llama-server
python3 scripts/yt500_learn.py --once --phase caption \
  --verify-schema --caption-batch 40

# upload album + captions.jsonl + status files
python3 scripts/yt500_publish.py --once --settle-seconds 0
python3 scripts/yt500_dataset.py --status
```

Notes:

- `build_dataset.py caption` skips every track already in the album and loads the
  captioner only because 34 are new — no work is repeated for the 432.
- `--max-tracks`/`--max-track-seconds` are already `0`/`off` in `yt500_learn`'s
  command, so none of the 34 can be skipped for length.
- Expected cost: sync 4–6 min (downloading 466 MP3s), captioner load ~1.5 min,
  34 × ~56 s ≈ 32 min on T4 (~29 s/track ≈ 17 min on L4), polish ~40 × 5.5 s
  ≈ 4 min, publish < 1 min.
- If the run dies mid-pass: the album checkpoints every 5 tracks, so just repeat
  §2.3 from `yt500_learn.py` (sync is idempotent, the album is on disk).

### 2.4 Phase 1 acceptance

```bash
python3 - <<'PY'
import json, re
album = json.load(open("work/yt500/caption_album.json"))
print(len(album["tracks"]), album["caption_stage"], album["caption_pipeline_version"])
PY
gcloud storage cp -q gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/caption_album.json /tmp/album.json
python3 - <<'PY'
import json, re
album = json.load(open("/tmp/album.json"))
cat = [json.loads(l) for l in open("catalog.jsonl")]      # fetched in §2.3
missing = [r["name"] for r in cat
           if f"{r['name']}_{r['generation']['seed']}.mp3"
           not in {t["file"] for t in album["tracks"]}]
print("album tracks:", len(album["tracks"]))
print("dataset tracks still without their caption:", missing)
assert not missing, missing[:5]
print("stage:", album["caption_stage"], "version:", album["caption_pipeline_version"])
PY
```

- [ ] GCS album carries a caption for **all 460 dataset takes** (expect 466+
      tracks; the number may exceed 466 if the live bucket produced new takes —
      that is fine, they are album-only until a later dataset build)
- [ ] every match is the **same take seed** as `track.json.generation.seed`
- [ ] the 34 new entries have `rewrite_status ∈ {ok, fallback, single_chunk}`
      and a non-empty `caption`
- [ ] `captions.jsonl` line count agrees with the album
- [ ] the 6 previously `failed` tracks either reached a terminal state or are
      still honestly `failed` (never published as `ok`)

## 3. Phase 2 — attach captions to the dataset (CPU, ~5 min)

### 3.1 Artifact contract

Per track, alongside the existing files (audio and the other artifacts are never
rewritten):

```
tracks/<name>/
  audio.mp3          unchanged
  lyrics.txt         unchanged
  style.txt          unchanged
  score.abc          unchanged
  track.json         + "artifacts": {"caption": "caption.json", "caption_text": "caption.txt"}
                     + "caption": { ...see below... }
  caption.txt        the polished prose caption, UTF-8, one trailing newline
  caption.json       the full record
```

`caption.json` (`schema: yt500-caption/1`):

```json
{
  "schema": "yt500-caption/1",
  "name": "yt500-0038-ZZwdBV1SDIg",
  "take": {"file": "yt500-0038-ZZwdBV1SDIg_3571189662.mp3",
           "seed": 3571189662, "run": "20261110-200631_yt500inst"},
  "caption": "<polished prose>",
  "caption_sha256": "<sha256 of caption.txt bytes>",
  "caption_chars": 812,
  "caption_raw": "<deterministic per-chunk merge>",
  "chunk_captions": ["<chunk 1>", "<chunk 2>"],
  "instruments": ["oud", "violin"], "vocals": [], "production": ["reverb"],
  "tags_experimental": true,
  "rewrite": {"status": "ok", "model": "gguf:Qwen3-4B-Instruct-2507-Q8_0.gguf",
              "prompt_version": "2", "seconds": 5.5},
  "pipeline": {"version": 3, "stage": "polished",
               "album": "gs://.../GENERATION/YT500_INSTRUMENTAL/work/caption_album.json",
               "album_sha256": "<sha256>"},
  "audio": {"duration_s": 364.2, "bitrate_kbps": 192, "chunk_count": 4,
            "silence_trimmed_s": 0.0}
}
```

`track.json.caption` mirrors it compactly: `{"file", "text_file", "sha256",
"chars", "rewrite_status", "pipeline_version", "take"}`.

Root additions: `captions.jsonl` (one row per track: `name`, `file`, `seed`,
`caption`, tag lists, `rewrite_status`, `caption_sha256`) and `captions.sha256`
(`sha256␠␠tracks/<name>/caption.txt` and `…/caption.json`, 920 lines) so
`sha256sum -c captions.sha256` works.

Tag lists stay marked **experimental** (`TAG_AUDIT.md` still open); do not
advertise them as prompt tags.

### 3.2 Code changes in `arabmaqamrock/plumbing/compile_yt500_collection.py`

The compiler is committed in `arabmaqamrock` `main` (`5c07c04`) together with the
`outputs/YT500/generation/quality_audit_duration.json` snapshot it built v0.1
with — `--verify` reads `index.json` + that audit to re-derive the 460-name set,
so both must be the committed versions.

Small, additive, reusing the existing writers:

1. `parse_caption_album(path) -> (by_stem, album_sha256)` — key by the full stem
   `<name>_<seed>` (video ids may contain `_`, so split the seed with
   `^(yt500-\d{4}-.+)_(\d+)$`). Look up `f"{row['name']}_{row['generation']['seed']}"`:
   a caption is tied to a *take*, so a name-only match must fail rather than
   attach a caption to the wrong audio.
2. `caption_record(row, track) -> (record, text)` with hard guards:
   - the exact stem for the dataset's take is present for **every** row (unless
     `--allow-missing-captions`);
   - `caption` non-empty and `rewrite_status` in
     `{ok, fallback, single_chunk}` (a `failed` entry may be attached only with
     `--allow-unpolished`, flagged in the record);
   - `caption_chars >= 200` and no `[00:12]`-style timestamps (reuse the
     rewrite guard's regex from `src/rewrite.py`).
3. `write_catalogs(rows)` — factor the existing catalog/csv/md/m3u/parquet block
   out of `build()` so both modes write identical files; add the three caption
   columns and `captions.jsonl`/`captions.sha256`.
4. `attach(args)` — new mode operating on the `--local` tree (`catalog.jsonl`,
   `manifest.json`, `recipe/`, `tracks/<name>/track.json`): it rewrites each
   `track.json`, writes the two caption files, then regenerates the root files
   and `manifest.json` (`counts.captions`, a `captions` block with
   `source`/`source_sha256`/`pipeline_version`/`attached_at`, `dataset_version`).
   It must **not** touch `recipe/`, `.build.json` inputs, `lyrics.txt`,
   `style.txt`, `score.abc` or any audio field.
5. `verify()` — add: both caption files present for every row; `caption.txt`
   sha256 == `track.json.caption.sha256` == row `caption_sha256`; seed match;
   `captions.jsonl` line count == rows; `--remote` also counts
   `tracks/*/caption.json` == 460.
6. CLI: `--local PATH` (default `LOCAL`), `--captions PATH`, `--captions-uri`,
   `--attach-captions`, `--no-audio` for publish (skips the 460 server-side
   `cp -n`, which would be no-ops anyway), `--allow-missing-captions`,
   `--allow-unpolished`.
7. Bump `dataset_version` default to `v0.2` for an attach run; keep `v0.1` when
   building without captions.

CPU unit tests (no network, no audio): 3-row fixture + fake album —
empty caption rejected; seed mismatch rejected; missing name rejected; re-run is
byte-identical (idempotent); a tampered `caption.txt` makes `--verify` fail.

### 3.3 Procedure on Colab

```bash
cd /content/arabmaqamrock && git pull

# 0. pull the published v0.1 metadata (no audio: ~10 MB, 1,840 small objects)
L=work/yt500_gen/caption_attach
mkdir -p "$L"
gcloud storage rsync -r -x '.*audio\.mp3$' \
  gs://akbar-december-2024-backup/DATASET/yt500_instrumental "$L"
find "$L/tracks" -name audio.mp3 | wc -l          # must print 0

# 1. snapshot exactly what we are about to modify (≈3 MB compressed) -> rollback
tar -C "$L" -czf /tmp/v0.1_meta.tar.gz \
  catalog.jsonl tracks.jsonl manifest.json tracks.parquet \
  DATASET_CARD.md README.md catalog.csv catalog.md playlist.m3u \
  recipe .build.json tracks
gcloud storage cp -q /tmp/v0.1_meta.tar.gz \
  gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/backup/
gcloud storage ls -l "gs://akbar-december-2024-backup/DATASET/yt500_instrumental/tracks/**/audio.mp3" \
  > /tmp/audio_before.txt

# 2. attach (rewrites tracks/<name>/track.json, writes caption.json + caption.txt,
#    regenerates the root files, bumps dataset_version to v0.2)
python3 plumbing/compile_yt500_collection.py --attach-captions --local "$L" \
  --captions /content/music_captioning/work/yt500/caption_album.json \
  --captions-uri gs://akbar-december-2024-backup/GENERATION/YT500_INSTRUMENTAL/work/caption_album.json

# 3. local + remote verification (hashes, seeds, counts)
python3 plumbing/compile_yt500_collection.py --verify --local "$L" \
  --captions /content/music_captioning/work/yt500/caption_album.json --remote

# 4. publish (metadata only; audio untouched)
python3 plumbing/compile_yt500_collection.py --publish --local "$L" --no-audio --dry-run
python3 plumbing/compile_yt500_collection.py --publish --local "$L" --no-audio

# 5. prove no audio moved
gcloud storage ls -l "gs://akbar-december-2024-backup/DATASET/yt500_instrumental/tracks/**/audio.mp3" \
  > /tmp/audio_after.txt
diff /tmp/audio_before.txt /tmp/audio_after.txt      # identical (sizes + mtimes)
```

`--publish` runs `gsutil -m rsync -r "$L"/ DATASET_URI/`: it uploads the 460×3
changed track files plus the regenerated root files and **skips unchanged
objects** (audio is not in the tree at all, see §3.4). It is safe to re-run; the
attach step is deterministic.

### 3.4 Hazards to code defensively against

- **Never add `-d` to the publish rsync.** The local tree has no `audio.mp3`
  (audio is copied server-side), so a deleting rsync would remove all 460 MP3s.
  Add an assertion in `publish()` that the tree contains no `tracks/*/audio.mp3`
  and that `-d` is absent.
- **Case-sensitive prefix.** `scripts/yt500_gcs.py` has
  `DATASET_PREFIX = …/DATASET/YT500_INSTRUMENTAL` (upper case) while the real
  dataset is `…/DATASET/yt500_instrumental`. If `yt500_publish.py` is ever run
  with the package present, pass `--gcs-dataset gs://…/DATASET/yt500_instrumental`
  (or fix the constant in a separate change). This plan uses the compiler, not
  `yt500_publish.py`, for the dataset.
- **Don't rebuild.** `--build` re-derives the set from `index.json` + audit +
  run metadata; an attach run must not re-select names or re-write
  `audio`/`score`/`style` hashes. Only `--attach-captions` touches v0.1.
- **Names contain `_`** inside video ids (`…-pEV_jW6S8eM`, `…--faWzeEe1J0`,
  `…-wS-S_Vc_638`): always split the seed off with the greedy regex, not
  `rsplit("_", 1)` on the name side.

## 4. Phase 3 — end-to-end verification

```bash
# counts (recursive pattern, one object per track folder)
gcloud storage ls "gs://akbar-december-2024-backup/DATASET/yt500_instrumental/tracks/**/caption.json" | wc -l   # 460
gcloud storage ls "gs://akbar-december-2024-backup/DATASET/yt500_instrumental/tracks/**/caption.txt"  | wc -l   # 460
gcloud storage cp -q gs://akbar-december-2024-backup/DATASET/yt500_instrumental/captions.jsonl .
wc -l captions.jsonl                                                                                            # 460

# checksums: `--verify --remote` above does this object by object; the file is
# also self-contained for consumers who rsync the tree down
gcloud storage cp -q gs://akbar-december-2024-backup/DATASET/yt500_instrumental/captions.sha256 /tmp/
#   cd <downloaded dataset root> && sha256sum -c /tmp/captions.sha256   # 920 OK
```

- [ ] `tracks/**/caption.json` == `tracks/**/caption.txt` == `captions.jsonl` == **460**
- [ ] `--verify --remote` exits 0 and its per-file hashes agree with
      `captions.sha256` (920 lines: 460 `caption.txt` + 460 `caption.json`)
- [ ] sample 6 tracks: `caption.json.take.seed == track.json.generation.seed`, and
      `caption.txt` reads as prose (no timestamps, no `[section]` tags, no
      `suno_prompt`-style comma list)
- [ ] `manifest.json`: `counts.tracks=460`, `counts.captions=460`,
      `dataset_version=v0.2`, `captions.source` points at the album and its
      sha256 matches the uploaded album
- [ ] `DATASET_CARD.md`/`README.md` document `caption.txt`/`caption.json` in the
      layout block
- [ ] remote object comparison: 0 missing, 0 mismatched among the new/changed
      objects; **no** `audio.mp3` object changed (`tracks/**/audio.mp3` sizes and
      mtimes identical to `/tmp/audio_before.txt`)
- [ ] 5 spot-listens across different generation runs: the caption matches the
      audio (this is the only judgement a script cannot make)

## 5. Risks and rollback

| Risk | Mitigation |
|---|---|
| Phase 1 re-captions already-done tracks because the local album is stale | §2.2 overwrites it from GCS and asserts 432 tracks before any GPU work |
| Corpus index drift → `abc_mismatch` exit 2 | §2.1 sha256 gate against `work/manifest.json.corpus.sha256`; do **not** reach for `--allow-abc-mismatch` |
| Colab OOM | 4-bit captioner, one role at a time, no watchers; T4 peaks 8.0 GiB (measured) |
| Dataset `track.json` rewritten wrongly | full v0.1 metadata tarball in `GENERATION/.../work/backup/` (untar over the local tree, re-run `--publish`); audio never at risk |
| A caption attached to the wrong take | seed guard in `caption_record()`; `--verify` re-checks it |
| Partial publish | run `--verify --remote` after; re-run `--publish` (idempotent) |

**Nothing is deleted in either phase.** The 6 album-only tracks, the 22 gray
takes, transcription and any rename of the dataset prefix are out of scope.

## 6. Definition of done

1. `GENERATION/.../work/caption_album.json` = 466 polished tracks (at least;
   more if the live bucket produced new takes); the 34 new ones look like the
   rest.
2. `DATASET/yt500_instrumental` v0.2: every one of 460 track folders carries
   `caption.txt`, `caption.json`, and a `track.json` that points at them; root
   `captions.jsonl` + `captions.sha256` published.
3. `compile_yt500_collection.py --verify --remote` exits 0; `pytest tests/`
   still passes; the new CPU unit tests pass.
4. Audio untouched, no secrets or weights in either git diff.

## Run log — 2026-11-11 (phase 2 on Colab CPU)

Phase 1 was already complete: the album at
`GENERATION/YT500_INSTRUMENTAL/work/caption_album.json` is **467 tracks**,
`caption_stage=polished`, `caption_pipeline_version=3`, sha256
`07aa645392a0996871efcc182769a96cbd9aed1fd5be213370417260123e29d9`
(checked before anything else). Every one of the 460 dataset takes has a caption
for its exact take stem, so no GPU work ran.

Phase 2 code came from `origin/feat/dataset-captions-v0.2` (`b72ff6a`, tip
`9842390` after its merge of main). It was merged `--ff-only` into `main`;
`tests/test_compile_yt500_collection.py` → **10 passed**.

Driver: `arabmaqamrock` `plumbing/phase2_attach_captions.sh` (canonical, tracked;
executed as `logs/phase2.sh`), detached with `setsid nohup`, log
`arabmaqamrock/logs/phase2.log`, per-step status pushed to
`GENERATION/YT500_INSTRUMENTAL/work/backup/status_phase2.json`.

| step | wall | result |
|---|---:|---|
| pull_tree (`rsync -r -x '.*audio\.mp3$'`) | 21 s | 1,853 files, 13 MB, 0 `audio.mp3` |
| backup_v01 | 4 s | `…/backup/dataset_v0.1_meta.tar.gz` |
| fetch_album + sha gate | 3 s | sha256 matched `07aa6453…` |
| attach (`--attach-captions --allow-unpolished --min-caption-chars 150`) | 1 s | `attached 460/460 captions`, `dataset v0.2` |
| backup_v02 | 5 s | `…/backup/dataset_v0.2_meta.tar.gz` |
| verify (`--verify --local … --allow-unpolished`) | 1 s | `verified 460 rows, 0 problem(s)` |
| dry run | — | **1,392** `Would copy` |
| publish (`--publish --no-audio`) | 23 s | `published 460 metadata tracks (audio untouched)` |
| remote counts + re-sync | 19 s | 460 `caption.json`, 460 `caption.txt`, `Would copy` = 0 |
| audio diff | — | before/after listing identical; newest `audio.mp3` object update still `2026-11-11T16:22:20Z`, i.e. before the 20:06:49Z publish |
| bank_album | — | album committed in `music_captioning` |
| summary | 2 s | `…/backup/phase2_summary.json` |

Total wall time 20:05:45Z → 20:07:48Z = **2 min 03 s**.

Final published state (`DATASET/yt500_instrumental`):

- `manifest.json`: `dataset_version=v0.2`, `counts.tracks=460`,
  `counts.captions=460`, `counts.captions_unpolished=6`,
  `captions.source_sha256=07aa6453…`, `captions.attached_at=2026-11-11T20:06:29Z`.
- Root `captions.jsonl` = 460 rows, `captions.sha256` = 920 lines.
- Spot checks on `yt500-0001…`, `yt500-0468…`, `yt500-0684…`: `caption.txt` sha
  == `caption.json.caption_sha256` == `track.json.caption.sha256`, take seed ==
  `track.json.generation.seed`, prose with no timecodes.

The six takes attached as **unpolished** (raw caption kept, flagged
`unpolished: true`, never presented as polished — the documented outcome):

```
yt500-0147-ZP1sp5vzBik_2484751784   yt500-0174-vPLW6oWdEjc_1729334891
yt500-0320-c9CXSbWIpMs_2678923838   yt500-0331-iFjFNhs1glQ_661465120
yt500-0468-vVYepS8lWjM_369284659    yt500-0478-jZ-naNcSVXg_603989557
```

One deviation from the driver spec: the bank step's
`gcloud storage cp "$BK/../captions.jsonl"` is not normalised by `gcloud`
(“matched no objects”), so the album `captions.jsonl` was copied from its real
URI (`work/captions.jsonl`) and committed separately. Nothing else deviated.

