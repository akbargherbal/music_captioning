# Music Captioning

Given **any music track**, produce a prose description plus **experimental**
term lists describing its **instruments**, **vocals**, and **production**.

Pipeline:

```
audio ─▶ ACE-Step Captioner ─▶ prose caption
      ─▶ experimental deterministic term extraction
```

The captioner is [`ACE-Step/acestep-captioner`](https://huggingface.co/ACE-Step/acestep-captioner)
(MIT, based on Qwen2.5-Omni-7B). It runs through **HuggingFace `transformers` in
4-bit (nf4)** on the T4, using the reference audio pipeline. The term extraction
is plain Python — **no hosted LLM**.

> **The extracted terms are experimental, not a finished Suno prompt.**
> A comma-joined `suno_prompt` string used to be emitted by default. It was
> removed because it omits genre/mood, misses plurals and phrase forms, and
> unions contradictory chunk captions. See `docs/TAG_AUDIT.md`. The string is
> now opt-in (`--experimental-suno`) and marked `suno_experimental`.

A llama.cpp GGUF backend is also supported; see *Alternative backend* below.

## Requirements

- Google Colab with a **T4 GPU** (sm_75). No bf16, no FlashAttention.
- `ffmpeg` / `ffprobe` on `PATH` (used to split long tracks).
- Python 3.10+.

## Quickstart (fresh Colab runtime, start to finish)

Every step is idempotent — re-running skips what already exists. Colab already
provides `torch`, `ffmpeg` and `ffprobe`; only the extra packages are installed.
Both Hugging Face repos are public, so **no HF token is needed**.

```bash
# 0. Get the code (skip if already on this runtime / on Drive).
git clone https://github.com/akbargherbal/music_captioning_oct_2026.git
cd music_captioning_oct_2026

# 1. Dependencies (torch itself is preinstalled on Colab).
pip install -q -r requirements.txt

# 2. Weights: 4-bit captioner (~6.4 GB) + spk_dict.pt (skipped if cached).
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit", local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt", local_dir="models/Ace-Step-Captioner-4bit")
PY

# 3. Run the pipeline on a track (upload one, or point at a path).
python scripts/demo.py /content/elisa_maktooba_leek.mp3 --out outputs/elisa.json

# 4. Optional: CPU-only tests, no model or GPU needed.
python -m pytest tests/ -q
```

Or open `notebooks/colab_setup.ipynb` and **Run all** — it does exactly the
same thing, cell by cell.

> **The runtime is ephemeral.** `models/` lives on the Colab VM and is lost on
> restart, so a fresh runtime re-downloads ~6.4 GB. To avoid that, persist it on
> Drive before step 2, e.g.
> `ln -s /content/drive/MyDrive/music_captioning_models models` (mount Drive
> first). Otherwise just re-run step 2 — it is safe and resumable.

## Example

Input: `elisa_maktooba_leek.mp3` (317 s).

<!-- EXAMPLE_START -->
Measured on a Colab T4: model load **36.2 s**, six 60 s chunks captioned in
**68.5 s total**, peak VRAM **8.0 GiB**. Full numbers in `docs/RESULTS.md`.

A representative chunk caption:

> An emotional Arabic pop ballad driven by a powerful female vocal performance.
> The arrangement blends traditional and modern elements, featuring a prominent
> oud playing melodic lines over a bed of lush, cinematic synth strings and pads.
> A steady, mid-tempo beat combines electronic drum machine elements with the
> distinct sound of a darbuka, while a subtle synth bass provides the foundation.
> The vocalist delivers the lyrics with significant passion and vibrato, her
> voice treated with a spacious reverb that enhances the track's dramatic and
> longing mood.

Resulting experimental tags:

```json
{
  "instruments": ["nylon-string guitar", "string section", "percussion", "oud",
                  "synth strings", "drum machine", "darbuka", "synth bass",
                  "tabla", "synth pads", "kick", "violin", "accordion"],
  "vocals": ["female vocal", "singing", "vibrato", "melismatic", "melisma", "vocals"],
  "production": ["arpeggiated", "melody", "lush", "cinematic", "bassline",
                 "world-fusion", "melodic", "spacious", "reverb", "clean",
                 "punchy", "dynamic", "lo-fi", "vintage", "sample"],
  "tags_experimental": true
}
```
<!-- EXAMPLE_END -->

## Output format

By default `scripts/demo.py` **only prints**; nothing is written to disk. Pass
`--out PATH` (e.g. `--out outputs/elisa.json`) to also save the JSON — parent
directories are created and `outputs/` is gitignored.

`scripts/demo.py --json` prints:

```json
{
  "instruments": ["..."],
  "vocals": ["..."],
  "production": ["..."],
  "caption": "the raw prose caption",
  "tags_experimental": true
}
```

The three lists are **experimental extracted terms**, not a generation prompt.
Only terms the caption actually mentions are emitted (no guessing, e.g. no vocal
gender unless stated). Long tracks are split into ~60 s chunks, captioned
separately, and merged with order-preserving de-duplication.

The legacy comma-joined string is still available for experiments:

```bash
python scripts/demo.py SONG.mp3 --experimental-suno
```

That adds `"suno_prompt"` and `"suno_experimental": true` to the output. Do not
treat it as a supported Suno prompt until `docs/TAG_AUDIT.md` is resolved.

## Resumable batch across Colab sessions

Colab runtimes are ephemeral (free tier ~2–3 h; disk is lost on disconnect).
The batch runner treats **GitHub as the durable store**: after *every* album it
writes the result, updates `outputs/index.json`, and commits + pushes
immediately.

```bash
python scripts/batch.py --artist ELISSA --max-minutes 90   # or omit --artist for all
python scripts/batch.py --artist ELISSA --list             # show pending vs done
```

On a fresh runtime, just re-run the same command: it `git pull`s, reads the
manifest, and **skips every album already done**. Only the in-flight album is
lost if you get disconnected. Flags: `--list`, `--force` (redo), `--max-albums N`,
`--max-minutes N` (stop cleanly between albums), `--no-push`. Audio is cached in
`work/` (gitignored) and re-downloaded only when missing.

**Failures are isolated per track.** If one track errors, it is recorded in the
album JSON under `failed_tracks` and the manifest is marked `partial`; the rest
of the album is still written and pushed. `--retry-failed` re-captions **only**
the failed tracks. A single bad track never discards an album's work.

**Long tracks are skipped, not failed.** Tracks longer than `--max-track-seconds`
(default 480 = 8 min) are listed under `skipped_tracks` — handy for concert
recordings or long mixes. Set `0` to disable the cap.

**Low-quality tracks are skipped too.** `--min-bitrate-kbps` (default 128;
`0` disables) drops low-bitrate files (e.g. 32/64 kbps) before captioning. For a
local `--root` whose directory name isn't the artist label you want, pass
`--artist-name NAME`.

### Surviving kills (use this for long runs)

`batch.py` isolates per-track/per-album *failures*, but it cannot survive its own
process being killed (host OOM, a native crash, or the session reaping the job) —
nothing inside a dead process can log or continue. Run it under the supervisor,
detached, so a kill simply resumes:

```bash
setsid nohup python3 -u scripts/supervise.py --log /content/batch.log -- \
  --root "/content/Fadl Shaker.Full Discography.BY.RoMaNTiCPoET" \
  --artist-name FADL_SHAKER --min-bitrate-kbps 128 --max-track-seconds 480 \
  --max-minutes 0 >/content/supervisor.out 2>&1 &
```

It restarts the batch after any non-zero exit (including SIGKILL), and only
gives up after repeated *quick* failures. Watch `/content/batch.log`.

## Benchmarking

Time captioning for one file or a whole album (model loaded once, reused):

```bash
python scripts/benchmark.py "/content/Elissa - 2008 Ayami Beek" \
  --out outputs/Elissa_2008_Ayami_Beek.json
```

Writes one JSON with per-track chunk timings, caption seconds, peak VRAM and
experimental tags (no `suno_prompt` by default), plus album totals and a
`realtime_factor` (caption seconds ÷ audio seconds). See `docs/RESULTS.md` for
measured numbers.

## Alternative backend: llama.cpp GGUF

The original pinned GGUF runtime is kept in `src/captioner.py` +
`scripts/build_llamacpp.sh`. It needs a **CUDA build** of the patched fork
(`tpsjr7/llama.cpp`, branch `ted/fix-qwen-audio-cleanup-merge`) because stock
llama.cpp mishandles this model's audio input. There is **no Linux CUDA
prebuilt**, so this path compiles for `sm_75`.

```bash
bash scripts/build_llamacpp.sh
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["acestep-captioner-Q4_K_M.gguf", "acestep-captioner-mmproj-Q8_0.gguf"]:
    print(hf_hub_download("dernet/acestep-captioner-GGUF", f, local_dir="models"))
PY
python scripts/demo.py SONG.mp3 --backend llamacpp
```

## Layout

```
src/captioner_hf.py  # transformers 4-bit backend (current)
src/captioner.py     # llama.cpp GGUF wrapper (alternative)
src/tags.py          # lexicon + experimental term extraction
src/lexicon.json     # instruments / vocals / production terms
scripts/demo.py      # end-to-end CLI
scripts/benchmark.py # per-track timing + VRAM benchmark
scripts/build_llamacpp.sh
tests/               # pure-Python, no GPU/model needed
docs/RESULTS.md      # real measured VRAM + timing
```

## Testing

```bash
python -m pytest tests/ -q
```

Runs on CPU with no model present (uses saved caption fixtures).

## Notes & limitations

- The captioner is deterministic (greedy decoding) and capped at 400 tokens.
- **Term extraction is experimental.** Known gaps: no genre/mood category,
  plural/phrase misses (`trumpets`, `female lead vocal`, `pads`), contradictions
  across chunks, and no negation handling. See `docs/TAG_AUDIT.md` for the
  5-track audit and corpus numbers. These lists are not a Suno prompt.
- VRAM/time are hardware-specific; see `docs/RESULTS.md` for measured numbers.
- The 4-bit checkpoint is a community conversion (`Urabewe`); the base model and
  prompt are unchanged.
- Rejected for this T4 (do not switch silently): MOSS-Music-8B (bf16 ~18 GB) and
  the original ACE-Step F32/BF16 11B weights.

## Attribution

- ACE-Step Captioner and its GGUF conversion: MIT (`ACE-Step`, `dernet`).
- 4-bit conversion: `Urabewe`.
- Underlying architecture: Qwen2.5-Omni.
- Alternative patched runtime: `tpsjr7/llama.cpp`, `koda-dernet/acestep-captioner`.
