# Results — real runs

> Rule from `AGENTS.md`: do not claim the pipeline works until it has been run
> end to end on a real T4. This file records actual commands and measurements.

## Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4, 15360 MiB VRAM, compute capability 7.5 (sm_75) |
| CUDA driver / toolkit | 13.0 (driver 580.82.07) |
| Python | 3.13 |
| Runtime | `transformers` 5.18 + `bitsandbytes` 0.50.2 (`device_map="auto"`, fp16 compute, SDPA) |
| Model | `Urabewe/Ace-Step-Captioner-4bit` (nf4, ~6.4 GB) + `spk_dict.pt` from `ACE-Step/acestep-captioner` |
| Prompt | `*Task* Describe this audio in detail` |
| Decoding | greedy (`do_sample=False`), `thinker_max_new_tokens=400`, `generation_mode="text"` |

### Runtime change (2026-10-09, user-approved)

The original plan was the llama.cpp GGUF runtime. We built the patched fork
(`tpsjr7/llama.cpp`, branch `ted/fix-qwen-audio-cleanup-merge`) but stopped at
~32% because the CUDA build is slow on Colab's 2 vCPUs and there is no shortcut
for this GPU:

- llama.cpp `b10796` publishes **no Linux CUDA prebuilt** (Windows CUDA only).
- The upstream **Vulkan** Linux prebuilt is unusable here: the Colab image ships
  `libvulkan.so.1` but **no NVIDIA Vulkan ICD**.
- The GGUF author's fix is a 1.88 MB `libmtmd.so` overlay for `b10796`, which
  still needs a base llama.cpp build for CUDA.

The user approved switching to the transformers 4-bit route, which uses the
reference audio pipeline (no audio patch needed).

## Run 1 — `elisa_maktooba_leek.mp3`

- Source file: `/content/elisa_maktooba_leek.mp3`
- Duration: 317.0 s (5:17)
- Chunking: 60 s segments → 6 chunks

### Measurements

| Metric | Value |
|---|---|
| Model load | 36.2 s |
| Chunk caption times | 16.7, 9.7, 11.5, 10.6, 10.2, 9.8 s |
| Total caption time (6 chunks) | 68.5 s |
| Peak VRAM | **8229 MiB (~8.0 GiB)** |
| Status | ✅ end-to-end success, no errors |

The first chunk is slower (CUDA kernel warmup); steady-state is ~10 s per 60 s
of audio.

### Model output (chunk 2 of 6)

> An emotional Arabic pop ballad driven by a powerful female vocal performance.
> The arrangement blends traditional and modern elements, featuring a prominent
> oud playing melodic lines over a bed of lush, cinematic synth strings and
> pads. A steady, mid-tempo beat combines electronic drum machine elements with
> the distinct sound of a darbuka, while a subtle synth bass provides the
> foundation. The vocalist delivers the lyrics with significant passion and
> vibrato, her voice treated with a spacious reverb that enhances the track's
> dramatic and longing mood.

### Extracted tags (merged across 6 chunks, experimental)

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

These lists are **experimental** and are not a supported Suno prompt. The
comma-joined string is opt-in only (`--experimental-suno`); see
`docs/TAG_AUDIT.md` for why it was removed from the default output.

### Notes / errors encountered

- `from_pretrained` **always** loads `spk_dict.pt` (speaker map) even when the
  talker is unused; `Urabewe/Ace-Step-Captioner-4bit` omits it. Fetch it from the
  base repo into the model dir.
- The checkpoint ships a `quantization_config` with **bf16 compute**, which the
  T4 cannot do natively, and it overrides any config passed to `from_pretrained`.
  We patch `config.quantization_config["bnb_4bit_compute_dtype"] = "float16"`
  before loading.
- `generate` renamed `return_audio` → `generation_mode`; use
  `generation_mode="text"` to skip the talker/vocoder entirely.
- The reference pipeline needs no 30-second-boundary workaround (that bug was
  specific to stock llama.cpp).

## Run 2 — album benchmark (`Elissa - 2008 Ayami Beek`)

- Input: `/content/Elissa - 2008 Ayami Beek` (11 tracks, 320 kbps MP3)
- Output: `outputs/Elissa_2008_Ayami_Beek.json`
- Model loaded **once**, reused; 60 s chunks
- Reproduce: `python scripts/benchmark.py "/content/Elissa - 2008 Ayami Beek" --out outputs/Elissa_2008_Ayami_Beek.json`

### Totals

| Metric | Value |
|---|---|
| Tracks | 11 (0 errors) |
| Total audio | 3034.3 s (50.6 min) |
| Chunks | 56 |
| Caption time | 617.0 s (10.3 min) |
| Wall time (captioning only) | 623.4 s |
| Model load (once) | 39.3 s |
| Realtime factor | **0.203** (≈ 4.9× faster than real time) |
| Peak VRAM | 8228 MiB (~8.0 GiB) |

### Per-track

| Track | Audio (s) | Chunks | Caption (s) |
|---|---:|---:|---:|
| 01. Ayami Bik | 279.7 | 5 | 57.9 |
| 02. Khod Balak Alaya | 329.2 | 6 | 63.8 |
| 03. Betmon | 244.5 | 5 | 52.0 |
| 04. Ya Alam | 313.4 | 6 | 61.4 |
| 05. Law Ma Tegi | 274.2 | 5 | 56.2 |
| 06. Mish Ktir Alayk | 268.9 | 5 | 55.4 |
| 07. Adik Erift | 290.6 | 5 | 55.5 |
| 08. Ala Hobbak | 218.1 | 4 | 47.9 |
| 09. Ana Bastaghrib Aleh | 268.1 | 5 | 52.8 |
| 10. Sahar Einy | 275.0 | 5 | 58.3 |
| 11. Awakhir El Shity | 272.6 | 5 | 55.8 |
| **Total** | **3034.3** | **56** | **617.0** |

### Example tags (track 01, "Ayami Bik")

```
instruments: violin, synth, drum machine, synth bass, kick, darbuka, ney, synth strings, synth pads
vocals:      female vocal, harmonies, melisma, vibrato
production:  melody, four-on-the-floor, groove, melodic, clean, dynamic, punchy, bassline, layered, reverb, lush, spacious
```

## Run 3 — chunk-length experiment (2026-11-06)

Measurement only; no model, prompt, quantization or default-runtime change.
Full write-up: [CHUNK_LENGTH_EXPERIMENT.md](CHUNK_LENGTH_EXPERIMENT.md).

- Track: `/content/fadhl_shaker_selected/حبيتك.mp3`, 266.256 s
- Model load 47.8 s, 6,519 MiB

| Config | Chunks | Caption s | Peak alloc / reserved | Words | Notes |
|---|---:|---:|---|---:|---|
| 60 s fixed (current) | 5 | 54.1 | 8,211 / 8,888 MiB | 378 | 5 whole-song framings, 4 near-identical |
| one-shot 266 s | 1 | — | — | — | **OOM** at prefill (4.66 GiB requested) |
| 2 × 133.13 s | 2 | 36.3 | 9,803 / 10,044 MiB | 162 | fabricated coda at the cut |
| 3 × 88.75 s | 3 | 46.8 | 8,462 / 9,890 MiB | 285 | no fabricated coda |

Single-window ceiling: 190 s OK (12,838 MiB), 210 s OOM. The ~10 GiB line is
≈140 s. Time never binds (190 s captions in 27 s).

Two corrections to earlier assumptions:

1. The decoder runs in **bf16, not fp16** (`config.dtype` is bfloat16 and wins
   over the `bnb_4bit_compute_dtype` patch), which disqualifies the
   memory-efficient SDPA kernel and causes the one-shot OOM. `AGENTS.md` says
   fp16 compute; on this stack it is not. Forcing fp16 failed
   (`Input type (c10::BFloat16) and bias type (c10::Half) should be the same`).
2. The "ends abruptly, followed by a long silence" tail text is **not** a
   hallucination: the file fades from ~254 s and is digital silence for the last
   ~5.5 s (7.1 s below -50 dB). The real fabrication is at mid-song cuts.

A longer window does not yield a longer description: 133 / 150 / 170 / 190 s
captions are 82–91 % identical word-for-word, so one-shot would not fix the
repetition.

## Run 4 — silence trim + equal chunks + merge (2026-11-06)

Implements the fix recommended in Run 3 and validates it on the same track.
Full write-up: [CHUNK_MERGE.md](CHUNK_MERGE.md). Model, prompt, quantization and
decoding are unchanged; `pytest tests/ -q` -> 128 passed.

- Track: `/content/fadhl_shaker_selected/حبيتك.mp3`, 266.256 s
- Model load 43.9 s

| Config | Chunks | Caption s | Peak alloc / reserved | Words | Framings | Coda |
|---|---:|---:|---|---:|---:|---|
| 60 s fixed, blind join (Run 3 baseline) | 5 | 54.1 | 8,211 / 8,888 MiB | 378 | 5 | one, describing the silent tail |
| trim + 3 x 86.408 s + merge | 3 | 52.0 | 8,441 / 9,800 MiB | **210** | **1** | none; "tape-stop" absent |

- Silence: 266.256 s -> 259.224 s (7.032 s trailing silence removed).
- Per-chunk generation: 19.7 / 13.7 / 14.6 s; actual chunk lengths 86.424 / 86.400 / 86.400 s.
- The merge dropped only the two repeated section openers; all 10 remaining
  sentences were kept. Tags from the merged caption equal the union of the
  per-chunk tags.

One implementation detail measured here: a `-c copy` trim ends at 259.224 s, not
the computed 259.189 s, so chunking must plan on the *measured* trimmed duration
or it emits a 0.03 s fragment that ffprobe cannot even read. Both backends now
measure first, and `_split_audio` discards sub-second trailing fragments.

## Run 5 — GGUF vs fp16 rewrite A/B (2026-11-06)

First real comparison of the two rewrite engines, run on the stored
`chunk_captions` of `fadhl_shaker_selected` (24 tracks, no re-captioning) via
`scripts/rewrite_ab.py`. Same model (`Qwen3-4B-Instruct-2507`), same system
prompt, same verifier; only the engine and the weight format differ.

| Backend | Weights | ok | fallback | s/track | Mean words | Term coverage |
|---|---|---:|---:|---:|---:|---:|
| `hf` | fp16 transformers | 24/24 | 0 | 18.2 | 147.9 | 80.3% |
| `llamacpp` | Q8_0 GGUF (4.0 GB) | 24/24 | 0 | **5.8** | 149.8 | **81.8%** |

- Total: 437.6 s (hf) vs 139.4 s (llamacpp) — **3.1× faster** end to end.
- The first track absorbs warmup: 153.0 s (hf) and 24.6 s (llama-server start);
  excluding it, 12.4 vs 5.0 s/track, still ~2.5×.
- Quality did not drop: the quantized run kept slightly *more* instrument/vocal
  terms (81.8% vs 80.3%) and returned marginally longer prose (149.8 vs 147.9
  words), with zero verification fallbacks in both.
- Per-track coverage is within a few points either way; no track regressed
  sharply (worst delta: 87.5% vs 93.8% on `حظك ياقلبي`).
- Peak RAM/VRAM were not sampled in this run (it used `049bbde`, before the
  sampler landed in `87ffd49`); re-running on the newer commit adds those
  columns.

Interpretation: Q8_0 preserves the detail we verify, so the earlier worry that
quantization would drop instruments did not materialize at this size. The
engine, not the weights, accounts for most of the speedup.

## GPU smoke test

Kept out of `pytest` (manual, needs GPU + model). Reproduce with:

```bash
python scripts/demo.py /content/elisa_maktooba_leek.mp3 --json
```

CPU-only tag-extraction tests: `python -m pytest tests/ -q` (27 passed).
