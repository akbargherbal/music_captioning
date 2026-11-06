# Chunk-length experiment — 60 s vs one-shot vs rebalanced

Date: 2026-11-06
Scope: measurement/debugging only. **No model, quantization, prompt or default
runtime change.** `src/` and `scripts/` were not modified.

## Question

The current pipeline slices every track into fixed 60 s chunks, captions each
chunk independently with `*Task* Describe this audio in detail`, then joins the
captions with `" ".join(...)`. Two problems motivated this experiment:

1. every chunk yields a complete, self-contained "whole song" description, so a
   5-minute track becomes 4-7 near-identical descriptions concatenated;
2. fixed 60 s slicing leaves tiny final chunks, which were observed producing
   invented "sting / notification / video game" text in a 24-track run.

Is a single pass over the whole track possible on a T4, and what does it cost?

## Environment

| Item | Value |
|---|---|
| GPU | Tesla T4, 15360 MiB, sm_75, driver 580.82.07 / CUDA 13.0 |
| Python / torch | 3.13 / `2.11.0+cu130` |
| transformers | 5.22.0 |
| bitsandbytes / accelerate | 0.50.2 / 1.15.0 |
| Model | `Urabewe/Ace-Step-Captioner-4bit` (nf4) + `spk_dict.pt` from `ACE-Step/acestep-captioner` |
| Prompt / decoding | `*Task* Describe this audio in detail`, greedy, `thinker_max_new_tokens=400`, `generation_mode="text"` |
| Track | `/content/fadhl_shaker_selected/حبيتك.mp3`, 266.256 s, 258 kbps |

`pytest tests/ -q` → 69 passed. Model load 47.8 s, weights 6,519 MiB.

## Token budget (preprocess only, no forward)

| Input | input_ids | audio tokens |
|---|---:|---:|
| full track 266.3 s | 6,684 | 6,655 |
| 60 s chunk | 1,529 | 1,500 |
| 26.3 s chunk | 683 | 654 |

Audio is ~25 tokens/s, so a full track uses 6.6k of the 32,768-token context.
**Context length is not the limiting factor.**

## Baseline — 60 s × 5 (reproduced)

5 chunks, 54.1 s total caption time, peak 8,211 MiB allocated / 8,888 MiB reserved.

| # | window | words | caption s | peak alloc |
|---|---|---|---:|---:|
| 1 | 0.0–60.0 s | 82 | 13.1 | 8,206 MiB |
| 2 | 60.0–120.0 s | 81 | 10.5 | 8,211 MiB |
| 3 | 120.0–180.0 s | 69 | 10.3 | 8,211 MiB |
| 4 | 180.0–240.0 s | 81 | 11.8 | 8,211 MiB |
| 5 | 240.0–266.26 s | 65 | 7.6 | 7,900 MiB |

Joined = 378 words, **5 whole-song framings**, 4 of them the same template
("An energetic Arabic pop track driven by a powerful, …"). Adjacent sections
contradict each other on tempo ("mid-tempo" vs "driving"/"fast") and on
instrumentation ("orchestral / piano" vs "electronic dance beat").

## One-shot — 266 s: OOM, no caption produced

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.66 GiB. GPU 0 has a total
capacity of 14.56 GiB of which 2.37 GiB is free. Including non-PyTorch memory, this process
has 12.19 GiB in use. Of the allocated memory 11.80 GiB is allocated by PyTorch, and
262.25 MiB is reserved by PyTorch but unallocated.
```

Fails in `sdpa_attention_forward` during decoder **prefill**, 7.4 s in. Reproduced
in a clean process that loaded the model and immediately captioned the full track,
so it is not leftover allocator state from the chunked run.

### Root cause: the decoder is running in bf16, not fp16

`HFCaptioner._load()` patches `bnb_4bit_compute_dtype="float16"`, but the
checkpoint's top-level `dtype: bfloat16` wins for every non-quantized tensor:

```
next(model.parameters()).dtype -> torch.bfloat16
model.config.dtype             -> torch.bfloat16
```

PyTorch then refuses the memory-efficient SDPA kernel:

```
Memory efficient kernel not used because: ... Expected query, key and value to all be
of dtype: {Half, Float}. Got Query dtype: c10::BFloat16, Key dtype: c10::BFloat16,
and Value dtype: c10::BFloat16 instead.
```

Forcing that kernel gives `RuntimeError: No available kernel. Aborting execution.`,
so prefill falls back to the math backend, which materialises 28 heads × 6655 ×
6655 fp16 score matrices (~2.5 GiB each, two live at once) — hence the 4.66 GiB
request. **`AGENTS.md` states fp16 compute; on this stack it is bf16.**

Forcing fp16 is not a one-line change:

- `model.to(torch.float16)` → `ValueError: You cannot cast a bitsandbytes model in a new dtype...`
- `from_pretrained(..., dtype=torch.float16)` → params still report bf16, then
  `RuntimeError: Input type (c10::BFloat16) and bias type (c10::Half) should be the same`.

Treat "make fp16 actually take effect" as an open task, not a known fix.

## Rebalanced splits

`n = round(duration / chunk_seconds)`, `segment_time = duration / n`, same
`_split_audio()` and `caption_chunk()`:

| config | chunks | total s | peak alloc / reserved | words | framings |
|---|---|---:|---|---:|---:|
| 2 × 133.13 s | 2 | 36.3 | 9,803 / 10,044 MiB | 162 | 2 |
| 3 × 88.75 s | 3 | 46.8 | 8,462 / 9,890 MiB | 285 | 3 |

## Largest single window (bracket)

| clip | caption s | peak alloc / reserved | result |
|---|---:|---|---|
| 133.0 s | 23.6 | 9,801.6 / 10,040 MiB | OK |
| 150.0 s | 22.0 | 10,600.1 / 10,954 MiB | OK |
| 170.0 s | 25.5 | 11,656.0 / 12,094 MiB | OK |
| 190.0 s | 27.2 | 12,837.7 / 13,412 MiB | OK |
| 210.0 s | — | — | OOM (`Tried to allocate 746.00 MiB`, 651 MiB free) |
| 266.3 s | — | — | OOM (4.66 GiB requested) |

Ceiling is between 190 s and 210 s. The ~10 GiB line is ≈140 s
(133 s = 9.80 GiB, 150 s = 10.60 GiB). Time never binds: 190 s takes 27 s.

### A longer window does not produce a longer description

The 133 / 150 / 170 / 190 s captions are **82–91 % identical** word-for-word
(`difflib.SequenceMatcher` on word sequences) and keep the same five-sentence
shape; only the closing sentence changes. The 190 s caption never mentions
anything from 133–190 s. The model saturates at roughly two minutes of audio and
emits a stock summary of the opening.

## The "ends abruptly, followed by a long silence" tail is not a hallucination

Baseline chunk 5 (240.0–266.26 s) ends with that sentence. Measured audio:

```
silence_start: 259.089  silence_end: 266.217  silence_duration: 7.12802  (-50 dB)
RMS 254.2s -32.6 dBFS -> 258.7s -59.1 -> 260.7s -97.9 -> 262.7s digital silence
```

The file fades out from ~254 s and is digital silence for the last ~5.5 s
(7.1 s below -50 dB). The silence claim is **correct**; only "abruptly"
misdescribes a 6 s fade. Chunk 5 is 26.3 s, so 27 % of it really is silence.

The genuine fabrication appears at a **mid-song cut**: the 2 × 133.13 s split
invented "The track concludes abruptly with a tape-stop sound effect" at the
133.13 s boundary, where the audio simply continues. Every independently
captioned window is given an ending; only the last one can be true.

## Conclusions

1. **One-shot is not viable on this T4** for a 4.5-minute track (OOM at 266 s;
   hard ceiling ≈190 s).
2. **One-shot would not fix the repetition anyway.** The defect is a fixed
   per-window frame, not duplicated text: `133 → 190 s` is 82–91 % the same
   sentence sequence, and every section opens by re-stating the whole song.
3. **Rebalanced chunks remove the tiny-tail problem**; a silence trim removes
   the silence-captioning problem; neither removes the repeated framing.

## Recommended fix (implemented)

These three changes are now implemented in `src/merge.py` and
`src/captioner*.py`; measured results are in
[CHUNK_MERGE.md](CHUNK_MERGE.md).

1. **Trim leading/trailing silence before splitting.** Deterministic and cheap;
   on this track it removes 7.1 s of silence that is currently fed to the model
   as if it were music.
2. **Equal chunks via `n = round(duration / target)`** so no sub-10 s tail is
   ever produced. ~90 s (3 sections, 8.46 GiB) is the safe default; ~133 s
   (2 sections, 9.80 GiB) is the maximum-savings option but leaves little
   headroom and fabricated a coda at the cut.
3. **Merge deterministically.** Keep the first section's framing, drop later
   sections' "An energetic Arabic pop track…" openers, and drop coda claims
   from every non-final section. This is the step that actually removes the
   repetition (378 → ~200 words, one framing), with no extra model call.

## Reproduction

```bash
python -m pytest tests/ -q
# baseline + one-shot + rebalanced, all via the library:
python - <<'PY'
from pathlib import Path
from src.captioner_hf import HFCaptioner
from src.captioner import _split_audio, audio_duration
track = Path("/content/fadhl_shaker_selected/حبيتك.mp3")
cap = HFCaptioner(model_dir=Path("models/Ace-Step-Captioner-4bit"))
print(cap.caption_audio(track))                       # 60 s chunks
print(cap.caption_chunk(track))                       # one-shot: OOM on T4
n = round(audio_duration(track) / 89.0)               # rebalanced
print([cap.caption_chunk(c) for c in _split_audio(track, Path("/tmp/x"), audio_duration(track)/n)])
PY
```

Artifacts from the session (ephemeral, not committed): `/content/debug/*.json`,
`/content/debug/run.log`, `/content/debug/bracket.log`.
