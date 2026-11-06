# Stateless caption rewrite

Date: 2026-11-06
Status: wired into `scripts/batch.py` as the per-album polish phase. The default
engine is **Q8_0 GGUF under `llama-server`** (3.1× faster than fp16 with equal or
better coverage on 24 real tracks); `--rewrite-backend hf` keeps the fp16 path.
The model and its checks are unit tested.

## Why this exists

Each audio window is captioned as if it were a complete song, so the
concatenated chunk captions repeat the framing, re-describe the same instruments
with different words, and contain filler. Removing those sentences with string
similarity (Jaccard) only catches near-identical wording and grows a pile of
regex rules. Instead we give the raw text to a small instruct model and ask it to
tidy it up.

## Design: plain text in, plain text out, stateless

The model never sees JSON, field names, timestamps, or per-chunk metadata. It
receives exactly two messages:

```
system: The text below describes one music track. It was written in several
        passes, so it repeats itself and contains filler. Rewrite it as one
        clean, natural description of the track: keep every distinct detail,
        remove the repetition and padding, and add nothing that is not in the
        text. Output only the rewritten description, with no headings, lists,
        or commentary.

user:   <the chunk captions, joined by blank lines>
```

Every call rebuilds the message list from scratch; there is no conversation
history and no state carried between tracks. Greedy decoding, fixed prompt
(`REWRITE_PROMPT_VERSION`), so the same input and model version give the same
message layout and decoding settings.

## Verification (not rewriting)

After generation, a deterministic pass rejects an output when any of these hold:

| Check | Rejection |
|---|---|
| length | under 50 or over 400 words |
| unchanged | identical to the concatenated source |
| markup | bullets, numbered lines, `#`, or a JSON object |
| commentary | "here is", "as an AI", "summary:", ... |
| invented instruments/vocals | an instrument or vocal term whose words do not appear in the source (`sitar` when only `oud` was mentioned) |
| false ending | a coda claim when the final source section makes none |

On rejection nothing is deleted: the untouched source text is returned with
`status="fallback"` and the reason, so the album JSON records what happened.

Term matching is lexical containment on stemmed words, not sentence similarity.
It only guards against invented facts; it does not decide what to remove.
Production terms are excluded from the check: the lexicon also contains
adjectives and hyphenated variants ("melodic", "textures", "reverb-drenched"),
which are prose rather than recording credits and caused needless fallbacks.

To re-run only the polish phase on stored captions (for example to A/B a newer
model) without re-captioning:

```bash
python scripts/batch.py --root /content/MyArtist --artist-name MY_ARTIST \
  --repolish --rewrite-model ibm-granite/granite-4.1-3b
```

## Model choice

Hallucination rate when summarizing a document (Vectara HHEM leaderboard,
lower is better; measured 2026):

| Model | Params | 4-bit RAM | Hallucination | Answer rate |
|---|---:|---:|---:|---:|
| Phi-4 | 14B | ~8.5 GB | 3.7% | 80.7% |
| Gemma 3 12B | 12B | ~7.5 GB | 4.4% | 97.4% |
| Qwen3-8B | 8B | ~5 GB | 4.8% | 99.9% |
| **Qwen3-4B** | 4B | ~2.5 GB | **5.7%** | 99.9% |
| Gemma 3 4B | 4B | ~3 GB | 6.4% | 67.3% |
| Phi-4-mini | 3.8B | ~2.5 GB | 23.5% | 92.5% |
| Ministral 3-8B | 8B | ~5 GB | 21.7% | 99.1% |

Default: **`Qwen/Qwen3-4B-Instruct-2507`** (Apache-2.0). Since 2026-11-06 it runs
as **Q8_0 GGUF under `llama-server`**, not fp16 transformers: on the same 24
stored tracks it was 3.1× faster (5.8 vs 18.2 s/track) with slightly better term
coverage (81.8% vs 80.3%) and zero fallbacks in both (Run 5 in
[RESULTS.md](RESULTS.md)). `--rewrite-backend hf` keeps the fp16 path as a
fallback; `--rewrite-quantization` then applies to that path only. `Qwen3-8B` at
4-bit is the quality fallback if 4B turns out not to be good enough. Phi-4-mini
and Llama-3.1-8B are rejected on their measured rates.

fp16 remains the reference: a 4B model is ~8 GiB and fits alone on a 15 GiB T4
(the captioner is unloaded first), but it costs roughly three times the polish
time and double the VRAM of the quantized GGUF.

The GGUF weights are not fetched by the HF prefetch thread; stage them once with
`hf download unsloth/Qwen3-4B-Instruct-2507-GGUF
Qwen3-4B-Instruct-2507-Q8_0.gguf --local-dir models` (or the GCS cache command in
[GGUF_REWRITE.md](GGUF_REWRITE.md)). `scripts/batch.py` checks both the weights
and `llama-server` at startup and exits before captioning if either is missing.

Caveat: those benchmarks summarize news articles; this task condenses already
written English and is easier. No model is at zero, which is why the
verification pass remains.

## Where it runs (per album, two phases)
`scripts/batch.py` never interleaves the models:

1. **Caption** every track of the album with the audio captioner. Each track
   stores `chunk_captions` (raw model output per chunk) and `caption_raw` (the
   deterministic merge).
2. Unload the captioner, then **polish** every track with the rewrite model.
   Single-chunk tracks are left as they are. The caption-only album is already
   checkpointed (written, manifest updated, committed) with
   `caption_stage: "captioned"`, and is checkpointed again as `"polished"` after
   the rewrite phase.

If the rewrite model fails to load, the album stays `captioned`, the raw
captions remain committed, and the next run reuses them instead of re-captioning
the whole album.

`--repolish` re-runs only this second phase and rebuilds the album list from
`outputs/*/*.json`, so it needs neither the audio source nor a reachable
`--root`: the stored `chunk_captions` are enough. It is the cheap way to change
the rewrite engine or prompt across an existing discography.

The raw captions are never deleted, so a bad result can be traced to the
captioner (look at `chunk_captions`) or to the rewrite model (compare
`caption_raw` and `caption`, and read `rewrite_status` / `rewrite_reasons`).

## Manual experiment

To try the rewrite on one track without touching the batch output:

```bash
python scripts/demo.py "/content/fadhl_shaker_selected/حبيتك.mp3" \
  --keep-chunk-captions -o /tmp/habeetak.json

# default: GGUF under llama-server
python scripts/rewrite_demo.py /tmp/habeetak.json \
  --backend llamacpp --gguf models/Qwen3-4B-Instruct-2507-Q8_0.gguf

# fp16 reference
python scripts/rewrite_demo.py /tmp/habeetak.json --backend hf

# A/B the two engines on stored captions (no re-captioning)
python scripts/rewrite_ab.py outputs/fadhl_shaker_selected/*.json --backends hf,llamacpp
```

The script prints the status, rejection reasons, timing, and the rewrite. A
plain text file with one caption per paragraph works too.

## What is not done yet

- Peak RAM/VRAM for the GGUF backend has not been measured (the A/B ran on an
  earlier commit); re-running `scripts/rewrite_ab.py` on current `main` reports
  both.
- `scripts/demo.py` still shows the deterministic merge for a single file; only
  the batch pipeline runs the polish phase.
- `CAPTION_PIPELINE_VERSION` stays 3: switching the engine did not change the
  prompt or the model, and the single fp16-polished album gets `--repolish`ed so
  the discography ends up on one engine.
