# Stateless caption rewrite

Date: 2026-11-06
Status: prototype. The rewrite module and its checks are implemented and unit
tested; the batch/demo pipeline still uses the deterministic merge until the
model is validated on real chunk captions on a Colab T4.

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
| invented terms | a lexicon term whose words do not appear in the source (`sitar` when only `oud` was mentioned) |
| false ending | a coda claim when the final source section makes none |

On rejection nothing is deleted: the untouched source text is returned with
`status="fallback"` and the reason, so the album JSON records what happened.

Term matching is lexical containment on stemmed words, not sentence similarity.
It only guards against invented facts; it does not decide what to remove.

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

Default: **`Qwen/Qwen3-4B-Instruct-2507`** (Apache-2.0) at 8-bit. It fits beside
the captioner (~8.4 GiB + ~4.3 GiB on a 15 GiB T4), answers reliably, and its
hallucination rate is close to the 8B/12B models. `Qwen3-8B` at 4-bit is the
fallback if 4B quality is not good enough; it requires unloading the captioner
first. Phi-4-mini and Llama-3.1-8B are rejected on their measured rates.

Caveat: those benchmarks summarize news articles; this task condenses already
written English and is easier. No model is at zero, which is why the
verification pass remains.

## Usage (prototype)

On Colab, capture the raw chunk captions for one track:

```bash
python scripts/demo.py "/content/fadhl_shaker_selected/حبيتك.mp3" \
  --keep-chunk-captions -o /tmp/habeetak.json
```

Then rewrite them (the model is downloaded on first use):

```bash
python scripts/rewrite_demo.py /tmp/habeetak.json
python scripts/rewrite_demo.py /tmp/habeetak.json --quantization 4bit
python scripts/rewrite_demo.py /tmp/habeetak.json --model Qwen/Qwen3-8B
```

A plain text file with one caption per paragraph works too. The script prints
the status, rejection reasons, timing, and the rewrite.

## What is not done yet

- The batch/demo pipeline still calls `merge_section_captions`; the rewrite is
  not wired in.
- `CAPTION_PIPELINE_VERSION` stays at 2 until the rewrite becomes part of the
  output, then it must move to 3 (older albums re-queue automatically).
- Quality on real captions (and 4B vs 8B) must be compared before wiring.
