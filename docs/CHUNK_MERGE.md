# Chunk trim + rebalanced split + deterministic merge

Date: 2026-11-06
Status: implemented on branch `fix/chunk-merge`; measured on a Colab T4.

This implements the three recommendations from
[CHUNK_LENGTH_EXPERIMENT.md](CHUNK_LENGTH_EXPERIMENT.md). The model,
quantization, prompt (`*Task* Describe this audio in detail`), decoding settings
and lexicon are unchanged.

## What changed

| Part | Where | Behaviour |
|---|---|---|
| Chunk planner | `src/captioner.py::plan_chunks` | `n = int(duration/target + 0.5)`, raised until a segment fits `max_chunk_s`; returns equal segments so there is no tiny tail |
| Silence trim | `src/captioner.py::parse_silence` / `detect_silence` / `trim_silence_edges` | ffmpeg `silencedetect` at -50 dB for 0.5 s, keep 0.1 s padding, only trim if it removes > 0.5 s |
| Deterministic merge | `src/merge.py` | one whole-song framing, no coda claims from non-final sections, near-duplicates (Jaccard >= 0.6) dropped |
| Wiring | `src/captioner.py`, `src/captioner_hf.py`, `scripts/{demo,batch,benchmark}.py` | trim -> measure -> plan -> split -> caption -> merge; `--no-trim-silence`; `chunk_count` / `silence_trimmed_s` in batch output |

Defaults:

| Backend | `chunk_seconds` | `max_chunk_seconds` | Why |
|---|---:|---:|---|
| `hf` (transformers) | 90.0 | 120.0 | 3 windows for a 4.5 min track, ~8.5 GiB; measured OOM ceiling ~190 s |
| `llamacpp` (GGUF) | 60.0 | 90.0 | the GGUF author validated one-minute clips |

`--chunk-seconds` defaults to `None` in `demo.py` / `batch.py` so each backend
applies its own default. `scripts/benchmark.py` is HF-only and defaults to 90 s.

## Edge cases found while implementing

Both were found by the validation run, not by reasoning:

1. **Planning must use the measured trimmed duration.** A `-c copy` trim snaps to
   frame boundaries, so the trimmed file was 259.224 s, not the computed
   `266.256 - 7.067 = 259.189 s`. Planning on the computed value made a 4th
   chunk of ~0.03 s, and that fragment was not even readable by ffprobe
   (`Invalid frame size (672): Could not seek to 1982`). Both backends now
   measure the trimmed file before planning.
2. **Sub-second split fragments are dropped.** `_split_audio` now discards a
   trailing fragment under 1 s (frame-snap debris): captioning it is how tiny
   chunks produce invented text, and a mid-frame copy may be corrupt.

## Validation — `حبيتك.mp3`

Input `/content/fadhl_shaker_selected/حبيتك.mp3`, 266.256 s, 258 kbps.
`pytest tests/ -q` -> **128 passed**. Model load 43.9 s.

| Metric | Value |
|---|---|
| Original duration | 266.256 s |
| Trimmed duration | 259.224 s |
| `silence_trimmed_s` | 7.032 s (trailing silence at 259.089 s) |
| Chunk plan | 3 x 86.408 s |
| Actual chunk durations | 86.424 / 86.400 / 86.400 s |
| Per-chunk generation | 19.7 / 13.7 / 14.6 s (total 52.0 s) |
| Peak VRAM | 8,441 MiB allocated / 9,800 MiB reserved |

Caption shape:

| Metric | 60 s x 5, blind join | 3 x 86.4 s + merge |
|---|---|---|
| Words | 378 | 210 (246 before the merge) |
| Whole-song framings | 5 | **1** |
| Coda claims in non-final sections | 0 | **0** |
| Coda describing the silent tail | 1 ("ends abruptly, followed by a long silence") | 0 |
| Mid-song "tape-stop" coda | seen in the 2 x 133 s config of Run 3 | absent |
| Duplicate sentences | 4 near-identical openers | none (max body Jaccard 0.16) |

For reference, running the same merge over the old 60 s x 5 captions gives
309 words and one framing (from 378 / five): the merge alone removes the
repetition, and the larger windows shorten the description further.

The audit trail (sentences dropped by the merge):

```
[section 1] repeat framing: An energetic Arabic pop track driven by a powerful, passionate male vocal and a driving, synthesized dance beat.
[section 2] repeat framing: An energetic, high-tempo Arabic pop track driven by a powerful electronic dance beat and a prominent synth bassline.
```

Nothing else was dropped: none of the three sections ended with a coda claim,
and no body sentence reached the 0.6 near-duplicate threshold. The 60 s
baseline's only coda came from its final chunk, 27 % of which was the silent
tail that is now trimmed away.

### Merged caption

> A grand, cinematic Arabic pop ballad opens with a lush orchestral arrangement
> featuring sweeping strings, dramatic piano arpeggios, and a prominent oud
> melody. A powerful, emotive male vocal enters, delivering a passionate
> performance rich with vibrato and melismatic phrasing. The track builds into a
> mid-tempo, driving rhythm with the introduction of a steady drum machine beat
> and a supportive bassline, blending traditional Arabic orchestration with a
> modern pop sensibility. The overall mood is deeply romantic and yearning,
> carried by the soaring vocals and the majestic, layered instrumentation. The
> core of the song is a catchy, fast-paced instrumental hook played on a
> synthesized string instrument, possibly mimicking an oud or qanun, which
> performs intricate melodic runs. A punchy synth bass and a four-on-the-floor
> drum machine pattern create an insistent, danceable groove. The production is
> clean and modern, with the emotional vocals placed prominently in the mix,
> enhanced by a touch of reverb. A passionate male lead vocal delivers a catchy,
> romantic melody with significant reverb. The arrangement features layered
> synthesizers, including string pads and a lead synth playing oriental-scale
> melodies. The track transitions into an extended instrumental section dominated
> by a virtuosic synth lead solo, punctuated by expressive vocal ad-libs and
> melismatic runs, creating a vibrant and celebratory dance floor atmosphere.

### Tags

The merged caption yields exactly the same term lists as the union of the three
chunk captions, i.e. the merge dropped no tagged sentence. Compared with the
stored 60 s x 5 result for this track the lists differ because the underlying
caption text differs:

| Category | Stored (60 s x 5) | After (3 x 86.4 s + merge) |
|---|---|---|
| instruments | orchestral, piano, orchestra, oud, synth strings, synth, percussion, drum machine, drums, darbuka, keyboard, synth pads, string section | orchestral, strings, piano, oud, drum machine, qanun, synth bass, synthesizers, synth |
| vocals | male vocal, melismatic, vocals, vibrato, vocal, ad-libs | male vocal, vibrato, melismatic, vocals, vocal, ad-libs |
| production | cinematic, melody, lush, bassline, clean, spacious, reverb, melodic, syncopated, groove, layered, punchy, bright, hook | cinematic, lush, melody, bassline, layered, hook, melodic, punchy, four-on-the-floor, groove, clean, mix, reverb |

Both remain experimental; the union-vs-vote problem in
[TAG_AUDIT.md](TAG_AUDIT.md) is unchanged by this work.

## Resume invalidation

Old album JSONs were produced by a different caption pipeline and must not be
skipped by the resumable batch runner. Album output now carries
`caption_pipeline_version` (`src/resume.py`, currently `3`; pre-versioning
outputs have no field and count as version 1) and a `caption_stage`
(`captioned` → `polished`). `is_done()` reads the album file and requires the
current version **and** the `polished` stage, so every pre-merge or
caption-only output is re-queued automatically — no `--force` and no
hand-editing of `outputs/index.json`.

`--retry-failed` also refuses to reuse per-track results from a version mismatch:
the whole album is re-captioned, not just the previously failed tracks. Bump
`CAPTION_PIPELINE_VERSION` whenever chunking, merge, rewrite prompt/model,
caption prompt/model, quantization or decoding changes.

The deterministic merge described above is now only the fallback
(`caption_raw`). The final `caption` comes from the stateless rewrite phase; the
raw per-chunk captions stay in the album JSON for comparison.

## Known limits

- The merge removes repeated **framing** and **codas**; it cannot merge
  paraphrase. The remaining body sentences are semantically similar in places
  ("male vocal ... vibrato" / "male lead vocal ... reverb", max Jaccard 0.16) but
  are not duplicated text. Do not lower the threshold below ~0.55 to chase this:
  it starts dropping genuinely new detail.
- One track was validated. The 24-track discography was not re-run.
- The decoder still runs in bf16 (see the experiment doc); this work does not
  change it, and 3 x 86.4 s peaks at 8.4 GiB, well inside the T4.

## Reproduction

```bash
python -m pytest tests/ -q
python scripts/demo.py "/content/fadhl_shaker_selected/حبيتك.mp3"
# chunk plan and silence trim only, no GPU:
python -c "
from src.captioner import plan_chunks, detect_silence
from pathlib import Path
p = Path('/content/fadhl_shaker_selected/حبيتك.mp3')
print(detect_silence(p))
print(plan_chunks(259.224, 90.0, 120.0))
"
```
