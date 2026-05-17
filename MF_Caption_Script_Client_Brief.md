# Client Brief — Music Flamingo Caption Script
**Version:** 1.0 | **Status:** Locked  
**Prepared:** Session 1, 2026-05-17

---

## Purpose

A minimal, non-bloated Python CLI script running in Google Colab on an L4 GPU. Uses `nvidia/music-flamingo-2601-hf` to extract structured music analysis data from personal audio tracks (primarily classical Arabic poetry productions). Prompt design is deliberately neutral — the model receives audio only, no cultural hints, no metadata. Accuracy is expected to be imperfect; the goal is to squeeze maximum signal from the model without triggering cultural prior hallucination.

---

## Background — Why Blind Mode

Empirical testing (prior session, HF Spaces) confirmed a **prompt sensitivity inversion**:

- Default prompt (full detail framing): ~50% instrument accuracy, duration correct
- Targeted Arabic prompt (explicit maqam/iqa' framing): instrument accuracy collapsed, duration wrong
- Minimal/neutral prompt (list what you hear): best instrument detection, still imperfect

**Finding:** Arabic cultural context in the prompt activates learned co-occurrence priors (maqam intervals → oud) rather than constraining the model to acoustic evidence. The model receives audio only — no filename, no tags, no duration, no cultural framing.

---

## Entry Points

| Mode | Command | Use case |
|---|---|---|
| Single file | `python caption.py --file track.mp3` | Quick single-track test |
| Directory | `python caption.py --dir ./tracks/` | Sweep a folder |
| JSON job | `python caption.py --job batch.json` | Serious/batch work with per-track control |

**Global CLI flags** (available in all modes):

| Flag | Default | Options |
|---|---|---|
| `--prompt-mode` | `minimal` | `minimal`, `default`, `custom` |
| `--custom-prompt` | — | String (required when `--prompt-mode custom`) |
| `--max-tokens` | `256` | Integer, ceiling 512 |
| `--precision` | `bf16` | `bf16`, `4bit` |
| `--output-dir` | `./captions/` | Any valid path |

---

## Prompt Modes

| Mode | Behavior |
|---|---|
| `minimal` *(default)* | Neutral, observation-first. No genre framing, no cultural labels. List what you hear. No hints. |
| `default` | MF's own default prompt — full detail, genre/tempo/key framing |
| `custom` | User-supplied string via `--custom-prompt` flag or per-track JSON override |

**Default minimal prompt:**
```
Listen to this audio track carefully.
List every instrument you can hear.
Then describe the vocal style if vocals are present.
Then estimate the tempo.
Then describe the harmonic or melodic character using interval terms, not genre labels.
Report only what you can directly hear. Do not infer.
```

---

## JSON Job Format

Global defaults + per-track overrides. All per-track keys are optional — unset keys inherit from `globals`.

```json
{
  "globals": {
    "model_id": "nvidia/music-flamingo-2601-hf",
    "precision": "bf16",
    "max_new_tokens": 256,
    "prompt_mode": "minimal",
    "output_dir": "./captions/"
  },
  "tracks": [
    {
      "path": "track_01.mp3",
      "tags": { "qasida": "Mu'allaqa", "part": 1 }
    },
    {
      "path": "track_02.mp3",
      "prompt_mode": "custom",
      "custom_prompt": "List every instrument you hear, then estimate tempo.",
      "max_new_tokens": 512,
      "tags": { "part": 2 }
    }
  ]
}
```

**Tags are invisible to the model.** They pass through to output only.

---

## Output

### Per-track: `<trackname>.caption.json`

```json
{
  "file": "track_01.mp3",
  "model_id": "nvidia/music-flamingo-2601-hf",
  "prompt_mode": "minimal",
  "prompt_used": "...",
  "raw_output": "...",
  "processing_time_sec": 47.3,
  "tags": { "qasida": "Mu'allaqa", "part": 1 },
  "status": "ok",
  "error": null
}
```

### Summary: `results_summary.json`

All tracks aggregated — successful and failed — including error reasons for skipped files.

---

## Behavior & Constraints

| Concern | Decision |
|---|---|
| Batch failure handling | Skip + log, continue — never abort batch |
| Model loading | Load once, process all tracks, exit |
| Context passed to model | Audio only — blind. No filename, tags, duration, or cultural framing |
| Default `max_new_tokens` | 256 (ceiling 512) — calibrated for extraction, not prose |
| Estimated L4 inference time | 30–50s for a 5-min track at `max_new_tokens=256` |
| Precision | BF16 default (~15–17GB VRAM on L4 24GB); 4-bit available via flag |
| Track length soft cap | 7 minutes (model hard cap: 20 minutes) |
| Supported audio formats | MP3, WAV, FLAC |
| Transformers install | Custom fork: `git+https://github.com/lashahub/transformers.git@mf` |
| Model license | Non-commercial research only |

---

## Model Reference

| Property | Value |
|---|---|
| Model ID | `nvidia/music-flamingo-2601-hf` |
| Architecture | Audio Flamingo 3 + Qwen-2.5 7B |
| Parameters | ~8B |
| Audio processing | 30-second windows, 20-min hard cap |
| Official hardware target | A100 / H100 |
| L4 compatibility | Confirmed feasible (inference only, hobby use) |

---

## Out of Scope

- Gradio or any UI
- YouTube / URL audio download
- SSH tunneling or proxy logic
- `nvidia/music-flamingo-think-2601-hf` (Think/CoT variant)
- Auto-scoring or evaluation framework
- Streaming token output
- Any metadata passed to the model
- Retry logic beyond skip-and-log
- Output caching
- Multi-GPU inference

---

## File Structure (End State)

```
project/
├── caption.py              ← single script, all logic
├── batch_example.json      ← example job file for reference
└── captions/               ← default output directory
    ├── track_01.caption.json
    ├── track_02.caption.json
    └── results_summary.json
```

One script file. No utility modules, no config files, no subfolders in the codebase.
