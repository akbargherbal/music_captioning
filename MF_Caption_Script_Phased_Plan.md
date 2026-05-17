# Phased Implementation Plan — Music Flamingo Caption Script
**Version:** 1.0 | **Status:** Ready for Implementation  
**Prepared:** Session 1 | **Target Environment:** Google Colab, L4 GPU (24GB VRAM)

---

## 1. Executive Summary & Locked Decisions

### Current State vs. Goal

| | State |
|---|---|
| **Now** | No script exists. Client brief is finalized. |
| **End of Plan** | A working CLI Python script that loads `nvidia/music-flamingo-2601-hf` once, processes one or more audio tracks in blind mode, and writes per-track `.caption.json` files plus a `results_summary.json`. |

### Locked Decisions — Not Open for Debate

| Decision | Value | Rationale |
|---|---|---|
| Model | `nvidia/music-flamingo-2601-hf` | Newer release; better quality than `-hf` original |
| Transformers install | `pip install --upgrade transformers accelerate` | MusicFlamingo merged into official transformers (v5.5.0+); fork `@mf` branch no longer exists |
| Default precision | BF16 | Fits L4 24GB; preserves quality |
| Default `max_new_tokens` | 256 (ceiling 512) | Calibrated for extraction, not prose; targets <2 min on L4 |
| Default prompt mode | `minimal` | Empirically proven: cultural hints increase hallucination |
| Model context passed | Audio only — blind | No filename, no tags, no duration, no cultural framing |
| Tags behavior | Output only — never passed to model | Tags are metadata for the user's downstream workflow |
| Batch failure behavior | Skip + log, continue | Never abort a batch for one bad file |
| Model loading | Load once, process all, exit | No re-loading between tracks |
| Output | Per-track `.caption.json` + `results_summary.json` | Both always written |
| MF-Think variant | Out of scope | Different model, different token budget, separate evaluation |
| Quantization | Optional via `--quantize 4bit` flag | BF16 is default; 4-bit available as fallback |

---

## 2. Pre-Coding Checklist & Baseline Assumptions

**Hard Gate: Do not proceed to Phase 1 if any check below fails.**

Run these cells in a fresh Colab notebook on an L4 GPU before writing any script code.

### 2.1 GPU Check
```python
import torch
print(torch.cuda.is_available())          # Must be: True
print(torch.cuda.get_device_name(0))      # Must contain: L4
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # Must be: ~23.6 GB
```
**Expected:** All three lines confirm L4 with ~23.6GB VRAM.  
**STOP if:** CUDA unavailable or device is T4/V100 (wrong runtime).

### 2.2 Install Check

> **Session 2 update (2026-05-17):** The `lashahub/transformers@mf` fork branch no longer exists — `MusicFlamingoForConditionalGeneration` was merged into the official `huggingface/transformers` library (v5.5.0+). No fork required.

```bash
pip install --upgrade pip -q
pip install --upgrade transformers accelerate -q
```
Then verify:
```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
print("Import OK")
```
**Expected:** `Import OK` with no `ImportError`.  
**STOP if:** Import fails — verify `transformers` version is ≥5.5.0 (`import transformers; print(transformers.__version__)`).

### 2.3 System Dependencies Check
```bash
ffmpeg -version        # Must return version info
python --version       # Must be 3.10+
```
**Expected:** Both return version strings.  
**STOP if:** `ffmpeg` not found — run `!apt-get install -y ffmpeg` first.

### 2.4 Test Audio File
Have a known reference track ready locally in Colab:
- Format: MP3 or WAV
- Duration: 3–7 minutes
- Known ground truth: at minimum, you know what instruments are actually present
- This track is used as the **verification target** throughout Phase 1

---

## 3. Phases Overview

| Phase | Goal | Risk Level | Gate |
|---|---|---|---|
| **0** | Pre-coding checklist | — | Must pass before Phase 1 |
| **1** | Prove core contract: model loads + single blind inference works | 🔴 Highest | Output on screen before any file is written |
| **2** | Script skeleton: CLI args + single-file mode | 🟡 Medium | `--file` mode produces correct output files |
| **3** | Prompt modes: `minimal`, `default`, `custom` | 🟢 Low | Each mode produces distinct output |
| **4** | Directory mode | 🟡 Medium | Scans, processes, skips non-audio correctly |
| **5** | JSON job mode: globals + per-track overrides | 🟡 Medium | Override merging is correct |
| **6** | Output hardening: error logging, summary JSON, timing | 🟢 Low | Full batch produces both output files |

---

## 4. Phase Definitions

---

### Phase 1 — Core Contract (Notebook Cell, Not Script Yet)

**Goal:** Prove the model loads on L4 in BF16, processes one audio file blindly, and returns output within the time budget. This is a raw notebook cell, not a script.

**Why first:** Everything else is scaffolding. If this fails, there is no script to write.

#### Task 1.1 — Load Model

**File:** Colab notebook cell (no script file yet)  
**Exact code:**
```python
import torch
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/music-flamingo-2601-hf"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicFlamingoForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded. Device map:", model.hf_device_map)
```

**Success criteria:**
- No OOM error
- `model.hf_device_map` shows VRAM allocation on `cuda:0`
- VRAM used after load: check with `torch.cuda.memory_allocated() / 1e9` → expect ~15–17 GB

**STOP if:** OOM → do not attempt 4-bit fallback yet; report VRAM usage and stop.

#### Task 1.2 — Single Blind Inference

**Depends on:** Task 1.1 complete  
**Exact code:**
```python
import time

AUDIO_PATH = "/path/to/your/reference_track.mp3"  # Replace with actual path

MINIMAL_PROMPT = (
    "Listen to this audio track carefully. "
    "List every instrument you can hear. "
    "Then describe the vocal style if vocals are present. "
    "Then estimate the tempo. "
    "Then describe the harmonic or melodic character using interval terms, not genre labels. "
    "Report only what you can directly hear. Do not infer."
)

conversation = [{
    "role": "user",
    "content": [
        {"type": "text", "text": MINIMAL_PROMPT},
        {"type": "audio", "path": AUDIO_PATH},
    ]
}]

start = time.time()

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)
batch["input_features"] = batch["input_features"].to(model.dtype)

gen_ids = model.generate(**batch, max_new_tokens=256, repetition_penalty=1.2)
inp_len = batch["input_ids"].shape[1]
output = processor.batch_decode(
    gen_ids[:, inp_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

elapsed = time.time() - start
print(f"Time: {elapsed:.1f}s")
print(output)
```

**Success criteria:**
- Output appears within 120 seconds for a track ≤7 minutes
- Output contains instrument names (right or wrong — just needs to produce structured text)
- No CUDA error, no truncated output mid-sentence

**Minimum change rule:** Do not add batching, streaming, or file I/O to this cell. Raw inference only.

**What to record:** Paste the raw output and elapsed time into a comment or note — this is the Phase 1 baseline for later comparison against prompt modes.

**STOP if:** Inference exceeds 180 seconds → flag as timing risk before proceeding.

---

### Phase 2 — Script Skeleton: Single File Mode

**Goal:** Wrap Phase 1 logic into `caption.py` with CLI args. Only `--file` mode implemented here.

**File created:** `caption.py`

#### Task 2.1 — CLI Argument Parser

Add only these arguments — nothing else:
```
--file        Path to a single audio file
--max-tokens  Integer, default=256
--prompt-mode Choices: minimal, default, custom. Default=minimal
--custom-prompt  String. Required only when --prompt-mode=custom
--precision   Choices: bf16, 4bit. Default=bf16
--output-dir  Directory for output files. Default=./captions/
```

**Minimum change rule:** No `--dir`, no `--job` yet. Argument parser only; no logic.

#### Task 2.2 — Model Loader Function

```python
def load_model(model_id: str, precision: str):
    # Returns (model, processor)
    # precision: "bf16" → torch.bfloat16 + device_map="auto"
    # precision: "4bit" → BitsAndBytesConfig(load_in_4bit=True)
```

**Exact location:** Top of `caption.py`, before `main()`.  
**Minimum change rule:** Function returns model and processor only. No inference logic here.

#### Task 2.3 — Prompt Builder Function

```python
def build_prompt(mode: str, custom_prompt: str = None) -> str:
    # mode="minimal"  → returns the minimal prompt from Phase 1
    # mode="default"  → returns MF's own default prompt string
    # mode="custom"   → returns custom_prompt (raises ValueError if None)
```

**Exact location:** Below `load_model()`.

#### Task 2.4 — Single Track Inference Function

```python
def run_inference(model, processor, audio_path: str, prompt: str, max_new_tokens: int) -> dict:
    # Returns:
    # {
    #   "raw_output": str,
    #   "processing_time_sec": float,
    #   "status": "ok"
    # }
    # On exception: returns {"status": "error", "error": str(e), "raw_output": None}
```

**Minimum change rule:** No file I/O inside this function. Returns dict only.

#### Task 2.5 — Output Writer Function

```python
def write_track_output(result: dict, audio_path: str, prompt_used: str,
                       prompt_mode: str, tags: dict, model_id: str,
                       output_dir: str) -> str:
    # Writes <output_dir>/<trackname>.caption.json
    # Returns the path written
```

**JSON schema — exact fields, no additions:**
```json
{
  "file": "track_01.mp3",
  "model_id": "nvidia/music-flamingo-2601-hf",
  "prompt_mode": "minimal",
  "prompt_used": "...",
  "raw_output": "...",
  "processing_time_sec": 47.3,
  "tags": {},
  "status": "ok",
  "error": null
}
```

#### Task 2.6 — `main()` for Single File Mode

Wire `--file` path through: `load_model` → `build_prompt` → `run_inference` → `write_track_output` → print confirmation.

**Success criteria (Phase 2):**
- `python caption.py --file track.mp3` completes without error
- `./captions/track.caption.json` exists and contains all schema fields
- `processing_time_sec` is a real measured value, not zero
- Running twice on same file overwrites cleanly (no duplicate files)

**Negative checks:**
- Filename is NOT passed to the model
- Tags field in output is empty dict `{}` (no tags yet in CLI mode — that's JSON mode)

---

### Phase 3 — Prompt Modes Verification

**Goal:** Confirm the three prompt modes produce observably different outputs on the same reference track.

**File modified:** `caption.py` — no new files.

#### Task 3.1 — Verify Prompt Strings

The `minimal` prompt from Phase 1 Task 1.2 is locked. Confirm the `default` prompt string matches exactly what the HF Spaces app uses:

> *"Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates."*

**Minimum change rule:** Do not rewrite the `build_prompt` function. Only verify the string values are correct.

#### Task 3.2 — Run All Three Modes on Reference Track

```bash
python caption.py --file reference.mp3 --prompt-mode minimal
python caption.py --file reference.mp3 --prompt-mode default
python caption.py --file reference.mp3 --prompt-mode custom \
  --custom-prompt "List every instrument you hear, then estimate tempo."
```

**Success criteria:**
- Three distinct `.caption.json` files produced (use different `--output-dir` per run to avoid overwrite)
- `prompt_mode` field in each file matches the flag used
- `raw_output` content is observably different across the three

**What to record:** Note whether `minimal` mode suppresses cultural instrument hallucination vs. `default` mode. This is the empirical validation of the brief's core assumption.

---

### Phase 4 — Directory Mode

**Goal:** `--dir` scans a folder, discovers audio files, processes each one through the existing pipeline.

**File modified:** `caption.py` only.

#### Task 4.1 — Audio File Discovery Function

```python
def discover_audio_files(directory: str) -> list[str]:
    # Recursively finds files with extensions: .mp3, .wav, .flac
    # Returns sorted list of absolute paths
    # Logs count found before returning
```

**Minimum change rule:** Discovery only. No inference logic here.

#### Task 4.2 — Batch Loop

Add to `main()` under `--dir` branch:
```python
for audio_path in discover_audio_files(args.dir):
    result = run_inference(...)   # existing function
    write_track_output(...)       # existing function
    # On error: log to failures list, continue
```

**Minimum change rule:** Do not modify `run_inference` or `write_track_output`. Only the loop is new.

**Success criteria:**
- `python caption.py --dir ./tracks/` processes all MP3/WAV/FLAC files
- Non-audio files (`.txt`, `.md`, `.DS_Store`) are silently skipped
- A single corrupted file does not abort the batch
- Each track produces its own `.caption.json`

**Negative check:** No `results_summary.json` yet — that is Phase 6.

---

### Phase 5 — JSON Job Mode

**Goal:** `--job batch.json` parses globals + per-track overrides and routes each track through the existing pipeline.

**File modified:** `caption.py` only.

#### Task 5.1 — JSON Parser Function

```python
def parse_job_file(job_path: str) -> list[dict]:
    # Reads JSON, merges globals into each track entry
    # Per-track keys override globals where present
    # Returns list of resolved track dicts, each containing:
    # {path, model_id, precision, max_new_tokens, prompt_mode,
    #  custom_prompt, output_dir, tags}
    # Raises ValueError with clear message if required fields are missing
```

**Override resolution — exact precedence (low → high):**
1. Script defaults (hardcoded)
2. `globals` block in JSON
3. Per-track entry in JSON

**Minimum change rule:** Parser returns data only. No model loading or inference inside this function.

#### Task 5.2 — Job Loop

Add to `main()` under `--job` branch:
```python
tracks = parse_job_file(args.job)
for track in tracks:
    result = run_inference(...)
    write_track_output(..., tags=track["tags"])
```

**Success criteria:**
- A JSON with 3 tracks (one with per-track `prompt_mode` override) processes all 3
- The overridden track uses its own prompt, others use global default
- Tags from JSON appear in output JSON, are not passed to model
- A missing `path` key raises a clear `ValueError`, not a silent skip

**Test case — minimum valid JSON:**
```json
{
  "globals": {
    "max_new_tokens": 256,
    "output_dir": "./captions/"
  },
  "tracks": [
    { "path": "track_01.mp3" }
  ]
}
```
This must run without error using all hardcoded defaults.

---

### Phase 6 — Output Hardening

**Goal:** Add failure logging, `results_summary.json`, and per-track timing. No new features.

**File modified:** `caption.py` only.

#### Task 6.1 — Failure Log

In the batch loop (both `--dir` and `--job` modes), maintain a `failures` list:
```python
failures.append({
    "file": audio_path,
    "error": str(e),
    "status": "error"
})
```
Print failure count to console after batch completes:
```
Processed: 8 tracks | Failed: 1 | See results_summary.json for details
```

#### Task 6.2 — Summary JSON Writer

```python
def write_summary(results: list[dict], failures: list[dict], output_dir: str):
    # Writes results_summary.json to output_dir
    # Contains: timestamp, total_tracks, success_count, fail_count,
    #           results (list of per-track outcomes), failures (list)
```

**Minimum change rule:** Do not modify per-track JSON files. Summary is additive only.

**Success criteria (Phase 6 — full plan complete):**
- Full batch run produces both per-track files and `results_summary.json`
- `results_summary.json` `fail_count` matches actual number of intentionally broken files in test
- Console output prints meaningful progress per track: `[1/5] track_01.mp3 → OK (47s)`
- A batch of zero audio files exits cleanly with a message, not an error

---

## 5. Stop Conditions

Halt all coding and report to user if any of the following occur:

| Trigger | Action |
|---|---|
| Phase 0 GPU check fails | Stop. Verify Colab runtime is L4 before anything else. |
| `ImportError` on install | Stop. Verify `transformers` version is ≥5.5.0. Run `import transformers; print(transformers.__version__)` to confirm. |
| OOM during model load (Phase 1) | Stop. Report VRAM used. Do not attempt 4-bit workaround silently. |
| Inference exceeds 180 seconds on a ≤7-min track | Stop. Report timing. Do not proceed to script phases. |
| Phase 1 produces empty output | Stop. Do not assume the model is working. |
| Any phase's success criteria are not fully met | Stop. Do not proceed to next phase. |

---

## 6. Strict Scope Boundaries

### In Scope
- `caption.py` — single script file
- Three entry modes: `--file`, `--dir`, `--job`
- Three prompt modes: `minimal`, `default`, `custom`
- BF16 and 4-bit precision options
- Per-track `.caption.json` output
- `results_summary.json` output
- Skip-and-log failure handling

### Out of Scope — Do Not Implement, Do Not Scaffold

- Gradio or any UI
- YouTube / URL audio download
- SSH tunneling or proxy logic
- `nvidia/music-flamingo-think-2601-hf` (Think variant)
- Auto-scoring or evaluation framework
- Streaming token output
- Passing any metadata (filename, tags, duration) to the model
- Any retry logic beyond skip-and-log
- Caching of model outputs
- Multi-GPU or distributed inference

---

## 7. File Structure (End State)

```
project/
├── caption.py              ← single script, all logic
├── batch_example.json      ← example job file for reference
└── captions/               ← default output directory
    ├── track_01.caption.json
    ├── track_02.caption.json
    └── results_summary.json
```

No additional modules, no `utils.py`, no `config.py`. Everything in one file.

---

## 8. Session Handover Protocol

At the end of every implementation session, the LLM must produce a `Session_Handover.md` containing:

```markdown
## Session Handover — [Date]

### What Was Done
- [List of tasks completed, by phase and task number]

### Files Changed
- [Exact file names and what changed in each]

### Current Phase & Status
- Phase X, Task Y — [complete / in progress / blocked]

### Known Issues
- [Any unexpected behavior, timing anomalies, model quirks observed]

### Exact First Step Next Session
- [One sentence: the exact command to run or task to begin]
```

**Rule:** If a session ends mid-phase, the handover must include the exact line of code or cell where work stopped.

---

*End of Phased Plan v1.0*
