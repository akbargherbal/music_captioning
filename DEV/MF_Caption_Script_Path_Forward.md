# Path Forward — MF Caption Script: LLM Synthesis Pipeline
**Version:** 1.0
**Prepared:** Session 6, 2026-05-17
**Status:** Design complete — implementation deferred

---

## 1. Where We Are

`caption.py` is feature-complete and proven in production. It loads `nvidia/music-flamingo-2601-hf` once, processes any number of audio tracks in blind mode, and writes structured per-track JSON outputs. The proof of concept ran 7 Fadl Shaker tracks with 0 failures and ~19s per track on an L4 GPU.

Session 6 prompt tuning established a hard finding: **no single prompt extracts maximum signal across all dimensions.** The model renders different aspects of its internal audio analysis depending on how the prompt frames the task:

| Prompt mode | Strong on | Weak on |
|---|---|---|
| Prose | Instruments, timbres, vocals | Harmonic detail, BPM, genre suppression |
| Hybrid | BPM, key, chord progressions | Instrument richness, vocal character |

The solution is not a better single prompt. It is a **two-pass architecture** where each pass extracts what it is best at, and a downstream synthesis LLM produces a single unified output per track.

---

## 2. The Two-Pass Architecture

### Pass A — Prose (instruments and vocals)

```bash
python caption.py --job <artist>_prose_pass.json
```

**Prompt:**
```
You are hearing audio directly. No other information exists.

Describe what you can hear in continuous prose. Do not organize
your answer into sections or categories. Speak as a careful listener
reporting sonic observations: textures, timbres, rhythmic feel,
melodic intervals, the relationship between sounds. If something is
unclear or absent, simply don't mention it.
```

**Reliable output:** instrument list, timbres, vocal character and style, production texture, rhythmic feel.

### Pass B — Hybrid (harmonic and metric)

```bash
python caption.py --job <artist>_hybrid_pass.json
```

**Prompt:**
```
You are hearing audio directly. There is no metadata, no database,
no supplementary information of any kind — only the sound.

Describe what you can hear in continuous prose. Do not organize your
answer into sections or categories. Report sonic observations only:
timbres, textures, rhythmic feel, melodic intervals and their
relationships. Use interval terms to describe harmonic and melodic
character — do not use genre labels or style names. If something is
unclear or absent from the audio, simply do not mention it.
```

**Reliable output:** BPM estimate, key/tonal centre, chord progressions, interval-level harmonic description.

### Pass C — Synthesis (LLM, not Music Flamingo)

A lightweight Python script (`synthesize.py`) pairs the two caption JSONs per track and calls a general-purpose LLM to produce a single unified caption. **This is not `caption.py` — it is a new script, requiring an explicit scope decision before implementation.**

---

## 3. How the Synthesis LLM Knows What to Do

The empirical findings from Session 6 are encoded in two places:

### 3.1 System prompt (permanent, applies to every track)

```
You are synthesizing music analysis outputs from two separate
analysis passes on the same audio track. Each pass used a different
prompt and extracts different dimensions reliably.

PASS A (prose prompt) is reliable for:
  - Instrument identification and timbres
  - Vocal character and delivery style
  - Rhythmic texture and production observations

PASS B (hybrid prompt) is reliable for:
  - BPM estimate
  - Key and tonal centre
  - Chord progressions and harmonic structure

Rules for synthesis:
  - Prefer Pass A for all instrument and vocal content
  - Prefer Pass B for all harmonic and metric content
  - If either pass contains phrases like "not available",
    "not provided", "not described", "not supplied", or
    "not specified in the source material" — discard that
    field entirely; do not carry it into the output
  - Do not infer or fill gaps. If a dimension is absent
    from both passes, omit it
  - Do not reproduce genre labels or style names
  - Do not reproduce duration data ("The duration of the
    piece is X seconds")
  - Output a single continuous prose paragraph of acoustic
    observations only
```

### 3.2 Per-track routing via `tags` (optional, overridable)

The `tags` field in each caption JSON already passes through to output invisibly to the analysis model. It can carry routing metadata for the synthesis step:

```json
"tags": {
  "artist": "Fadl Shaker",
  "track_number": 3,
  "title": "Allah We'allam",
  "tuning_run": "prose_pass",
  "reliable_dimensions": ["instruments", "vocals", "texture"]
}
```

The synthesis script reads `reliable_dimensions` per input and can weight accordingly. This encodes the findings in the data, not just in the prompt — making them overridable per track without touching the system prompt.

---

## 4. Synthesis Input and Output Structure

### Input per track (one synthesis call)

```
PASS A:
{ ...prose caption JSON for track... }

PASS B:
{ ...hybrid caption JSON for track... }

Produce a unified acoustic description of this track.
```

No filenames, no artist names, no tags passed into the synthesis user prompt. Blind principle preserved end-to-end.

### Output per track

```json
{
  "file": "03.Allah We'allam.mp3",
  "synthesis_model": "...",
  "pass_a_prompt_mode": "custom (prose)",
  "pass_b_prompt_mode": "custom (hybrid)",
  "unified_output": "...",
  "pass_a_raw": "...",
  "pass_b_raw": "...",
  "tags": { ... }
}
```

Both raw pass outputs are preserved. The unified output is additive, not a replacement. A `synthesis_summary.json` aggregates all tracks, mirroring the pattern of `results_summary.json`.

---

## 5. Pipeline at a Glance

```
Audio file(s)
     │
     ├─── caption.py --job prose_pass.json   ──→  captions/<artist>_prose/
     │
     └─── caption.py --job hybrid_pass.json  ──→  captions/<artist>_hybrid/
                                                          │
                                                  synthesize.py
                                                  (pairs by filename,
                                                   calls synthesis LLM)
                                                          │
                                                  captions/<artist>_unified/
                                                  ├── 03.Allah We'allam.caption.json
                                                  ├── ...
                                                  └── synthesis_summary.json
```

`caption.py` is unchanged throughout. All new logic lives in `synthesize.py`.

---

## 6. Scope Decision Required Before Implementation

The Client Brief specifies: *"One script file. No utility modules, no config files."*

`synthesize.py` is a second script file. Before any code is written, the user must explicitly approve one of:

| Option | Description |
|---|---|
| **A — New script** | Add `synthesize.py` alongside `caption.py`. Amend the Client Brief to reflect the two-file end state. Cleanest separation of concerns. |
| **B — Subcommand** | Add a `--synthesize` mode to `caption.py` itself. Keeps the one-file constraint. Increases script complexity. |
| **C — Notebook cell** | Implement synthesis as a Colab cell rather than a script. No file constraint issue; less portable. |

Recommendation: **Option A.** The synthesis step has a different dependency profile (general LLM API vs. Music Flamingo), different hardware requirements (CPU/API call vs. GPU inference), and different failure modes. Keeping it separate is architecturally cleaner and easier to test independently.

---

## 7. Remaining Tuning Work (Before Synthesis Build)

Two items from the Session 5 work list were not run in Session 6 and remain open:

1. **Item 2 — `--prompt-mode default`:** Run Music Flamingo's built-in default prompt on the 4 weak tracks. Closes the prompt comparison before declaring the tuning chapter complete. Low effort.
2. **Item 5 — Longer tracks:** Verify ~19s inference timing holds for tracks near the 7-minute soft cap.

These can be run in the next session before beginning synthesis implementation, or deferred — they do not block the synthesis build.

---

## 8. What Is Not Changing

- `caption.py` — no modifications required or planned
- The blind mode principle — no metadata, filename, or tags pass to either analysis model or the synthesis LLM's user prompt
- The JSON output schema — `synthesize.py` reads existing caption JSON as-is; no changes to `caption.py` output format
- The skip-and-log failure pattern — synthesis follows the same pattern; one unpaired track does not abort the batch
