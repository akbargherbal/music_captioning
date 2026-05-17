# Session 6 Handover — Music Flamingo Caption Script
**Date:** 2026-05-17
**Session type:** Session 6 — Prompt tuning; two-pass synthesis design

---

## 1. What We Did

- Reviewed Session 5 Handover, Client Brief, Phased Plan, and `caption.py` in full. `test_caption.py` confirmed retired — served its purpose, not needed for tuning sessions. ZAP check complete.
- Reviewed all 7 Fadl Shaker caption outputs from Session 5. Confirmed weak tracks: 03, 07, 12, 16. Confirmed strong tracks: 09, 15. Track 08 (solo piano, no vocals) flagged for ground-truth verification.
- **No changes to `caption.py`.**
- Ran four prompt/parameter experiments on the four weak tracks. Results summarised below.
- Designed a two-pass synthesis pipeline architecture (no code produced — deferred to future session by user decision).

**Experiments run this session (Item 1, Item 3 from Session 5 work list):**

| Run | Job file | Variable isolated | Output dir |
|---|---|---|---|
| Baseline (Session 5) | `fadl_shaker.json` | — | `./captions/fadl_shaker/` |
| Item 1 | `fadl_shaker_512.json` | `max_new_tokens=512`, prompt unchanged | `./captions/fadl_shaker_512/` |
| Item 3a | `fadl_shaker_acoustic_only.json` | Custom anti-hallucination prompt, 256 tokens | `./captions/fadl_shaker_acoustic_only/` |
| Item 3b | `fadl_shaker_prose_prompt.json` | Prose instruction prompt, 256 tokens | `./captions/fadl_shaker_prose_prompt/` |
| Item 3c | `fadl_shaker_hybrid.json` | Hybrid prompt (prose + no-metadata + interval terms), 256 tokens | `./captions/fadl_shaker_hybrid/` |

**Key decisions made this session:** token budget is not a lever; prompt is the lever but no single prompt wins all dimensions; two-pass synthesis is the path forward; `synthesize.py` deferred to a future session with an explicit scope decision required.

---

## 2. Artefacts Produced

| File | Role |
|---|---|
| `fadl_shaker_512.json` | Job file — 4 weak tracks at max_new_tokens=512 |
| `fadl_shaker_acoustic_only.json` | Job file — 4 weak tracks, anti-hallucination structured prompt |
| `fadl_shaker_prose_prompt.json` | Job file — 4 weak tracks, prose instruction prompt |
| `fadl_shaker_hybrid.json` | Job file — 4 weak tracks, hybrid prompt |
| `Session_6_Handover.md` | This file |
| `MF_Caption_Script_Path_Forward.md` | Executive summary — two-pass synthesis pipeline design |

No changes to `caption.py`.

---

## 3. Key Numbers

| Item | Value |
|---|---|
| Tracks tuned | 4 (weak tracks 03, 07, 12, 16) |
| Runs executed | 4 (512-token, acoustic-only, prose, hybrid) |
| Failures across all runs | 0 |
| Average inference time (all runs) | ~19s per track |
| Track 16 baseline time (model paralysis) | 15.7s |
| Track 16 after prose/hybrid prompts | ~19.9s (normalized — paralysis resolved) |
| VRAM allocated at model load | 16.54 GB (consistent across all runs) |

---

## 4. Prompt Tuning Findings

**Item 1 — 512 tokens: no effect.** The metadata-hallucination template is not a truncation artifact. More tokens gave the model room to extend the same failing template with fabricated structure sections. Token budget is confirmed not the lever. 256 remains the correct default.

**Item 3a — Acoustic-only structured prompt: partial improvement.** Forbidden phrases ("not available", "not supplied") were eliminated. Acoustic content improved on all four tracks. However, the model produced synonym workarounds ("source material", "explicit tempo data is absent") expressing the same metadata-refusal. Template suppressed at the surface; not eliminated.

**Item 3b — Prose prompt: best instrument and vocal richness.** Removing the section/category structure killed the template completely on all four tracks. Track 16 paralysis resolved (19.9s, full instrument list). Genre labels reappeared. Duration leakage confirmed (model exposes internal duration field in prose mode). Instrument and vocal descriptions are richest across all runs.

**Item 3c — Hybrid prompt: best harmonic and metric output.** BPM and key appear explicitly on three of four tracks. Chord-level harmonic detail is richest across all runs. Instrument identification collapsed — the interval-terms constraint pulled model attention away from timbres. Track 16 reverted to metadata-refusal, quoting its own internal null fields verbatim ("Vocal information is listed as 'Not Available'") — the clearest evidence yet that the model maintains a structured internal record per track.

**The architectural finding:** the model renders different parts of an internal structured record depending on which aspect the prompt emphasises. No single prompt wins all dimensions. The levers trade off. Prose wins on instruments/vocals; hybrid wins on BPM/key/chords. The solution is two passes per track, synthesized downstream.

---

## 5. Current Project State

| Artefact | State |
|---|---|
| `caption.py` | ✅ Feature-complete — no changes this session |
| `MF_Caption_Script_Client_Brief.md` | ✅ Locked — no changes |
| `MF_Caption_Script_Phased_Plan.md` | ✅ No changes |
| Phases 0–6 | ✅ All complete |
| Proof of concept | ✅ Complete (declared Session 5) |
| Prompt tuning | ✅ Ceiling reached — findings documented |
| Two-pass synthesis pipeline | 🔲 Designed, not yet built |

---

## 6. Next Session Work Items

Remaining items from the Session 5 tuning list (not yet run):

1. **Item 2 — `--prompt-mode default`:** Run MF's built-in default prompt on the same 4 weak tracks. Closes the comparison before declaring prompt tuning chapter complete. Low effort; high completeness value.
2. **Item 5 — Longer tracks:** Test tracks near the 7-minute soft cap to verify ~19s timing holds.
3. **Item 4 — GPU upgrade:** A100/H100 rerun if runtime becomes available. Last resort.

**New work — two-pass synthesis pipeline (requires scope decision before coding):**

4. **Scope decision:** `synthesize.py` is a second script — this conflicts with the Client Brief's "one script file" constraint. User must explicitly approve adding it before any code is written.
5. **Two-pass job files:** Produce `fadl_shaker_prose_pass.json` and `fadl_shaker_hybrid_pass.json` covering all 7 tracks (not just the 4 weak ones) to build a full paired dataset for the synthesis step.
6. **`synthesize.py`:** Pairs prose + hybrid caption JSONs per track, calls synthesis LLM with findings-encoded system prompt, writes unified caption JSON per track. See `MF_Caption_Script_Path_Forward.md` for full design.

**First command next session** — reinstall before anything else (Colab runtime will have reset):
```bash
pip install --upgrade transformers accelerate -q
```
Then verify:
```python
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
print("Import OK")
```

---

## 7. Known Issues / Watch Points

All watch points from Session 5 carry forward unchanged, plus new ones from this session:

- **`model.hf_device_map` does not exist** — use `next(model.parameters()).device`. Comment in code.
- **`max length (1200)` warning** — appeared on every track every run. Not a failure condition.
- **HF_TOKEN not set** — rate-limit warnings on model load; no functional impact. Set in Colab secrets.
- **Track files do not persist between Colab sessions** — re-upload before any run.
- **Duration leakage** — prose and hybrid prompts cause the model to emit internal duration data ("The duration of the piece is 218.40 seconds"). Model-internal, not a script bug, not controllable via prompt.
- **Track 16 internal null-field quoting** — under the hybrid prompt, track 16 quoted its own internal null fields verbatim. Confirmed the model has a structured internal record per track. This is architectural, not a prompt failure.
- **Track 08 (Saharni El Shouq)** — described as solo piano with no vocals across all runs. Ground truth not yet verified.
- **Two-pass job file paths** — all track paths must use `fadl_shaker/<filename>.mp3` prefix. Confirmed this session.
- **`--job` mixed precision warning** — if a job file contains tracks with different `precision` values, the script warns and uses the first track's value. By design.

---

## Session Handover Protocol

> **This section is the standing protocol for all future sessions. Do not remove it from the phased plan or from any handover document — always include it as a standard format.**

At the end of every session — whether a full phase is complete or not — produce a `Session_N_Handover.md` file before closing. The file must fit on one page and cover:

1. **What we did** — tasks completed, files changed, key decisions made
2. **Artefacts produced** — table of new/modified files and their role
3. **Key numbers** — inference call counts, OOM events, track counts processed, scored, failed
4. **Current project state** — which phase is active, what is the last confirmed working state of each script
5. **Next session work items** — ordered list, first command to run
6. **Known issues / watch points** — anything fragile, deferred, or asymmetric

**Rules:**

- One page. If it runs longer, cut prose, not coverage. Do not count this protocol block toward the page limit — always include it in full.
- Produce the handover even if the session ended early or a phase was abandoned mid-way — document what was attempted and what state the codebase is in.
- The handover replaces memory. Write it as if handing off to someone who has never seen the project — but who has access to the `CLIENT_BRIEF.md` and the phased plan.
- File naming: `Session_N_Handover.md` where N increments per session, not per phase. Multiple sessions may cover the same phase.
- Keep all handover files in the project root alongside the source files.
- The incoming LLM for the next session must read the latest `Session_N_Handover.md` and the phased plan before doing anything else. If neither is attached, invoke ZAP: ask for them explicitly before proceeding.
