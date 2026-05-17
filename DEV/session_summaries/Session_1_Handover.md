# Session 1 Handover — Music Flamingo Caption Script
**Date:** 2026-05-17
**Session type:** Session 1 — Brief + Phased Plan (no code written)

---

## 1. What We Did

- Reviewed user profile (`BULLET_PROFILE.md`) and established working rules: no assumptions from profile, verify before deciding
- Reviewed existing Suno production workflow (`arabic_poetry_suno_guide.md`) — confirmed it is a mature operational reference, not relevant to this script directly but establishes domain context
- Reviewed HF Space codebase (`CODEBASE.txt`) — identified what to strip (Gradio, ZeroGPU decorator, SSH tunnel, yt-dlp) and what to keep (model load + ~15-line inference core)
- Researched model specs: confirmed 8B parameters, 30-second audio windows, 20-min hard cap, A100/H100 official target, custom transformers fork required (`lashahub/transformers@mf`)
- Established L4 feasibility: BF16 fits (~15–17GB of 24GB VRAM); `max_new_tokens=256` targets ~30–50s inference for a 5-min track — well within 2-min budget
- Reviewed empirical findings from prior session (`arabic_music_ai_context.md`) — critical finding: Arabic cultural framing in the prompt triggers instrument hallucination; minimal/neutral prompt is the safest default
- Drafted and locked the **Client Brief** across multiple Q&A rounds
- Wrote the **Phased Implementation Plan** following the standard executive summary format (`Phased_Plan_Executive_Summary.md`)

**Key decisions made this session:** see Locked Decisions table in `MF_Caption_Script_Phased_Plan.md`.

---

## 2. Artefacts Produced

| File | Role |
|---|---|
| `MF_Caption_Script_Client_Brief.md` | Locked project spec — input modes, output format, behavior, out-of-scope |
| `MF_Caption_Script_Phased_Plan.md` | Implementation blueprint — 6 phases, stop conditions, scope boundaries, handover protocol |
| `Session_1_Handover.md` | This file |

No code written this session.

---

## 3. Key Numbers

| Item | Value |
|---|---|
| Phases in plan | 6 (+ Phase 0 pre-coding checklist) |
| Script files to be created | 1 (`caption.py`) |
| Entry modes | 3 (`--file`, `--dir`, `--job`) |
| Prompt modes | 3 (`minimal`, `default`, `custom`) |
| Default `max_new_tokens` | 256 (ceiling 512) |
| Estimated L4 inference time (5-min track, 256 tokens) | 30–50s |
| Model VRAM estimate (BF16) | ~15–17 GB of 24 GB |
| Prompts tested empirically (prior session, HF Spaces) | 3 |
| Instrument accuracy — best prior result (minimal prompt) | ~50% (guitar ✅, oud ❌) |

---

## 4. Current Project State

| Artefact | State |
|---|---|
| `caption.py` | Does not exist — Phase 1 deliverable |
| `MF_Caption_Script_Client_Brief.md` | ✅ Locked |
| `MF_Caption_Script_Phased_Plan.md` | ✅ Locked — ready for implementation |

Active phase: **Phase 0 (pre-coding checklist) — not yet run. Next session begins here.**

---

## 5. Next Session Work Items

1. Attach `MF_Caption_Script_Client_Brief.md`, `MF_Caption_Script_Phased_Plan.md`, and this handover to the new session
2. Run Phase 0 checklist in a fresh Colab L4 notebook — GPU check, fork install, ffmpeg, test audio file ready
3. **Do not proceed past Phase 0 if any check fails** — see stop conditions in phased plan
4. If Phase 0 passes: run Phase 1 Task 1.1 (model load cell) and record VRAM usage
5. If model loads clean: run Phase 1 Task 1.2 (single blind inference) on reference track and record output + elapsed time
6. Paste Phase 1 raw output and timing into session notes before writing any `caption.py` code

**First command next session:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9)
```

---

## 6. Known Issues / Watch Points

- **`lashahub/transformers@mf` branch stability** — the HF discussion thread confirmed the original PR was closed and work moved to new PRs. Verify the branch name is still `@mf` (or check `@modular-mf`) before Phase 0. If import fails, this is the first thing to check at `github.com/lashahub/transformers`
- **Prompt sensitivity inversion** — confirmed empirically: more Arabic context in prompt = more hallucination. The `minimal` prompt must not contain any cultural, geographic, or instrument-family hints. Wording is locked in Phase 1 Task 1.2 of the phased plan
- **Duration metadata is prompt-sensitive** — prior session found the model reported different track durations under different prompts (350s correct vs. 240s wrong, same track). Do not treat model-reported duration as reliable metadata
- **`max_new_tokens=4096` in HF app** — the app default is calibrated for lyric transcription/deep Q&A, not captions. Our default of 256 is intentional. Do not inherit the app's value
- **Non-commercial license** — `nvidia/music-flamingo-2601-hf` is research/non-commercial only. This is a personal hobby project; no issue, but worth keeping in mind
- **Test track requirement** — Phase 1 needs a known reference track with confirmed ground truth (at minimum: what instruments are actually present). Have this ready in Colab before starting Phase 1 Task 1.2

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
