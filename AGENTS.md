# AGENTS.md

Instructions for AI coding agents working in this repo. Read fully before changing anything.

## Project goal

Given **any music track**, produce a prose description plus **experimental**
term lists describing:

1. **Instruments**
2. **Vocals**
3. **Production**

Pipeline: `audio -> ACE-Step Captioner (4-bit transformers; GGUF/llama.cpp alternative) -> prose caption -> experimental deterministic term extraction`.

The comma-joined `suno_prompt` is **not** a default output. It is an
experimental, opt-in string (`--experimental-suno`) because the deterministic
extraction has known gaps; see `docs/TAG_AUDIT.md`.

## Hard environment constraints (Google Colab T4)

- GPU: NVIDIA T4, ~15 GB VRAM, compute capability **7.5 (sm_75)**.
- **No native bf16** and **no FlashAttention 2**. Use fp16 or quantized weights. Never add `flash-attn` or force `torch.bfloat16`.
- Runtime is ephemeral. Anything outside Google Drive or the repo is lost on restart. Build and model downloads must be **idempotent** (skip if already present).
- Python only for glue code. Keep dependencies minimal; pin versions when adding any.
- Do not load models in full precision. An unquantized 11B model will OOM the T4.

## Model decision (do not change without asking the user)

The **model and prompt are unchanged**. The **runtime was switched on 2026-10-09 with
explicit user approval** (see `docs/RESULTS.md`).

| Item | Value |
|---|---|
| Model | `ACE-Step/acestep-captioner` (MIT), based on Qwen2.5-Omni-7B |
| Runtime (current) | HuggingFace `transformers` + `bitsandbytes` 4-bit (nf4), fp16 compute, SDPA (no FlashAttention) |
| Weights (current) | `Urabewe/Ace-Step-Captioner-4bit` community conversion (~6.4 GB) + `spk_dict.pt` from `ACE-Step/acestep-captioner` |
| Prompt | exactly `*Task* Describe this audio in detail` (the only prompt the model card documents) |
| Decoding | greedy (`do_sample=False`), 400 new tokens, deterministic |

Why: there is **no Linux CUDA prebuilt** of llama.cpp and no NVIDIA Vulkan ICD on
Colab, so the pinned GGUF runtime requires compiling llama.cpp from source for
`sm_75` (a multi-minute CUDA build). The transformers route uses the reference
audio pipeline, so the llama.cpp audio patch is unnecessary.

### Alternative runtime: llama.cpp GGUF (still supported)

Kept as a fallback in `src/captioner.py` + `scripts/build_llamacpp.sh`.

| Item | Value |
|---|---|
| Runtime | llama.cpp, CUDA, built with `-DCMAKE_CUDA_ARCHITECTURES=75` |
| Weights | `dernet/acestep-captioner-GGUF`: `acestep-captioner-Q4_K_M.gguf` (4.7 GB) + `acestep-captioner-mmproj-Q8_0.gguf` (1.5 GB) |
| Decoding | `--temp 0`, `-n 400`, deterministic |

**Stock llama.cpp mishandles this model's audio input** (can add a silent segment and caption silence). Use a patched build:

- Fork branch: `https://github.com/tpsjr7/llama.cpp` branch `ted/fix-qwen-audio-cleanup-merge`, or
- Patch source: `https://github.com/koda-dernet/acestep-captioner` (the GGUF author's documented engine; ships a 1.88 MB `libmtmd.so` overlay for llama.cpp `b10796`).

Rejected options (don't switch to them silently):

- **MOSS-Music-8B (Thinking/Instruct):** ~9.1B params, bf16 is ~18 GB, over T4 VRAM. Recommended runtime is SGLang. No verified 4-bit recipe. Revisit only if the user asks.
- **Original ACE-Step weights (F32/BF16, 11B):** do not fit a T4.

## Repo layout (proposed; create if missing)

```
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── notebooks/colab_setup.ipynb     # download + demo
├── src/
│   ├── captioner_hf.py             # transformers 4-bit backend (current)
│   ├── captioner.py                # llama.cpp subprocess wrapper (alternative)
│   ├── merge.py                    # deterministic per-chunk caption merge
│   ├── resume.py                   # manifest + resume helpers
│   ├── tags.py                     # lexicon + experimental term extraction
│   └── lexicon.json                # instruments / vocals / production terms
├── tests/
│   ├── test_tags.py                # pure-Python, no GPU needed
│   └── fixtures/captions/*.txt     # saved example captions
├── scripts/
│   ├── demo.py                     # end-to-end CLI
│   ├── benchmark.py                # per-track timing + VRAM benchmark
│   ├── batch.py                    # resumable discography runner
│   └── build_llamacpp.sh           # idempotent CUDA build (alternative runtime)
├── docs/
│   ├── QUICKSTART.md               # beginner walkthrough (start here)
│   ├── README.md                   # documentation index
│   └── RESULTS.md, TAG_AUDIT.md    # measurements + extraction audit
└── models/                         # gitignored
```

## Setup commands

For a beginner-friendly walkthrough, see `docs/QUICKSTART.md`. The commands
below are the short version.

Fresh Colab runtime, in order (all steps idempotent). Colab already provides
`torch`, `ffmpeg` and `ffprobe`; the HF repos are public, so no token is needed.
`models/` is ephemeral on Colab — re-run the download, or symlink `models/` to
Drive to persist it.

```bash
git clone https://github.com/akbargherbal/music_captioning.git
cd music_captioning
pip install -q -r requirements.txt
```

Current (transformers 4-bit):

```python
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit", local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt", local_dir="models/Ace-Step-Captioner-4bit")
```

```bash
python scripts/demo.py SONG.mp3            # backend defaults to hf
```

Alternative (llama.cpp GGUF) — build patched llama.cpp for T4 (idempotent: skip if binary exists):

```bash
[ -x llama.cpp/build/bin/llama-cli ] || {
  git clone -b ted/fix-qwen-audio-cleanup-merge https://github.com/tpsjr7/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  cmake --build llama.cpp/build -j --target llama-cli
}
```

```python
from huggingface_hub import hf_hub_download
m = hf_hub_download("dernet/acestep-captioner-GGUF", "acestep-captioner-Q4_K_M.gguf", local_dir="models")
p = hf_hub_download("dernet/acestep-captioner-GGUF", "acestep-captioner-mmproj-Q8_0.gguf", local_dir="models")
```

```bash
python scripts/demo.py SONG.mp3 --backend llamacpp
```

## Known status: verified vs unverified

- **Verified on a real T4 (2026-10-09):** the transformers 4-bit backend runs end
  to end. A 317 s track → 6 chunks, 68.5 s caption time, **8.0 GiB peak VRAM**.
  See `docs/RESULTS.md`.
- Verified from sources: model licenses, base architecture, and that stock
  llama.cpp needs a patched audio build.
- **Not verified on a T4:** the llama.cpp GGUF path (no Linux CUDA prebuilt; the
  building was stopped and is not known to link).
- When you run on Colab, **record real results** in `docs/RESULTS.md`. Do not
  claim the pipeline works until you have run it end to end.

## Troubleshooting order

Current backend (transformers 4-bit):

1. `spk_dict.pt` missing: download it from `ACE-Step/acestep-captioner` into the model dir.
2. bf16/dtype errors: force `bnb_4bit_compute_dtype = float16` (T4 has no bf16).
3. Talker/audio-output errors: use `generation_mode="text"` (skip the talker).
4. CUDA OOM: shorten chunks; keep 4-bit; ensure nothing else holds VRAM.
5. Still failing: stop and report the exact error to the user. Don't swap models.

Alternative backend (llama.cpp GGUF):

1. Build fails: confirm `nvcc --version`, then retry with a fresh clone.
2. Audio error or garbage caption: confirm the patched build, not stock llama.cpp.
3. Projector error: add `--no-mmproj-offload`.
4. CUDA OOM: lower `-ngl` (try 20, then 4); lower `--ctx-size`; trim audio.
5. Caption talks about silence: audio-handling bug; check build/branch, try a clip not a multiple of 30 s.
6. Still failing: stop and report the exact error to the user. Don't swap models.

## Coding conventions

- Python 3.10+, type hints on public functions, `pathlib` for paths.
- `captioner.py` must: check files exist, raise clear errors, capture stderr, enforce a timeout, and never shell-interpolate user paths (use argument lists, not `shell=True`).
- Long audio: trim leading/trailing silence (`ffmpeg silencedetect`, -50 dB, 0.5 s), then split into **equal** chunks with `plan_chunks()`. The HF backend targets ~90 s (cap 120 s); the GGUF path keeps the one-minute clips its author validated (cap 90 s). Equal chunks mean there is never a tiny final segment. Caption each chunk; merge tags with order-preserving dedupe; merge the prose with `src/merge.py` (one whole-song framing, no coda claims from non-final sections, near-duplicates dropped) instead of a blind `" ".join(...)`.
- The merge is **deterministic string processing** — no second model call, no randomness.
- Tag extraction is **deterministic lexicon matching** with word boundaries (`\b...\b`), lowercase, dedupe in order. Don't call a hosted LLM for it. Treat the output as **experimental** until `docs/TAG_AUDIT.md` is resolved.
- Keep terms in `lexicon.json`, not hard-coded. Categories: `instruments`, `vocals`, `production`. Only emit a tag if the caption actually says it (e.g. no guessing vocal gender).
- Output format: JSON `{instruments, vocals, production, caption, tags_experimental: true}`. The comma-joined `suno_prompt` is opt-in (`experimental_suno=True` / `--experimental-suno`), marked `suno_experimental`, and must not be advertised as a supported prompt.

## Testing

- `pytest tests/` must pass on CPU with no model present. Tests use saved caption text fixtures.
- Test cases to keep: empty caption, repeated terms (dedupe), multiword terms ("acoustic guitar" must not also yield only "guitar" wrongly), terms that are substrings of others (`bass` vs `bass guitar`), case insensitivity.
- GPU smoke test is a separate manual notebook cell, never part of `pytest`.

## Batch runs & resumability (Colab is ephemeral)

- GitHub is the durable store; local disk is not. `scripts/batch.py` must
  checkpoint after **every album**: atomic write of
  `outputs/<ARTIST>/<album>.json`, update `outputs/index.json`, then
  `git add` / `commit` / `push` immediately.
- A fresh runtime resumes by `git pull` + reading the manifest; never redo an
  album whose JSON is present. `--force` is the only way to redo.
- Audio is skip-if-present under `work/` (gitignored) and re-fetchable from
  GCS, so only the in-flight album is ever lost.
- Writes must be atomic (temp file + `os.replace`) so a killed session cannot
  leave a corrupt result.
- Isolate per-track failures: log the track, keep the rest of the album, mark
  the manifest `partial`, and re-run only the failures via `--retry-failed`.
  Never let one bad track discard an album's work.
- Skip (don't process) tracks longer than `--max-track-seconds` (default 480 =
  8 min): record them under `skipped_tracks`. Skips are neither successes nor
  failures — they don't affect album status. `0` disables the cap.
- Skip low-quality tracks below `--min-bitrate-kbps` (default 128; `0`
  disables) the same way — never caption 32/64 kbps files. For a local root,
  `--artist-name` overrides the artist label (default: the root dir name).
- Unattended/long runs must go through `scripts/supervise.py`, detached
  (`setsid nohup ... &`). The batch's own handlers cannot catch the whole
  process being killed (OOM/native crash/session reap); the supervisor restarts
  the batch after any non-zero exit so the run resumes instead of stopping.

## Guardrails

- **Never commit** model weights, audio files, HF tokens, or API keys. `models/` and `audio/` are gitignored. Use Colab Secrets for `HF_TOKEN` if needed.
- Only download from the Hugging Face repos and git remotes named above. Ask before adding any other source.
- Respect licenses: ACE-Step Captioner and its GGUF conversion are MIT; keep attribution notices. Qwen2.5-Omni is the underlying architecture.
- Don't process audio the user hasn't provided; don't upload user audio to any external service.
- Don't change the model, quantization, prompt, or runtime without explicit user approval.
- Don't rewrite unrelated files or reformat the whole repo in a change.

## Definition of done

- [ ] `pytest tests/` passes
- [ ] Notebook runs top to bottom on a fresh Colab T4 runtime
- [ ] `docs/RESULTS.md` has real measured VRAM and time for at least one clip
- [ ] README shows the quickstart and example output
- [ ] No secrets, weights or audio in the git diff

## Communication

Be concise. State what you ran, what happened, and what is still unverified. If a command fails, show the exact error and your next step instead of guessing.
