# GGUF rewrite experiment

Status: prepared, not run. The current pipeline uses fp16 transformers and is
already acceptable (24/24 rewritten, ~15 s/track on a T4). This document records
the next experiment: stock llama.cpp + a GGUF rewrite model, to see whether a
quantized runtime is faster without losing detail.

## Why GGUF

GGUF is not a drop-in for transformers: it needs llama.cpp. The payoff is a
different inference engine (CUDA kernels tuned for quantized weights) and smaller
memory. Q4_K_M also loses more quality than fp16/Q8_0; the only way to know if
that matters here is to run the same stored captions through it.

## Build once on a CPU-only VM

The build does **not** need a GPU: `nvcc` cross-compiles for an explicit
`--cuda-arch`. Build on any CPU VM (or a CPU Colab runtime with the CUDA
toolkit), publish to GCS, then Colab only downloads. This mirrors
`arabmaqamrock/plumbing/fetch_binary.sh`.

```bash
# CPU-only VM: no GPU required
./plumbing/fetch_llamacpp_text.sh --build --publish --cuda-arch 75
```

Builds `llama-cli` + `llama-server` into `llama.cpp-text/` and uploads:

```
gs://akbar-december-2024-backup/music_captioning/binaries/sm75-t4/
  llamacpp-text.tar.gz
  llamacpp-text.tar.gz.sha256
  manifest.json          # llama.cpp commit, CUDA, flags, size, sha256, built_at
```

Pin the source commit in `plumbing/llamacpp.pin` before publishing if you want
the build to be reproducible later (empty = track upstream main).

## Fetch on Colab (no build)

```bash
./plumbing/fetch_llamacpp_text.sh
```

Detects the GPU (`sm75-t4` on a T4), downloads the tarball, verifies SHA256,
extracts it, loads `cuda_loader_path.sh`, and smoke-tests `llama-cli --version`.
If the fetch, checksum, or smoke test fails, it falls back to building from
source.

- `--force` re-stages even if binaries exist.
- `--dry-run` prints the plan.
- `--cuda-arch 75` is required only when building without a GPU.

### CUDA runtime mismatch

A binary built against a different CUDA major than the Colab image needs the
matching runtime libs. `cuda_loader_path.sh` prepends the `nvidia-*` wheel
directories to `LD_LIBRARY_PATH`; the `manifest.json` records `cuda_toolkit` and
`llamacpp_commit` so a mismatch can be diagnosed. The fetch script sources the
loader before its smoke test.

### GGUF weights cache

Same idea: copy from GCS, else download from Hugging Face and publish:

```bash
GGUF=Qwen3-4B-Instruct-2507-Q8_0.gguf
PREFIX="gs://akbar-december-2024-backup/music_captioning"

gcloud storage cp "$PREFIX/gguf/$GGUF" models/ \
  || { hf download unsloth/Qwen3-4B-Instruct-2507-GGUF "$GGUF" --local-dir models \
       && gcloud storage cp "models/$GGUF" "$PREFIX/gguf/$GGUF"; }
```

`GCP_PREFIX` comes from `plumbing/env.sh`; override `GCP_ROOT` for another
bucket.

## Candidate GGUFs (checked 2026-11-06)

| Repo | Quant | Approx. size | Note |
|---|---|---:|---|
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `Q8_0` | ~4.3 GB | closest to fp16; 700k downloads |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `Q6_K` | ~3.3 GB | middle ground |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `Q4_K_M` | ~2.5 GB | smallest; expect some quality loss |
| `ibm-granite/granite-4.1-3b-GGUF` | `Q8_0` | ~3.2 GB | official IBM conversion, 2026 model |
| `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` | `Q8_0` | ~4.3 GB | alternative conversion |

Start with Q8_0: it should match the fp16 output closely, so any difference is
the runtime rather than quantization. Only then compare Q6_K/Q4_K_M.

## Plan

1. Build the binaries (above).
2. Download one Q8_0 GGUF with `hf download <repo> <file>`.
3. Add `LlamaCppRewriter` (`src/rewriter_llama.py`): start `llama-server` once
   with `--jinja` (uses the GGUF's chat template), `-ngl 99`, small context, then
   send one stateless chat completion per track. It implements the same
   `rewrite(messages)` protocol as `HFRewriter`.
4. Benchmark on the stored `chunk_captions` of
   `outputs/fadhl_shaker_selected/*.json` (no re-captioning) and compare
   instrument/vocal/production coverage, fallback count, and seconds per track
   against the fp16 run.
5. Keep it only if it matches fp16 quality and is clearly faster; otherwise
   delete the experiment.

## Guardrails

- The GGUF repos above are third-party conversions except IBM's. They are
  trusted by download count, not verified; compare outputs against the fp16
  baseline before adopting one.
- Keep using the same system prompt and the same verification checks; only the
  inference engine changes.
