# GGUF rewrite experiment

Status: **binaries published and the backend implemented** (2026-11-06); the
A/B benchmark has not run yet. The current pipeline uses fp16 transformers and is
already acceptable (24/24 rewritten, ~15 s/track on a T4). This document records
the experiment: stock llama.cpp + a GGUF rewrite model, to see whether a
quantized runtime is faster without losing detail.

## Published build (sm75-t4, 2026-11-06)

Built on a CPU-only Colab VM with CUDA toolkit 13.0 (no GPU present; `nvcc`
cross-compiled for `sm_75`), from llama.cpp commit
`954de248ac6907dd05dc19f0f511ab18f2807a54` (pinned in `plumbing/llamacpp.pin`).

| | |
|---|---|
| Tarball | 104.97 MiB, contains only `build/bin/llama-cli` + `llama-server` |
| sha256 | `e75ffce18236def9b654c82f2031f38e24f3e8e0cd84e4d369905108e7378d0b` |
| GCS | `binaries/sm75-t4/{llamacpp-text.tar.gz,.sha256,manifest.json,build.log.gz}` |
| Verified | checksum re-downloaded and matched; archive has exactly 2 binaries |

**Portability:** the binary is linked against the build host's glibc (this build:
glibc 2.38 / `GLIBCXX_3.4.32`, i.e. Ubuntu 24.04) and against `libcudart.so.13` /
`libcublas.so.13`. A Colab GPU image with an older glibc or a CUDA 12 runtime
fails the `llama-cli --version` smoke test, which makes
`fetch_llamacpp_text.sh` fall back to building locally instead of using a broken
cache entry. Rebuild and republish after a Colab base-image change.

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

## Running the A/B

`src/rewriter_llama.py` implements the same `rewrite(messages)` protocol as
`HFRewriter`: it starts one `llama-server` (`--jinja`, `-ngl 99`, greedy,
`-c 4096`) and sends one stateless chat completion per track. Nothing about the
prompt or the verification changes.

```bash
# one album, both backends, no re-captioning
python scripts/rewrite_ab.py outputs/fadhl_shaker_selected/*.json \
  --backends hf,llamacpp \
  --gguf models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
  --out /tmp/rewrite_ab.json
```

It reports, per backend: ok/fail counts, seconds per track, average words, and
**term coverage** (share of source instrument/vocal terms that survive). Coverage
is the proxy for lost detail; speed alone is not enough to switch.

Quick single-file check:

```bash
python scripts/rewrite_demo.py track.json --backend llamacpp \
  --gguf models/Qwen3-4B-Instruct-2507-Q8_0.gguf
```

To run the real batch with the GGUF backend:

```bash
python scripts/batch.py --root ... --repolish \
  --rewrite-backend llamacpp \
  --rewrite-gguf models/Qwen3-4B-Instruct-2507-Q8_0.gguf
```

## Decision rule

Keep GGUF only if it (a) keeps the same ok/fallback profile, (b) loses clearly
less detail than fp16 on coverage, and (c) is measurably faster per track.
Otherwise the fp16 `HFRewriter` stays the default and this stays an experiment.

## Guardrails

- The GGUF repos above are third-party conversions except IBM's. They are
  trusted by download count, not verified; compare outputs against the fp16
  baseline before adopting one.
- Keep using the same system prompt and the same verification checks; only the
  inference engine changes.
