#!/usr/bin/env bash
# Idempotent CUDA build of stock llama.cpp for text GGUF models on a Colab T4.
#
# This is separate from scripts/build_llamacpp.sh, which builds the patched fork
# needed for the audio captioner. Text rewriting needs no audio patch, so stock
# upstream is enough.
#
# Optional GCS cache (survives Colab resets):
#   export LLAMACPP_TEXT_CACHE_GS=gs://YOUR_BUCKET/cache/llamacpp-text
# The tarball is keyed by llama.cpp ref + CUDA version + sm_75, so a changed
# Colab image does not silently reuse an incompatible binary. Re-running is a
# no-op once the binaries exist.
#
#   LLAMACPP_TEXT_FORCE_BUILD=1   ignore the cache and rebuild
#   LLAMACPP_TEXT_DIR=...         build directory (default: ./llama.cpp-text)
#   LLAMACPP_TEXT_REF=...         upstream branch/tag (default: master)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMACPP_TEXT_DIR="${LLAMACPP_TEXT_DIR:-$REPO_ROOT/llama.cpp-text}"
LLAMACPP_TEXT_REF="${LLAMACPP_TEXT_REF:-master}"
REMOTE="${LLAMACPP_TEXT_REMOTE:-https://github.com/ggml-org/llama.cpp}"
CACHE_GS="${LLAMACPP_TEXT_CACHE_GS:-}"
FORCE_BUILD="${LLAMACPP_TEXT_FORCE_BUILD:-0}"
ARCH=75
CLI="$LLAMACPP_TEXT_DIR/build/bin/llama-cli"
SERVER="$LLAMACPP_TEXT_DIR/build/bin/llama-server"

# CUDA version decides binary compatibility. nvcc is the build toolkit; fall
# back to torch's CUDA version when nvcc is absent.
CUDA_VER=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
fi
if [ -z "$CUDA_VER" ] && command -v python >/dev/null 2>&1; then
  CUDA_VER="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
fi
CUDA_VER="${CUDA_VER:-unknown}"
KEY="llamacpp-text-${LLAMACPP_TEXT_REF//\//-}-cu${CUDA_VER}-sm${ARCH}"
TARBALL="$KEY.tar.gz"

if [ "${1:-}" = "--print-cache-key" ]; then
  echo "$KEY"
  exit 0
fi

if [ "$FORCE_BUILD" != "1" ] && [ -x "$CLI" ] && [ -x "$SERVER" ]; then
  echo "[build] already built: $CLI"
  exit 0
fi

# -- try the GCS cache -------------------------------------------------------
if [ "$FORCE_BUILD" != "1" ] && [ -n "$CACHE_GS" ]; then
  echo "[cache] looking for $CACHE_GS/$TARBALL"
  if gcloud storage ls "$CACHE_GS/$TARBALL" >/dev/null 2>&1; then
    echo "[cache] downloading $TARBALL"
    if gcloud storage cp "$CACHE_GS/$TARBALL" "/tmp/$TARBALL" >/dev/null 2>&1; then
      rm -rf "$LLAMACPP_TEXT_DIR/build/bin"
      mkdir -p "$LLAMACPP_TEXT_DIR"
      tar -xzf "/tmp/$TARBALL" -C "$LLAMACPP_TEXT_DIR"
      chmod +x "$LLAMACPP_TEXT_DIR/build/bin/"* 2>/dev/null || true
      if [ -x "$CLI" ] && [ -x "$SERVER" ] && "$CLI" --version >/dev/null 2>&1; then
        echo "[cache] restored build from GCS"
        exit 0
      fi
      echo "[cache] restored binaries do not run here; rebuilding"
      rm -rf "$LLAMACPP_TEXT_DIR/build"
    else
      echo "[cache] download failed; building"
    fi
  else
    echo "[cache] miss; building"
  fi
fi

# -- build -------------------------------------------------------------------
if [ ! -d "$LLAMACPP_TEXT_DIR/.git" ]; then
  echo "[build] cloning $REMOTE ($LLAMACPP_TEXT_REF) -> $LLAMACPP_TEXT_DIR"
  git clone --depth 1 --branch "$LLAMACPP_TEXT_REF" "$REMOTE" "$LLAMACPP_TEXT_DIR"
else
  echo "[build] reusing existing clone at $LLAMACPP_TEXT_DIR"
fi

echo "[build] configuring (CUDA, sm_75)"
cmake -S "$LLAMACPP_TEXT_DIR" -B "$LLAMACPP_TEXT_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF

echo "[build] compiling llama-cli and llama-server (this takes a while)"
cmake --build "$LLAMACPP_TEXT_DIR/build" -j"$(nproc)" \
  --target llama-cli llama-server

echo "[build] done: $CLI"
echo "[build] done: $SERVER"

# -- upload to the cache -----------------------------------------------------
if [ -n "$CACHE_GS" ]; then
  if tar -czf "/tmp/$TARBALL" -C "$LLAMACPP_TEXT_DIR" build/bin; then
    echo "[cache] uploading $CACHE_GS/$TARBALL"
    if ! gcloud storage cp "/tmp/$TARBALL" "$CACHE_GS/$TARBALL"; then
      echo "[cache] WARNING: upload failed (build still usable locally)"
    fi
  else
    echo "[cache] WARNING: could not create $TARBALL"
  fi
fi
