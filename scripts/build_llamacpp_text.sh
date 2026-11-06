#!/usr/bin/env bash
# Idempotent CUDA build of stock llama.cpp for text GGUF models on a Colab T4.
#
# This is separate from scripts/build_llamacpp.sh, which builds the patched fork
# needed for the audio captioner. Text rewriting needs no audio patch, so stock
# upstream is enough. Re-running is a no-op once the binaries exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMACPP_TEXT_DIR="${LLAMACPP_TEXT_DIR:-$REPO_ROOT/llama.cpp-text}"
LLAMACPP_TEXT_REF="${LLAMACPP_TEXT_REF:-master}"
REMOTE="${LLAMACPP_TEXT_REMOTE:-https://github.com/ggml-org/llama.cpp}"
CLI="$LLAMACPP_TEXT_DIR/build/bin/llama-cli"
SERVER="$LLAMACPP_TEXT_DIR/build/bin/llama-server"

if [ -x "$CLI" ] && [ -x "$SERVER" ]; then
  echo "[build] already built: $CLI"
  exit 0
fi

if [ ! -d "$LLAMACPP_TEXT_DIR/.git" ]; then
  echo "[build] cloning $REMOTE ($LLAMACPP_TEXT_REF) -> $LLAMACPP_TEXT_DIR"
  git clone --depth 1 --branch "$LLAMACPP_TEXT_REF" "$REMOTE" "$LLAMACPP_TEXT_DIR"
else
  echo "[build] reusing existing clone at $LLAMACPP_TEXT_DIR"
fi

echo "[build] configuring (CUDA, sm_75)"
cmake -S "$LLAMACPP_TEXT_DIR" -B "$LLAMACPP_TEXT_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF

echo "[build] compiling llama-cli and llama-server (this takes a while)"
cmake --build "$LLAMACPP_TEXT_DIR/build" -j"$(nproc)" \
  --target llama-cli llama-server

echo "[build] done: $CLI"
echo "[build] done: $SERVER"
