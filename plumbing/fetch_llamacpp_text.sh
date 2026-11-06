#!/usr/bin/env bash
#
# Stage stock llama.cpp (llama-cli + llama-server) for the machine you are on.
#
# Priority: fetch the prebuilt for the local GPU's arch from this repo's GCS
# prefix (fast, no compile), fall back to building from source. A build needs
# NO GPU: nvcc cross-compiles device code for an explicit --cuda-arch. It needs
# the CUDA toolkit, cmake and g++ (ccache is used when available).
#
# Layout in GCS (same idea as arabmaqamrock/plumbing/fetch_binary.sh):
#   <GCP_PREFIX>/binaries/<arch-tag>/llamacpp-text.tar.gz
#                                    /llamacpp-text.tar.gz.sha256
#                                    /manifest.json
#
# Usage:
#   ./plumbing/fetch_llamacpp_text.sh                 # fetch for the local GPU
#   ./plumbing/fetch_llamacpp_text.sh --cuda-arch 75  # explicit arch (CPU build)
#   ./plumbing/fetch_llamacpp_text.sh --build         # build instead of fetching
#   ./plumbing/fetch_llamacpp_text.sh --build --publish --cuda-arch 75
#   ./plumbing/fetch_llamacpp_text.sh --force         # re-stage even if present
#   ./plumbing/fetch_llamacpp_text.sh --dry-run       # print the plan only
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
. "$PLUMBING_DIR/arch.sh"

LLAMACPP_DIR="${LLAMACPP_TEXT_DIR:-$REPO_ROOT/llama.cpp-text}"
BIN_DIR="$LLAMACPP_DIR/build/bin"
CLI="$BIN_DIR/llama-cli"
SERVER="$BIN_DIR/llama-server"
ARCH_FILE="$LLAMACPP_DIR/.arch"
PIN_FILE="$PLUMBING_DIR/llamacpp.pin"
LLAMACPP_REPO="${LLAMACPP_TEXT_REMOTE:-https://github.com/ggml-org/llama.cpp}"
TAR_NAME="llamacpp-text.tar.gz"

FORCE=0; DO_BUILD=0; DO_PUBLISH=0; DRY=0; CC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda-arch) CC="${2:?--cuda-arch needs a value}"; shift 2 ;;
    --build)     DO_BUILD=1; shift ;;
    --publish)   DO_PUBLISH=1; shift ;;
    --force)     FORCE=1; shift ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '3,28p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[fetch_llamacpp] %s\n' "$*"; }
RUN() { if [ "$DRY" = 1 ]; then echo "  dry-run: $*"; else "$@"; fi; }

# --- resolve the target arch -------------------------------------------------
[ -n "$CC" ] || CC="$(detect_compute_cap)"
TAG="$(arch_tag_for_cc "$CC")"

if [ -x "$CLI" ] && [ -x "$SERVER" ] && [ "$FORCE" != 1 ] && [ "$DO_BUILD" != 1 ]; then
  log "already staged (arch: $(cat "$ARCH_FILE" 2>/dev/null || echo '?')); --force to re-stage"
  exit 0
fi

# --- 1. prebuilt from GCS (default) ------------------------------------------
fetch_prebuilt() {
  if [ -z "$TAG" ]; then
    log "no GPU detected and no --cuda-arch: cannot choose a prebuilt"
    return 1
  fi
  local remote="$GCP_PREFIX/binaries/$TAG"
  log "fetching prebuilt $TAG  <-  $remote/$TAR_NAME"
  if [ "$DRY" = 1 ]; then
    echo "  dry-run: would fetch + sha256-verify + extract to $LLAMACPP_DIR"
    return 0
  fi

  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/llamacpp-fetch.XXXXXX")"
  if ! gcs_cp "$remote/$TAR_NAME" "$tmp/$TAR_NAME" >/dev/null 2>&1; then
    log "no prebuilt there (or no GCS access) — will build"
    rm -rf "$tmp"; return 1
  fi
  if gcs_cat "$remote/$TAR_NAME.sha256" > "$tmp/$TAR_NAME.sha256" 2>/dev/null; then
    local want have
    want="$(awk '{print $1}' "$tmp/$TAR_NAME.sha256")"
    have="$(sha256sum "$tmp/$TAR_NAME" | awk '{print $1}')"
    if [ "$want" != "$have" ]; then
      log "sha256 MISMATCH (want $want, got $have) — discarding, will build"
      rm -rf "$tmp"; return 1
    fi
    log "sha256 ok: $have"
  else
    log "WARNING: no .sha256 published for $TAG — staging unverified"
  fi

  mkdir -p "$LLAMACPP_DIR"
  rm -rf "$BIN_DIR"
  tar -xzf "$tmp/$TAR_NAME" -C "$LLAMACPP_DIR"
  chmod +x "$BIN_DIR/"* 2>/dev/null || true
  rm -rf "$tmp"

  # Loader path + smoke test: a cached binary from another CUDA image must not
  # be accepted silently.
  . "$REPO_ROOT/cuda_loader_path.sh"
  if [ ! -x "$CLI" ] || ! "$CLI" --version >/dev/null 2>&1; then
    log "restored binaries do not run here (CUDA mismatch?) — will build"
    return 1
  fi
  printf '%s\n' "$TAG" > "$ARCH_FILE"
  log "staged $LLAMACPP_DIR ($TAG)"
  return 0
}

# --- 2. build from source (fallback / --build) -------------------------------
build_from_source() {
  if [ -z "$CC" ]; then
    log "--build needs --cuda-arch <CC> (no GPU here to detect it)"
    exit 2
  fi
  for tool in git cmake nvcc g++; do
    command -v "$tool" >/dev/null 2>&1 || {
      log "MISSING '$tool' — a build needs the CUDA toolkit (nvcc), cmake and g++"
      exit 1
    }
  done

  local pin
  pin="$(grep -v '^[[:space:]]*#' "$PIN_FILE" 2>/dev/null | tr -d '[:space:]' | head -n1 || true)"
  if [ ! -d "$LLAMACPP_DIR/.git" ]; then
    log "cloning llama.cpp"
    RUN git clone --depth 1 "$LLAMACPP_REPO" "$LLAMACPP_DIR"
  else
    log "reusing llama.cpp checkout at $LLAMACPP_DIR"
  fi
  RUN git -C "$LLAMACPP_DIR" fetch --depth 1 origin
  if [ -n "$pin" ]; then
    log "checking out pinned llama.cpp commit $pin"
    RUN git -C "$LLAMACPP_DIR" fetch --depth 1 origin "$pin" || true
    RUN git -C "$LLAMACPP_DIR" checkout --quiet "$pin"
  else
    log "no pin (plumbing/llamacpp.pin empty) — tracking llama.cpp main"
    RUN git -C "$LLAMACPP_DIR" reset --hard FETCH_HEAD
  fi

  local cmake_args=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CC"
    -DGGML_NATIVE=OFF
    -DLLAMA_CURL=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_TESTS=OFF
  )
  if command -v ccache >/dev/null 2>&1; then
    cmake_args+=(
      -DCMAKE_C_COMPILER_LAUNCHER=ccache
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
    )
    log "ccache enabled"
  else
    log "ccache not found — build will be slower"
  fi

  log "building llama-cli + llama-server (--cuda-arch $CC)"
  RUN cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" "${cmake_args[@]}"
  RUN cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" \
        --target llama-cli llama-server

  RUN mkdir -p "$BIN_DIR"
  RUN sh -c "printf '%s\n' '$TAG' > '$ARCH_FILE'"
  log "built $CLI ($TAG, llama.cpp $(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
}

publish_binary() {
  [ -x "$CLI" ] || { log "--publish: no bin/llama-cli to publish"; exit 1; }
  [ -n "$TAG" ] || { log "--publish: unknown arch tag"; exit 2; }
  local remote="$GCP_PREFIX/binaries/$TAG"
  local commit cuda h size tmp
  commit="$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  cuda="$( { nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1; } || echo unknown)"
  log "publishing $TAG -> $remote/ (llama.cpp ${commit:0:12}, cuda ${cuda:-unknown})"
  if [ "$DRY" = 1 ]; then
    echo "  dry-run: would upload $TAR_NAME + .sha256 + manifest.json"
    return 0
  fi

  tmp="$(mktemp -d "${TMPDIR:-/tmp}/llamacpp-publish.XXXXXX")"
  tar -czf "$tmp/$TAR_NAME" -C "$LLAMACPP_DIR" build/bin
  h="$(sha256sum "$tmp/$TAR_NAME" | awk '{print $1}')"
  size="$(stat -c %s "$tmp/$TAR_NAME")"
  printf '%s  %s\n' "$h" "$TAR_NAME" > "$tmp/$TAR_NAME.sha256"

  python3 - "$TAG" "$CC" "$h" "$size" "$commit" "${cuda:-unknown}" "$remote" \
      "$LLAMACPP_DIR" > "$tmp/manifest.json" <<'PY'
import json, sys, datetime
tag, cc, h, size, commit, cuda, src, build_dir = sys.argv[1:9]
print(json.dumps({
    "arch_tag": tag,
    "cuda_arch": cc,
    "llamacpp_commit": commit,
    "cuda_toolkit": cuda,
    "build_flags": f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cc} "
                   "-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF",
    "size_bytes": int(size),
    "sha256": h,
    "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": src,
    "build_dir": build_dir,
}, indent=2) + "\n")
PY

  gcs_cp "$tmp/$TAR_NAME" "$remote/$TAR_NAME" >/dev/null
  gcs_cp "$tmp/$TAR_NAME.sha256" "$remote/$TAR_NAME.sha256" >/dev/null
  gcs_cp "$tmp/manifest.json" "$remote/manifest.json" >/dev/null
  rm -rf "$tmp"
  log "published $TAG (sha256 $h)"
}

# --- dispatch ----------------------------------------------------------------
if [ "$DRY" = 1 ]; then
  log "arch=$CC tag=${TAG:-?} build=$DO_BUILD publish=$DO_PUBLISH"
  [ "$DO_BUILD" = 1 ] && { build_from_source; [ "$DO_PUBLISH" = 1 ] && publish_binary; exit 0; }
  fetch_prebuilt || { log "(dry-run) would fall back to building from source"; }
  exit 0
fi

if [ "$DO_BUILD" = 1 ]; then
  build_from_source
  [ "$DO_PUBLISH" = 1 ] && publish_binary
elif ! fetch_prebuilt; then
  log "falling back to a source build"
  build_from_source
  [ "$DO_PUBLISH" = 1 ] && publish_binary
elif [ "$DO_PUBLISH" = 1 ]; then
  publish_binary
fi

. "$REPO_ROOT/cuda_loader_path.sh"
"$CLI" --version 2>/dev/null || true
