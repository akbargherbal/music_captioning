#!/usr/bin/env bash
# CUDA loader path for the prebuilt llama.cpp binaries.
#
# A binary built on a CPU VM against one CUDA toolkit links libcudart/libcublas
# with that toolkit's SONAME. A Colab image with a different CUDA major does not
# expose those SONAMEs on the default loader path, so the process dies at load:
#
#   error while loading shared libraries: libcudart.so.12: cannot open shared object file
#
# The matching .so files ship inside the pip nvidia-*-cuXX wheels; this prepends
# their directories to LD_LIBRARY_PATH. It is a no-op when they already resolve,
# and idempotent across nested invocations.
#
# Usage:
#   .    ./cuda_loader_path.sh                 # set LD_LIBRARY_PATH in this shell
#   bash ./cuda_loader_path.sh                 # print the export it would set
#   bash ./cuda_loader_path.sh --check <bin>   # exit 1 if deps still missing

_cuda_loader_roots() {
  local d
  for d in /usr/local/lib/python3.*/dist-packages/nvidia \
           /usr/local/lib/python3.*/site-packages/nvidia \
           /usr/lib/python3/dist-packages/nvidia; do
    [ -d "$d" ] && printf '%s\n' "$d"
  done
  python3 - <<'PY' 2>/dev/null
import glob, site, sysconfig
roots = set()
try:
    roots.update(site.getsitepackages())
except Exception:
    pass
for k in ("purelib", "platlib"):
    p = sysconfig.get_paths().get(k)
    if p:
        roots.add(p)
try:
    roots.add(site.getusersitepackages())
except Exception:
    pass
for r in sorted(roots):
    nr = r + "/nvidia"
    if glob.glob(nr + "/*/lib") or glob.glob(nr + "/*/lib64"):
        print(nr)
PY
}

cuda_loader_path() {
  case "${LD_LIBRARY_PATH:-}" in
    *nvidia/*/lib*|*nvidia/*/lib64*) return 0 ;;
  esac
  local nroot d dirs="" first
  for nroot in $(_cuda_loader_roots | sort -u); do
    for d in "$nroot"/*/lib "$nroot"/*/lib64; do
      [ -d "$d" ] && dirs="$dirs:$d"
    done
  done
  # A locally installed toolkit (CPU build VM, or a Colab image that ships one).
  for d in /usr/local/cuda*/lib64 /usr/local/cuda*/lib; do
    [ -d "$d" ] && dirs="$dirs:$d"
  done
  # /usr/lib64-nvidia carries libcuda.so.1 on Colab when it is not on the default path.
  dirs="${dirs}:/usr/lib64-nvidia"
  dirs="${dirs#:}"
  first="${dirs%%:*}"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$first:"*) return 0 ;;
  esac
  export LD_LIBRARY_PATH="$dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

cuda_loader_check() {
  local bin="${1:?usage: cuda_loader_check <binary>}" miss
  miss="$(ldd "$bin" 2>/dev/null | grep 'not found' || true)"
  if [ -n "$miss" ]; then
    printf 'cuda_loader: %s still has missing dependencies:\n%s\n' "$bin" "$miss" >&2
    return 1
  fi
  return 0
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  if [ "${1:-}" = "--check" ]; then
    cuda_loader_path
    cuda_loader_check "${2:?usage: $0 --check <binary>}"
    exit $?
  fi
  cuda_loader_path
  printf 'export LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
  exit 0
fi

cuda_loader_path
