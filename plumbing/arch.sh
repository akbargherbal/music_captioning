#!/usr/bin/env bash
# Compute capability <-> arch tag. Sourced (no side effects on source).
#
#   cc="$(detect_compute_cap)"          # 75 / 80 / 89  (empty with no GPU)
#   tag="$(arch_tag_for_cc "$cc")"      # sm75-t4 / sm80-a100 / sm89-l4
#
# The tag names the GCS folder binaries/<tag>/ and is baked into the build's
# manifest. Keep it in sync with the GCS layout in docs/GGUF_REWRITE.md.

detect_compute_cap() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -n1 | tr -d '. ' | tr -dc '0-9'
}

arch_tag_for_cc() {
  case "$1" in
    "")  echo "" ;;
    75)  echo sm75-t4 ;;
    80)  echo sm80-a100 ;;
    89)  echo sm89-l4 ;;
    *)   echo "sm$1" ;;
  esac
}

cc_for_arch_tag() {
  printf '%s\n' "${1#sm}" | cut -d- -f1
}
