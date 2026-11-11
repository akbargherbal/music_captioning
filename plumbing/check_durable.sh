#!/usr/bin/env bash
# Refuse to start work from a tree that is not on the remote.
#
#   ./plumbing/check_durable.sh
#   ./plumbing/check_durable.sh docs/YT500_CAPTION_COMPLETION_PLAN.md
#   REPO=../arabmaqamrock ./plumbing/check_durable.sh plumbing/compile_yt500_collection.py
#   ./plumbing/check_durable.sh --allow-untracked      # ignore scratch files
#
# Why: local disk and the git working tree are ephemeral. A fresh Colab VM
# clones origin, so a file that was never pushed does not exist for the next
# session (see AGENTS.md, "Durable work: push before you run").
#
# Exits 1 when the branch has no upstream, is behind or ahead of it, a tracked
# file is modified, an untracked non-ignored file exists, or a file named on the
# command line is not tracked. Exit 2 means the check itself could not run.
set -uo pipefail

own=$(cd "$(dirname "$0")/.." && pwd)
REPO=${REPO:-$own}
allow_untracked=0
files=()
for arg in "$@"; do
  case "$arg" in
    --allow-untracked) allow_untracked=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) files+=("$arg") ;;
  esac
done

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "check_durable: not a git repo: $REPO" >&2
  exit 2
}
git -C "$REPO" fetch -q origin || {
  echo "check_durable: cannot fetch origin (offline? no credentials?)" >&2
  exit 2
}

branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
problems=0
fail() { echo "FAIL: $*"; problems=$((problems + 1)); }

upstream=$(git -C "$REPO" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
if [ -z "$upstream" ]; then
  fail "$branch has no upstream (push it with -u)"
else
  ahead=$(git -C "$REPO" rev-list --count "$upstream..HEAD")
  behind=$(git -C "$REPO" rev-list --count "HEAD..$upstream")
  [ "$ahead" -eq 0 ] || fail "$ahead unpushed commit(s) on $branch vs $upstream"
  [ "$behind" -eq 0 ] || fail "$behind commit(s) behind $upstream (pull first)"
fi

dirty=$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)
if [ "$dirty" -ne 0 ]; then
  fail "$dirty modified tracked file(s):"
  git -C "$REPO" status --short --untracked-files=no | sed 's/^/      /'
fi

if [ "$allow_untracked" -eq 0 ]; then
  untracked=$(git -C "$REPO" status --porcelain -uall | grep -c '^??' || true)
  if [ "${untracked:-0}" -ne 0 ]; then
    fail "$untracked untracked non-ignored file(s) — commit them or use --allow-untracked:"
    git -C "$REPO" status --porcelain -uall | grep '^??' | head -20 | sed 's/^/      /'
  fi
fi

for f in "${files[@]}"; do
  if ! git -C "$REPO" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    fail "$f is not tracked in $REPO (unpublished, or it belongs to another repo)"
  fi
done

if [ "$problems" -ne 0 ]; then
  echo "check_durable: $problems problem(s) in $REPO ($branch)"
  echo "Next: git -C $REPO add <the files you mean to keep> && commit && push."
  echo "      Use 'add -A' only if every file listed above belongs in git;"
  echo "      build output and scratch files should be gitignored or left out."
  exit 1
fi
echo "durable: $REPO is clean and $branch == $upstream"
