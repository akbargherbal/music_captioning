#!/usr/bin/env bash
# Shared environment for the plumbing scripts (source me).
#
#   . plumbing/env.sh
#
# Sets REPO_ROOT / PLUMBING_DIR, GCP_ROOT (bucket root) and GCP_PREFIX (this
# repo's prefix under it), plus gcs_cp / gcs_cat / gcs_ls wrappers. Prefers
# `gcloud storage` and falls back to `gsutil`.

PLUMBING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PLUMBING_DIR/.." && pwd)"
export PLUMBING_DIR REPO_ROOT

: "${GCP_ROOT:=gs://akbar-december-2024-backup}"
export GCP_ROOT
export GCP_PREFIX="${GCP_ROOT%/}/music_captioning"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp()    { gcloud storage cp "$@"; }
  gcs_cat()   { gcloud storage cat "$@"; }
  gcs_ls()    { gcloud storage ls "$@"; }
  export GCS_CLI=gcloud
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp()    { gsutil cp "$@"; }
  gcs_cat()   { gsutil cat "$@"; }
  gcs_ls()    { gsutil ls "$@"; }
  export GCS_CLI=gsutil
else
  echo "plumbing/env.sh: need 'gcloud' or 'gsutil' on PATH" >&2
  export GCS_CLI=
fi
