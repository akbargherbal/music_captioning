#!/usr/bin/env python3
"""Resumable batch captioning across ephemeral Colab runtimes.

GitHub is the durable store. Each album runs in two strictly separated phases:

  1. **Caption** every track (audio captioner only), keeping the raw per-chunk
     text in ``chunk_captions`` and the deterministic merge in ``caption_raw``.
  2. **Polish** every track (small rewrite model only): the captioner is
     unloaded first, so the two models never work at the same time.

Then the album JSON is written atomically, the manifest updated, and the work
committed/pushed immediately. The raw chunk captions stay in the JSON, so a bad
result can be traced to the captioner or to the rewrite model.

On a fresh runtime, run the same command: it pulls, reads the manifest, and
skips every album already polished. Nothing is redone — only the in-flight
album is lost if the runtime dies.

A single bad track does not sink its album: the track is logged under
`failed_tracks`, the rest of the album is written/committed, and the album is
marked `partial` in the manifest. Re-run with `--retry-failed` to re-caption
only the failed tracks. Only if *every* track fails is the album left pending.

Usage:
    python scripts/batch.py --artist ELISSA --list          # show pending
    python scripts/batch.py --artist ELISSA --max-minutes 90
    python scripts/batch.py                                 # all artists
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.resume import (  # noqa: E402
    CAPTION_PIPELINE_VERSION,
    CAPTION_STAGE_CAPTIONED,
    CAPTION_STAGE_POLISHED,
    SkippedTrack,
    album_output_path,
    album_status,
    atomic_write_json,
    group_local_tracks,
    group_tracks,
    is_current_caption_output,
    load_manifest,
    pending_albums,
    run_tracks,
    skip_reason,
)
from src.tags import CATEGORIES  # noqa: E402
from src.rewriter_hf import DEFAULT_MODEL_ID as DEFAULT_REWRITE_MODEL  # noqa: E402

DEFAULT_ROOT = "gs://akbar-december-2024-backup/DISCOGRAPHY"

# Repo-local fallback identity. A fresh Colab runtime has no global
# user.name/user.email, so `git commit` fails and every album checkpoint is lost
# (writing JSON + manifest without ever pushing). Set these if unset.
FALLBACK_GIT_NAME = "akbargherbal"
FALLBACK_GIT_EMAIL = "akbargherbal@users.noreply.github.com"


def prefetch_model(model_id: str) -> threading.Thread:
    """Download the rewrite weights in the background (no GPU, no loading).

    The polish phase runs after captioning; having the files already cached
    removes a multi-minute download from the critical path. If the download is
    still running when polish starts, the caller joins this thread first.
    """

    def run() -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model_id)
            print(f"[prefetch] {model_id} ready", flush=True)
        except Exception as exc:  # noqa: BLE001 - never kill the batch
            print(
                f"[prefetch] {model_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    thread = threading.Thread(target=run, name="rewrite-prefetch", daemon=True)
    thread.start()
    return thread


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ensure_git_identity(repo: Path) -> None:
    """Set a repo-local commit identity when none is configured."""
    if not _run(["git", "-C", str(repo), "config", "user.name"]).stdout.strip():
        _run(["git", "-C", str(repo), "config", "user.name", FALLBACK_GIT_NAME])
    if not _run(["git", "-C", str(repo), "config", "user.email"]).stdout.strip():
        _run(["git", "-C", str(repo), "config", "user.email", FALLBACK_GIT_EMAIL])


def list_gs_mp3(root: str) -> list[str]:
    """List every .mp3 under ``root`` recursively via the gcloud CLI."""
    pattern = root.rstrip("/") + "/**/*.mp3"
    proc = _run(["gcloud", "storage", "ls", "-r", pattern])
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud storage ls failed: {proc.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().endswith(".mp3")]


def download_track(uri: str, dest: Path) -> None:
    """Copy one track to ``dest`` unless it is already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(["gcloud", "storage", "cp", uri, str(dest)])
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud cp failed for {uri}: {proc.stderr.strip()}")


def git_checkpoint(repo: Path, message: str, push: bool, retries: int = 3) -> None:
    """Commit everything and (optionally) push, retrying transient failures."""
    ensure_git_identity(repo)
    _run(["git", "-C", str(repo), "add", "-A"])
    if _run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0:
        print("  (no changes to commit)")
        return
    if _run(["git", "-C", str(repo), "commit", "-q", "-m", message]).returncode != 0:
        raise RuntimeError("git commit failed")
    print(f"  committed: {message}")
    if not push:
        return
    for attempt in range(1, retries + 1):
        proc = _run(["git", "-C", str(repo), "push", "origin", "HEAD"])
        if proc.returncode == 0:
            print("  pushed to origin")
            return
        print(f"  push failed (attempt {attempt}/{retries}): {proc.stderr.strip()[:200]}")
        time.sleep(2 * attempt)
    print("  WARNING: could not push; work is committed locally only")


def caption_album(
    captioner,
    entry: dict,
    workdir: Path,
    existing: dict | None = None,
    retry_only_failed: bool = False,
    max_track_seconds: float = 0.0,
    min_bitrate_kbps: float = 0.0,
    experimental_suno: bool = False,
) -> dict:
    """Phase 1: caption **every** track of the album, keeping the raw text.

    Nothing is polished here. Each track stores the untouched per-chunk
    captions (``chunk_captions``) and the deterministic merge (``caption_raw``)
    so the rewrite phase, and a human, can compare what the captioner said with
    what the editor produced.
    """
    from src.captioner import audio_stream_info
    from src.captioner_hf import caption_and_tag
    from src.tags import TAGS_EXPERIMENTAL, merge_tags, to_suno

    # Never reuse per-track results from an older caption pipeline.
    if existing is not None and not is_current_caption_output(existing):
        existing = None
        retry_only_failed = False

    artist, album = entry["artist"], entry["album"]
    album_dir = workdir / artist / album

    def caption_one(filename: str, uri: str) -> dict:
        if uri.startswith("gs://"):
            local = album_dir / filename
            download_track(uri, local)
        else:
            local = Path(uri)
        duration, bitrate = audio_stream_info(local)
        reason = skip_reason(duration, bitrate,
                             max_track_seconds=max_track_seconds,
                             min_bitrate_kbps=min_bitrate_kbps)
        if reason:
            raise SkippedTrack(reason)
        t0 = time.time()
        result = caption_and_tag(
            captioner, local, experimental_suno=experimental_suno
        )
        caption_s = time.time() - t0
        chunks = list(result.get("chunk_captions") or [])
        run = getattr(captioner, "last_run", {}) or {}
        track_obj = {
            "file": filename,
            "duration_s": round(duration or 0.0, 1),
            "bitrate_kbps": round(bitrate) if bitrate else None,
            "chunk_count": run.get("chunk_count"),
            "silence_trimmed_s": run.get("silence_trimmed_s"),
            "caption_seconds": round(caption_s, 1),
            "instruments": result["instruments"],
            "vocals": result["vocals"],
            "production": result["production"],
            # Kept for audit: what the captioner wrote, before any editing.
            "chunk_captions": chunks,
            "caption_raw": result["caption"],
            "caption": "",
            "rewrite_status": "pending",
            "tags_experimental": TAGS_EXPERIMENTAL,
        }
        if experimental_suno:
            track_obj["suno_prompt"] = result["suno_prompt"]
            track_obj["suno_experimental"] = True
        return track_obj

    track_results, failed, skipped = run_tracks(
        entry["tracks"], caption_one,
        existing=existing, retry_only_failed=retry_only_failed,
        echo=lambda m: print(f"    {m}", flush=True),
    )

    if not track_results and failed:
        first = f"; first error: {failed[0]['error']}" if failed else ""
        raise RuntimeError(f"no tracks captioned ({len(failed)} failed){first}")

    if failed:
        status = "partial"
    elif skipped and not track_results:
        status = "skipped"
    else:
        status = "ok"

    total_audio = sum(t["duration_s"] for t in track_results)
    total_caption = sum(t["caption_seconds"] for t in track_results)
    merged = merge_tags([
        {cat: t[cat] for cat in CATEGORIES} for t in track_results
    ])
    album_obj = {
        "artist": artist,
        "album": album,
        "source_dir": entry["source_dir"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "track_count": len(track_results),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "total_audio_seconds": round(total_audio, 1),
        "total_caption_seconds": round(total_caption, 1),
        "instruments": merged["instruments"],
        "vocals": merged["vocals"],
        "production": merged["production"],
        "tags_experimental": TAGS_EXPERIMENTAL,
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "caption_stage": CAPTION_STAGE_CAPTIONED,
        "tracks": track_results,
        "failed_tracks": failed,
        "skipped_tracks": skipped,
    }
    if experimental_suno:
        album_obj["suno_prompt"] = to_suno(merged)
        album_obj["suno_experimental"] = True
    return album_obj


def polish_album(album_obj: dict, rewriter) -> dict:
    """Phase 2: rewrite every multi-chunk track with the small model.

    Runs **after** every caption in the album exists (the captioner is unloaded
    by the caller). Single-chunk tracks are already one coherent description and
    are left untouched. A rejected rewrite falls back to ``caption_raw`` and
    records the reason; the raw chunk text is never removed.
    """
    from src.rewrite import REWRITE_PROMPT_VERSION, rewrite_caption

    tracks = album_obj.get("tracks") or []
    if rewriter is None:
        for track in tracks:
            track["caption"] = track.get("caption_raw", "")
            track["rewrite_status"] = "disabled"
        album_obj["caption_stage"] = CAPTION_STAGE_CAPTIONED
        album_obj["rewrite"] = {
            "model": None,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "disabled": len(tracks),
        }
        return album_obj

    summary: dict[str, object] = {
        "model": getattr(rewriter, "name", None),
        "prompt_version": REWRITE_PROMPT_VERSION,
        "ok": 0,
        "fallback": 0,
        "error": 0,
        "single_chunk": 0,
    }
    rewrite_seconds = 0.0
    for track in tracks:
        # Reused track dicts may carry results/reasons from an earlier polish;
        # clear them so a successful rewrite does not keep stale rejection text.
        for key in ("rewrite_reasons", "rewrite_model", "rewrite_prompt_version"):
            track.pop(key, None)

        chunks = list(track.get("chunk_captions") or [])
        if len(chunks) <= 1:
            track["caption"] = chunks[0] if chunks else track.get("caption_raw", "")
            track["rewrite_status"] = "single_chunk"
            summary["single_chunk"] = int(summary["single_chunk"]) + 1
            continue

        started = time.time()
        result = rewrite_caption(
            chunks, rewriter, fallback=track.get("caption_raw")
        )
        elapsed = time.time() - started
        rewrite_seconds += elapsed
        track["caption"] = result.text
        track["rewrite_seconds"] = round(elapsed, 1)
        track["rewrite_model"] = result.rewriter
        track["rewrite_prompt_version"] = result.prompt_version
        if result.reasons:
            track["rewrite_reasons"] = list(result.reasons)
        if result.ok:
            track["rewrite_status"] = "ok"
            summary["ok"] = int(summary["ok"]) + 1
        elif any(r.startswith("rewriter error:") for r in result.reasons):
            # The model itself failed (load/OOM); keep the raw caption and
            # leave the album at 'captioned' so the next run retries only this.
            track["rewrite_status"] = "error"
            summary["error"] = int(summary["error"]) + 1
        else:
            track["rewrite_status"] = "fallback"
            summary["fallback"] = int(summary["fallback"]) + 1

    attempted = int(summary["ok"]) + int(summary["fallback"]) + int(summary["error"])
    if attempted and int(summary["ok"]) == 0 and int(summary["error"]) == attempted:
        # Every rewrite errored: do not mark the album done, and do not lose the
        # captioning work already on disk.
        album_obj["caption_stage"] = CAPTION_STAGE_CAPTIONED
        summary["status"] = "failed"
    else:
        album_obj["caption_stage"] = CAPTION_STAGE_POLISHED
        summary["status"] = "ok"
    album_obj["rewrite"] = summary
    album_obj["total_rewrite_seconds"] = round(rewrite_seconds, 1)
    return album_obj


def process_album(
    captioner,
    entry: dict,
    workdir: Path,
    outputs: Path,
    existing: dict | None = None,
    retry_only_failed: bool = False,
    max_track_seconds: float = 0.0,
    min_bitrate_kbps: float = 0.0,
    experimental_suno: bool = False,
    rewriter=None,
    rewrite: bool = True,
    checkpoint: Callable[[dict, str], None] | None = None,
    before_polish: Callable[[], None] | None = None,
) -> dict:
    """Caption **all** tracks, then polish **all** tracks, then write the album.

    The two phases never interleave: the captioner is unloaded before the
    rewrite model is used. ``checkpoint(obj, stage)`` is called after the caption
    phase so the caller can persist/commit the raw captions before polishing;
    that way a rewrite crash never loses the captioning work. A track that raises
    is recorded under ``failed_tracks`` and skipped; a track over
    ``max_track_seconds`` or below ``min_bitrate_kbps`` is recorded under
    ``skipped_tracks``. Only if *every* non-skipped track fails does the album
    count as failed.
    """
    album_obj = caption_album(
        captioner, entry, workdir,
        existing=existing,
        retry_only_failed=retry_only_failed,
        max_track_seconds=max_track_seconds,
        min_bitrate_kbps=min_bitrate_kbps,
        experimental_suno=experimental_suno,
    )

    if checkpoint is not None:
        checkpoint(album_obj, CAPTION_STAGE_CAPTIONED)

    if rewrite and rewriter is not None:
        if before_polish is not None:
            before_polish()  # e.g. wait for the background weight download
        unload = getattr(captioner, "unload", None)
        if callable(unload):
            unload()
        album_obj = polish_album(album_obj, rewriter)
        unload = getattr(rewriter, "unload", None)
        if callable(unload):
            unload()
    else:
        album_obj = polish_album(album_obj, None)

    atomic_write_json(
        album_output_path(outputs, entry["artist"], entry["album"]), album_obj
    )
    return album_obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="GCS prefix (gs://...) or a local directory to scan")
    parser.add_argument("--artist", action="append", default=None,
                        help="only this artist (repeatable); default: all")
    parser.add_argument("--artist-name", default=None,
                        help="override the artist label for a local --root "
                             "(default: the root directory name)")
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT / "work")
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs" / "index.json")
    parser.add_argument("--hf-model", type=Path, default=REPO_ROOT / "models" / "Ace-Step-Captioner-4bit")
    parser.add_argument(
        "--rewrite-model", default=DEFAULT_REWRITE_MODEL,
        help="small instruct model used in the polish phase (default: "
             "Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--rewrite-quantization", choices=("none", "4bit", "8bit"), default="none",
        help="default none = plain fp16 (a 4B model fits alone on the T4 and "
             "loads fastest); quantization is only a memory fallback",
    )
    parser.add_argument(
        "--rewrite-backend", choices=("hf", "llamacpp"), default="hf",
        help="hf = transformers fp16 (default); llamacpp = GGUF under "
             "llama-server (needs --rewrite-gguf)",
    )
    parser.add_argument(
        "--rewrite-gguf", type=Path, default=None,
        help="path to the GGUF weights used by --rewrite-backend llamacpp",
    )
    parser.add_argument(
        "--llamacpp-server", type=Path,
        default=REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server",
        help="llama-server binary staged by plumbing/fetch_llamacpp_text.sh",
    )
    parser.add_argument(
        "--rewrite-max-new-tokens", type=int, default=384,
        help="hard cap for one rewrite",
    )
    parser.add_argument(
        "--no-rewrite", action="store_true",
        help="caption only (debug); albums stay at stage 'captioned' and are "
             "re-captioned on the next normal run",
    )
    parser.add_argument(
        "--no-prefetch-rewrite", action="store_true",
        help="do not download the rewrite weights in the background during "
             "captioning (they are fetched when polish starts)",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=None,
        help="target chunk length (default: 90 for the hf backend); chunks are "
             "rebalanced so there is never a tiny final one",
    )
    parser.add_argument(
        "--no-trim-silence", action="store_true",
        help="keep leading/trailing silence instead of trimming it before chunking",
    )
    parser.add_argument("--max-track-seconds", type=float, default=480.0,
                        help="skip tracks longer than this (default 480 = 8 min; "
                             "0 disables the cap). Guards against concerts/long mixes.")
    parser.add_argument("--min-bitrate-kbps", type=float, default=128.0,
                        help="skip tracks below this audio bitrate (default 128; "
                             "0 disables). Filters out low-quality 32/64 kbps files.")
    parser.add_argument("--max-minutes", type=float, default=100.0,
                        help="stop cleanly between albums after this long (0 = no limit)")
    parser.add_argument("--max-albums", type=int, default=0, help="cap albums this run (0 = no cap)")
    parser.add_argument("--no-push", action="store_true", help="commit but do not push")
    parser.add_argument("--force", action="store_true", help="redo albums already in the manifest")
    parser.add_argument(
        "--repolish", action="store_true",
        help="re-run only the polish phase on every album that has a JSON "
             "(reuses the stored chunk_captions; useful to A/B a new rewrite model)",
    )
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-queue albums marked partial (some tracks failed) and "
                             "re-caption only the failed tracks")
    parser.add_argument("--experimental-suno", action="store_true",
                        help="also write the unsupported suno_prompt string "
                             "(default: omit; see docs/TAG_AUDIT.md)")
    parser.add_argument("--list", action="store_true", help="show pending albums and exit")
    args = parser.parse_args(argv)

    push = not args.no_push

    if push:
        proc = _run(["git", "-C", str(REPO_ROOT), "pull", "--ff-only"])
        if proc.returncode != 0:
            print(f"WARNING: git pull failed: {proc.stderr.strip()[:200]}")

    if args.root.startswith("gs://"):
        albums = group_tracks(list_gs_mp3(args.root), args.root)
    else:
        albums = group_local_tracks(args.root, artist=args.artist_name)
    if args.artist:
        wanted = set(args.artist)
        albums = {k: v for k, v in albums.items() if v["artist"] in wanted}

    manifest = load_manifest(args.manifest)
    pending = pending_albums(albums, manifest, args.outputs,
                             force=args.force, retry_failed=args.retry_failed)
    if args.repolish:
        pending = [
            key for key in albums
            if album_output_path(
                args.outputs, albums[key]["artist"], albums[key]["album"]
            ).exists()
        ]
    done = [k for k in albums if k not in pending]

    print(f"Caption pipeline version: {CAPTION_PIPELINE_VERSION}")
    print(f"Albums found: {len(albums)} | done: {len(done)} | pending: {len(pending)}")
    if args.list:
        for key in pending:
            print(f"  PENDING  {key}  ({len(albums[key]['tracks'])} tracks)")
        for key in done:
            st = album_status(key, manifest)
            note = f"  [{st}]" if st and st != "ok" else ""
            print(f"  done     {key}{note}")
        return 0
    if not pending:
        print("Nothing to do — all albums already captured.")
        return 0

    from src.captioner_hf import HFCaptioner

    chunk_args = (
        {} if args.chunk_seconds is None else {"chunk_seconds": args.chunk_seconds}
    )
    captioner = HFCaptioner(
        model_dir=args.hf_model,
        trim_silence=not args.no_trim_silence,
        **chunk_args,
    )
    rewriter = None
    if not args.no_rewrite:
        if args.rewrite_backend == "llamacpp":
            if args.rewrite_gguf is None:
                parser.error("--rewrite-backend llamacpp needs --rewrite-gguf PATH")
            from src.rewriter_llama import LlamaCppRewriter

            rewriter = LlamaCppRewriter(
                model_path=args.rewrite_gguf,
                server_bin=args.llamacpp_server,
                max_new_tokens=args.rewrite_max_new_tokens,
            )
        else:
            from src.rewriter_hf import HFRewriter

            rewriter = HFRewriter(
                model_id=args.rewrite_model,
                quantization=args.rewrite_quantization,
                max_new_tokens=args.rewrite_max_new_tokens,
            )
    prefetch_thread = None
    prefetch_id = getattr(rewriter, "prefetch_model_id", None)
    if rewriter is not None and prefetch_id and not args.no_prefetch_rewrite:
        print(f"Prefetching {prefetch_id} in the background...", flush=True)
        prefetch_thread = prefetch_model(prefetch_id)
    t0 = time.time()
    if not args.repolish:
        captioner._load()
        print(f"MODEL LOAD {time.time() - t0:.1f}s", flush=True)
    if rewriter is not None:
        mode = getattr(rewriter, "quantization", "gguf")
        print(f"Rewrite model: {rewriter.name} ({mode})", flush=True)

    started = time.time()
    completed = 0
    durations: list[float] = []

    def record_album(key: str, entry: dict, album_obj: dict, stage: str,
                     message: str) -> None:
        atomic_write_json(
            album_output_path(args.outputs, entry["artist"], entry["album"]),
            album_obj,
        )
        manifest["albums"][key] = {
            "path": f"{entry['artist']}/{entry['album']}.json",
            "source_dir": entry["source_dir"],
            "track_count": album_obj["track_count"],
            "failed_count": album_obj.get("failed_count", 0),
            "skipped_count": album_obj.get("skipped_count", 0),
            "status": album_obj.get("status", "ok"),
            "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
            "caption_stage": stage,
            "rewrite_fallback_count": (album_obj.get("rewrite") or {}).get("fallback", 0),
            "rewrite_error_count": (album_obj.get("rewrite") or {}).get("error", 0),
            "total_caption_seconds": album_obj["total_caption_seconds"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(args.manifest, manifest)
        git_checkpoint(REPO_ROOT, message, push=push)

    for i, key in enumerate(pending, 1):
        if args.max_albums and completed >= args.max_albums:
            print(f"Reached --max-albums={args.max_albums}; stopping.")
            break
        elapsed_min = (time.time() - started) / 60.0
        est = (sum(durations) / len(durations)) if durations else 0.0
        if args.max_minutes and durations and elapsed_min + est > args.max_minutes:
            print(f"Time budget {args.max_minutes:.0f} min reached "
                  f"(elapsed {elapsed_min:.1f} min); stopping cleanly.")
            break

        entry = albums[key]
        print(f"\n[{i}/{len(pending)}] {key} ({len(entry['tracks'])} tracks)", flush=True)

        prior_obj = None
        retry_only = False
        prior_path = album_output_path(args.outputs, entry["artist"], entry["album"])
        if prior_path.exists() and (args.repolish or not args.force):
            try:
                data = json.loads(prior_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if (
                args.repolish
                and isinstance(data, dict)
                and is_current_caption_output(data)
                and data.get("tracks")
            ):
                # Reuse every stored caption and only redo the polish phase.
                prior_obj = data
                retry_only = True
            elif isinstance(data, dict) and is_current_caption_output(data):
                if args.retry_failed:
                    prior_obj = data
                    retry_only = data.get("status") != "ok"
                elif data.get("caption_stage") == CAPTION_STAGE_CAPTIONED:
                    # Captioning is already paid for but polishing failed or was
                    # disabled: reuse every successful caption and retry only the
                    # failed tracks, instead of re-captioning the whole album.
                    prior_obj = data
                    retry_only = True

        def checkpoint(obj, stage, _key=key, _entry=entry):
            # Persist+push the raw captions before the polish phase, so a rewrite
            # crash never loses the captioning work.
            record_album(
                _key, _entry, obj, stage,
                f"captions({_key}): {obj['track_count']} tracks [{stage}]",
            )

        album_t0 = time.time()
        try:
            album_obj = process_album(
                captioner, entry, args.workdir, args.outputs,
                existing=prior_obj, retry_only_failed=retry_only,
                max_track_seconds=args.max_track_seconds,
                min_bitrate_kbps=args.min_bitrate_kbps,
                experimental_suno=args.experimental_suno,
                rewriter=rewriter,
                checkpoint=checkpoint if not args.repolish else None,
                before_polish=(
                    prefetch_thread.join if prefetch_thread is not None else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"  ERROR on {key}: {type(exc).__name__}: {exc}")
            continue

        album_min = (time.time() - album_t0) / 60.0
        durations.append(album_min)
        completed += 1
        n_failed = album_obj.get("failed_count", 0)
        n_skipped = album_obj.get("skipped_count", 0)
        rw = album_obj.get("rewrite") or {}
        n_fallback = int(rw.get("fallback", 0) or 0)
        n_error = int(rw.get("error", 0) or 0)
        print(f"  done in {album_min:.1f} min "
              f"({album_obj['total_caption_seconds']}s captioning, "
              f"{album_obj.get('total_rewrite_seconds', 0)}s polishing)",
              flush=True)
        if n_skipped:
            print(f"  note: {n_skipped} track(s) skipped "
                  f"(over --max-track-seconds={args.max_track_seconds:.0f})", flush=True)
        if n_failed:
            print(f"  WARNING: {n_failed} track(s) failed; re-run with "
                  f"--retry-failed to retry just them.", flush=True)
        if n_fallback:
            print(f"  WARNING: {n_fallback} rewrite(s) fell back to the raw "
                  f"caption; see rewrite_reasons in the JSON", flush=True)
        if n_error:
            print(f"  WARNING: {n_error} rewrite(s) errored; album stays "
                  f"'captioned' and will be retried without re-captioning",
                  flush=True)
        record_album(
            key, entry, album_obj,
            album_obj.get("caption_stage", CAPTION_STAGE_CAPTIONED),
            f"captions({key}): {album_obj['track_count']} tracks"
            + (f", {n_failed} failed" if n_failed else "")
            + (f", {n_skipped} skipped" if n_skipped else "")
            + (f", {n_fallback} rewrite-fallback" if n_fallback else "")
            + f" ({len(done) + completed}/{len(albums)})",
        )

    remaining = len(pending) - completed
    print(f"\nRun complete: {completed} album(s) this session, ~{remaining} still pending.")
    print(f"Total wall time: {(time.time() - started) / 60.0:.1f} min")

    problematic = {
        k: v for k, v in (manifest.get("albums") or {}).items()
        if isinstance(v, dict) and v.get("status", "ok") != "ok"
    }
    if problematic:
        print(f"\n{len(problematic)} album(s) need attention:")
        for k, v in sorted(problematic.items()):
            bits = []
            if v.get("failed_count"):
                bits.append(f"{v['failed_count']} failed")
            if v.get("skipped_count"):
                bits.append(f"{v['skipped_count']} skipped")
            print(f"  [{v.get('status', '?')}] {k}: " + ", ".join(bits))
        print("  (re-run with --retry-failed to re-caption failed tracks)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
