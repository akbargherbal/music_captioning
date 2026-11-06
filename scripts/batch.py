#!/usr/bin/env python3
"""Resumable batch captioning across ephemeral Colab runtimes.

GitHub is the durable store. For every album we:
  1. download its tracks from GCS (skip files already cached),
  2. caption them,
  3. atomically write outputs/<ARTIST>/<album>.json,
  4. update outputs/index.json,
  5. git add/commit/push immediately.

On a fresh runtime, run the same command: it pulls, reads the manifest, and
skips every album already pushed. Nothing is redone — only the in-flight album
is lost if the runtime dies.

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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.resume import (  # noqa: E402
    SkippedTrack,
    album_key,
    album_output_path,
    album_status,
    atomic_write_json,
    group_local_tracks,
    group_tracks,
    is_done,
    load_manifest,
    pending_albums,
    run_tracks,
    skip_reason,
)
from src.tags import CATEGORIES  # noqa: E402

DEFAULT_ROOT = "gs://akbar-december-2024-backup/DISCOGRAPHY"

# Repo-local fallback identity. A fresh Colab runtime has no global
# user.name/user.email, so `git commit` fails and every album checkpoint is lost
# (writing JSON + manifest without ever pushing). Set these if unset.
FALLBACK_GIT_NAME = "akbargherbal"
FALLBACK_GIT_EMAIL = "akbargherbal@users.noreply.github.com"


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
) -> dict:
    """Caption one album, **isolating per-track failures**.

    A track that raises is recorded under ``failed_tracks`` and skipped; the
    remaining tracks are still written and committed. A track longer than
    ``max_track_seconds`` (e.g. a concert) or below ``min_bitrate_kbps`` (low
    quality) is recorded under ``skipped_tracks`` instead. Only if *every*
    non-skipped track fails does the album count as failed — in that case
    nothing is written, so it stays pending.
    """
    from src.captioner import audio_stream_info
    from src.captioner_hf import caption_and_tag
    from src.tags import TAGS_EXPERIMENTAL, merge_tags, to_suno

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
        result.pop("chunk_captions", None)
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
            "caption": result["caption"],
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
        "tracks": track_results,
        "failed_tracks": failed,
        "skipped_tracks": skipped,
    }
    if experimental_suno:
        album_obj["suno_prompt"] = to_suno(merged)
        album_obj["suno_experimental"] = True
    atomic_write_json(album_output_path(outputs, artist, album), album_obj)
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
    done = [k for k in albums if k not in pending]

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
    t0 = time.time()
    captioner._load()
    print(f"MODEL LOAD {time.time() - t0:.1f}s", flush=True)

    started = time.time()
    completed = 0
    durations: list[float] = []
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
        if args.retry_failed:
            prior_path = album_output_path(args.outputs, entry["artist"], entry["album"])
            if prior_path.exists():
                try:
                    prior_obj = json.loads(prior_path.read_text(encoding="utf-8"))
                    retry_only = prior_obj.get("status") != "ok"
                except (json.JSONDecodeError, OSError):
                    prior_obj = None

        album_t0 = time.time()
        try:
            album_obj = process_album(
                captioner, entry, args.workdir, args.outputs,
                existing=prior_obj, retry_only_failed=retry_only,
                max_track_seconds=args.max_track_seconds,
                min_bitrate_kbps=args.min_bitrate_kbps,
                experimental_suno=args.experimental_suno,
            )
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"  ERROR on {key}: {type(exc).__name__}: {exc}")
            continue

        manifest["albums"][key] = {
            "path": f"{entry['artist']}/{entry['album']}.json",
            "source_dir": entry["source_dir"],
            "track_count": album_obj["track_count"],
            "failed_count": album_obj.get("failed_count", 0),
            "skipped_count": album_obj.get("skipped_count", 0),
            "status": album_obj.get("status", "ok"),
            "total_caption_seconds": album_obj["total_caption_seconds"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(args.manifest, manifest)

        album_min = (time.time() - album_t0) / 60.0
        durations.append(album_min)
        completed += 1
        n_failed = album_obj.get("failed_count", 0)
        n_skipped = album_obj.get("skipped_count", 0)
        print(f"  done in {album_min:.1f} min "
              f"({album_obj['total_caption_seconds']}s captioning)", flush=True)
        if n_skipped:
            print(f"  note: {n_skipped} track(s) skipped "
                  f"(over --max-track-seconds={args.max_track_seconds:.0f})", flush=True)
        if n_failed:
            print(f"  WARNING: {n_failed} track(s) failed; re-run with "
                  f"--retry-failed to retry just them.", flush=True)
        git_checkpoint(
            REPO_ROOT,
            f"captions({key}): {album_obj['track_count']} tracks"
            + (f", {n_failed} failed" if n_failed else "")
            + (f", {n_skipped} skipped" if n_skipped else "")
            + f" ({len(done) + completed}/{len(albums)})",
            push=push,
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
