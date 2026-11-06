"""Resume/continuity helpers for batch captioning across ephemeral runtimes.

GitHub is the durable store: after every album the caller commits the output
JSON plus a manifest, so a fresh Colab runtime can skip completed work.

Pure helpers only (no torch/gcloud), so they are unit-testable on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MANIFEST_VERSION = 1

# Bump whenever caption semantics change: chunking, merge, prompt, model,
# quantization or decoding. Album JSONs written by an older pipeline (or by the
# pre-versioning code, which has no field at all) are treated as *not done*, so
# `scripts/batch.py` re-captions them instead of skipping them.
#   1 (implicit) = no version field: fixed 60 s chunks + blind join
#   2            = silence trim + rebalanced ~90 s chunks + deterministic merge
CAPTION_PIPELINE_VERSION = 2


class SkippedTrack(Exception):
    """Raised by a track callback to skip a track *without* failing its album.

    Used for (e.g.) concert-length tracks past the ``--max-track-seconds`` cap.
    """


def album_key(artist: str, album: str) -> str:
    return f"{artist}/{album}"


def album_output_path(outputs_root: Path | str, artist: str, album: str) -> Path:
    """Where an album's result JSON lives, relative to the outputs root."""
    return Path(outputs_root) / artist / f"{album}.json"


def parse_gs_track(uri: str, root: str) -> dict[str, str] | None:
    """Split ``gs://.../<artist>/<album>/<file>`` into its parts.

    Returns ``None`` for anything that is not an audio file under ``root``.
    """
    root = root.rstrip("/")
    if not uri.startswith(root + "/"):
        return None
    rel = uri[len(root) + 1:]
    parts = rel.split("/")
    if len(parts) < 3:
        return None
    artist, album = parts[0], parts[1]
    filename = "/".join(parts[2:])
    if Path(filename).suffix.lower() not in AUDIO_SUFFIXES:
        return None
    return {
        "artist": artist,
        "album": album,
        "filename": filename,
        "uri": uri,
        "source_dir": f"{root}/{artist}/{album}",
    }


def group_tracks(uris: list[str], root: str) -> dict[str, dict[str, object]]:
    """Group track URIs into albums keyed by ``artist/album``."""
    albums: dict[str, dict[str, object]] = {}
    for uri in uris:
        info = parse_gs_track(uri, root)
        if info is None:
            continue
        key = album_key(info["artist"], info["album"])
        entry = albums.setdefault(
            key,
            {
                "artist": info["artist"],
                "album": info["album"],
                "source_dir": info["source_dir"],
                "tracks": [],
            },
        )
        entry["tracks"].append({"filename": info["filename"], "uri": info["uri"]})
    for entry in albums.values():
        entry["tracks"].sort(key=lambda t: t["filename"])  # type: ignore[index]
    return albums


def group_local_tracks(
    root: Path | str, artist: str | None = None
) -> dict[str, dict[str, object]]:
    """Group audio files under a local root into albums.

    Supported layouts:
      ``root/<file>``                    -> artist = root.name, album = root.name
      ``root/<album>/<file>``            -> artist = root.name, album = <album>
      ``root/<artist>/<album>/<file>``   -> artist = <artist>

    ``artist`` overrides the derived artist name.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    albums: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if len(parts) == 1:
            a, alb, fname, src = artist or root.name, root.name, parts[0], root
        elif len(parts) == 2:
            a, alb, fname, src = artist or root.name, parts[0], parts[1], root / parts[0]
        else:
            a = artist or parts[0]
            alb, fname = parts[1], str(Path(*parts[2:]))
            src = root / parts[0] / parts[1]
        key = album_key(a, alb)
        entry = albums.setdefault(
            key,
            {"artist": a, "album": alb, "source_dir": str(src), "tracks": []},
        )
        entry["tracks"].append({"filename": fname, "uri": str(path)})
    for entry in albums.values():
        entry["tracks"].sort(key=lambda t: t["filename"])  # type: ignore[index]
    return albums


def load_manifest(path: Path | str) -> dict[str, object]:
    """Load the manifest, returning an empty one if missing/corrupt."""
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("version", MANIFEST_VERSION)
                data.setdefault("albums", {})
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": MANIFEST_VERSION, "albums": {}}


def is_current_caption_output(data: object) -> bool:
    """True when an album JSON was written by the current caption pipeline."""
    return (
        isinstance(data, dict)
        and data.get("caption_pipeline_version") == CAPTION_PIPELINE_VERSION
    )


def is_done(key: str, manifest: dict[str, object], outputs_root: Path | str) -> bool:
    """An album is done only if it exists **and** uses the current pipeline.

    The album file is the source of truth: a manifest entry alone is not enough,
    because outputs produced by an older chunking/merge pipeline (or by the
    pre-versioning code) must be re-captioned. Missing/corrupt JSON is not done.
    """
    entry = (manifest.get("albums") or {}).get(key)  # type: ignore[union-attr]
    if not isinstance(entry, dict):
        return False
    rel = entry.get("path")
    if not rel:
        return False
    path = Path(outputs_root) / rel
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return is_current_caption_output(data)


def album_status(key: str, manifest: dict[str, object]) -> str | None:
    """Manifest status for an album: ``"ok"``, ``"partial"``, ``"error"`` or None."""
    entry = (manifest.get("albums") or {}).get(key)  # type: ignore[union-attr]
    if isinstance(entry, dict):
        return entry.get("status", "ok")
    return None


def pending_albums(
    albums: dict[str, dict[str, object]],
    manifest: dict[str, object],
    outputs_root: Path | str,
    force: bool = False,
    retry_failed: bool = False,
) -> list[str]:
    """Album keys still to process, in stable order.

    ``force`` redoes everything. ``retry_failed`` additionally re-queues albums
    whose manifest status is not ``"ok"`` (they had failed tracks).
    """
    out: list[str] = []
    for key in sorted(albums):
        if force or not is_done(key, manifest, outputs_root):
            out.append(key)
        elif retry_failed and album_status(key, manifest) != "ok":
            out.append(key)
    return out


def skip_reason(
    duration_s: float | None,
    bitrate_kbps: float | None,
    max_track_seconds: float = 0.0,
    min_bitrate_kbps: float = 0.0,
) -> str | None:
    """Why a track should be skipped, or ``None`` to process it.

    Bitrate (quality) is checked first, then duration. Missing values (``None``)
    never trigger a skip, so unknown metadata is processed rather than dropped.
    """
    if min_bitrate_kbps and bitrate_kbps and bitrate_kbps < min_bitrate_kbps:
        return f"{bitrate_kbps:.0f} kbps < {min_bitrate_kbps:.0f} kbps"
    if max_track_seconds and duration_s and duration_s > max_track_seconds:
        return f"{duration_s:.0f}s > {max_track_seconds:.0f}s cap"
    return None


def run_tracks(
    tracks: list[dict],
    caption_one: Callable[[str, str], dict],
    existing: dict | None = None,
    retry_only_failed: bool = False,
    echo: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Caption tracks one by one, isolating per-track failures.

    ``caption_one(filename, uri)`` returns a per-track result (or raises):

    * raising :class:`SkippedTrack` -> recorded under ``skipped`` (e.g. the
      track is longer than the configured cap),
    * raising anything else -> recorded under ``failed``.

    Neither case discards the album. With ``retry_only_failed``, tracks that
    already succeeded in ``existing`` are kept and only previously-failed
    tracks are re-run.

    Returns ``(results, failed, skipped)``.
    """
    prior = existing or {}
    prior_tracks = {t["file"]: t for t in prior.get("tracks", [])}
    failed_files = {f.get("file") for f in prior.get("failed_tracks", [])}
    results: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    for track in tracks:
        filename = track["filename"]
        if retry_only_failed and filename in prior_tracks and filename not in failed_files:
            results.append(prior_tracks[filename])
            continue
        try:
            result = caption_one(filename, track["uri"])
        except SkippedTrack as exc:
            if echo:
                echo(f"{filename}: SKIPPED - {exc}")
            skipped.append({"file": filename, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - isolate a bad track
            msg = f"{type(exc).__name__}: {exc}"
            if echo:
                echo(f"{filename}: FAILED - {msg}")
            failed.append({"file": filename, "error": msg})
            continue
        if echo:
            echo(f"{filename}: {result.get('caption_seconds', 0):.1f}s")
        results.append(result)
    return results, failed, skipped


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def atomic_write_json(path: Path | str, data: object) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
