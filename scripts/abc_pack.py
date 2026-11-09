#!/usr/bin/env python3
"""Pack raw SheetSage2 transcripts into per-album ABC JSON + the resume index.

CPU only. Reads a raw output tree shaped like::

    <raw-root>/<ARTIST>/<album>/<stem>/{score.abc, meta.json, *.lab, ...}

and writes::

    outputs/<ARTIST>/abc/<album>.json
    outputs/abc_index.json

The album JSON is the §10.1 schema, including the §9 QC block and the derived
``abc_melody`` (chords stripped, §15.3). The raw ABC is evidence: this script
never edits it. ``qc.parse=fail`` rows are kept and quarantined downstream.

It also works on a flat pool (e.g. the 3-track pilot) by matching each stem back
to the manifest, which is how a pilot that spans albums gets packed.

ABC text is written locally only. Whether it is committed is a policy decision
(§15.1) and is deliberately **not** made here.

Usage:
    python3 scripts/abc_pack.py --outputs outputs --manifest work/manifest.jsonl
    python3 scripts/abc_pack.py --raw-dir /content/work/out/pilot --artist ELISSA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from abcscore import (  # noqa: E402
    normalize_title,
    qc_track,
    sha256_file,
    sha256_text,
    strip_chords,
)

DEFAULT_RAW_ROOT = Path("/content/work/out")
GCS_TRANSCRIPTS = "gs://akbar-december-2024-backup/TRANSCRIPTS"
MODEL = "m-a-p/SheetSage2"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record") == "header":
                continue
            rows.append(row)
    return rows


def load_prior_tracks(outputs: Path, skip: set[str]) -> list[dict]:
    """All tracks already packed in other album JSONs (for duplicate_of)."""
    prior: list[dict] = []
    for path in sorted(outputs.glob("*/abc/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = f"{data.get('artist')}/{data.get('album')}"
        if key in skip:
            continue
        for track in data.get("tracks", []):
            prior.append({**track,
                          "artist": (track.get("source_artist")
                                     or data.get("artist")),
                          "album": data.get("album")})
    return prior


def discover(raw_root: Path, flat_dir: Path | None,
             manifest: list[dict], artist: str | None,
             album: str | None) -> dict[str, list[tuple[Path, dict]]]:
    """Group raw track directories by ``ARTIST/album``."""
    by_album_stem = {(r["artist"], r["album"], Path(r["file"]).stem): r
                     for r in manifest}
    by_stem: dict[str, list[dict]] = {}
    for row in manifest:
        by_stem.setdefault(Path(row["file"]).stem, []).append(row)

    groups: dict[str, list[tuple[Path, dict]]] = {}
    if flat_dir is not None:
        for track_dir in sorted(p for p in flat_dir.iterdir() if p.is_dir()):
            matches = by_stem.get(track_dir.name, [])
            if len(matches) > 1:
                # Disambiguate with --artist / --album when filenames collide.
                matches = [m for m in matches
                           if (not artist or m["artist"] == artist)
                           and (not album or m["album"] == album)]
            if len(matches) != 1:
                raise SystemExit(
                    f"{track_dir}: cannot match stem to exactly one manifest row "
                    f"({len(matches)} matches); use --artist/--album")
            row = matches[0]
            groups.setdefault(f"{row['artist']}/{row['album']}", []).append(
                (track_dir, row))
        return groups

    artists = [artist] if artist else sorted(
        {r["artist"] for r in manifest})
    for art in artists:
        art_dir = raw_root / art
        if not art_dir.is_dir():
            continue
        albums = [album] if album else sorted(
            p.name for p in art_dir.iterdir() if p.is_dir())
        for alb in albums:
            alb_dir = art_dir / alb
            if not alb_dir.is_dir():
                continue
            for track_dir in sorted(p for p in alb_dir.iterdir() if p.is_dir()):
                row = by_album_stem.get((art, alb, track_dir.name))
                if row is None:
                    # The stem may have been suffixed by --on-collision.
                    candidates = [r for r in manifest
                                  if r["artist"] == art and r["album"] == alb
                                  and track_dir.name.startswith(Path(r["file"]).stem)]
                    row = candidates[0] if len(candidates) == 1 else None
                if row is None:
                    continue
                groups.setdefault(f"{art}/{alb}", []).append((track_dir, row))
    return groups


def build_track(track_dir: Path, row: dict, artist: str, album: str,
                raw_gcs: str | None = None,
                raw_archive: bool = False) -> dict | None:
    meta_path = track_dir / "meta.json"
    abc_path = track_dir / "score.abc"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    duration = meta.get("duration_s") or row.get("duration_s")
    if raw_gcs:
        raw_uri = (f"{raw_gcs.rstrip('/')}/{track_dir.name}.tar.gz"
                   if raw_archive else
                   f"{raw_gcs.rstrip('/')}/{track_dir.name}/")
    else:
        raw_uri = f"{GCS_TRANSCRIPTS}/{artist}/{album}/{track_dir.name}/"
    source_artist = row.get("source_artist") or artist
    base = {
        "track_id": row["track_id"],
        "artist": source_artist,
        "source_artist": source_artist,
        "album": album,
        "video_id": row.get("video_id"),
        "source_rank": row.get("source_rank"),
        "youtube_title": row.get("youtube_title"),
        "country": row.get("country"),
        "tier": row.get("tier"),
        "file": row["file"],
        "title": row.get("title") or Path(row["file"]).stem,
        "track_no": row.get("track_no"),
        "duration_s": round(float(duration), 3) if duration else None,
        "audio_sha256": meta.get("source_wav_sha256"),
        "source_uri": row.get("source_uri"),
        "stem": track_dir.name,
        "raw_gcs": raw_uri,
        "gpu_seconds": meta.get("seconds"),
        "transcribe_status": meta.get("status"),
    }
    if meta.get("status") != "ok" or not abc_path.is_file():
        base.update({
            "abc": None, "abc_sha256": None, "abc_melody": None,
            "abc_melody_sha256": None, "has_chords": False,
            "instrumental": "(music)" in row["file"].lower(),
            "lyrics_eligible": "(music)" not in row["file"].lower(),
            "qc": {"parse": "fail", "flags": ["meta_status:" + str(meta.get("status"))],
                   "detail": meta.get("error", "")},
        })
        return base

    abc = abc_path.read_text(encoding="utf-8")
    abc_sha = sha256_file(abc_path)
    if meta.get("abc_sha256") and meta["abc_sha256"] != abc_sha:
        base["qc"] = {"parse": "fail",
                      "flags": ["abc_hash_mismatch"],
                      "detail": "score.abc does not match meta.abc_sha256"}
        base.update({"abc": abc, "abc_sha256": abc_sha, "has_chords": False,
                     "instrumental": False, "lyrics_eligible": True,
                     "abc_melody": None, "abc_melody_sha256": None})
        return base

    qc = qc_track(track_dir, duration, file=row["file"], abc=abc)
    melody = strip_chords(abc)
    base.update({
        "abc": abc,
        "abc_sha256": abc_sha,
        "abc_melody": melody,
        "abc_melody_sha256": sha256_text(melody),
        "key": qc["key"],
        "tempo_bpm": qc["tempo_bpm"],
        "meters": qc["meters"],
        "sections": qc["sections"],
        "notes_vocal": qc["notes_vocal"],
        "notes_ins": qc["notes_ins"],
        "has_chords": qc["has_chords"],
        "instrumental": qc["instrumental"],
        "lyrics_eligible": not qc["instrumental"],
        "qc": qc,
    })
    return base


def assign_duplicates(tracks: list[dict], prior: list[dict],
                      artist: str) -> None:
    """Set ``duplicate_of`` by audio hash first, normalized title second.

    For a synthetic group key such as ``YT500`` the real artist lives in
    ``source_artist``; using it here prevents two different artists with the
    same song title from being collapsed.
    """
    def key_artist(row: dict) -> str:
        return str(row.get("source_artist") or row.get("artist") or artist)

    hash_index: dict[str, str] = {}
    title_index: dict[tuple[str, str], str] = {}
    for row in prior:
        if row.get("audio_sha256"):
            hash_index.setdefault(row["audio_sha256"], row["track_id"])
        title_index.setdefault(
            (key_artist(row), normalize_title(row["title"])),
            row["track_id"])
    for row in tracks:
        dup = None
        if row.get("audio_sha256") and row["audio_sha256"] in hash_index:
            dup = hash_index[row["audio_sha256"]]
        else:
            dup = title_index.get((key_artist(row), normalize_title(row["title"])))
        row["duplicate_of"] = dup
        if row.get("audio_sha256"):
            hash_index.setdefault(row["audio_sha256"], row["track_id"])
        title_index.setdefault(
            (key_artist(row), normalize_title(row["title"])), row["track_id"])


def pack_album(key: str, items: list[tuple[Path, dict]], outputs: Path,
               prior: list[dict], expected_ids: set[str],
               raw_gcs: str | None = None,
               raw_archive: bool = False) -> dict:
    artist, album = key.split("/", 1)
    tracks: list[dict] = []
    failed: list[dict] = []
    for track_dir, row in items:
        track = build_track(track_dir, row, artist, album, raw_gcs,
                            raw_archive=raw_archive)
        if track is None:
            failed.append({"file": row["file"], "error": "missing meta.json"})
            continue
        tracks.append(track)
        if track["qc"]["parse"] == "fail":
            failed.append({"file": track["file"],
                           "error": track["qc"].get("detail", "parse failed")})
    tracks.sort(key=lambda t: (t.get("track_no") or 0, t["file"]))
    assign_duplicates(tracks, prior, artist)

    missing = sorted(expected_ids - {t["track_id"] for t in tracks})
    total_audio = sum(t.get("duration_s") or 0 for t in tracks)
    total_gpu = sum(t.get("gpu_seconds") or 0 for t in tracks)
    metas = []
    for track_dir, _ in items:
        meta_path = track_dir / "meta.json"
        if meta_path.is_file():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    revision = next((m.get("model_revision") for m in metas
                     if m.get("model_revision")), None)
    mode = "melody" if all(m.get("melody_only") for m in metas if m) else "full"
    ok = sum(1 for t in tracks
             if t["qc"]["parse"] == "ok" and not t["qc"].get("flags"))
    flagged = sum(1 for t in tracks
                  if t["qc"]["parse"] == "ok" and t["qc"].get("flags"))
    # ``failed`` already contains both missing-meta tracks and parse failures.
    failed_count = len(failed)
    status = "done" if not failed and not missing else "partial"

    album_obj = {
        "version": 1,
        "artist": artist,
        "album": album,
        "generated_at": utcnow(),
        "model": MODEL,
        "model_revision": revision,
        "mode": mode,
        "dtype": next((m.get("dtype") for m in metas if m.get("dtype")), None),
        "device": next((m.get("device") for m in metas if m.get("device")), None),
        "status": status,
        "total_audio_seconds": round(total_audio, 1),
        "total_gpu_seconds": round(total_gpu, 1),
        "rtf": round(total_audio / total_gpu, 3) if total_gpu else None,
        "tracks": tracks,
        "failed_tracks": failed,
        "missing_track_ids": missing,
    }
    out_path = outputs / artist / "abc" / f"{album}.json"
    atomic_write_json(out_path, album_obj)

    return {
        "path": f"{artist}/abc/{album}.json",
        "status": status,
        "track_count": len(tracks),
        "ok": ok,
        "flagged": flagged,
        "failed": failed_count,
        "missing_track_ids": missing,
        "total_audio_seconds": round(total_audio, 1),
        "total_gpu_seconds": round(total_gpu, 1),
        "rtf": round(total_audio / total_gpu, 3) if total_gpu else None,
        "completed_at": album_obj["generated_at"],
        "model_revision": revision,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "work" / "manifest.jsonl")
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="flat directory of track dirs (e.g. the pilot output)")
    ap.add_argument("--artist", default=None)
    ap.add_argument("--album", default=None)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--raw-gcs", default=None,
                    help="GCS URI to record as raw_gcs for every track in the "
                         "album (e.g. the per-album tar.gz); default is the "
                         "per-track directory")
    ap.add_argument("--raw-archive", action="store_true",
                    help="with --raw-gcs, record <raw-gcs>/<stem>.tar.gz "
                         "instead of <raw-gcs>/<stem>/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    groups = discover(args.raw_root, args.raw_dir, manifest,
                      args.artist, args.album)
    if not groups:
        print("nothing to pack", file=sys.stderr)
        return 1

    index_path = args.index or (args.outputs / "abc_index.json")
    index: dict = {"version": 1, "albums": {}}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index.setdefault("version", 1)
        index.setdefault("albums", {})

    expected_by_album: dict[str, set[str]] = {}
    for row in manifest:
        expected_by_album.setdefault(
            f"{row['artist']}/{row['album']}", set()).add(row["track_id"])

    prior = load_prior_tracks(args.outputs, set(groups))
    summary = []
    for key in sorted(groups):
        items = groups[key]
        if args.dry_run:
            print(f"{key}: {len(items)} track(s) -> "
                  f"outputs/{key.split('/', 1)[0]}/abc/{key.split('/', 1)[1]}.json")
            continue
        block = pack_album(key, items, args.outputs, prior,
                           expected_by_album.get(key, set()), args.raw_gcs,
                           raw_archive=args.raw_archive)
        index["albums"][key] = block
        # Newly packed rows become prior for the next album in this run.
        packed = json.loads(
            (args.outputs / block["path"]).read_text(encoding="utf-8"))
        prior.extend(packed["tracks"])
        summary.append({"album": key, **block})
        print(f"{key}: {block['track_count']} tracks, "
              f"{block['ok']} ok / {block['flagged']} flagged / "
              f"{block['failed']} failed, rtf={block['rtf']}")

    if args.dry_run:
        return 0
    index["updated_at"] = utcnow()
    atomic_write_json(index_path, index)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
