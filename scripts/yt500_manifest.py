#!/usr/bin/env python3
"""Build the frozen manifest for the ``yt_top_500_music_playlist`` corpus.

The bucket already contains a complete download manifest at
``state/latest/manifest.json`` (495 tracks, all ``done``).  This script turns it
into the JSONL contract consumed by ``scripts/abc_batch.py`` and freezes the
derived decisions (shards, near-duplicate links, pilot set).

Inputs
------
* ``--bucket`` (default) reads
  ``<bucket>/state/latest/manifest.json`` with ``gcloud storage cat``.
* ``--bucket-manifest PATH`` reads a local copy instead (offline/tests).
* ``--report PATH`` optionally reads the bucket's ``report.json`` for totals.

Outputs (under ``--out``'s parent, default ``outputs/YT500/``)
--------------------------------------------------------------
``manifest.jsonl``   header row + one track row per source object
``manifest.sha256``  sha256 of manifest.jsonl
``duplicates.json``  the same-artist/same-title groups and canonical choice
``shards.json``      track ids and audio seconds per shard
``pilot.json``       the 10-track pilot selection

Usage
-----
    python3 scripts/yt500_manifest.py \
      --bucket gs://akbar-december-2024-backup/yt_top_500_music_playlist \
      --out outputs/YT500/manifest.jsonl
    python3 scripts/yt500_manifest.py --check outputs/YT500/manifest.jsonl
    python3 scripts/yt500_manifest.py --download-pilot \
      outputs/YT500/pilot.json --pilot-dir work/yt500/pilot
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUCKET = "gs://akbar-december-2024-backup/yt_top_500_music_playlist"
DEFAULT_OUT = Path("outputs/YT500/manifest.jsonl")
REQUIRED_TRACK_FIELDS = (
    "record", "track_id", "video_id", "source_rank", "source_artist",
    "title", "file", "source_uri", "source_bytes", "source_sha256",
    "duration_s", "bitrate_kbps", "format", "shard", "output_stem",
    "artist", "album", "status",
)
PILOT_VIDEO_IDS = ("Pz59wizOqLw", "QS0v8uxzfJM", "ikcfOn9VzYo")
LIVE_MARKERS = ("live", "حفلة", "فبراير", "concert")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def read_gcs_json(uri: str) -> dict:
    proc = run(["gcloud", "storage", "cat", uri])
    if proc.returncode != 0:
        raise SystemExit(
            f"gcloud storage cat failed for {uri}: "
            f"{(proc.stderr or proc.stdout).strip()[-800:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON from {uri}: {exc}") from exc


def load_bucket_manifest(bucket: str, local: Path | None) -> tuple[dict, str | None]:
    """Return (manifest, source_uri)."""
    if local is not None:
        return json.loads(local.read_text(encoding="utf-8")), None
    uri = f"{bucket.rstrip('/')}/state/latest/manifest.json"
    return read_gcs_json(uri), uri


def load_report(bucket: str, local: Path | None
                ) -> tuple[dict | None, str | None, str | None]:
    """Return (report, source_uri, sha256_of_raw_bytes)."""
    if local is not None:
        raw = local.read_text(encoding="utf-8")
        try:
            return json.loads(raw), str(local), hashlib.sha256(
                raw.encode("utf-8")).hexdigest()
        except json.JSONDecodeError:
            return None, str(local), None
    uri = f"{bucket.rstrip('/')}/state/latest/report.json"
    proc = run(["gcloud", "storage", "cat", uri])
    if proc.returncode != 0:
        return None, None, None
    try:
        return (json.loads(proc.stdout), uri,
                hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest())
    except json.JSONDecodeError:
        return None, uri, None


def normalize_title(title: str) -> str:
    """Lowercase, collapse punctuation/whitespace; keep Arabic letters/digits."""
    text = title.casefold().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _track_bytes(track: dict) -> int:
    return int(track.get("file_bytes") or 0)


def _track_seconds(track: dict) -> float:
    for key in ("actual_duration", "duration_seconds", "duration_s"):
        value = track.get(key)
        if value is not None:
            return float(value)
    return 0.0


def assign_shards(rows: list[dict], count: int) -> dict[str, list[dict]]:
    """Balance by track count first, then audio seconds; deterministic."""
    if count <= 0:
        raise SystemExit("--shards must be > 0")
    shards: dict[str, list[dict]] = {
        f"shard-{i + 1:02d}": [] for i in range(count)
    }
    loads = {name: 0.0 for name in shards}
    for row in sorted(rows, key=lambda r: (-_track_seconds(r),
                                           int(r.get("rank") or 0),
                                           str(r.get("video_id") or ""))):
        name = min(shards, key=lambda s: (len(shards[s]), loads[s], s))
        shards[name].append(row)
        loads[name] += _track_seconds(row)
    for name in shards:
        shards[name].sort(key=lambda r: (int(r.get("rank") or 0),
                                         str(r.get("video_id") or "")))
    return shards


def assign_duplicates(rows: list[dict]) -> list[dict]:
    """Link same-artist/same-title groups. All rows stay transcribed."""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["source_artist"].casefold(),
                normalize_title(row["title"]))].append(row)

    report: list[dict] = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda r: (int(r.get("source_rank") or 0),
                                                str(r["video_id"])))
        group_id = f"{canonical['source_artist']} | {canonical['title']}"
        for row in members:
            row["duplicate_group"] = group_id
            if row is canonical:
                row["canonical"] = True
                row["duplicate_of"] = None
            else:
                row["canonical"] = False
                row["duplicate_of"] = canonical["track_id"]
        report.append({
            "group": group_id,
            "canonical": canonical["track_id"],
            "reason": "same artist + normalized title; different audio bytes",
            "members": [
                {
                    "track_id": r["track_id"],
                    "video_id": r["video_id"],
                    "rank": r["source_rank"],
                    "duration_s": round(_track_seconds(r), 3),
                    "source_sha256": r["source_sha256"],
                    "duplicate_of": r["duplicate_of"],
                }
                for r in sorted(members, key=lambda x: int(x["source_rank"]))
            ],
        })
    return report


def build_rows(manifest: dict, shard_count: int) -> tuple[list[dict], dict[str, list[dict]]]:
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise SystemExit("bucket manifest has no tracks")
    raw = sorted(tracks, key=lambda t: (int(t.get("rank") or 0),
                                       str(t.get("video_id") or "")))
    shards = assign_shards(raw, shard_count)

    rows: list[dict] = []
    for shard, members in shards.items():
        for track in members:
            uri = str(track["gcs_uri"])
            filename = Path(uri).name
            stem = Path(filename).stem
            duration = _track_seconds(track)
            row = {
                "record": "track",
                "track_id": f"YT500/{shard}/{stem}",
                "source_track_id": track.get("track_id"),
                "video_id": track["video_id"],
                "source_rank": int(track["rank"]),
                "source_artist": track["artist"],
                "title": track["title"],
                "youtube_title": track.get("youtube_title"),
                "title_normalized": normalize_title(track["title"]),
                "country": track.get("country"),
                "tier": track.get("tier"),
                "file": filename,
                "source_uri": uri,
                "info_json_uri": track.get("info_json_uri"),
                "source_bytes": _track_bytes(track),
                "source_sha256": track["sha256"],
                "duration_s": round(duration, 6),
                "bitrate_kbps": round(float(track.get("bitrate") or 0) / 1000, 1),
                "format": (Path(filename).suffix.lstrip(".") or "m4a").lower(),
                "duplicate_group": None,
                "duplicate_of": None,
                "canonical": True,
                "shard": shard,
                "output_stem": stem,
                "status": "pending",
            }
            # The batch driver groups by artist/album; YT500/<shard> is the
            # checkpoint.  source_artist keeps the real artist for the packer.
            row["artist"] = "YT500"
            row["album"] = shard
            rows.append(row)
    rows.sort(key=lambda r: (int(r["source_rank"]), r["video_id"]))
    return rows, shards


def make_pilot(rows: list[dict], limit: int = 10) -> list[dict]:
    """Deterministic, deliberately varied pilot selection."""
    by_duration = sorted(rows, key=lambda r: (_track_seconds(r),
                                              int(r["source_rank"])))
    if not by_duration:
        return []
    indices = [0, len(by_duration) // 2,
               min(len(by_duration) - 1, int(len(by_duration) * 0.95)),
               len(by_duration) - 1]
    picks: list[dict] = []
    seen: set[str] = set()

    def add(row: dict, reason: str) -> None:
        if row["video_id"] in seen:
            return
        seen.add(row["video_id"])
        picks.append({"reason": reason, **row})

    for idx, reason in zip(indices, ("shortest", "median", "p95", "max")):
        add(by_duration[idx], reason)

    by_video = {r["video_id"]: r for r in rows}
    for vid in PILOT_VIDEO_IDS:
        row = by_video.get(vid)
        if row is not None:
            add(row, "duplicate/live case")

    for predicate, reason in (
        (lambda r: r.get("tier") == "Tier 3", "tier 3"),
        (lambda r: r.get("source_artist") == "Elissa", "artist spread"),
        (lambda r: _track_seconds(r) < 180, "short"),
        (lambda r: "_" in str(r.get("video_id")), "video_id underscore"),
    ):
        row = next((r for r in by_duration if predicate(r)), None)
        if row is not None:
            add(row, reason)

    # Fill if the predicates collapsed onto already-picked rows.
    step = max(1, len(by_duration) // max(limit, 1))
    for row in by_duration[::step]:
        if len(picks) >= limit:
            break
        add(row, "spread")
    return picks[:limit]


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def write_outputs(rows: list[dict], shards: dict[str, list[dict]],
                  manifest: dict, source_manifest_uri: str | None,
                  source_manifest_sha: str, report_sha: str | None,
                  out: Path, version: str, bucket: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dup_report = assign_duplicates(rows)

    counts = {
        "n_source": len(rows),
        "n_transcribed": len(rows),
        "n_shards": len(shards),
        "n_duplicate_groups": len(dup_report),
        "n_duplicate_tracks": sum(len(g["members"]) for g in dup_report),
        "total_audio_seconds": round(sum(_track_seconds(r) for r in rows), 3),
        "total_bytes": sum(int(r["source_bytes"]) for r in rows),
        "unique_video_id": len({r["video_id"] for r in rows}),
        "unique_sha256": len({r["source_sha256"] for r in rows}),
    }
    header = {
        "record": "header",
        "corpus": "yt_top_500_music_playlist",
        "corpus_version": version,
        "created_utc": utcnow(),
        "source_prefix": bucket.rstrip("/"),
        "source_manifest": source_manifest_uri,
        "source_manifest_sha256": source_manifest_sha,
        "report_sha256": report_sha,
        "counts": counts,
    }
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out.parent / "manifest.sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")

    write_json_atomic(out.parent / "duplicates.json", {
        "version": 1,
        "generated_utc": utcnow(),
        "policy": "transcribe all 495; duplicate_of is metadata, not a skip",
        "groups": dup_report,
    })
    write_json_atomic(out.parent / "shards.json", {
        "version": 1,
        "generated_utc": utcnow(),
        "shard_count": len(shards),
        "shards": {
            name: {
                "track_ids": [r["track_id"] for r in members],
                "count": len(members),
                "audio_seconds": round(
                    sum(_track_seconds(r) for r in members), 3),
            }
            for name, members in sorted(shards.items())
        },
    })
    pilot = make_pilot(rows)
    write_json_atomic(out.parent / "pilot.json", {
        "version": 1,
        "generated_utc": utcnow(),
        "note": "10 deliberately varied tracks; download with --download-pilot",
        "tracks": pilot,
    })


def check_manifest(path: Path, expected_tracks: int | None = None,
                   expected_seconds: float | None = None,
                   expected_bytes: int | None = None) -> int:
    if not path.is_file():
        print(f"[FAIL] manifest not found: {path}", file=sys.stderr)
        return 1
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines or lines[0].get("record") != "header":
        print("[FAIL] first line is not a header row", file=sys.stderr)
        return 1
    header, rows = lines[0], lines[1:]
    problems: list[str] = []

    def expect(cond: bool, message: str) -> None:
        if not cond:
            problems.append(message)

    expect(header.get("corpus") == "yt_top_500_music_playlist",
           "header corpus is wrong")
    expect(len(rows) >= 1, "no track rows")
    for field in REQUIRED_TRACK_FIELDS:
        expect(all(field in row for row in rows), f"missing field: {field}")
    for field in ("track_id", "video_id", "source_uri", "output_stem"):
        vals = [str(r.get(field)) for r in rows]
        expect(len(vals) == len(set(vals)), f"duplicate {field}")
    expect(len({r["file"] for r in rows}) == len(rows), "duplicate file")
    expect(all("/" not in str(r["file"]) for r in rows),
           "file contains a path separator")
    expect(all(len(str(r["source_sha256"])) == 64 for r in rows),
           "source_sha256 is not 64 hex chars")
    expect(all(str(r.get("record")) == "track" for r in rows),
           "non-track row in the body")

    shards = collections.Counter(r["shard"] for r in rows)
    expect(len(shards) >= 1, "no shards")
    base, remainder = divmod(len(rows), len(shards))
    allowed_sizes = {base, base + (1 if remainder else 0)}
    expect(all(n in allowed_sizes for n in shards.values()),
           f"unbalanced shards: {sorted(shards.values())} "
           f"(allowed {sorted(allowed_sizes)})")
    expect(len({r["track_id"] for r in rows}) == len(rows),
           "track_id is not unique")

    track_ids = {r["track_id"] for r in rows}
    for row in rows:
        dup = row.get("duplicate_of")
        if dup:
            expect(dup in track_ids, f"{row['track_id']} duplicate_of missing")
        group = row.get("duplicate_group")
        if group:
            expect(sum(1 for x in rows if x.get("duplicate_group") == group) >= 2,
                   f"singleton duplicate_group: {group}")

    total_seconds = sum(float(r["duration_s"]) for r in rows)
    total_bytes = sum(int(r["source_bytes"]) for r in rows)
    expect(len(rows) == header.get("counts", {}).get("n_source"),
           "header n_source does not match rows")
    if expected_tracks is not None:
        expect(len(rows) == expected_tracks,
               f"expected {expected_tracks} tracks, got {len(rows)}")
    if expected_seconds:
        expect(abs(total_seconds - expected_seconds) / expected_seconds < 0.01,
               f"total duration {total_seconds:.1f}s differs from "
               f"{expected_seconds:.1f}s by >1%")
    if expected_bytes:
        expect(abs(total_bytes - expected_bytes) / expected_bytes < 0.01,
               f"total bytes {total_bytes} differs from {expected_bytes} by >1%")

    print(f"manifest : {path}")
    print(f"tracks   : {len(rows)}")
    print(f"shards   : {len(shards)}  sizes={sorted(shards.values())}")
    print(f"audio    : {total_seconds / 3600:.2f} h")
    print(f"bytes    : {total_bytes} ({total_bytes / 2**30:.2f} GiB)")
    print(f"dupes    : {len({r['duplicate_group'] for r in rows if r.get('duplicate_group')})} groups")
    if problems:
        print("\n".join(f"[FAIL] {p}" for p in problems), file=sys.stderr)
        return 1
    print("[OK] manifest checks passed")
    return 0


def download_pilot(pilot_path: Path, dest: Path) -> int:
    data = json.loads(pilot_path.read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    for track in data["tracks"]:
        target = dest / track["file"]
        if target.is_file() and sha256_file(target) == track["source_sha256"]:
            print(f"[skip] {target.name}")
            continue
        proc = run(["gcloud", "storage", "cp", track["source_uri"], str(target)])
        if proc.returncode != 0:
            print(f"[FAIL] {track['source_uri']}: "
                  f"{(proc.stderr or proc.stdout).strip()[-500:]}", file=sys.stderr)
            return 1
        if sha256_file(target) != track["source_sha256"]:
            print(f"[FAIL] sha256 mismatch: {target}", file=sys.stderr)
            return 1
        print(f"[ok] {target.name}")
    print(f"pilot files: {len(data['tracks'])} in {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--bucket-manifest", type=Path, default=None,
                    help="local copy of state/latest/manifest.json")
    ap.add_argument("--report", type=Path, default=None,
                    help="local copy of state/latest/report.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shards", type=int, default=25)
    ap.add_argument("--version", default="2026-11-09")
    ap.add_argument("--check", type=Path, default=None,
                    help="validate an existing manifest.jsonl and exit")
    ap.add_argument("--expected-tracks", type=int, default=495)
    ap.add_argument("--expected-seconds", type=float, default=137991.0)
    ap.add_argument("--expected-bytes", type=int, default=2228785610)
    ap.add_argument("--download-pilot", type=Path, default=None,
                    help="pilot.json to download")
    ap.add_argument("--pilot-dir", type=Path, default=Path("work/yt500/pilot"))
    args = ap.parse_args(argv)

    if args.check is not None:
        return check_manifest(args.check, args.expected_tracks,
                              args.expected_seconds, args.expected_bytes)
    if args.download_pilot is not None:
        return download_pilot(args.download_pilot, args.pilot_dir)

    manifest, source_uri = load_bucket_manifest(args.bucket, args.bucket_manifest)
    report, _report_uri, report_sha = load_report(args.bucket, args.report)
    if source_uri is None:
        source_sha = sha256_file(args.bucket_manifest)  # type: ignore[arg-type]
        source_uri = str(args.bucket_manifest)
    else:
        proc = run(["gcloud", "storage", "cat", source_uri])
        if proc.returncode != 0:
            raise SystemExit(f"cannot hash source manifest: {source_uri}")
        source_sha = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()

    rows, shards = build_rows(manifest, args.shards)
    write_outputs(rows, shards, manifest, source_uri, source_sha, report_sha,
                  args.out, args.version, args.bucket)
    print(f"wrote {args.out} ({len(rows)} tracks, {len(shards)} shards)")
    print(f"duplicates: {args.out.parent / 'duplicates.json'}")
    print(f"shards    : {args.out.parent / 'shards.json'}")
    print(f"pilot     : {args.out.parent / 'pilot.json'}")
    return check_manifest(args.out, args.expected_tracks,
                          args.expected_seconds, args.expected_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
