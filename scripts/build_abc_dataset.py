#!/usr/bin/env python3
"""Join packed ABC + existing captions into train/val JSONL (§10.2).

CPU only, no model, no network.

Split rules, in order:

1. **Album level.** A row's split is its album's split; tracks are never split
   individually. The fixed human-chosen val albums are late-era ELISSA and the
   Fadl Shaker compilation.
2. **Dedupe groups stay together.** Albums connected by a duplicate (same
   ``audio_sha256``, else the packer's ``duplicate_of``) are unioned. If any
   album in a component is a val album, the whole component is val. This is
   deliberately stricter than "put the val album in val": it is the only way
   the studio/live/compilation copies of one song cannot straddle a split.
3. **Quarantine.** ``qc.parse=fail`` rows are written with
   ``"split": "quarantine"``, never dropped.

Rows are never silently dropped for being duplicates: each keeps
``duplicate_of`` and ``dedupe_group`` so a downstream view can collapse them.
``lyrics_eligible=false`` rows stay in the ABC-conditioned dataset.

Usage:
    python3 scripts/build_abc_dataset.py --outputs outputs --out datasets/elissa_fadl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from abcscore import normalize_title  # noqa: E402
from src.tags import TAGS_EXPERIMENTAL  # noqa: E402

DEFAULT_VAL_ALBUMS = (
    "ELISSA/Elissa - 2012  As3ad Wa7da",  # note the double space
    "FADL_SHAKER/15.The Best Of (Al Rawae3).2010",
)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["track_id"]] = row
    return out


def load_packed(outputs: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(outputs.glob("*/abc/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for track in data.get("tracks", []):
            rows.append({**track, "artist": data["artist"],
                         "album": data["album"]})
    return rows


def load_captions(outputs: Path) -> dict[tuple[str, str, str], dict]:
    """``(artist, album, file) -> {"status", "row"}``."""
    out: dict[tuple[str, str, str], dict] = {}
    for path in sorted(outputs.glob("*/*.json")):
        if path.parent.name == "abc":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "album" not in data:
            continue
        artist, album = data.get("artist"), data["album"]
        for track in data.get("tracks", []):
            out[(artist, album, track["file"])] = {
                "status": "ok", "row": track}
        for track in data.get("skipped_tracks", []):
            out[(artist, album, track["file"])] = {
                "status": "skipped_long", "row": None}
    return out


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def dedupe_keys(rows: list[dict]) -> dict[str, str]:
    """``track_id -> group id`` for hash duplicates and ``duplicate_of`` links."""
    ids = {row["track_id"] for row in rows}
    uf = UnionFind()
    for track_id in ids:
        uf.find(track_id)
    by_hash: dict[str, str] = {}
    for row in rows:
        track_id = row["track_id"]
        digest = row.get("audio_sha256")
        if digest:
            if digest in by_hash:
                uf.union(by_hash[digest], track_id)
            else:
                by_hash[digest] = track_id
        target = row.get("duplicate_of")
        if target in ids:
            uf.union(target, track_id)
    return {track_id: uf.find(track_id) for track_id in ids}


def assign_splits(rows: list[dict], val_albums=set(DEFAULT_VAL_ALBUMS)) -> dict[str, str]:
    """Return ``track_id -> split`` under the album + dedupe rules."""
    if not rows:
        return {}
    album_of = {row["track_id"]: f"{row['artist']}/{row['album']}" for row in rows}
    uf = UnionFind()
    for album in album_of.values():
        uf.find(album)

    # Connect albums that share a duplicate group.
    groups: dict[str, list[str]] = defaultdict(list)
    for track_id, group in dedupe_keys(rows).items():
        groups[group].append(track_id)
    for members in groups.values():
        albums = {album_of[mid] for mid in members}
        first = next(iter(albums))
        for album in albums:
            uf.union(first, album)

    val_roots = {uf.find(a) for a in val_albums if a in uf.parent}
    splits: dict[str, str] = {}
    for row in rows:
        album = album_of[row["track_id"]]
        split = "val" if uf.find(album) in val_roots else "train"
        if row.get("qc", {}).get("parse") == "fail":
            split = "quarantine"
        splits[row["track_id"]] = split
    return splits


def dedupe_groups(rows: list[dict]) -> dict[str, list[str]]:
    """``group -> track_ids`` for groups with more than one member."""
    groups: dict[str, list[str]] = defaultdict(list)
    for track_id, key in dedupe_keys(rows).items():
        groups[key].append(track_id)
    return {k: v for k, v in groups.items() if len(v) > 1}


def build_rows(outputs: Path, manifest: dict[str, dict],
               val_albums=set(DEFAULT_VAL_ALBUMS),
               captions_root: Path | None = None) -> tuple[list[dict], dict]:
    packed = load_packed(outputs)
    captions = load_captions(captions_root or outputs)
    splits = assign_splits(packed, val_albums)
    groups = dedupe_keys(packed)
    multi_groups = dedupe_groups(packed)

    rows: list[dict] = []
    for track in packed:
        mrow = manifest.get(track["track_id"], {})
        cap = captions.get((track["artist"], track["album"], track["file"]), {})
        cap_row = cap.get("row") or {}
        caption = cap_row.get("caption")
        flags = list(track.get("qc", {}).get("flags", []))
        if cap.get("status") == "skipped_long":
            caption = None
            flags.append("caption_skipped_long")
        elif not cap:
            flags.append("no_caption_row")

        rows.append({
            "track_id": track["track_id"],
            "artist": track["artist"],
            "album": track["album"],
            "title": track.get("title"),
            "duration_s": track.get("duration_s"),
            "audio_sha256": track.get("audio_sha256"),
            "audio_uri": mrow.get("source_uri") or track.get("source_uri"),
            "caption": caption,
            "caption_status": cap.get("status"),
            "tags_experimental": TAGS_EXPERIMENTAL,
            "instruments": cap_row.get("instruments", []),
            "vocals": cap_row.get("vocals", []),
            "production": cap_row.get("production", []),
            "abc": track.get("abc"),
            "abc_sha256": track.get("abc_sha256"),
            "abc_melody": track.get("abc_melody"),
            "abc_melody_sha256": track.get("abc_melody_sha256"),
            "has_chords": bool(track.get("has_chords")),
            "instrumental": bool(track.get("instrumental")),
            "lyrics_eligible": bool(track.get("lyrics_eligible",
                                              not track.get("instrumental"))),
            "lyrics": None,          # §11 deferred
            "lyrics_source": None,
            "split": splits[track["track_id"]],
            "qc_parse": track.get("qc", {}).get("parse"),
            "qc_flags": sorted(set(flags)),
            "duplicate_of": track.get("duplicate_of"),
            "dedupe_group": groups[track["track_id"]],
        })
    rows.sort(key=lambda r: (r["artist"], r["album"],
                             r.get("title") or "", r["track_id"]))

    stats = {
        "packed_tracks": len(packed),
        "rows_written": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_artist": dict(Counter(r["artist"] for r in rows)),
        "with_caption": sum(r["caption"] is not None for r in rows),
        "with_chords": sum(r["has_chords"] for r in rows),
        "with_lyrics": sum(r["lyrics"] is not None for r in rows),
        "lyrics_eligible": sum(r["lyrics_eligible"] for r in rows),
        "abc_bytes_median": (statistics.median(
            len(r["abc"].encode("utf-8")) for r in rows if r["abc"])
            if any(r["abc"] for r in rows) else 0),
        "dedupe_groups": len(multi_groups),
        "dedupe_rows_in_groups": sum(len(v) for v in multi_groups.values()),
        "flag_histogram": dict(Counter(f for r in rows for f in r["qc_flags"])),
    }
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "work" / "manifest.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "datasets" / "elissa_fadl")
    ap.add_argument("--captions-root", type=Path, default=None,
                    help="where the existing caption JSONs live "
                         "(default: --outputs)")
    ap.add_argument("--val-album", action="append", default=None,
                    help="ARTIST/album held out for val (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    val_albums = set(args.val_album or DEFAULT_VAL_ALBUMS)
    manifest = load_manifest(args.manifest)
    rows, stats = build_rows(args.outputs, manifest, val_albums,
                             args.captions_root)

    if not args.dry_run:
        for split in ("train", "val", "quarantine"):
            atomic_write_jsonl(args.out / f"{split}.jsonl",
                               [r for r in rows if r["split"] == split])
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not rows:
        print("no packed ABC rows found; run abc_pack.py first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
