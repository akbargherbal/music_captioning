#!/usr/bin/env python3
"""
Build a music-flamingo captioning input JSON from a given audio directory.

Usage:
    python build_captions_json.py <audio_dir> [options]

Options:
    -o, --output <file>      Write JSON to file instead of stdout
    -r, --recursive          Recurse into subdirectories
    --base-dir <dir>         Root for relative track paths (default: parent of audio_dir)
    --no-tags                Skip mutagen entirely; use filename parsing only

Examples:
    # Flat album directory, output to file
    python build_captions_json.py "FADL_SHAKER/15.The Best Of" -o output.json

    # Full discography, recurse into all albums
    python build_captions_json.py "FADL_SHAKER" -r -o output.json

    # Custom base dir so paths read as "MUSIC/FADL_SHAKER/..."
    python build_captions_json.py "FADL_SHAKER" --base-dir "MUSIC" -o output.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".wav", ".wma", ".opus", ".ape", ".alac",
}

GLOBALS = {
    "model_id": "nvidia/music-flamingo-2601-hf",
    "precision": "bf16",
    "max_new_tokens": 512,
    "output_dir": "./captions/fadl_shaker_512/",
}


# ── Tag extraction ────────────────────────────────────────────────────────────

def get_tags_mutagen(path: Path) -> dict:
    """
    Extract artist / title / track_number via mutagen (easy=True interface).
    Returns an empty dict if mutagen is not installed or the file has no tags.
    """
    try:
        from mutagen import File as MutagenFile  # type: ignore
    except ImportError:
        return {}

    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception:
        return {}

    if audio is None:
        return {}

    tags: dict = {}

    def first(key):
        val = audio.get(key)
        return str(val[0]) if val else None

    artist = first("artist")
    if artist:
        tags["artist"] = artist

    title = first("title")
    if title:
        tags["title"] = title

    tracknumber = first("tracknumber")
    if tracknumber:
        # May arrive as "3/12" — keep only the left part
        num_str = tracknumber.split("/")[0].strip()
        try:
            tags["track_number"] = int(num_str)
        except ValueError:
            pass

    return tags


def parse_filename(stem: str) -> dict:
    """
    Best-effort tag extraction from the filename stem alone.
    Handles common patterns:
        "03.Song Title"  →  track 3,  title "Song Title"
        "03 - Song Title"
        "03_Song Title"
        "Song Title"     →  title "Song Title" (no track number)
    """
    tags: dict = {}
    m = re.match(r"^(\d+)[.\s_-]+(.+)$", stem)
    if m:
        try:
            tags["track_number"] = int(m.group(1))
        except ValueError:
            pass
        tags["title"] = m.group(2).strip()
    else:
        tags["title"] = stem
    return tags


def build_tags(path: Path, use_mutagen: bool) -> dict:
    """
    Merge mutagen tags (preferred) with filename-derived fallbacks.
    Only includes keys that could actually be resolved.
    """
    tags: dict = get_tags_mutagen(path) if use_mutagen else {}

    # Fill in anything mutagen didn't provide from the filename
    fn_tags = parse_filename(path.stem)
    for key, val in fn_tags.items():
        if key not in tags:
            tags[key] = val

    return tags


# ── Directory walk ────────────────────────────────────────────────────────────

def collect_tracks(audio_dir: Path, recursive: bool, base_dir: Path, use_mutagen: bool) -> list:
    """Return a sorted list of track dicts for every audio file found."""
    if recursive:
        candidates = sorted(audio_dir.rglob("*"))
    else:
        candidates = sorted(audio_dir.iterdir())

    tracks = []
    for f in candidates:
        if not f.is_file():
            continue
        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        rel_path = f.relative_to(base_dir)
        tags = build_tags(f, use_mutagen)

        tracks.append({
            "path": rel_path.as_posix(),  # forward slashes on all platforms
            "tags": tags,
        })

    return tracks


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a music-flamingo captioning JSON from an audio directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recurse into subdirectories")
    parser.add_argument("--base-dir", default=None,
                        help="Root directory for relative track paths "
                             "(default: parent of audio_dir)")
    parser.add_argument("--no-tags", action="store_true",
                        help="Skip mutagen; derive tags from filenames only")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.is_dir():
        sys.exit(f"Error: '{audio_dir}' is not a directory.")

    base_dir = Path(args.base_dir).resolve() if args.base_dir else audio_dir.parent

    # Warn if mutagen is requested but unavailable
    use_mutagen = not args.no_tags
    if use_mutagen:
        try:
            import mutagen  # noqa: F401
        except ImportError:
            print(
                "Warning: mutagen not installed — falling back to filename parsing.\n"
                "         Install with:  pip install mutagen",
                file=sys.stderr,
            )
            use_mutagen = False

    tracks = collect_tracks(audio_dir, args.recursive, base_dir, use_mutagen)

    if not tracks:
        print(f"Warning: no audio files found in '{audio_dir}'.", file=sys.stderr)

    result = {"globals": GLOBALS, "tracks": tracks}
    out_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out_json, encoding="utf-8")
        print(f"Written {len(tracks)} track(s) → {args.output}", file=sys.stderr)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
