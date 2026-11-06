#!/usr/bin/env python3
"""Benchmark captioning time (and VRAM) for one file or a whole directory.

Usage:
    python scripts/benchmark.py "/content/Elissa - 2008 Ayami Beek" --out outputs/Elissa_2008_Ayami_Beek.json

The model is loaded once and reused for every track. For each track we record
duration, chunk count, per-chunk seconds, caption time, and peak VRAM, then
write one combined JSON plus an album-level `realtime_factor`
(caption seconds / audio seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
DEFAULT_HF_MODEL = REPO_ROOT / "models" / "Ace-Step-Captioner-4bit"


def collect_audio(target: Path) -> list[Path]:
    """Return the audio files for a file or directory (sorted, non-recursive)."""
    if target.is_dir():
        return sorted(
            p for p in target.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES
        )
    if target.is_file():
        return [target]
    raise FileNotFoundError(f"input not found: {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="audio file or directory of audio")
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL)
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    import torch
    from src.captioner import _split_audio, audio_duration
    from src.captioner_hf import HFCaptioner
    from src.tags import TAGS_EXPERIMENTAL, extract_tags, merge_tags

    tracks = collect_audio(args.input)
    if not tracks:
        raise SystemExit(f"no audio files found in {args.input}")
    print(f"Tracks: {len(tracks)}")

    captioner = HFCaptioner(
        model_dir=args.hf_model, chunk_seconds=args.chunk_seconds
    )
    t0 = time.time()
    captioner._load()
    load_s = time.time() - t0
    print(f"MODEL LOAD {load_s:.1f}s")

    results: list[dict[str, object]] = []
    bench_t0 = time.time()
    for i, track in enumerate(tracks, 1):
        entry: dict[str, object] = {"file": track.name}
        try:
            duration = audio_duration(track)
            torch.cuda.reset_peak_memory_stats()
            with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
                chunks = _split_audio(track, Path(tmp), args.chunk_seconds)
                captions, chunk_times = [], []
                for chunk in chunks:
                    tc = time.time()
                    captions.append(captioner.caption_chunk(chunk))
                    chunk_times.append(time.time() - tc)
            merged = merge_tags([extract_tags(c) for c in captions])
            entry.update({
                "duration_s": round(duration, 1),
                "chunks": len(captions),
                "chunk_seconds": [round(x, 1) for x in chunk_times],
                "caption_seconds": round(sum(chunk_times), 1),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "instruments": merged["instruments"],
                "vocals": merged["vocals"],
                "production": merged["production"],
                "tags_experimental": TAGS_EXPERIMENTAL,
                "caption": " ".join(c.strip() for c in captions),
            })
            print(f"[{i}/{len(tracks)}] {track.name}: {entry['caption_seconds']}s "
                  f"({entry['chunks']} chunks, audio {entry['duration_s']}s)")
        except Exception as exc:  # keep going if one track fails
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i}/{len(tracks)}] {track.name}: ERROR {entry['error']}")
            traceback.print_exc()
        results.append(entry)

    wall = time.time() - bench_t0
    total_audio = sum(r.get("duration_s", 0) for r in results)
    total_caption = sum(r.get("caption_seconds", 0) for r in results)
    output = {
        "album": args.input.name,
        "source_dir": str(args.input),
        "model_dir": str(args.hf_model),
        "chunk_seconds": args.chunk_seconds,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_seconds": round(load_s, 1),
        "track_count": len(results),
        "total_audio_seconds": round(total_audio, 1),
        "total_caption_seconds": round(total_caption, 1),
        "total_wall_seconds": round(wall, 1),
        "realtime_factor": round(total_caption / total_audio, 3) if total_audio else None,
        "peak_vram_mib": max(
            (r.get("peak_vram_mib", 0) for r in results), default=0
        ),
        "tracks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWROTE {args.out}")
    print(f"TOTAL audio {total_audio:.1f}s | caption {total_caption:.1f}s | "
          f"wall {wall:.1f}s | RTF {output['realtime_factor']} | "
          f"peak {output['peak_vram_mib']} MiB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
