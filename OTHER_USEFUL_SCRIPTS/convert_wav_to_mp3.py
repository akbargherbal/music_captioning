#!/usr/bin/env python3
"""
convert_wav_to_mp3.py

Convert a single .wav file or a directory of .wav files to .mp3.

Defaults mirror Audacity's standard export settings:
  - Bit rate mode : Constant (CBR)
  - Bit rate       : 192 kbps  (good quality, reasonable size)
  - Sample rate    : preserved from source
  - Channels       : preserved from source

Requirements:
    pip install pydub
    # plus ffmpeg on PATH:
    #   macOS  → brew install ffmpeg
    #   Ubuntu → sudo apt install ffmpeg
    #   Windows→ https://ffmpeg.org/download.html  (add bin/ to PATH)

Usage:
    # Single file → same directory
    python convert_wav_to_mp3.py audio.wav

    # Single file → custom output path
    python convert_wav_to_mp3.py audio.wav -o output.mp3

    # Whole directory → same directory (alongside originals)
    python convert_wav_to_mp3.py ./wavs/

    # Whole directory → separate output directory
    python convert_wav_to_mp3.py ./wavs/ -o ./mp3s/

    # Custom bit rate
    python convert_wav_to_mp3.py ./wavs/ --bitrate 320k
"""

import argparse
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit(
        "pydub is not installed. Run:  pip install pydub\n"
        "You also need ffmpeg on your PATH."
    )


# ---------------------------------------------------------------------------
# Defaults (Audacity "standard export" equivalent)
# ---------------------------------------------------------------------------
DEFAULT_BITRATE = "192k"   # CBR 192 kbps — transparent for most content
DEFAULT_FORMAT  = "mp3"


def convert_file(
    src: Path,
    dst: Path,
    bitrate: str = DEFAULT_BITRATE,
) -> None:
    """Convert a single WAV file to MP3."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_wav(str(src))

    audio.export(
        str(dst),
        format=DEFAULT_FORMAT,
        bitrate=bitrate,
        tags={},          # strip embedded metadata; remove line to preserve
        parameters=[
            "-q:a", "0",  # highest VBR quality (only matters in VBR mode)
        ],
    )
    print(f"  ✓  {src}  →  {dst}  [{bitrate} CBR]")


def resolve_output_path(src: Path, output: Path | None, is_dir_mode: bool) -> Path:
    """
    Work out where the converted file should land.

    - Single-file mode, no -o  → same dir as source, .mp3 extension
    - Single-file mode, -o dir → that dir / same stem .mp3
    - Single-file mode, -o file→ exactly that path
    - Dir mode, no -o          → alongside original
    - Dir mode, -o dir         → mirror into that dir
    """
    stem = src.stem + ".mp3"

    if output is None:
        return src.with_suffix(".mp3")

    if output.is_dir() or (is_dir_mode and not output.suffix):
        return output / stem

    # Explicit file path (single-file mode only)
    return output


def process_directory(
    src_dir: Path,
    output: Path | None,
    bitrate: str,
    recursive: bool,
) -> int:
    """Walk a directory and convert every .wav found."""
    pattern = "**/*.wav" if recursive else "*.wav"
    wav_files = sorted(src_dir.glob(pattern))

    if not wav_files:
        print(f"No .wav files found in {src_dir}")
        return 0

    print(f"Found {len(wav_files)} .wav file(s) in '{src_dir}'")

    converted = 0
    errors    = 0

    for wav in wav_files:
        # Preserve sub-directory structure when output dir is given
        if output:
            relative = wav.relative_to(src_dir)
            dst = (output / relative).with_suffix(".mp3")
        else:
            dst = wav.with_suffix(".mp3")

        try:
            convert_file(wav, dst, bitrate)
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗  {wav}  →  ERROR: {exc}")
            errors += 1

    print(f"\nDone — {converted} converted, {errors} failed.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WAV file(s) to MP3 (Audacity-style defaults).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a .wav file or a directory containing .wav files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Output file (single-file mode) or output directory (dir mode). "
            "Defaults to the same location as the source."
        ),
    )
    parser.add_argument(
        "--bitrate",
        default=DEFAULT_BITRATE,
        metavar="RATE",
        help=(
            "MP3 bit rate, e.g. 128k / 192k / 256k / 320k. "
            f"Default: {DEFAULT_BITRATE}"
        ),
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recurse into sub-directories (directory mode only).",
    )

    args = parser.parse_args()
    src: Path = args.source.expanduser().resolve()

    if not src.exists():
        sys.exit(f"Error: '{src}' does not exist.")

    # ── Directory mode ──────────────────────────────────────────────────────
    if src.is_dir():
        out_dir: Path | None = None
        if args.output:
            out_dir = args.output.expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

        exit_code = process_directory(src, out_dir, args.bitrate, args.recursive)
        sys.exit(exit_code)

    # ── Single-file mode ────────────────────────────────────────────────────
    if src.suffix.lower() != ".wav":
        sys.exit(f"Error: '{src}' is not a .wav file.")

    dst = resolve_output_path(src, args.output, is_dir_mode=False)

    try:
        convert_file(src, dst, args.bitrate)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Conversion failed: {exc}")


if __name__ == "__main__":
    main()
