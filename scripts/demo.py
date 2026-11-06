#!/usr/bin/env python3
"""End-to-end demo: audio -> caption -> experimental term lists.

Usage:
    python scripts/demo.py /path/to/song.mp3            # transformers 4-bit (default)
    python scripts/demo.py /path/to/song.mp3 --backend llamacpp

The deterministic tags are **experimental** (see docs/TAG_AUDIT.md). Pass
``--experimental-suno`` to additionally print/save the unsupported prompt string.

The hf backend needs models/Ace-Step-Captioner-4bit; the llamacpp backend needs
the built llama-cli and the two GGUF files (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CLI = REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
DEFAULT_MODEL = MODELS_DIR / "acestep-captioner-Q4_K_M.gguf"
DEFAULT_MMPROJ = MODELS_DIR / "acestep-captioner-mmproj-Q8_0.gguf"
DEFAULT_HF_MODEL = MODELS_DIR / "Ace-Step-Captioner-4bit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="path to an audio file")
    parser.add_argument(
        "--backend", choices=("hf", "llamacpp"), default="hf",
        help="hf = transformers 4-bit (default); llamacpp = GGUF build",
    )
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    parser.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL)
    parser.add_argument(
        "--chunk-seconds", type=float, default=None,
        help="target chunk length (default: 90 for hf, 60 for llamacpp); chunks "
             "are rebalanced so there is never a tiny final one",
    )
    parser.add_argument(
        "--no-trim-silence", action="store_true",
        help="keep leading/trailing silence instead of trimming it before chunking",
    )
    parser.add_argument(
        "--json", action="store_true", help="print only the JSON result"
    )
    parser.add_argument(
        "--keep-chunk-captions", action="store_true",
        help="keep the per-chunk captions in the JSON (for rewrite experiments)",
    )
    parser.add_argument(
        "--experimental-suno", action="store_true",
        help="also emit the unsupported suno_prompt string (see docs/TAG_AUDIT.md)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="write the JSON result to this path (parent dirs are created)",
    )
    args = parser.parse_args(argv)

    chunk_args = (
        {} if args.chunk_seconds is None else {"chunk_seconds": args.chunk_seconds}
    )
    if args.backend == "hf":
        from src.captioner_hf import HFCaptioner, caption_and_tag
        captioner = HFCaptioner(
            model_dir=args.hf_model,
            trim_silence=not args.no_trim_silence,
            **chunk_args,
        )
    else:
        from src.captioner import Captioner, caption_and_tag
        captioner = Captioner(
            llama_cli=args.llama_cli, model=args.model, mmproj=args.mmproj,
            trim_silence=not args.no_trim_silence, **chunk_args,
        )

    result = caption_and_tag(
        captioner, args.audio, experimental_suno=args.experimental_suno
    )
    if not args.keep_chunk_captions:
        result.pop("chunk_captions", None)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved: {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Caption:\n" + str(result["caption"]) + "\n")
        print("Experimental tags (not a finished prompt):")
        print(json.dumps(
            {k: result[k] for k in ("instruments", "vocals", "production")},
            ensure_ascii=False,
            indent=2,
        ))
        if args.experimental_suno:
            print("\nExperimental Suno prompt (unsupported; see docs/TAG_AUDIT.md):")
            print(str(result["suno_prompt"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
