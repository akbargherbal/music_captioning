#!/usr/bin/env python3
"""Rewrite repeated chunk captions into one description with a small LLM.

Input is either a ``demo.py --keep-chunk-captions`` JSON or a plain text file
whose paragraphs are the per-chunk captions (blank line separated). The model
receives only the raw text: no JSON, no field names, no metadata. Each call is
stateless (fresh system+user messages, no history).

Usage:
    python scripts/rewrite_demo.py captions.json
    python scripts/rewrite_demo.py captions.txt --quantization 4bit
    python scripts/rewrite_demo.py captions.json --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_sections(path: Path) -> list[str]:
    """Chunk captions from a demo JSON, or blank-line-separated text."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        sections = data.get("chunk_captions") if isinstance(data, dict) else None
        if not sections:
            raise SystemExit(
                f"{path} has no chunk_captions; re-run demo.py with "
                "--keep-chunk-captions"
            )
        return [str(s) for s in sections]
    return [part.strip() for part in raw.split("\n\n") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="demo JSON or section text file")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B-Instruct-2507",
        help="small instruct model used only for rewriting",
    )
    parser.add_argument(
        "--quantization", choices=("8bit", "4bit", "none"), default="8bit",
        help="8-bit is the default: 4-bit raises hallucination slightly",
    )
    parser.add_argument(
        "--backend", choices=("hf", "llamacpp"), default="hf",
        help="hf = transformers (default); llamacpp = GGUF under llama-server",
    )
    parser.add_argument(
        "--gguf", type=Path, default=None,
        help="GGUF weights for --backend llamacpp",
    )
    parser.add_argument(
        "--server-bin", type=Path,
        default=REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server",
        help="llama-server binary staged by plumbing/fetch_llamacpp_text.sh",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--show-source", action="store_true", help="print the raw chunk text too"
    )
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from src.rewrite import build_source_text, rewrite_caption

    sections = load_sections(args.input)
    source = build_source_text(sections)
    print(f"Sections: {len(sections)} | source words: {len(source.split())}")

    if args.backend == "llamacpp":
        if args.gguf is None:
            raise SystemExit("--backend llamacpp needs --gguf PATH")
        from src.rewriter_llama import LlamaCppRewriter

        rewriter = LlamaCppRewriter(
            model_path=args.gguf,
            server_bin=args.server_bin,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        from src.rewriter_hf import HFRewriter

        rewriter = HFRewriter(
            model_id=args.model,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
        )
    result = rewrite_caption(sections, rewriter)

    print(f"Status: {result.status} | model: {result.rewriter} | "
          f"prompt v{result.prompt_version}")
    if result.reasons:
        print("Rejected:", "; ".join(result.reasons))
    print(f"Output words: {result.output_words}")
    if rewriter.last_run:
        run = rewriter.last_run
        print(f"Time: {run.get('seconds')} s | quantization: {run.get('quantization')}")

    if args.show_source:
        print("\n--- source ---\n" + source)
    print("\n--- rewrite ---\n" + result.text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.text, encoding="utf-8")
        print(f"\nSaved: {args.out}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
