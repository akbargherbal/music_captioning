#!/usr/bin/env python3
"""
caption.py — Music Flamingo caption script
Model: nvidia/music-flamingo-2601-hf
Phase 4: --file and --dir modes implemented.
Phase 5–6 (--job, summary JSON) not yet implemented.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MAX_TOKENS_CEILING = 512

# Locked in Phase 1 — do not edit without updating the phased plan
MINIMAL_PROMPT = (
    "Listen to this audio track carefully. "
    "List every instrument you can hear. "
    "Then describe the vocal style if vocals are present. "
    "Then estimate the tempo. "
    "Then describe the harmonic or melodic character using interval terms, not genre labels. "
    "Report only what you can directly hear. Do not infer."
)

# Matches the HF Spaces default — Task 3.1 verified string
DEFAULT_PROMPT = (
    "Describe this track in full detail - tell me the genre, tempo, and key, "
    "then dive into the instruments, production style, and overall mood it creates."
)


# ---------------------------------------------------------------------------
# Task 2.2 — Model Loader
# ---------------------------------------------------------------------------

def load_model(model_id: str, precision: str):
    """
    Load MusicFlamingo model and processor.

    Args:
        model_id:  HuggingFace model ID string.
        precision: "bf16" (default) or "4bit".

    Returns:
        (model, processor) tuple — model is in eval mode.

    Raises:
        ValueError: unknown precision value.
    """
    print(f"Loading processor from {model_id!r} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model (precision={precision}) ...")
    if precision == "bf16":
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif precision == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        raise ValueError(
            f"Unknown precision: {precision!r}. Must be 'bf16' or '4bit'."
        )

    model.eval()

    # Note: model.hf_device_map does not exist on this class — use parameter device instead
    device = next(model.parameters()).device
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model ready. Device: {device} | VRAM allocated: {vram_gb:.2f} GB")

    return model, processor


# ---------------------------------------------------------------------------
# Task 2.3 — Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(mode: str, custom_prompt: str = None) -> str:
    """
    Return the prompt string for the given mode.

    Args:
        mode:          "minimal" | "default" | "custom"
        custom_prompt: Required when mode="custom".

    Returns:
        Prompt string.

    Raises:
        ValueError: unknown mode, or custom mode with no prompt supplied.
    """
    if mode == "minimal":
        return MINIMAL_PROMPT
    elif mode == "default":
        return DEFAULT_PROMPT
    elif mode == "custom":
        if not custom_prompt:
            raise ValueError(
                "--custom-prompt is required when --prompt-mode is 'custom'."
            )
        return custom_prompt
    else:
        raise ValueError(
            f"Unknown prompt mode: {mode!r}. Must be 'minimal', 'default', or 'custom'."
        )


# ---------------------------------------------------------------------------
# Task 2.4 — Single Track Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """
    Run blind inference on one audio file.

    The model receives audio only — no filename, no tags, no duration,
    no cultural framing. Context passed to the model is audio + prompt text.

    Args:
        model:          Loaded MusicFlamingo model.
        processor:      Matching AutoProcessor.
        audio_path:     Absolute or relative path to the audio file.
        prompt:         Prompt string (built by build_prompt).
        max_new_tokens: Token budget for generation.

    Returns:
        On success: {"raw_output": str, "processing_time_sec": float, "status": "ok"}
        On failure: {"raw_output": None, "processing_time_sec": None,
                     "status": "error", "error": str}
    """
    try:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }]

        start = time.time()

        batch = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        batch["input_features"] = batch["input_features"].to(model.dtype)

        gen_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
        )

        inp_len = batch["input_ids"].shape[1]
        raw_output = processor.batch_decode(
            gen_ids[:, inp_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.time() - start
        return {
            "raw_output": raw_output,
            "processing_time_sec": round(elapsed, 1),
            "status": "ok",
            "error": None,
        }

    except Exception as e:
        return {
            "raw_output": None,
            "processing_time_sec": None,
            "status": "error",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Task 2.5 — Output Writer
# ---------------------------------------------------------------------------

def write_track_output(
    result: dict,
    audio_path: str,
    prompt_used: str,
    prompt_mode: str,
    tags: dict,
    model_id: str,
    output_dir: str,
) -> str:
    """
    Write <output_dir>/<trackname>.caption.json.

    JSON schema is fixed — no extra fields added here.
    output_dir is created if it does not exist.
    Existing files are overwritten silently.

    Returns:
        Path of the file written.
    """
    os.makedirs(output_dir, exist_ok=True)

    track_name = Path(audio_path).name
    out_name   = Path(audio_path).stem + ".caption.json"
    out_path   = os.path.join(output_dir, out_name)

    payload = {
        "file":                track_name,
        "model_id":            model_id,
        "prompt_mode":         prompt_mode,
        "prompt_used":         prompt_used,
        "raw_output":          result.get("raw_output"),
        "processing_time_sec": result.get("processing_time_sec"),
        "tags":                tags if tags is not None else {},
        "status":              result.get("status"),
        "error":               result.get("error"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# Task 4.1 — Audio File Discovery
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac"}


def discover_audio_files(directory: str) -> list:
    """
    Recursively discover audio files under directory.

    Walks the full tree; silently skips anything that is not .mp3, .wav,
    or .flac (including hidden files, .DS_Store, .txt, .md, etc.).

    Args:
        directory: Path to the folder to scan.

    Returns:
        Sorted list of absolute path strings.

    Raises:
        SystemExit: directory does not exist or is not a directory.
    """
    if not os.path.isdir(directory):
        sys.exit(f"Error: directory not found: {directory!r}")

    found = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(os.path.abspath(os.path.join(root, fname)))

    found.sort()
    print(f"Found {len(found)} audio file(s) in {directory!r}")
    return found


# ---------------------------------------------------------------------------
# Task 2.1 — CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description=(
            "Blind audio analysis using nvidia/music-flamingo-2601-hf. "
            "Phase 4: --file and --dir modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="Path to a single audio file (MP3, WAV, or FLAC).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        metavar="DIR",
        help="Directory to scan recursively for audio files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        dest="max_tokens",
        metavar="N",
        help="Maximum new tokens to generate (default: 256, ceiling: 512).",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["minimal", "default", "custom"],
        default="minimal",
        dest="prompt_mode",
        help="Prompt strategy: minimal (default), default, or custom.",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        dest="custom_prompt",
        metavar="STRING",
        help="Prompt string. Required when --prompt-mode is 'custom'.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "4bit"],
        default="bf16",
        help="Model precision (default: bf16).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./captions/",
        dest="output_dir",
        metavar="DIR",
        help="Directory for output files (default: ./captions/).",
    )

    return parser


# ---------------------------------------------------------------------------
# Task 2.6 — main() — single file mode wiring
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Shared pre-flight validation ---

    # Exactly one entry point must be supplied
    entry_points = [ep for ep in (args.file, args.dir) if ep]
    if len(entry_points) == 0:
        parser.error(
            "No entry point specified. "
            "Use --file <path> or --dir <directory>. "
            "(--job is not yet implemented — Phase 5.)"
        )
    if len(entry_points) > 1:
        parser.error("Specify only one of --file or --dir, not both.")

    # Clamp max_tokens to ceiling
    if args.max_tokens > MAX_TOKENS_CEILING:
        print(
            f"Warning: --max-tokens {args.max_tokens} exceeds ceiling "
            f"of {MAX_TOKENS_CEILING}. Clamping."
        )
        args.max_tokens = MAX_TOKENS_CEILING

    # Validate custom prompt supplied when required
    if args.prompt_mode == "custom" and not args.custom_prompt:
        parser.error("--custom-prompt is required when --prompt-mode is 'custom'.")

    # --- Load model once, regardless of mode ---
    model, processor = load_model(MODEL_ID, args.precision)

    # --- Build prompt (same for all tracks in this invocation) ---
    prompt = build_prompt(args.prompt_mode, args.custom_prompt)

    # -----------------------------------------------------------------------
    # Task 2.6 — --file mode
    # -----------------------------------------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            sys.exit(f"Error: file not found: {args.file!r}")

        print(f"\nProcessing: {args.file}")
        result = run_inference(model, processor, args.file, prompt, args.max_tokens)

        out_path = write_track_output(
            result=result,
            audio_path=args.file,
            prompt_used=prompt,
            prompt_mode=args.prompt_mode,
            tags={},
            model_id=MODEL_ID,
            output_dir=args.output_dir,
        )

        if result["status"] == "ok":
            print(f"OK ({result['processing_time_sec']}s) → {out_path}")
        else:
            print(f"ERROR → {out_path}")
            print(f"  {result.get('error')}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Task 4.2 — --dir mode
    # -----------------------------------------------------------------------
    elif args.dir:
        audio_files = discover_audio_files(args.dir)

        if not audio_files:
            print("No audio files found. Nothing to do.")
            return

        total    = len(audio_files)
        failures = []

        for idx, audio_path in enumerate(audio_files, start=1):
            label = f"[{idx}/{total}] {Path(audio_path).name}"
            print(f"\n{label}")

            result = run_inference(
                model, processor, audio_path, prompt, args.max_tokens
            )

            out_path = write_track_output(
                result=result,
                audio_path=audio_path,
                prompt_used=prompt,
                prompt_mode=args.prompt_mode,
                tags={},
                model_id=MODEL_ID,
                output_dir=args.output_dir,
            )

            if result["status"] == "ok":
                print(f"  OK ({result['processing_time_sec']}s) → {out_path}")
            else:
                print(f"  ERROR → {out_path}")
                print(f"    {result.get('error')}")
                failures.append({"file": audio_path, "error": result.get("error")})
            # Always continue — never abort batch for one bad file

        print(
            f"\nDone. Processed: {total} | "
            f"OK: {total - len(failures)} | "
            f"Failed: {len(failures)}"
        )
        if failures:
            for f in failures:
                print(f"  FAILED: {f['file']} — {f['error']}")


if __name__ == "__main__":
    main()
