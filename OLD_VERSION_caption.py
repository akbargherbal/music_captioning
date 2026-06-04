# OLD VERSION
#!/usr/bin/env python3
"""
caption.py — Music Flamingo caption script
Model: nvidia/music-flamingo-2601-hf
Phases 2–6 complete: --file, --dir, and --job modes implemented.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MAX_TOKENS_CEILING = 512

# Script-level defaults — lowest precedence in all override chains.
# Phase 5 (Task 5.1) uses these as the base for JSON job merging.
SCRIPT_DEFAULTS = {
    "model_id": MODEL_ID,
    "precision": "bf16",
    "max_new_tokens": 256,
    "prompt_mode": "minimal",
    "custom_prompt": None,
    "output_dir": "./captions/",
}

# Locked in Phase 1 — do not edit without updating the phased plan.
MINIMAL_PROMPT = (
    "Listen to this audio track carefully. "
    "List every instrument you can hear. "
    "Then describe the vocal style if vocals are present. "
    "Then estimate the tempo. "
    "Then describe the harmonic or melodic character using interval terms, not genre labels. "
    "Report only what you can directly hear. Do not infer."
)

# Matches the HF Spaces default — Task 3.1 verified string.
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
        raise ValueError(f"Unknown precision: {precision!r}. Must be 'bf16' or '4bit'.")

    model.eval()

    # Note: model.hf_device_map does not exist on this class — use parameter device instead.
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
        On success: {"raw_output": str, "processing_time_sec": float,
                     "status": "ok", "error": None}
        On failure: {"raw_output": None, "processing_time_sec": None,
                     "status": "error", "error": str}
    """
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

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
    out_name = Path(audio_path).stem + ".caption.json"
    out_path = os.path.join(output_dir, out_name)

    payload = {
        "file": track_name,
        "model_id": model_id,
        "prompt_mode": prompt_mode,
        "prompt_used": prompt_used,
        "raw_output": result.get("raw_output"),
        "processing_time_sec": result.get("processing_time_sec"),
        "tags": tags if tags is not None else {},
        "status": result.get("status"),
        "error": result.get("error"),
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
# Task 5.1 — JSON Job Parser  [Phase 5]
# ---------------------------------------------------------------------------


def parse_job_file(job_path: str) -> list[dict]:
    """
    Read a JSON job file and return a list of fully resolved track dicts.

    Override precedence (low → high):
        script defaults → globals block → per-track entry

    Each returned dict contains:
        path, model_id, precision, max_new_tokens,
        prompt_mode, custom_prompt, output_dir, tags

    Args:
        job_path: Path to the JSON job file.

    Returns:
        List of resolved track dicts, one per entry in the tracks array.

    Raises:
        FileNotFoundError:  job file does not exist.
        json.JSONDecodeError: job file is not valid JSON.
        ValueError:         tracks list is empty, or a track is missing 'path'.
    """
    if not os.path.isfile(job_path):
        raise FileNotFoundError(f"Job file not found: {job_path!r}")

    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    raw_globals = job.get("globals", {})
    tracks_raw = job.get("tracks", [])

    if not tracks_raw:
        raise ValueError("Job file contains no tracks.")

    resolved = []
    for i, track in enumerate(tracks_raw):
        if "path" not in track:
            raise ValueError(f"Track at index {i} is missing required key 'path'.")

        # Merge: script defaults ← globals ← per-track (rightmost wins)
        merged = {**SCRIPT_DEFAULTS, **raw_globals, **track}

        # Clamp max_new_tokens per track
        max_tokens = int(
            merged.get("max_new_tokens", SCRIPT_DEFAULTS["max_new_tokens"])
        )
        if max_tokens > MAX_TOKENS_CEILING:
            print(
                f"  Warning: track[{i}] ({track['path']!r}) max_new_tokens={max_tokens} "
                f"exceeds ceiling {MAX_TOKENS_CEILING}. Clamping."
            )
            max_tokens = MAX_TOKENS_CEILING

        resolved.append(
            {
                "path": str(merged["path"]),
                "model_id": str(merged.get("model_id", SCRIPT_DEFAULTS["model_id"])),
                "precision": str(merged.get("precision", SCRIPT_DEFAULTS["precision"])),
                "max_new_tokens": max_tokens,
                "prompt_mode": str(
                    merged.get("prompt_mode", SCRIPT_DEFAULTS["prompt_mode"])
                ),
                "custom_prompt": merged.get("custom_prompt") or None,
                "output_dir": str(
                    merged.get("output_dir", SCRIPT_DEFAULTS["output_dir"])
                ),
                "tags": (
                    merged.get("tags") if isinstance(merged.get("tags"), dict) else {}
                ),
            }
        )

    return resolved


# ---------------------------------------------------------------------------
# Task 6.2 — Summary JSON Writer  [Phase 6]
# ---------------------------------------------------------------------------


def write_summary(results: list[dict], output_dir: str) -> str:
    """
    Write results_summary.json to output_dir.

    Args:
        results:    List of per-track outcome records (one per track attempted).
                    Each record: {file, status, processing_time_sec,
                                  error, output_file, tags}
        output_dir: Destination directory (created if absent).

    Returns:
        Path of the summary file written.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results_summary.json")

    success_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - success_count

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tracks": len(results),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# Task 2.1 — CLI Argument Parser  (--job added Phase 5)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description=(
            "Blind audio analysis using nvidia/music-flamingo-2601-hf. "
            "Phases 2–6 complete: --file, --dir, and --job modes."
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
    # Phase 5 — Task 5.2
    parser.add_argument(
        "--job",
        type=str,
        metavar="JOB.json",
        help="Path to a JSON job file with globals and per-track overrides.",
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
        help="Directory for output files (default: ./captions/). Not used in --job mode.",
    )

    return parser


# ---------------------------------------------------------------------------
# Task 2.6 / 4.2 / 5.2 — main()
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Shared pre-flight validation ---

    # Exactly one entry point must be supplied.
    entry_points = [ep for ep in (args.file, args.dir, args.job) if ep]
    if len(entry_points) == 0:
        parser.error(
            "No entry point specified. "
            "Use --file <path>, --dir <directory>, or --job <file.json>."
        )
    if len(entry_points) > 1:
        parser.error("Specify only one of --file, --dir, or --job.")

    # Clamp max_tokens ceiling (applies to --file and --dir; --job clamps per-track).
    if args.max_tokens > MAX_TOKENS_CEILING:
        print(
            f"Warning: --max-tokens {args.max_tokens} exceeds ceiling "
            f"of {MAX_TOKENS_CEILING}. Clamping."
        )
        args.max_tokens = MAX_TOKENS_CEILING

    # Validate custom prompt for --file / --dir modes (--job validates per-track below).
    if args.job is None and args.prompt_mode == "custom" and not args.custom_prompt:
        parser.error("--custom-prompt is required when --prompt-mode is 'custom'.")

    # -----------------------------------------------------------------------
    # Phase 5 — Task 5.2: Parse job file before loading the model so a bad
    # job file fails fast without spending time on model loading.
    # -----------------------------------------------------------------------
    tracks = None
    job_summary_output_dir = args.output_dir  # fallback; overwritten below for --job

    if args.job:
        try:
            tracks = parse_job_file(args.job)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"Error parsing job file: {exc}")

        # Validate every custom-mode track has a prompt before loading the model.
        for i, track in enumerate(tracks):
            if track["prompt_mode"] == "custom" and not track["custom_prompt"]:
                sys.exit(
                    f"Error: track[{i}] ({track['path']!r}) uses "
                    f"prompt_mode='custom' but has no custom_prompt."
                )

        # Summary goes to the globals output_dir (first resolved track carries it
        # when the first track does not override output_dir).
        job_summary_output_dir = tracks[0]["output_dir"]

    # --- Load model once, regardless of mode ---
    # For --job mode, all tracks share the model loaded with the first track's
    # precision. Per-track precision differences are not supported — the model
    # is loaded once and cannot be reloaded mid-batch.
    if args.job:
        precision = tracks[0]["precision"]
        unique_precisions = {t["precision"] for t in tracks}
        if len(unique_precisions) > 1:
            print(
                f"Warning: job file contains mixed precision values {unique_precisions}. "
                f"Using {precision!r} from first track. "
                f"The model is loaded once and cannot be reloaded mid-batch."
            )
    else:
        precision = args.precision

    model, processor = load_model(MODEL_ID, precision)

    # -----------------------------------------------------------------------
    # Task 2.6 — --file mode  [Phase 2]
    # Phase 6: summary JSON written even for single-file runs.
    # -----------------------------------------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            sys.exit(f"Error: file not found: {args.file!r}")

        prompt = build_prompt(args.prompt_mode, args.custom_prompt)

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

        # Phase 6 — Task 6.2: summary for single-file run
        summary_record = {
            "file": Path(args.file).name,
            "status": result["status"],
            "processing_time_sec": result["processing_time_sec"],
            "error": result["error"],
            "output_file": out_path,
            "tags": {},
        }
        summary_path = write_summary([summary_record], args.output_dir)

        if result["status"] == "ok":
            print(f"OK ({result['processing_time_sec']}s) → {out_path}")
        else:
            print(f"ERROR → {out_path}")
            print(f"  {result.get('error')}")
        print(f"Summary → {summary_path}")

        if result["status"] != "ok":
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Task 4.2 — --dir mode  [Phase 4]
    # Phase 6 — Task 6.1: single-line progress per track.
    # Phase 6 — Task 6.2: results_summary.json written at end.
    # -----------------------------------------------------------------------
    elif args.dir:
        prompt = build_prompt(args.prompt_mode, args.custom_prompt)
        audio_files = discover_audio_files(args.dir)

        if not audio_files:
            print("No audio files found. Nothing to do.")
            return

        total = len(audio_files)
        summary_records = []

        for idx, audio_path in enumerate(audio_files, start=1):
            name = Path(audio_path).name
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

            # Phase 6 — Task 6.1: single-line progress
            if result["status"] == "ok":
                print(f"[{idx}/{total}] {name} → OK ({result['processing_time_sec']}s)")
            else:
                print(f"[{idx}/{total}] {name} → ERROR: {result.get('error')}")

            summary_records.append(
                {
                    "file": name,
                    "status": result["status"],
                    "processing_time_sec": result["processing_time_sec"],
                    "error": result["error"],
                    "output_file": out_path,
                    "tags": {},
                }
            )
            # Always continue — never abort batch for one bad file.

        # Phase 6 — Task 6.2: write summary
        summary_path = write_summary(summary_records, args.output_dir)
        success_count = sum(1 for r in summary_records if r["status"] == "ok")
        fail_count = total - success_count

        print(
            f"\nProcessed: {total} | OK: {success_count} | Failed: {fail_count} "
            f"| Summary → {summary_path}"
        )

    # -----------------------------------------------------------------------
    # Task 5.2 — --job mode  [Phase 5]
    # Phase 6 — Task 6.1: single-line progress per track.
    # Phase 6 — Task 6.2: results_summary.json written at end.
    # -----------------------------------------------------------------------
    elif args.job:
        total = len(tracks)
        summary_records = []

        for idx, track in enumerate(tracks, start=1):
            audio_path = track["path"]
            name = Path(audio_path).name

            # Missing file: skip + log, consistent with batch failure behavior in brief.
            if not os.path.isfile(audio_path):
                err = f"File not found: {audio_path!r}"
                print(f"[{idx}/{total}] {name} → ERROR: {err}")
                summary_records.append(
                    {
                        "file": name,
                        "status": "error",
                        "processing_time_sec": None,
                        "error": err,
                        "output_file": None,
                        "tags": track["tags"],
                    }
                )
                continue

            prompt = build_prompt(track["prompt_mode"], track["custom_prompt"])

            result = run_inference(
                model,
                processor,
                audio_path,
                prompt,
                track["max_new_tokens"],
            )

            out_path = write_track_output(
                result=result,
                audio_path=audio_path,
                prompt_used=prompt,
                prompt_mode=track["prompt_mode"],
                tags=track["tags"],
                model_id=MODEL_ID,
                output_dir=track["output_dir"],
            )

            # Phase 6 — Task 6.1: single-line progress
            if result["status"] == "ok":
                print(f"[{idx}/{total}] {name} → OK ({result['processing_time_sec']}s)")
            else:
                print(f"[{idx}/{total}] {name} → ERROR: {result.get('error')}")

            summary_records.append(
                {
                    "file": name,
                    "status": result["status"],
                    "processing_time_sec": result["processing_time_sec"],
                    "error": result["error"],
                    "output_file": out_path,
                    "tags": track["tags"],
                }
            )
            # Always continue — never abort batch for one bad file.

        # Phase 6 — Task 6.2: summary written to globals output_dir
        summary_path = write_summary(summary_records, job_summary_output_dir)
        success_count = sum(1 for r in summary_records if r["status"] == "ok")
        fail_count = total - success_count

        print(
            f"\nProcessed: {total} | OK: {success_count} | Failed: {fail_count} "
            f"| Summary → {summary_path}"
        )


if __name__ == "__main__":
    main()
