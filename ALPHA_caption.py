#!/usr/bin/env python3
"""
caption.py — Music Flamingo caption script
Model: nvidia/music-flamingo-2601-hf
Multi-prompt mode: runs all PROMPTS sequentially per track.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# --- Bypass buggy torchcodec audio loading and force stable librosa/soundfile fallback ---
try:
    import transformers
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.is_torchcodec_available = lambda: False
    transformers.utils.import_utils.is_torchcodec_available = lambda: False
except ImportError:
    pass

from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MAX_TOKENS_CEILING = 512

output_dir = input("Enter path to caption output directory:\n")

# Script-level defaults — lowest precedence in job configurations
SCRIPT_DEFAULTS = {
    "model_id": MODEL_ID,
    "precision": "bf16",
    "max_new_tokens": 256,
    "output_dir": output_dir,
}

# ---------------------------------------------------------------------------
# Prompts — inlined from prompts.py
# ---------------------------------------------------------------------------

PROMPTS = [
    """
This is an Arabic song.

Briefly describe:
- The most likely genre or genres.
- The regional musical style (Khaliji, Egyptian, Levantine, Iraqi, Maghrebi, Yemeni, Sudanese, Mixed, or other).
- Whether the song sounds traditional, modern, or a fusion.
- The approximate era or decade of the musical style.
- Up to five descriptive tags.

Keep the answer concise and organized with bullet points.
""",
    """
This is an Arabic song.

Analyze the melody and tonal characteristics.

Discuss:
- The most likely maqam.
- One or two alternative maqamat if applicable.
- Whether quarter tones are clearly present, absent, or uncertain.
- The tonal center or resting note if identifiable.
- The main melodic characteristics that support your conclusion.

If uncertain, say so explicitly instead of guessing.
Keep the answer concise and organized with bullet points.
""",
    """
This is an Arabic song.

Analyze the performance and arrangement.

Describe:
- The main instruments heard.
- Whether the arrangement is acoustic, electronic, or mixed.
- The vocal style.
- Whether the singer appears male, female, mixed, or unclear.
- Whether there is a solo singer or multiple vocalists.
- Any notable rhythmic characteristics.

Keep the answer concise and organized with bullet points.
""",
    """
This is an Arabic song.

Describe the listening experience.

Discuss:
- The dominant mood or emotions.
- The energy level (low, medium, or high).
- The danceability (low, medium, or high).
- The tempo (slow, medium, or fast).
- Up to three similar Arabic artists.
- Situations or contexts where listeners might enjoy this song.

Keep the answer concise and organized with bullet points.
""",
]

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------


def load_model(model_id: str, precision: str):
    """
    Load MusicFlamingo model and processor.

    Args:
        model_id:  HuggingFace model ID string.
        precision: "bf16" (default) or "4bit".

    Returns:
        (model, processor) tuple — model is in eval mode.
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

    model._precision = precision
    model.eval()

    device = next(model.parameters()).device
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model ready. Device: {device} | VRAM allocated: {vram_gb:.2f} GB")

    return model, processor


# ---------------------------------------------------------------------------
# Single Prompt Inference
# ---------------------------------------------------------------------------


def run_inference(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """
    Run blind inference on one audio file with a specific prompt.

    Args:
        model:          Loaded MusicFlamingo model.
        processor:      Matching AutoProcessor.
        audio_path:     Absolute or relative path to the audio file.
        prompt:         Prompt string.
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

        # Use autocast for bf16 models so mixed-precision ops (e.g. float32
        # positional-embedding buffers alongside bfloat16 conv weights) resolve
        # without manual dtype casting. No-op for 4bit (enabled=False).
        _precision = getattr(model, "_precision", "bf16")
        _device_type = next(model.parameters()).device.type
        _use_autocast = _precision == "bf16" and _device_type == "cuda"

        with torch.autocast(
            device_type=_device_type, dtype=torch.bfloat16, enabled=_use_autocast
        ):
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
# Multi-Prompt Sequential Inference Runner
# ---------------------------------------------------------------------------


def run_track_prompts(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int,
) -> dict:
    """
    Run all PROMPTS sequentially on one audio file.

    Args:
        model:          Loaded MusicFlamingo model.
        processor:      Matching AutoProcessor.
        audio_path:     Path to the audio file.
        max_new_tokens: Token budget per prompt evaluation.

    Returns:
        Dictionary summarizing all prompt executions and statuses.
    """
    responses = []
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        result = run_inference(model, processor, audio_path, prompt, max_new_tokens)

        responses.append(
            {
                "prompt_index": i,
                "prompt": prompt.strip(),
                "raw_output": result["raw_output"],
                "processing_time_sec": result["processing_time_sec"],
                "status": result["status"],
                "error": result["error"],
            }
        )

        if result["status"] == "error":
            print(f"    [Prompt {i} Error] {result['error']}")

        if result["processing_time_sec"] is not None:
            total_time += result["processing_time_sec"]

    ok_count = sum(1 for r in responses if r["status"] == "ok")
    if ok_count == len(PROMPTS):
        overall_status = "ok"
    elif ok_count == 0:
        overall_status = "error"
    else:
        overall_status = "partial"

    return {
        "responses": responses,
        "total_processing_time_sec": round(total_time, 1),
        "status": overall_status,
    }


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------


def write_track_output(
    track_result: dict,
    audio_path: str,
    tags: dict,
    model_id: str,
    output_dir: str,
) -> str:
    """
    Write <output_dir>/<trackname>.caption.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    out_name = Path(audio_path).stem + ".caption.json"
    out_path = os.path.join(output_dir, out_name)

    payload = {
        "file": Path(audio_path).name,
        "model_id": model_id,
        "responses": track_result["responses"],
        "total_processing_time_sec": track_result["total_processing_time_sec"],
        "tags": tags if tags is not None else {},
        "status": track_result["status"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# Audio File Discovery
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac"}


def discover_audio_files(directory: str) -> list:
    """
    Recursively discover audio files under directory.
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
# JSON Job Parser
# ---------------------------------------------------------------------------


def parse_job_file(job_path: str) -> list[dict]:
    """
    Read a JSON job file and return a list of fully resolved track dicts.
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

        # Merge: SCRIPT_DEFAULTS <- globals <- track
        merged = {**SCRIPT_DEFAULTS, **raw_globals, **track}

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
# Summary JSON Writer
# ---------------------------------------------------------------------------


def write_summary(results: list[dict], output_dir: str) -> str:
    """
    Write results_summary.json to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results_summary.json")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    partial_count = sum(1 for r in results if r["status"] == "partial")
    fail_count = len(results) - ok_count - partial_count

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tracks": len(results),
        "ok_count": ok_count,
        "partial_count": partial_count,
        "fail_count": fail_count,
        "prompts_per_track": len(PROMPTS),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# CLI Parser Builder
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description=(
            "Arabic music analysis using nvidia/music-flamingo-2601-hf. "
            "Runs all PROMPTS sequentially per track. "
            "Modes: --file, --dir, --job."
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
        help="Maximum new tokens to generate per prompt (default: 256, ceiling: 512).",
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
# main()
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Shared pre-flight validations
    entry_points = [ep for ep in (args.file, args.dir, args.job) if ep]
    if len(entry_points) == 0:
        parser.error(
            "No entry point specified. "
            "Use --file <path>, --dir <directory>, or --job <file.json>."
        )
    if len(entry_points) > 1:
        parser.error("Specify only one of --file, --dir, or --job.")

    if args.max_tokens > MAX_TOKENS_CEILING:
        print(
            f"Warning: --max-tokens {args.max_tokens} exceeds ceiling "
            f"of {MAX_TOKENS_CEILING}. Clamping."
        )
        args.max_tokens = MAX_TOKENS_CEILING

    tracks = None
    job_summary_output_dir = args.output_dir

    if args.job:
        try:
            tracks = parse_job_file(args.job)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"Error parsing job file: {exc}")
        job_summary_output_dir = tracks[0]["output_dir"]

    # Resolve precision layout
    if args.job:
        precision = tracks[0]["precision"]
        unique_precisions = {t["precision"] for t in tracks}
        if len(unique_precisions) > 1:
            print(
                f"Warning: job file contains mixed precision values {unique_precisions}. "
                f"Using {precision!r} from first track."
            )
    else:
        precision = args.precision

    model, processor = load_model(MODEL_ID, precision)

    # -----------------------------------------------------------------------
    # --file mode
    # -----------------------------------------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            sys.exit(f"Error: file not found: {args.file!r}")

        print(f"\nProcessing: {args.file}  ({len(PROMPTS)} prompts)")
        track_result = run_track_prompts(model, processor, args.file, args.max_tokens)

        out_path = write_track_output(
            track_result=track_result,
            audio_path=args.file,
            tags={},
            model_id=MODEL_ID,
            output_dir=args.output_dir,
        )

        summary_record = {
            "file": Path(args.file).name,
            "status": track_result["status"],
            "total_processing_time_sec": track_result["total_processing_time_sec"],
            "prompts_ok": sum(
                1 for r in track_result["responses"] if r["status"] == "ok"
            ),
            "prompts_total": len(PROMPTS),
            "output_file": out_path,
            "tags": {},
        }
        summary_path = write_summary([summary_record], args.output_dir)

        print(
            f"{track_result['status'].upper()} ({track_result['total_processing_time_sec']}s) → {out_path}"
        )
        print(f"Summary → {summary_path}")

        if track_result["status"] == "error":
            sys.exit(1)

    # -----------------------------------------------------------------------
    # --dir mode
    # -----------------------------------------------------------------------
    elif args.dir:
        audio_files = discover_audio_files(args.dir)

        if not audio_files:
            print("No audio files found. Nothing to do.")
            return

        total = len(audio_files)
        summary_records = []

        for idx, audio_path in enumerate(audio_files, start=1):
            name = Path(audio_path).name
            print(f"[{idx}/{total}] {name}  ({len(PROMPTS)} prompts)")
            track_result = run_track_prompts(
                model, processor, audio_path, args.max_tokens
            )

            out_path = write_track_output(
                track_result=track_result,
                audio_path=audio_path,
                tags={},
                model_id=MODEL_ID,
                output_dir=args.output_dir,
            )

            prompts_ok = sum(
                1 for r in track_result["responses"] if r["status"] == "ok"
            )
            print(
                f"  → {track_result['status'].upper()} ({prompts_ok}/{len(PROMPTS)} prompts, {track_result['total_processing_time_sec']}s)"
            )

            summary_records.append(
                {
                    "file": name,
                    "status": track_result["status"],
                    "total_processing_time_sec": track_result[
                        "total_processing_time_sec"
                    ],
                    "prompts_ok": prompts_ok,
                    "prompts_total": len(PROMPTS),
                    "output_file": out_path,
                    "tags": {},
                }
            )

        summary_path = write_summary(summary_records, args.output_dir)
        ok_count = sum(1 for r in summary_records if r["status"] == "ok")
        partial_count = sum(1 for r in summary_records if r["status"] == "partial")
        fail_count = total - ok_count - partial_count

        print(
            f"\nProcessed: {total} | OK: {ok_count} | Partial: {partial_count} | Failed: {fail_count} "
            f"| Summary → {summary_path}"
        )

    # -----------------------------------------------------------------------
    # --job mode
    # -----------------------------------------------------------------------
    elif args.job:
        total = len(tracks)
        summary_records = []

        for idx, track in enumerate(tracks, start=1):
            audio_path = track["path"]
            name = Path(audio_path).name

            if not os.path.isfile(audio_path):
                err = f"File not found: {audio_path!r}"
                print(f"[{idx}/{total}] {name} → ERROR: {err}")
                summary_records.append(
                    {
                        "file": name,
                        "status": "error",
                        "total_processing_time_sec": None,
                        "prompts_ok": 0,
                        "prompts_total": len(PROMPTS),
                        "error": err,
                        "output_file": None,
                        "tags": track["tags"],
                    }
                )
                continue

            print(f"[{idx}/{total}] {name}  ({len(PROMPTS)} prompts)")
            track_result = run_track_prompts(
                model,
                processor,
                audio_path,
                track["max_new_tokens"],
            )

            out_path = write_track_output(
                track_result=track_result,
                audio_path=audio_path,
                tags=track["tags"],
                model_id=MODEL_ID,
                output_dir=track["output_dir"],
            )

            prompts_ok = sum(
                1 for r in track_result["responses"] if r["status"] == "ok"
            )
            print(
                f"  → {track_result['status'].upper()} ({prompts_ok}/{len(PROMPTS)} prompts, {track_result['total_processing_time_sec']}s)"
            )

            summary_records.append(
                {
                    "file": name,
                    "status": track_result["status"],
                    "total_processing_time_sec": track_result[
                        "total_processing_time_sec"
                    ],
                    "prompts_ok": prompts_ok,
                    "prompts_total": len(PROMPTS),
                    "output_file": out_path,
                    "tags": track["tags"],
                }
            )

        summary_path = write_summary(summary_records, job_summary_output_dir)
        ok_count = sum(1 for r in summary_records if r["status"] == "ok")
        partial_count = sum(1 for r in summary_records if r["status"] == "partial")
        fail_count = total - ok_count - partial_count

        print(
            f"\nProcessed: {total} | OK: {ok_count} | Partial: {partial_count} | Failed: {fail_count} "
            f"| Summary → {summary_path}"
        )


if __name__ == "__main__":
    main()
