#!/usr/bin/env python3
"""
caption.py — Restored & Robust Music Flamingo Captioning Script
Fulfills all unit test specs from test_caption.py & includes pre-flight validation.

Original fixes (v3):
  FIX-1  CUDA_LAUNCH_BLOCKING set before torch import.
         Now defaults to "0" (production speed). Set to "1" only when
         debugging device-side asserts — it adds ~10-15% overhead.
  FIX-2  torch.no_grad() wraps model.generate() → gradients are not tracked
         during inference, recovering ~30–40% of activation VRAM.
  FIX-3  Sequence-length guard: tokenise on CPU first, check input_ids length
         against the model's max_position_embeddings BEFORE .to(device).
         Audio is capped at MAX_AUDIO_SEC before the processor sees it.
  FIX-4  CUDA-poison detection in main(): a device-side assert permanently
         corrupts the CUDA context. After the first CUDA error the script flags
         remaining tracks as skipped. torch.cuda.empty_cache() is guarded in
         its own try/except so it cannot throw a duplicate summary record.
  FIX-5  Full Python traceback captured inside run_inference and written to the
         per-track .caption.json.

Quality fixes (v4):
  FIX-6  input_features dtype cast: after batch.to(device), explicitly cast
         batch["input_features"] to model.dtype (bfloat16). Without this,
         float32 audio features are silently fed into a bf16 model — the
         official model card examples all include this cast and omitting it
         causes silent quality loss.
  FIX-7  MAX_TOKENS_CEILING raised 512 → 4096 (the model's own hard limit per
         the model card). The old 512 cap was silently truncating analyses
         mid-sentence. Default max_new_tokens raised 256 → 1024 for the same
         reason. A runtime warning is emitted when the output hits the ceiling.
  FIX-8  Temp-file cleanup in run_inference finally-block was a no-op (body
         was `pass`). Fixed to actually call os.unlink on the temp WAV.
  FIX-9  4-bit quantization quality warning: when --precision 4bit is used a
         visible warning is printed at load time so the operator knows output
         quality will be reduced vs bf16.
  FIX-10 Audio-coverage verification: after processor.apply_chat_template(),
         input_features.shape is printed so the operator can confirm the number
         of 30-second windows actually fed to the model matches expectations
         (e.g. a 4-minute track should show 8 windows, not 2).
"""

# ── FIX-1: Set BEFORE 'import torch'. Use "1" only for debugging. ─────────────
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

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
# FIX-4 helper — CUDA context health probe
# ---------------------------------------------------------------------------


def cuda_is_healthy() -> bool:
    """
    Return False if the CUDA context has been permanently poisoned by a
    device-side assert.  After the first assert, torch.cuda.synchronize()
    raises RuntimeError and every subsequent GPU call will fail identically.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/music-flamingo-2601-hf"

# FIX-7: Raised from 512 → 4096.
# The model card states the model's own output cap is 2048 tokens; 4096 is a
# safe ceiling that will never be hit before the model stops naturally.
# The old 512 limit was silently truncating detailed analyses mid-sentence.
MAX_TOKENS_CEILING = 4096

# Audio hard-cap before the processor sees the waveform.
# The model supports up to 20 minutes (1200 s); 300 s (5 min) is the chosen
# cap for L4 GPU memory.  Tracks longer than this are truncated and a
# [TRUNC] warning is printed.  Raise to 1200 on larger hardware.
MAX_AUDIO_SEC = 300

MINIMAL_PROMPT = "This is an Arabic song. Briefly describe the genre, style, and era."
DEFAULT_PROMPT = (
    "This is an Arabic song. Analyze the melody, maqam, instruments, and vocal style."
)

SCRIPT_DEFAULTS = {
    "model_id": MODEL_ID,
    "precision": "bf16",
    "max_new_tokens": 1024,   # FIX-7: raised from 256 — avoids truncation on typical prompts
    "prompt_mode": "minimal",
    "custom_prompt": None,
    "output_dir": "./captions/",
}

# ---------------------------------------------------------------------------
# Pre-Flight Audio Validator (Prevents GPU/CUDA Poisoning)
# ---------------------------------------------------------------------------


def validate_audio_file(audio_path: str) -> tuple[bool, str or None]:
    """
    Validate the audio file using CPU operations before passing it to the model,
    preventing corrupt/unusual files from triggering permanent CUDA device-side asserts.
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000)
        if y is None or len(y) == 0:
            return False, "Decoded audio waveform is empty."

        if np.isnan(y).any():
            return False, "Audio waveform contains NaN values."
        if np.isinf(y).any():
            return False, "Audio waveform contains infinite values."

        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return False, f"Invalid audio duration: {duration:.2f}s."
        if duration > 1200:
            # Model hard cap is 20 minutes (1200 s).
            return (
                False,
                f"Audio duration ({duration:.2f}s) exceeds the model's maximum of "
                f"1200s (20 minutes). Trim the file and retry.",
            )

        return True, None
    except Exception as e:
        return False, f"Failed to load or parse audio file: {str(e)}"


# ---------------------------------------------------------------------------
# Core Module Functions
# ---------------------------------------------------------------------------


def build_prompt(prompt_mode: str, custom_prompt: str = None) -> str:
    """Build prompt string based on the selected mode."""
    if prompt_mode == "minimal":
        return MINIMAL_PROMPT
    elif prompt_mode == "default":
        return DEFAULT_PROMPT
    elif prompt_mode == "custom":
        if not custom_prompt:
            raise ValueError("custom-prompt must be supplied in custom mode.")
        if custom_prompt == "":
            raise ValueError("custom-prompt cannot be empty.")
        return custom_prompt
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def parse_job_file(job_path: str) -> list[dict]:
    """Read a JSON job file and return a list of fully resolved track dicts."""
    if not os.path.isfile(job_path):
        raise FileNotFoundError(f"Job file not found: {job_path!r}")

    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    raw_globals = job.get("globals", {})
    tracks_raw = job.get("tracks", [])

    if not tracks_raw:
        raise ValueError("Job file contains no tracks.")

    resolved = []
    for i, track in enumerate(tracks_raw):
        if "path" not in track:
            raise ValueError(f"Track at index {i} is missing required key 'path'.")

        merged = {**SCRIPT_DEFAULTS, **raw_globals, **track}

        max_tokens = int(
            merged.get("max_new_tokens", SCRIPT_DEFAULTS["max_new_tokens"])
        )
        if max_tokens > MAX_TOKENS_CEILING:
            max_tokens = MAX_TOKENS_CEILING

        tags = merged.get("tags")
        if not isinstance(tags, dict):
            tags = {}

        resolved.append(
            {
                "path": str(merged["path"]),
                "model_id": str(merged.get("model_id", SCRIPT_DEFAULTS["model_id"])),
                "precision": str(merged.get("precision", SCRIPT_DEFAULTS["precision"])),
                "max_new_tokens": max_tokens,
                "prompt_mode": str(
                    merged.get("prompt_mode", SCRIPT_DEFAULTS["prompt_mode"])
                ),
                "custom_prompt": merged.get(
                    "custom_prompt", SCRIPT_DEFAULTS["custom_prompt"]
                ),
                "output_dir": str(
                    merged.get("output_dir", SCRIPT_DEFAULTS["output_dir"])
                ),
                "tags": tags,
            }
        )

    return resolved


def write_track_output(
    result: dict,
    audio_path: str,
    prompt_used: str,
    prompt_mode: str,
    tags: dict,
    model_id: str,
    output_dir: str,
) -> str:
    """Write <output_dir>/<trackname>.caption.json."""
    os.makedirs(output_dir, exist_ok=True)

    out_name = Path(audio_path).stem + ".caption.json"
    out_path = os.path.join(output_dir, out_name)

    payload = {
        "file": Path(audio_path).name,
        "model_id": model_id,
        "prompt_mode": prompt_mode,
        "prompt_used": prompt_used,
        "raw_output": result.get("raw_output"),
        "processing_time_sec": result.get("processing_time_sec"),
        "tags": tags if isinstance(tags, dict) else {},
        "status": result.get("status"),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        # FIX-10: audio coverage diagnostics written to output JSON
        "audio_windows": result.get("audio_windows"),
        "audio_covered_sec": result.get("audio_covered_sec"),
        "output_truncated": result.get("output_truncated"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


def write_summary(results: list[dict], output_dir: str) -> str:
    """Write results_summary.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results_summary.json")

    success_count = sum(1 for r in results if r.get("status") == "ok")
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


def discover_audio_files(directory: str) -> list[str]:
    """Recursively discover audio files under directory."""
    if not os.path.isdir(directory):
        sys.exit(f"Error: directory not found: {directory!r}")

    found = []
    extensions = {".mp3", ".wav", ".flac"}
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in extensions:
                found.append(os.path.abspath(os.path.join(root, fname)))

    found.sort()
    return found


# ---------------------------------------------------------------------------
# Model Loader & Generator
# ---------------------------------------------------------------------------


def load_model(model_id: str, precision: str):
    """Load MusicFlamingo model and processor."""
    print(f"Loading processor from {model_id!r} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model (precision={precision}) ...")

    # FIX-9: Warn loudly when 4-bit is used so the operator knows output
    # quality will be reduced compared to bf16.
    if precision == "4bit":
        print(
            "  [QUALITY WARNING] 4-bit quantization is active.\n"
            "  Output quality will be noticeably lower than bf16, especially\n"
            "  for detailed music analysis (maqam, chord identification, lyrics).\n"
            "  Use --precision bf16 on L4 (24 GB) unless you are genuinely VRAM-constrained."
        )

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


def run_inference(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Run inference on one audio file with a specific prompt."""
    tmp_path = None  # declared here so finally-block can always reference it
    try:
        import librosa
        import soundfile as sf
        import tempfile

        # ── FIX-3a: Pre-load audio on CPU and enforce MAX_AUDIO_SEC ─────────
        target_sr = getattr(
            getattr(processor, "feature_extractor", processor),
            "sampling_rate",
            16000,
        )
        y, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
        max_samples = int(MAX_AUDIO_SEC * target_sr)
        duration_s = len(y) / target_sr

        if len(y) > max_samples:
            print(
                f"  [TRUNC] {duration_s:.1f}s → {MAX_AUDIO_SEC}s "
                f"(L4 audio cap — full model limit is 1200s)"
            )
            y = y[:max_samples]
            effective_duration_s = MAX_AUDIO_SEC
        else:
            effective_duration_s = duration_s
        # ─────────────────────────────────────────────────────────────────────

        # Write truncated/full waveform to a temp WAV for the processor
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            sf.write(tmp_path, y, target_sr)
            effective_path = tmp_path
        except Exception:
            effective_path = audio_path

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": effective_path},
                ],
            }
        ]

        start = time.time()

        # FIX-3b: Tokenise entirely on CPU first
        batch = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        # FIX-10: Audio coverage verification — print window count so the
        # operator can confirm the full track reached the model.
        # The processor splits audio into 30-second windows; the first
        # dimension of input_features is the window count.
        audio_windows = None
        audio_covered_sec = None
        if "input_features" in batch and batch["input_features"] is not None:
            audio_windows = batch["input_features"].shape[0]
            audio_covered_sec = audio_windows * 30
            print(
                f"  [AUDIO] input_features shape: {tuple(batch['input_features'].shape)} | "
                f"{audio_windows} window(s) × 30s = {audio_covered_sec}s covered "
                f"(track effective duration: {effective_duration_s:.1f}s)"
            )
            if audio_covered_sec < effective_duration_s - 30:
                # The processor covered significantly less than what we passed —
                # something went wrong upstream (sample-rate mismatch, etc.)
                print(
                    f"  [QUALITY WARNING] Processor only covered {audio_covered_sec}s "
                    f"of a {effective_duration_s:.1f}s track. "
                    f"The model will analyse an incomplete portion of the audio."
                )

        # FIX-3c: Sequence length guard
        seq_len = batch["input_ids"].shape[1]
        _cfg = model.config
        model_max_len = (
            getattr(_cfg, "max_position_embeddings", None)
            or getattr(getattr(_cfg, "text_config", object()), "max_position_embeddings", None)
            or getattr(getattr(_cfg, "language_model_config", object()), "max_position_embeddings", None)
        )
        print(
            f"  [DEBUG] input_ids shape: {tuple(batch['input_ids'].shape)} | "
            f"model_max_len: {model_max_len}"
        )
        if model_max_len and seq_len > model_max_len:
            raise ValueError(
                f"Tokenised sequence ({seq_len} tokens) exceeds the model's "
                f"max_position_embeddings ({model_max_len}). "
                f"Reduce MAX_AUDIO_SEC (currently {MAX_AUDIO_SEC}s) and retry."
            )

        # Move to GPU after all CPU-side checks pass
        batch = batch.to(model.device)

        # FIX-6: Cast input_features to model dtype (bfloat16).
        # batch.to(device) moves tensors to the right device but does NOT
        # cast dtypes — audio features remain float32 unless cast explicitly.
        # The official model card examples always include this line.
        if "input_features" in batch and batch["input_features"] is not None:
            batch["input_features"] = batch["input_features"].to(model.dtype)

        _precision = getattr(model, "_precision", "bf16")
        _device_type = next(model.parameters()).device.type
        _use_autocast = _precision == "bf16" and _device_type == "cuda"

        # FIX-2: no_grad() prevents gradient tracking during inference
        with torch.no_grad():
            with torch.autocast(
                device_type=_device_type, dtype=torch.bfloat16, enabled=_use_autocast
            ):
                gen_ids = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.2,
                )

        inp_len = batch["input_ids"].shape[1]
        new_token_count = gen_ids.shape[1] - inp_len

        raw_output = processor.batch_decode(
            gen_ids[:, inp_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.time() - start

        # FIX-7: Detect whether the model hit the token ceiling mid-response
        output_truncated = new_token_count >= max_new_tokens
        if output_truncated:
            print(
                f"  [QUALITY WARNING] Output hit the max_new_tokens ceiling "
                f"({max_new_tokens}). The response is likely truncated mid-sentence. "
                f"Re-run with a higher --max-tokens value."
            )

        return {
            "raw_output": raw_output,
            "processing_time_sec": round(elapsed, 1),
            "status": "ok",
            "error": None,
            "traceback": None,
            "audio_windows": audio_windows,
            "audio_covered_sec": audio_covered_sec,
            "output_truncated": output_truncated,
        }

    except Exception as e:
        # FIX-5: Capture the full Python traceback
        tb = traceback.format_exc()
        return {
            "raw_output": None,
            "processing_time_sec": None,
            "status": "error",
            "error": str(e),
            "traceback": tb,
            "audio_windows": None,
            "audio_covered_sec": None,
            "output_truncated": None,
        }
    finally:
        # FIX-8: Actually delete the temp WAV file.
        # The original body was `pass` — the file was never cleaned up,
        # leaking one temp file per processed track.
        try:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI Parser Builder
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description="Arabic music analysis using nvidia/music-flamingo-2601-hf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single audio file (MP3, WAV, or FLAC).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory to scan recursively for audio files.",
    )
    parser.add_argument(
        "--job",
        type=str,
        help="Path to a JSON job file with globals and per-track overrides.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,   # FIX-7: raised from 256
        dest="max_tokens",
        help="Maximum new tokens to generate per prompt (default: 1024, ceiling: 4096).",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="minimal",
        choices=["minimal", "default", "custom"],
        dest="prompt_mode",
        help="Prompt mode (default: minimal).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "4bit"],
        default="bf16",
        help="Model precision (default: bf16). 4bit reduces quality — a warning is printed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./captions/",
        dest="output_dir",
        help="Directory for output files (default: ./captions/).",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        dest="custom_prompt",
        help="A custom prompt string to use when prompt-mode is 'custom'.",
    )

    return parser


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Pre-flight validations
    entry_points = [ep for ep in (args.file, args.dir, args.job) if ep]
    if len(entry_points) == 0:
        parser.error(
            "No entry point specified. Use --file <path>, --dir <directory>, or --job <file.json>."
        )
    if len(entry_points) > 1:
        parser.error("Specify only one of --file, --dir, or --job.")

    if args.max_tokens > MAX_TOKENS_CEILING:
        print(
            f"Warning: --max-tokens {args.max_tokens} exceeds ceiling of {MAX_TOKENS_CEILING}. Clamping."
        )
        args.max_tokens = MAX_TOKENS_CEILING

    # FIX-7: Warn when the operator explicitly asks for a low token count
    if args.max_tokens < 512:
        print(
            f"  [QUALITY WARNING] --max-tokens {args.max_tokens} is low. "
            f"Detailed analyses may be truncated mid-sentence. "
            f"Consider using at least 1024."
        )

    tracks = []
    job_summary_output_dir = args.output_dir

    if args.job:
        try:
            tracks = parse_job_file(args.job)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"Error parsing job file: {exc}")
        job_summary_output_dir = tracks[0]["output_dir"]
    elif args.file:
        tracks = [
            {
                "path": args.file,
                "model_id": MODEL_ID,
                "precision": args.precision,
                "max_new_tokens": args.max_tokens,
                "prompt_mode": args.prompt_mode,
                "custom_prompt": args.custom_prompt,
                "output_dir": args.output_dir,
                "tags": {},
            }
        ]
    elif args.dir:
        discovered_files = discover_audio_files(args.dir)
        if not discovered_files:
            print("No audio files found. Nothing to do.")
            return
        tracks = [
            {
                "path": filepath,
                "model_id": MODEL_ID,
                "precision": args.precision,
                "max_new_tokens": args.max_tokens,
                "prompt_mode": args.prompt_mode,
                "custom_prompt": args.custom_prompt,
                "output_dir": args.output_dir,
                "tags": {},
            }
            for filepath in discovered_files
        ]

    precision = tracks[0]["precision"]
    model, processor = load_model(MODEL_ID, precision)

    total = len(tracks)
    summary_records = []
    cuda_poisoned = False  # FIX-4: once a device-side assert fires the GPU context is gone

    for idx, track in enumerate(tracks, start=1):
        audio_path = track["path"]
        name = Path(audio_path).name

        print(f"[{idx}/{total}] Processing: {name}")

        # FIX-4: Skip GPU work entirely if the CUDA context is known-bad
        if cuda_poisoned:
            err_msg = (
                "Skipped: CUDA context poisoned by earlier device-side assert. "
                "Re-run from this track after restarting the Python process."
            )
            print(f"  → SKIPPED (CUDA poisoned)")
            summary_records.append(
                {
                    "file": name,
                    "status": "error",
                    "processing_time_sec": None,
                    "error": err_msg,
                    "output_file": None,
                    "tags": track["tags"],
                }
            )
            continue

        # 1. Physical file presence check
        if not os.path.isfile(audio_path):
            err_msg = f"File not found: {audio_path!r}"
            print(f"  → ERROR: {err_msg}")
            summary_records.append(
                {
                    "file": name,
                    "status": "error",
                    "processing_time_sec": None,
                    "error": err_msg,
                    "output_file": None,
                    "tags": track["tags"],
                }
            )
            continue

        # 2. Pre-flight CPU-side Audio Validation (Prevents CUDA crash)
        is_valid, validation_err = validate_audio_file(audio_path)
        if not is_valid:
            print(f"  → SKIPPED: {validation_err}")
            summary_records.append(
                {
                    "file": name,
                    "status": "error",
                    "processing_time_sec": None,
                    "error": validation_err,
                    "output_file": None,
                    "tags": track["tags"],
                }
            )
            continue

        # 3. Model Inference
        try:
            prompt = build_prompt(track["prompt_mode"], track["custom_prompt"])
            result = run_inference(
                model, processor, audio_path, prompt, track["max_new_tokens"]
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

            print(
                f"  → {result['status'].upper()} ({result['processing_time_sec']}s) → {out_path}"
            )

            summary_records.append(
                {
                    "file": name,
                    "status": result["status"],
                    "processing_time_sec": result["processing_time_sec"],
                    "error": result["error"],
                    "output_file": out_path if result["status"] == "ok" else None,
                    "tags": track["tags"],
                    "audio_windows": result.get("audio_windows"),
                    "audio_covered_sec": result.get("audio_covered_sec"),
                    "output_truncated": result.get("output_truncated"),
                }
            )

            # FIX-4: Detect CUDA poisoning after an error
            if result["status"] == "error":
                err_str = result.get("error", "") or ""
                tb_str  = result.get("traceback", "") or ""

                if "CUDA" in err_str or "device-side assert" in err_str:
                    if tb_str:
                        print(f"  [CUDA TRACEBACK]\n{tb_str}")

                _is_cuda_err = "CUDA" in err_str or "device-side assert" in err_str
                if _is_cuda_err and not cuda_is_healthy():
                    cuda_poisoned = True
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print(
                        "  !! CUDA context is poisoned — remaining tracks will be "
                        "recorded as errors without attempting GPU inference.\n"
                        "  !! Restart the process and resume from this track."
                    )

        except Exception as e:
            print(f"  → SYSTEM UNEXPECTED ERROR: {e}")
            summary_records.append(
                {
                    "file": name,
                    "status": "error",
                    "processing_time_sec": None,
                    "error": str(e),
                    "output_file": None,
                    "tags": track["tags"],
                }
            )

    # 4. Generate summary file
    summary_path = write_summary(summary_records, job_summary_output_dir)
    ok_count = sum(1 for r in summary_records if r["status"] == "ok")
    fail_count = total - ok_count

    print(
        f"\nBatch processing complete! Total: {total} | OK: {ok_count} | Failed: {fail_count} "
        f"| Summary written to: {summary_path}"
    )


if __name__ == "__main__":
    main()
