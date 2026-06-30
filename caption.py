#!/usr/bin/env python3
"""
caption.py — Music Flamingo multi-prompt captioning, crash-resumable.

For every audio file under --dir, runs EVERY prompt in prompts.CAPTIONING_PROMPTS
(genre, vocals, instrumentation, production, mastering — edit prompts.py to change
the set) and writes one JSON file per track:

    <output_dir>/<track_stem>.caption.json
    {
      "file": "01.mp3",
      "path": "/abs/path/01.mp3",
      "categories": {
        "genre":  {"status": "ok", "output": "...", ...},
        "vocals": {"status": "ok", "output": "...", ...},
        ...
      }
    }

CRASH SAFETY:
  The JSON for a track is rewritten to disk (atomically) immediately after EACH
  category finishes — not after the whole track, not after the whole run. If the
  Colab runtime dies mid-run, at most the one category that was in flight is lost.

RESUME:
  Re-running the same command skips any (track, category) pair whose JSON already
  shows status "ok", and picks up wherever it left off. Use --overwrite to force
  every category to be redone.

USAGE (Colab):
  !python caption.py --dir /content/drive/MyDrive/tracks --output-dir /content/captions
  # no args also works: defaults to scanning the current directory
  !python caption.py
"""

# ── Set BEFORE 'import torch'. Use "1" only for debugging device-side asserts. ──
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # allowed so --dry-run works without the GPU stack installed

from prompts import CAPTIONING_PROMPTS

# --- Bypass buggy torchcodec audio loading and force stable librosa/soundfile fallback ---
try:
    import transformers
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.is_torchcodec_available = lambda: False
    transformers.utils.import_utils.is_torchcodec_available = lambda: False
except ImportError:
    pass

try:
    from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
except ImportError:
    # Allowed so --dry-run can be used to sanity-check the pipeline without the
    # (large, GPU-only) model installed.
    MusicFlamingoForConditionalGeneration = None
    AutoProcessor = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MAX_TOKENS_CEILING = 4096
MAX_AUDIO_SEC = 300  # L4-safe audio cap; raise to 1200 (model max) on bigger GPUs
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

# NOTE: `model.config.max_position_embeddings` (and the *_config fallbacks) do NOT
# reflect this checkpoint's real usable context. On nvidia/music-flamingo-2601-hf that
# attribute resolves to ~1200, which is the *audio encoder's* per-window position count,
# not the language model's context length. The model card documents the real input
# ceiling as 24,000 tokens (long-context support via Rotary Time Embeddings), so that's
# what we gate generation against instead of trusting introspected config values.
MODEL_MAX_INPUT_TOKENS = 24000  # see HF model card: "Max Text Length: 24000 tokens" (input)


# ---------------------------------------------------------------------------
# CUDA context health probe
# ---------------------------------------------------------------------------


def cuda_is_healthy() -> bool:
    """False if a device-side assert has permanently poisoned the CUDA context."""
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Audio validation / prep
# ---------------------------------------------------------------------------


def validate_audio_file(audio_path: str) -> tuple[bool, str | None]:
    """CPU-side sanity check before anything touches the GPU."""
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
            return False, (
                f"Audio duration ({duration:.2f}s) exceeds the model's maximum of "
                f"1200s (20 minutes). Trim the file and retry."
            )
        return True, None
    except Exception as e:
        return False, f"Failed to load or parse audio file: {str(e)}"


def prepare_audio(audio_path: str, target_sr: int) -> tuple[str | None, float, str | None]:
    """
    Load + trim the track ONCE per track (not once per prompt) and write it to a
    temp WAV that every category's inference call reuses. Returns
    (tmp_wav_path, effective_duration_sec, error).
    """
    try:
        import librosa
        import soundfile as sf

        y, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
        max_samples = int(MAX_AUDIO_SEC * target_sr)
        duration_s = len(y) / target_sr

        if len(y) > max_samples:
            print(f"  [TRUNC] {duration_s:.1f}s -> {MAX_AUDIO_SEC}s (audio cap)")
            y = y[:max_samples]
            effective_duration_s = float(MAX_AUDIO_SEC)
        else:
            effective_duration_s = duration_s

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_path, y, target_sr)
        return tmp_path, effective_duration_s, None
    except Exception as e:
        return None, 0.0, f"Failed to prepare audio: {e}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_id: str, precision: str):
    print(f"Loading processor from {model_id!r} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model (precision={precision}) ...")
    if precision == "4bit":
        print(
            "  [QUALITY WARNING] 4-bit quantization is active. Output quality will be "
            "noticeably lower than bf16, especially for detailed analysis."
        )

    if precision == "bf16":
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    elif precision == "4bit":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
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
# Single-category inference (audio already prepared on disk)
# ---------------------------------------------------------------------------


def run_inference(
    model, processor, prepared_wav_path: str, prompt: str, max_new_tokens: int,
    effective_duration_s: float,
) -> dict:
    """Run one prompt against an already-prepared WAV file."""
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": prepared_wav_path},
                ],
            }
        ]

        start = time.time()
        batch = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        )

        audio_windows = None
        audio_covered_sec = None
        if "input_features" in batch and batch["input_features"] is not None:
            audio_windows = batch["input_features"].shape[0]
            audio_covered_sec = audio_windows * 30
            if audio_covered_sec < effective_duration_s - 30:
                print(
                    f"  [QUALITY WARNING] Processor only covered {audio_covered_sec}s "
                    f"of a {effective_duration_s:.1f}s track."
                )

        seq_len = batch["input_ids"].shape[1]
        if seq_len > MODEL_MAX_INPUT_TOKENS:
            raise ValueError(
                f"Tokenised sequence ({seq_len} tokens) exceeds Music Flamingo's "
                f"documented max input length ({MODEL_MAX_INPUT_TOKENS}). Reduce "
                f"MAX_AUDIO_SEC and retry."
            )

        batch = batch.to(model.device)
        if "input_features" in batch and batch["input_features"] is not None:
            batch["input_features"] = batch["input_features"].to(model.dtype)

        _precision = getattr(model, "_precision", "bf16")
        _device_type = next(model.parameters()).device.type
        _use_autocast = _precision == "bf16" and _device_type == "cuda"

        with torch.no_grad():
            with torch.autocast(device_type=_device_type, dtype=torch.bfloat16, enabled=_use_autocast):
                gen_ids = model.generate(**batch, max_new_tokens=max_new_tokens, repetition_penalty=1.2)

        inp_len = batch["input_ids"].shape[1]
        new_token_count = gen_ids.shape[1] - inp_len
        raw_output = processor.batch_decode(
            gen_ids[:, inp_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        elapsed = time.time() - start

        output_truncated = new_token_count >= max_new_tokens
        if output_truncated:
            print(f"  [QUALITY WARNING] Hit max_new_tokens ceiling ({max_new_tokens}); response may be truncated.")

        return {
            "status": "ok",
            "output": raw_output,
            "error": None,
            "traceback": None,
            "processing_time_sec": round(elapsed, 1),
            "audio_windows": audio_windows,
            "audio_covered_sec": audio_covered_sec,
            "output_truncated": output_truncated,
        }

    except Exception as e:
        return {
            "status": "error",
            "output": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time_sec": None,
            "audio_windows": None,
            "audio_covered_sec": None,
            "output_truncated": None,
        }


def run_inference_dry(prompt_key: str) -> dict:
    """Used by --dry-run to exercise the pipeline (file I/O, resume logic) without a GPU/model."""
    time.sleep(0.05)
    return {
        "status": "ok",
        "output": f"[DRY RUN placeholder output for '{prompt_key}']",
        "error": None,
        "traceback": None,
        "processing_time_sec": 0.1,
        "audio_windows": 1,
        "audio_covered_sec": 30,
        "output_truncated": False,
    }


# ---------------------------------------------------------------------------
# Per-track record: load / atomic save
# ---------------------------------------------------------------------------


def track_output_path(audio_path: str, output_dir: str) -> str:
    return os.path.join(output_dir, Path(audio_path).stem + ".caption.json")


def load_existing_record(out_path: str, audio_path: str) -> dict:
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record.setdefault("categories", {})
            return record
        except Exception:
            pass  # corrupt/partial file from an old run — start fresh
    return {
        "file": Path(audio_path).name,
        "path": str(Path(audio_path).resolve()),
        "model_id": MODEL_ID,
        "categories": {},
    }


def atomic_save_record(record: dict, out_path: str) -> None:
    """Write-to-temp-then-rename so a crash mid-write never corrupts the on-disk JSON."""
    record["last_updated"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_audio_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        sys.exit(f"Error: directory not found: {directory!r}")
    found = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(os.path.abspath(os.path.join(root, fname)))
    found.sort()
    return found


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------


def write_summary(output_dir: str, category_keys: list[str]) -> tuple[str, str]:
    """Scan every *.caption.json in output_dir and write a combined JSON + CSV summary."""
    json_records = []
    for p in sorted(Path(output_dir).glob("*.caption.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                json_records.append(json.load(f))
        except Exception:
            continue

    summary_json_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tracks": len(json_records),
                "categories": category_keys,
                "tracks": json_records,
            },
            f, indent=2, ensure_ascii=False,
        )

    summary_csv_path = os.path.join(output_dir, "results_summary.csv")
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "path"] + category_keys)
        for rec in json_records:
            row = [rec.get("file", ""), rec.get("path", "")]
            for key in category_keys:
                cat = rec.get("categories", {}).get(key, {})
                if cat.get("status") == "ok":
                    text = (cat.get("output") or "").replace("\n", " ").strip()
                else:
                    text = f"[{cat.get('status', 'missing')}] {cat.get('error') or ''}".strip()
                row.append(text)
            writer.writerow(row)

    return summary_json_path, summary_csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description="Run every prompt in prompts.CAPTIONING_PROMPTS against every audio file in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=str, default=".", help="Directory to scan recursively (default: current directory).")
    parser.add_argument("--file", type=str, default=None, help="Process a single file instead of --dir.")
    parser.add_argument("--output-dir", type=str, default="./captions/", dest="output_dir", help="Where per-track JSON + summary files go (default: ./captions/).")
    parser.add_argument("--categories", type=str, default="all", help="Comma-separated subset of prompts.py keys to run (default: all).")
    parser.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens", help="Max new tokens per prompt (default: 1024, ceiling: 4096).")
    parser.add_argument("--precision", type=str, choices=["bf16", "4bit"], default="bf16")
    parser.add_argument("--overwrite", action="store_true", help="Re-run categories even if already marked 'ok' in existing output.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Skip model loading/inference; exercise file I/O and resume logic with placeholder text.")
    return parser


# ---------------------------------------------------------------------------
# Per-track processing
# ---------------------------------------------------------------------------


def process_track(
    model, processor, audio_path: str, categories_to_run: list[str],
    output_dir: str, max_new_tokens: int, overwrite: bool, dry_run: bool,
    target_sr: int, cuda_poisoned: bool,
) -> bool:
    """Process every requested category for one track. Returns updated cuda_poisoned flag."""
    name = Path(audio_path).name
    out_path = track_output_path(audio_path, output_dir)
    record = load_existing_record(out_path, audio_path)

    pending = [
        c for c in categories_to_run
        if overwrite or record["categories"].get(c, {}).get("status") != "ok"
    ]
    if not pending:
        print(f"  -> already complete, skipping ({len(categories_to_run)} categories)")
        return cuda_poisoned

    if not os.path.isfile(audio_path):
        err = f"File not found: {audio_path!r}"
        for c in pending:
            record["categories"][c] = {"status": "error", "output": None, "error": err}
        atomic_save_record(record, out_path)
        print(f"  -> ERROR: {err}")
        return cuda_poisoned

    if not dry_run:
        is_valid, validation_err = validate_audio_file(audio_path)
        if not is_valid:
            for c in pending:
                record["categories"][c] = {"status": "error", "output": None, "error": validation_err}
            atomic_save_record(record, out_path)
            print(f"  -> SKIPPED: {validation_err}")
            return cuda_poisoned

    tmp_wav = None
    effective_duration_s = 0.0
    if not dry_run:
        tmp_wav, effective_duration_s, prep_err = prepare_audio(audio_path, target_sr)
        if prep_err:
            for c in pending:
                record["categories"][c] = {"status": "error", "output": None, "error": prep_err}
            atomic_save_record(record, out_path)
            print(f"  -> ERROR: {prep_err}")
            return cuda_poisoned

    try:
        for category in pending:
            if cuda_poisoned:
                record["categories"][category] = {
                    "status": "error", "output": None,
                    "error": "Skipped: CUDA context poisoned earlier in this run. Restart and re-run to resume.",
                }
                atomic_save_record(record, out_path)
                continue

            prompt = CAPTIONING_PROMPTS[category]
            print(f"  - {category} ...", end=" ", flush=True)

            if dry_run:
                result = run_inference_dry(category)
            else:
                result = run_inference(model, processor, tmp_wav, prompt, max_new_tokens, effective_duration_s)

            result["prompt_used"] = prompt
            record["categories"][category] = result
            atomic_save_record(record, out_path)  # <-- saved after EVERY category

            if result["status"] == "ok":
                print(f"ok ({result['processing_time_sec']}s)")
            else:
                print(f"ERROR: {result['error']}")
                err_str = result.get("error", "") or ""
                tb_str = result.get("traceback", "") or ""
                if not dry_run and ("CUDA" in err_str or "device-side assert" in err_str):
                    if tb_str:
                        print(f"  [CUDA TRACEBACK]\n{tb_str}")
                    if not cuda_is_healthy():
                        cuda_poisoned = True
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        print("  !! CUDA context poisoned — remaining categories/tracks will be skipped.")
    finally:
        if tmp_wav is not None and os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass

    return cuda_poisoned


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.max_tokens > MAX_TOKENS_CEILING:
        print(f"Warning: --max-tokens {args.max_tokens} exceeds ceiling of {MAX_TOKENS_CEILING}. Clamping.")
        args.max_tokens = MAX_TOKENS_CEILING

    all_keys = list(CAPTIONING_PROMPTS.keys())
    if args.categories == "all":
        categories_to_run = all_keys
    else:
        categories_to_run = [c.strip() for c in args.categories.split(",") if c.strip()]
        unknown = [c for c in categories_to_run if c not in CAPTIONING_PROMPTS]
        if unknown:
            parser.error(f"Unknown categories {unknown}. Available: {all_keys}")

    if args.file:
        tracks = [os.path.abspath(args.file)]
    else:
        tracks = discover_audio_files(args.dir)
        if not tracks:
            print(f"No audio files found under {args.dir!r}. Nothing to do.")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(tracks)} track(s). Categories: {categories_to_run}")

    if args.dry_run:
        model, processor, target_sr = None, None, 16000
        print("[DRY RUN] Skipping model load.")
    else:
        model, processor = load_model(MODEL_ID, args.precision)
        target_sr = getattr(getattr(processor, "feature_extractor", processor), "sampling_rate", 16000)

    cuda_poisoned = False
    for idx, audio_path in enumerate(tracks, start=1):
        print(f"[{idx}/{len(tracks)}] {Path(audio_path).name}")
        cuda_poisoned = process_track(
            model, processor, audio_path, categories_to_run, args.output_dir,
            args.max_tokens, args.overwrite, args.dry_run, target_sr, cuda_poisoned,
        )

    summary_json, summary_csv = write_summary(args.output_dir, all_keys)
    print(f"\nDone. Per-track JSON in {args.output_dir}")
    print(f"Summary: {summary_json}")
    print(f"Summary (CSV): {summary_csv}")


if __name__ == "__main__":
    main()
