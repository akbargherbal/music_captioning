import os

# ── Must be set before ANY other import ──────────────────────────────────────
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Keep this in case a future library version honours it, but don't rely on it:
# the current library has a hardcoded pre-Ampere branch that forces fp16 after
# checking this variable, so it has no effect on T4.  The real fix is the
# post-load cast in initialize_model() below.
os.environ["ACESTEP_DTYPE"] = "float32"
# Reduce VRAM fragmentation: allows the allocator to return memory to the OS
# and reuse non-contiguous free blocks, preventing OOM from fragmentation
# on constrained GPUs like the T4.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    VALID_KEYSCALES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)
from acestep.handler import AceStepHandler

DIT_MODELS = [
    "acestep-v15-turbo",
    "acestep-v15-sft",
    "acestep-v15-base",
    "acestep-v15-xl-turbo",
    "acestep-v15-xl-sft",
    "acestep-v15-xl-base",
]

XL_VRAM_REQUIREMENT_GB = 9
TURBO_DEFAULT_STEPS = 8
NON_TURBO_DEFAULT_STEPS = 50


def int_range(lo: int, hi: int):
    def _validate(value: str) -> int:
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected an integer, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} is out of range [{lo}, {hi}]")
        return v

    return _validate


def float_range(lo: float, hi: float):
    def _validate(value: str) -> float:
        try:
            v = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected a number, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} is out of range [{lo}, {hi}]")
        return v

    return _validate


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def is_lm_model_name(model_name: str) -> bool:
    return (model_name or "").strip().startswith("acestep-5Hz-lm-")


def is_turbo_model(model_name: str) -> bool:
    return "turbo" in (model_name or "").lower()


def is_xl_model(model_name: str) -> bool:
    return "-xl-" in (model_name or "").lower()


def repair_incomplete_dit_checkpoint(model_dir: Path) -> None:
    try:
        if not model_dir.is_dir():
            return
        silence_latent = model_dir / "silence_latent.pt"
        config_json = model_dir / "config.json"

        if silence_latent.is_file() and config_json.is_file():
            return

        suffix = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = model_dir.with_name(f"{model_dir.name}.incomplete-{suffix}")
        print(
            "Warning: Detected an incomplete DiT checkpoint folder. Moving it aside..."
        )
        shutil.move(str(model_dir), str(backup_dir))
    except Exception as exc:
        print(f"Warning: Could not repair checkpoint folder {model_dir}: {exc}")


def cast_handler_models_to_fp32(handler) -> int:
    """
    The ACE-Step library hard-codes float16 for pre-Ampere GPUs (T4 = compute 7.5)
    in its initialize_service() path, overriding any ACESTEP_DTYPE env-var.  fp16
    accumulates NaN/Inf during the 50-step DiT diffusion via softmax/LayerNorm/exp
    overflow, crashing at decode time with "Generation produced NaN or Inf latents".

    We cast the DiT and other modules to fp32, but leave the VAE in fp16.
    If the VAE is cast to fp32, its activations for 30s audio take ~704 MB,
    which exceeds the remaining VRAM on a T4 and causes an OOM.

    To prevent "mat1 and mat2 must have the same dtype" errors when the fp16 VAE
    outputs meet the fp32 DiT, we intercept all inputs to the DiT model and cast
    any fp16 tensors to fp32. We also patch tiled_decode to cast back to fp16.

    Returns the number of top-level modules recast (for the log message).
    """
    n = 0
    for name, attr in vars(handler).items():
        if isinstance(attr, nn.Module):
            if name == "vae":
                # Keep VAE in fp16 to save activation memory
                continue
            attr.float()
            n += 1

    def cast_fp16_to_fp32(obj):
        if torch.is_tensor(obj):
            if obj.dtype == torch.float16:
                return obj.float()
            return obj
        if isinstance(obj, dict):
            return {k: cast_fp16_to_fp32(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cast_fp16_to_fp32(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cast_fp16_to_fp32(v) for v in obj)
        return obj

    # 1. Patch DiT prepare_condition to accept fp16 and cast to fp32
    if hasattr(handler, "model") and hasattr(handler.model, "prepare_condition"):
        orig_prepare_condition = handler.model.prepare_condition

        def patched_prepare_condition(*args, **kwargs):
            args = cast_fp16_to_fp32(args)
            kwargs = cast_fp16_to_fp32(kwargs)
            return orig_prepare_condition(*args, **kwargs)

        handler.model.prepare_condition = patched_prepare_condition

    # 2. Patch DiT forward to accept fp16 and cast to fp32
    if hasattr(handler, "model"):

        def pre_hook(module, args, kwargs):
            return cast_fp16_to_fp32(args), cast_fp16_to_fp32(kwargs)

        handler.model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # 3. Patch tiled_decode to cast DiT's fp32 output back to fp16 for the VAE
    if hasattr(handler, "tiled_decode"):
        orig_tiled_decode = handler.tiled_decode

        def patched_tiled_decode(latents, *args, **kwargs):
            return orig_tiled_decode(latents.half(), *args, **kwargs)

        handler.tiled_decode = patched_tiled_decode

    return n


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate music with ACE-Step 1.5 (Colab/CUDA Optimized)"
    )

    parser.add_argument(
        "prompt", type=str, help="Text description of the music to generate"
    )
    parser.add_argument(
        "--duration",
        type=int_range(DURATION_MIN, DURATION_MAX),
        default=15,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--output", type=str, default="generated_music.wav", help="Output filename"
    )
    parser.add_argument(
        "--steps", type=int_range(1, 100), default=None, help="Inference steps"
    )
    parser.add_argument(
        "--guidance", type=float_range(1.0, 20.0), default=7.0, help="Guidance scale"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--lyrics-file",
        type=str,
        default=None,
        help="Path to text file containing lyrics",
    )
    parser.add_argument(
        "--bpm", type=int_range(BPM_MIN, BPM_MAX), default=None, help="Beats per minute"
    )
    parser.add_argument(
        "--key-scale", type=str, default="", help="Musical key (e.g., 'C major')"
    )
    parser.add_argument(
        "--time-signature", type=str, default="", help="Time signature (e.g., '4/4')"
    )
    parser.add_argument(
        "--vocal-language",
        type=str,
        choices=VALID_LANGUAGES,
        default="ar",
        help="Language code for vocals",
    )
    parser.add_argument(
        "--batch-size",
        type=int_range(1, 8),
        default=1,
        help="Number of variations to generate",
    )
    parser.add_argument(
        "--model", type=str, default="acestep-v15-turbo", help="DiT model variant"
    )
    parser.add_argument(
        "--lm-model", type=str, default="", help="Optional 5Hz LM model directory name"
    )
    parser.add_argument(
        "--infer-method",
        type=str,
        choices=["ode", "sde"],
        default="ode",
        help="Diffusion inference method",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="Diffusion sampler",
    )
    parser.add_argument(
        "--download-source",
        type=str,
        choices=["huggingface", "modelscope", "auto"],
        default="auto",
        help="Download source",
    )

    # ── Cover mode ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--source-audio",
        type=str,
        default=None,
        help=(
            "Path to source audio file. "
            "Providing this automatically switches the task to 'cover' unless "
            "--task-type is set explicitly."
        ),
    )
    parser.add_argument(
        "--cover-strength",
        type=float_range(0.0, 1.0),
        default=0.25,
        help=(
            "How closely the output follows the source audio's structure "
            "(melody contour, rhythm, song form). "
            "0.5–0.65 = creative reinterpretation, "
            "0.7–0.85 = melody shape preserved / style transformed (default: 0.75), "
            "0.9+ = very tight to original. Only used when --source-audio is set."
        ),
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["text2music", "cover"],
        default=None,
        help=(
            "Generation task. Defaults to 'text2music'. "
            "Set to 'cover' (or just supply --source-audio) for cover/instrumental mode."
        ),
    )
    parser.add_argument(
        "--max-cover-duration",
        type=int,
        default=60,
        help=(
            "Maximum source audio duration in seconds for cover mode (default: 60). "
            "Source audio longer than this is trimmed before processing to avoid "
            "VRAM exhaustion on the T4 when running in fp32. "
            "Set to 0 to disable trimming."
        ),
    )
    # ─────────────────────────────────────────────────────────────────────────

    return parser.parse_args()


def validate_key_scale(key_scale: str) -> None:
    if not key_scale:
        return
    normalized = key_scale.strip()
    if normalized not in VALID_KEYSCALES:
        print(f"Error: Invalid --key-scale value: {key_scale!r}")
        raise SystemExit(1)


def validate_time_signature(time_signature: str) -> None:
    if not time_signature:
        return
    parts = time_signature.strip().split("/")
    try:
        numerator = int(parts[0])
    except (ValueError, IndexError):
        print(f"Error: Cannot parse --time-signature: {time_signature!r}")
        raise SystemExit(1)
    if numerator not in VALID_TIME_SIGNATURES:
        print(f"Error: Invalid time-signature numerator {numerator!r}")
        raise SystemExit(1)


def load_lyrics(lyrics_file: str | None) -> str:
    if not lyrics_file:
        return ""
    try:
        with open(lyrics_file, encoding="utf-8") as f:
            lyrics = f.read().strip()
    except Exception as e:
        print(f"Error: Could not read lyrics file: {e}")
        raise SystemExit(1)

    if len(lyrics) > 4096:
        print("Error: Lyrics too long (maximum is 4096 characters).")
        raise SystemExit(1)
    return lyrics


def detect_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({gpu_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device} (Apple Silicon)")
    else:
        device = "cpu"
        print(f"Using device: {device} (No GPU acceleration)")
    return device


def get_project_root() -> str:
    return os.path.abspath(os.environ.get("ACESTEP_PROJECT_ROOT", os.getcwd()))


def initialize_model(
    model_name: str, device: str, project_root: str, prefer_source: str = "auto"
):
    print("Loading ACE-Step handler...")
    handler = AceStepHandler()

    model_path = Path(project_root) / "checkpoints" / model_name
    repair_incomplete_dit_checkpoint(model_path)

    # Pre-Ampere GPUs (T4 = compute 7.5) don't support FlashAttention.
    is_pre_ampere = device == "cuda" and torch.cuda.get_device_capability(0)[0] < 8
    use_flash_attention = device == "cuda" and not is_pre_ampere
    use_mlx_dit = False  # Disabled for Colab/Linux

    prefer_source_arg = None if prefer_source == "auto" else prefer_source

    model_load_start = time.time()
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=use_mlx_dit,
        prefer_source=prefer_source_arg,
    )

    # ── fp32 post-load cast ───────────────────────────────────────────────────
    # The library's pre-Ampere detection branch forces fp16 regardless of the
    # ACESTEP_DTYPE env-var (confirmed by "Pre-Ampere CUDA detected: using
    # float16" appearing in the log even after the env-var is set).
    #
    # The only reliable workaround without modifying the installed package is
    # to cast every nn.Module in the handler to float32 after load.  .float()
    # recasts all parameters and buffers in-place; subsequent forward passes
    # then execute in fp32, preventing the softmax/LayerNorm/exp overflow that
    # produces all-NaN latents ("Generation produced NaN or Inf latents").
    #
    # VRAM budget on T4 (14.6 GB):
    #   fp16 model ~6 GB → fp32 model ~12 GB, leaving ~2.6 GB free.
    #   Generation overhead at 60 s is ~1.1 GB → fits comfortably.
    if is_pre_ampere and success:
        n = cast_handler_models_to_fp32(handler)
        print(
            f"[fp32 cast] Recast {n} model module(s) to float32 "
            f"(pre-Ampere fp16 NaN workaround)."
        )
    # ─────────────────────────────────────────────────────────────────────────

    return handler, time.time() - model_load_start, success


def save_audio_outputs(result: dict, output_file: str) -> None:
    import soundfile as sf

    if "audios" not in result or not result["audios"]:
        print("Error: No audio generated")
        return

    audio_outputs = result["audios"]
    output_path = Path(output_file)

    def _write(audio_data, sample_rate: int, path: Path) -> None:
        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()
        if audio_data.ndim == 2:
            audio_data = audio_data.T
        sf.write(str(path), audio_data, sample_rate)
        print(f"Saved: {path}")

    if len(audio_outputs) == 1:
        _write(audio_outputs[0]["tensor"], audio_outputs[0]["sample_rate"], output_path)
    else:
        for i, audio_output in enumerate(audio_outputs):
            numbered = output_path.with_name(
                f"{output_path.stem}_{i + 1}{output_path.suffix}"
            )
            _write(audio_output["tensor"], audio_output["sample_rate"], numbered)


def trim_source_audio(src_path: str, max_seconds: int) -> tuple[str, bool]:
    """
    If the source audio exceeds max_seconds, write a trimmed copy to a temp file
    and return (temp_path, True).  Otherwise return (src_path, False).
    Caller is responsible for deleting the temp file when done.
    """
    import soundfile as sf
    import tempfile

    info = sf.info(src_path)
    if info.duration <= max_seconds:
        return src_path, False

    print(
        f"Info: Source audio is {info.duration:.1f}s — trimming to {max_seconds}s "
        f"to fit within T4 VRAM budget. Pass --max-cover-duration 0 to disable."
    )

    data, sr = sf.read(src_path, dtype="float32")
    max_frames = int(max_seconds * sr)
    trimmed = data[:max_frames] if data.ndim == 1 else data[:max_frames, :]

    suffix = Path(src_path).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    sf.write(tmp.name, trimmed, sr)
    return tmp.name, True


def main() -> None:
    args = parse_arguments()
    dit_model = (args.model or "").strip()
    lm_model = (args.lm_model or "").strip()

    # ── Resolve task type & cover-mode defaults ───────────────────────────────
    if args.task_type is None:
        task_type = "cover" if args.source_audio else "text2music"
    else:
        task_type = args.task_type

    if task_type == "cover":
        if not args.source_audio:
            print("Error: --task-type cover requires --source-audio <path>.")
            raise SystemExit(1)

        if args.duration == 15:
            args.duration = 60

        if dit_model == "acestep-v15-turbo":
            dit_model = "acestep-v15-sft"
            print(
                "Info: Switched model to 'acestep-v15-sft' for cover mode "
                "(turbo has lower structure fidelity). "
                "Override with --model acestep-v15-turbo if you prefer speed."
            )

        _tmp_audio_created = False
        if args.max_cover_duration and args.max_cover_duration > 0:
            args.source_audio, _tmp_audio_created = trim_source_audio(
                args.source_audio, args.max_cover_duration
            )
    else:
        _tmp_audio_created = False
    # ─────────────────────────────────────────────────────────────────────────

    if is_lm_model_name(dit_model) and not lm_model:
        lm_model = dit_model
        dit_model = "acestep-v15-turbo"

    if args.steps is None:
        args.steps = (
            TURBO_DEFAULT_STEPS
            if is_turbo_model(dit_model)
            else NON_TURBO_DEFAULT_STEPS
        )

    if is_turbo_model(dit_model) and args.steps > TURBO_DEFAULT_STEPS:
        args.steps = TURBO_DEFAULT_STEPS

    validate_key_scale(args.key_scale)
    validate_time_signature(args.time_signature)

    output_path = Path(args.output)
    if output_path.suffix.lower() not in {".wav", ".flac", ".mp3", ".opus", ".aac"}:
        args.output = str(output_path.with_suffix(".wav"))

    lyrics = load_lyrics(args.lyrics_file)
    total_start = time.time()
    device = detect_device()
    project_root = get_project_root()

    if lm_model:
        try:
            from acestep.model_downloader import ensure_lm_model

            ensure_lm_model(
                model_name=lm_model, checkpoints_dir=Path(project_root) / "checkpoints"
            )
        except Exception:
            pass

    handler, model_load_time, success = initialize_model(
        dit_model, device, project_root, args.download_source
    )
    if not success:
        raise SystemExit(1)

    print(f"\nGenerating: {args.prompt}")

    # Flush any cached but unallocated VRAM before the generation pass so that
    # the VAE encoder and DiT diffusion steps have maximum contiguous memory.
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    generation_start = time.time()
    result = handler.generate_music(
        captions=args.prompt,
        lyrics=lyrics,
        bpm=args.bpm,
        key_scale=args.key_scale,
        time_signature=args.time_signature,
        vocal_language=args.vocal_language,
        audio_duration=float(args.duration),
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_random_seed=(args.seed == -1),
        seed=args.seed,
        batch_size=args.batch_size,
        task_type=task_type,
        infer_method=args.infer_method,
        sampler_mode=args.sampler,
        **(
            {
                "src_audio": args.source_audio,
                "audio_cover_strength": args.cover_strength,
            }
            if task_type == "cover"
            else {}
        ),
    )
    generation_time = time.time() - generation_start

    save_audio_outputs(result, args.output)

    if _tmp_audio_created:
        try:
            os.unlink(args.source_audio)
        except Exception:
            pass

    if "audios" in result and result["audios"]:
        print(f"\nTiming Summary:")
        print(f"   Model loading:    {format_duration(model_load_time)}")
        print(f"   Music generation: {format_duration(generation_time)}")
        print(f"   Total elapsed:    {format_duration(time.time() - total_start)}")


if __name__ == "__main__":
    main()
