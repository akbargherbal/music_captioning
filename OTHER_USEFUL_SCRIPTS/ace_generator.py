import os

# ── Must be set before ANY other import ──────────────────────────────────────
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["ACESTEP_DTYPE"] = "float32"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Force float32 globally — library ignores ACESTEP_DTYPE on pre-Ampere GPUs
# ─────────────────────────────────────────────────────────────────────────────
# WHY THREE LAYERS OF PATCHES:
#
#   Layer 1 — Module.half() / Module.to():  stops weight downcasting for any
#             nn.Module that the library calls .half() or .to(fp16) on.
#
#   Layer 2 — Tensor.half() / Tensor.to(): stops *activation* tensors being
#             explicitly cast to fp16 by preprocessing code that lives outside
#             any nn.Module (e.g. the silence-latent creation path that
#             produces `refer_audio_acoustic_hidden_states_packed` as Half).
#             This is the direct cause of the crash you saw.
#
#   Layer 3 — F.linear safety net: if any fp16 tensor still slips through
#             (e.g. via torch.zeros(..., dtype=torch.float16) which we cannot
#             intercept without extreme measures), auto-cast the input to match
#             the weight dtype at the last possible moment instead of crashing.
# ─────────────────────────────────────────────────────────────────────────────

torch.set_default_dtype(torch.float32)

# ── Layer 1: Module-level patches ────────────────────────────────────────────
_original_module_half = torch.nn.Module.half
_original_module_to   = torch.nn.Module.to


def _patched_module_half(self):
    return self  # silently refuse to downcast


def _patched_module_to(self, *args, **kwargs):
    args = tuple(torch.float32 if a in (torch.float16, torch.half) else a for a in args)
    if kwargs.get("dtype") in (torch.float16, torch.half):
        kwargs["dtype"] = torch.float32
    return _original_module_to(self, *args, **kwargs)


torch.nn.Module.half = _patched_module_half
torch.nn.Module.to   = _patched_module_to

# ── Layer 2: Tensor-level patches ────────────────────────────────────────────
# These are needed because intermediate activation tensors (e.g. the silence
# latent passed into AceStepTimbreEncoder) are created as fp16 by library code
# that calls tensor.half() or tensor.to(torch.float16) *directly*, not via any
# nn.Module method.  Without these patches the Module-level patches above are
# bypassed entirely for those tensors.
_original_tensor_half = torch.Tensor.half
_original_tensor_to   = torch.Tensor.to


def _patched_tensor_half(self):
    return self.float()  # up-cast to fp32 instead of staying fp16


def _patched_tensor_to(self, *args, **kwargs):
    args = tuple(torch.float32 if a in (torch.float16, torch.half) else a for a in args)
    if kwargs.get("dtype") in (torch.float16, torch.half):
        kwargs["dtype"] = torch.float32
    return _original_tensor_to(self, *args, **kwargs)


torch.Tensor.half = _patched_tensor_half
torch.Tensor.to   = _patched_tensor_to

# ── Layer 3: F.linear safety net ─────────────────────────────────────────────
# Catches any fp16 tensor that was created via torch.zeros/ones/randn with an
# explicit dtype kwarg (which we cannot intercept without replacing the entire
# torch tensor factory, an extreme measure).  Silently up-casts the input to
# match the weight dtype so the matmul succeeds.
_original_F_linear = F.linear


def _safe_F_linear(input, weight, bias=None):
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return _original_F_linear(input, weight, bias)


F.linear = _safe_F_linear
# Also patch the module-level reference used by nn.Linear internally
import torch.nn.functional as _F_module
_F_module.linear = _safe_F_linear
# ─────────────────────────────────────────────────────────────────────────────

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
        default="en",
        help="Language code for vocals",
    )
    parser.add_argument(
        "--batch-size",
        type=int_range(1, 8),
        default=1,  # lowered from 2 — T4 has limited VRAM headroom in fp32
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
    # The library detects this and falls back to SDPA, but we also pass
    # use_flash_attention=False explicitly to avoid any race in detection.
    is_pre_ampere = (
        device == "cuda"
        and torch.cuda.get_device_capability(0)[0] < 8
    )
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
    return handler, time.time() - model_load_start, success


def cast_model_to_float32(handler) -> None:
    """
    from_pretrained() loads safetensor weights directly as fp16 parameters,
    bypassing all Module.to() / Module.half() patches applied at import time.
    Walk every submodule and cast any fp16 parameter or buffer to fp32.
    Using the *original* Module.to() avoids our own patch intercepting the call.
    """
    if not (hasattr(handler, "model") and handler.model is not None):
        return

    print("Casting model weights to float32 (fp16 safetensor fix)...")
    converted = 0
    for name, module in handler.model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.dtype in (torch.float16, torch.half):
                new_param = torch.nn.Parameter(
                    param.data.float(), requires_grad=param.requires_grad
                )
                setattr(module, param_name, new_param)
                converted += 1
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.dtype in (torch.float16, torch.half):
                setattr(module, buf_name, buf.float())
                converted += 1

    print(f"Model is now fully float32 ({converted} tensors converted).")


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


def main() -> None:
    args = parse_arguments()
    dit_model = (args.model or "").strip()
    lm_model = (args.lm_model or "").strip()

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

    cast_model_to_float32(handler)

    print(f"\nGenerating: {args.prompt}")

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
        task_type="text2music",
        infer_method=args.infer_method,
        sampler_mode=args.sampler,
    )
    generation_time = time.time() - generation_start

    save_audio_outputs(result, args.output)

    if "audios" in result and result["audios"]:
        print(f"\nTiming Summary:")
        print(f"   Model loading:    {format_duration(model_load_time)}")
        print(f"   Music generation: {format_duration(generation_time)}")
        print(f"   Total elapsed:    {format_duration(time.time() - total_start)}")


if __name__ == "__main__":
    main()
