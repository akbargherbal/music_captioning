import os

# ── Must be set before ANY other import ──────────────────────────────────────
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# T4 is pre-Ampere (compute 7.5); the library forces fp16 regardless of this
# env-var, so the real fix is the post-load cast in initialize_model().
os.environ["ACESTEP_DTYPE"] = "float32"
# Reduces VRAM fragmentation on constrained GPUs like the T4.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import shutil
import subprocess
import tempfile
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
    VALID_TIME_SIGNATURES,
)
from acestep.handler import AceStepHandler

# ── Constants ─────────────────────────────────────────────────────────────────
TURBO_DEFAULT_STEPS = 8
NON_TURBO_DEFAULT_STEPS = 50

# Default prompt engineered for a clean instrumental cover:
#   - Explicitly removes vocals via multiple phrasings (the model responds to
#     redundancy here — one phrase is often not enough).
#   - Stays genre-neutral so it doesn't fight the source material's character.
#   - "cinematic" biases toward full arrangement rather than sparse output.
DEFAULT_PROMPT = (
    "cinematic instrumental arrangement, no vocals, no singing, no voice, "
    "melodic and expressive, orchestral with piano and strings, "
    "clean mix, studio quality"
)

# Best balance between structure fidelity and style transformation.
# 0.25 (old BETA default) is too low — causes noise artefacts.
# 0.9 is too high — near-ignores the source.
DEFAULT_COVER_STRENGTH = 0.75

# Maximum source duration passed to ACE-Step (after Demucs separation).
# Demucs itself handles full-length files; this only limits the ACE-Step pass.
DEFAULT_MAX_DURATION = 60

# Demucs model: htdemucs is the standard 4-stem model.
# htdemucs_ft is fine-tuned and slightly better but slower — user can override.
DEFAULT_DEMUCS_MODEL = "htdemucs"
# ─────────────────────────────────────────────────────────────────────────────


# ── Argument helpers ──────────────────────────────────────────────────────────
def int_range(lo: int, hi: int):
    def _validate(value: str) -> int:
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected an integer, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} out of range [{lo}, {hi}]")
        return v
    return _validate


def float_range(lo: float, hi: float):
    def _validate(value: str) -> float:
        try:
            v = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected a number, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} out of range [{lo}, {hi}]")
        return v
    return _validate


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m {seconds % 60:.1f}s"
# ─────────────────────────────────────────────────────────────────────────────


# ── Demucs separation ─────────────────────────────────────────────────────────
def separate_instrumental(
    src_path: str,
    demucs_model: str = DEFAULT_DEMUCS_MODEL,
    output_dir: str | None = None,
) -> str:
    """
    Run Demucs on src_path with --two-stems=vocals, which produces exactly
    two stems: vocals.wav and no_vocals.wav (the full instrumental minus voice).

    Returns the path to no_vocals.wav.

    Uses a temp directory if output_dir is not specified; caller owns cleanup.
    """
    src = Path(src_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source audio not found: {src}")

    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="gamma_demucs_"))

    print(f"\n[demucs] Separating vocals from: {src.name}")
    print(f"[demucs] Model: {demucs_model}  |  Output dir: {work_dir}")

    cmd = [
        "python", "-m", "demucs",
        "--two-stems=vocals",       # produce only vocals + no_vocals
        "-n", demucs_model,         # Demucs 4.x uses -n, not --model
        "-o", str(work_dir),
        str(src),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,   # let Demucs print its own progress bar
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Demucs failed (exit code {exc.returncode}). "
            "Make sure it is installed: pip install demucs"
        ) from exc
    except FileNotFoundError:
        raise RuntimeError(
            "Could not find 'python -m demucs'. "
            "Install with: pip install demucs"
        )

    elapsed = time.time() - t0
    print(f"[demucs] Separation finished in {format_duration(elapsed)}")

    # Demucs writes to: <work_dir>/<model>/<track_stem>/no_vocals.wav
    track_stem = src.stem
    instrumental_path = work_dir / demucs_model / track_stem / "no_vocals.wav"

    if not instrumental_path.is_file():
        # Fallback: search recursively in case Demucs changed its layout
        candidates = list(work_dir.rglob("no_vocals.wav"))
        if not candidates:
            raise FileNotFoundError(
                f"Demucs ran but no_vocals.wav was not found under {work_dir}. "
                "Check Demucs output above for errors."
            )
        instrumental_path = candidates[0]
        print(f"[demucs] Warning: found stem at non-standard path: {instrumental_path}")

    print(f"[demucs] Instrumental stem: {instrumental_path}")
    return str(instrumental_path)


def trim_audio(src_path: str, max_seconds: int) -> tuple[str, bool]:
    """
    Trim audio to max_seconds if it exceeds that length.
    Returns (path, was_trimmed). Caller must delete the temp file if was_trimmed.
    """
    import soundfile as sf

    info = sf.info(src_path)
    if info.duration <= max_seconds:
        return src_path, False

    print(
        f"[trim] Instrumental is {info.duration:.1f}s — trimming to {max_seconds}s "
        f"for T4 VRAM budget. Use --max-duration 0 to disable."
    )
    data, sr = sf.read(src_path, dtype="float32")
    max_frames = int(max_seconds * sr)
    trimmed = data[:max_frames] if data.ndim == 1 else data[:max_frames, :]

    suffix = Path(src_path).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    sf.write(tmp.name, trimmed, sr)
    return tmp.name, True
# ─────────────────────────────────────────────────────────────────────────────


# ── Model helpers (ported + improved from BETA) ───────────────────────────────
def is_turbo_model(name: str) -> bool:
    return "turbo" in (name or "").lower()


def is_xl_model(name: str) -> bool:
    return "-xl-" in (name or "").lower()


def repair_incomplete_dit_checkpoint(model_dir: Path) -> None:
    try:
        if not model_dir.is_dir():
            return
        if (model_dir / "silence_latent.pt").is_file() and (model_dir / "config.json").is_file():
            return
        suffix = time.strftime("%Y%m%d-%H%M%S")
        backup = model_dir.with_name(f"{model_dir.name}.incomplete-{suffix}")
        print(f"[checkpoint] Incomplete checkpoint detected — moving aside to {backup.name}")
        shutil.move(str(model_dir), str(backup))
    except Exception as exc:
        print(f"[checkpoint] Warning: could not repair checkpoint folder: {exc}")


def cast_handler_models_to_fp32(handler) -> int:
    """
    T4 (compute 7.5) is pre-Ampere: the ACE-Step library hard-codes fp16 for it,
    which causes softmax/LayerNorm overflow → all-NaN latents at decode time.

    Fix: cast every module except the VAE to fp32 after load.
    The VAE stays in fp16 to stay within T4 VRAM (fp32 VAE needs ~700 MB extra).

    Dtype boundary bridging:
      - DiT prepare_condition: cast fp16 inputs → fp32
      - DiT forward pre-hook: cast fp16 inputs → fp32
      - tiled_decode: cast fp32 latents → fp16 before handing to VAE

    Returns number of top-level modules recast.
    """
    n = 0
    for name, attr in vars(handler).items():
        if isinstance(attr, nn.Module):
            if name == "vae":
                continue   # keep fp16 to save ~700 MB activation memory
            attr.float()
            n += 1

    def _to_fp32(obj):
        if torch.is_tensor(obj):
            return obj.float() if obj.dtype == torch.float16 else obj
        if isinstance(obj, dict):
            return {k: _to_fp32(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_to_fp32(v) for v in obj]
            return type(obj)(converted)
        return obj

    if hasattr(handler, "model") and hasattr(handler.model, "prepare_condition"):
        orig = handler.model.prepare_condition
        def patched_prepare_condition(*a, **kw):
            return orig(*_to_fp32(a), **_to_fp32(kw))
        handler.model.prepare_condition = patched_prepare_condition

    if hasattr(handler, "model"):
        def pre_hook(module, args, kwargs):
            return _to_fp32(args), _to_fp32(kwargs)
        handler.model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    if hasattr(handler, "tiled_decode"):
        orig_td = handler.tiled_decode
        def patched_tiled_decode(latents, *a, **kw):
            return orig_td(latents.half(), *a, **kw)
        handler.tiled_decode = patched_tiled_decode

    return n


def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA — {name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("[device] MPS — Apple Silicon")
        return "mps"
    print("[device] CPU (no GPU acceleration)")
    return "cpu"


def initialize_model(model_name: str, device: str, project_root: str, prefer_source: str = "auto"):
    print(f"\n[model] Loading ACE-Step handler ({model_name})...")
    handler = AceStepHandler()

    model_path = Path(project_root) / "checkpoints" / model_name
    repair_incomplete_dit_checkpoint(model_path)

    is_pre_ampere = device == "cuda" and torch.cuda.get_device_capability(0)[0] < 8
    use_flash_attention = device == "cuda" and not is_pre_ampere

    t0 = time.time()
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=False,
        prefer_source=None if prefer_source == "auto" else prefer_source,
    )

    if is_pre_ampere and success:
        n = cast_handler_models_to_fp32(handler)
        print(f"[fp32] Recast {n} module(s) to float32 (pre-Ampere NaN workaround).")

    return handler, time.time() - t0, success
# ─────────────────────────────────────────────────────────────────────────────


# ── Audio output ──────────────────────────────────────────────────────────────
def save_audio_outputs(result: dict, output_file: str) -> None:
    import soundfile as sf

    if not result.get("audios"):
        print("[output] Error: no audio generated.")
        return

    outputs = result["audios"]
    out_path = Path(output_file)

    def _write(tensor, sample_rate: int, path: Path) -> None:
        data = tensor.cpu().numpy() if torch.is_tensor(tensor) else tensor
        if data.ndim == 2:
            data = data.T
        sf.write(str(path), data, sample_rate)
        print(f"[output] Saved: {path}")

    if len(outputs) == 1:
        _write(outputs[0]["tensor"], outputs[0]["sample_rate"], out_path)
    else:
        for i, o in enumerate(outputs):
            numbered = out_path.with_name(f"{out_path.stem}_{i + 1}{out_path.suffix}")
            _write(o["tensor"], o["sample_rate"], numbered)
# ─────────────────────────────────────────────────────────────────────────────


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GAMMA — Demucs-first instrumental cover generator for ACE-Step 1.5.\n"
            "\n"
            "Pipeline:\n"
            "  1. Demucs separates vocals from the source track → no_vocals.wav\n"
            "  2. ACE-Step cover mode uses the clean instrumental stem as input\n"
            "  3. Output is a new instrumental arrangement following the source structure\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "source_audio",
        type=str,
        help="Path to the source audio file (any format Demucs accepts: mp3, wav, flac, …)",
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=(
            f"Text description of the desired output.\n"
            f"Default: \"{DEFAULT_PROMPT}\""
        ),
    )

    # ── Cover settings ────────────────────────────────────────────────────────
    parser.add_argument(
        "--cover-strength",
        type=float_range(0.0, 1.0),
        default=DEFAULT_COVER_STRENGTH,
        help=(
            "How much the diffusion process transforms the source.\n"
            "  0.50–0.65 = loose interpretation, maximum style freedom\n"
            "  0.65–0.80 = melody shape preserved, style transformed  ← sweet spot\n"
            "  0.80–0.95 = very tight to source structure\n"
            f"Default: {DEFAULT_COVER_STRENGTH}"
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=DEFAULT_MAX_DURATION,
        help=(
            f"Maximum seconds of the instrumental stem fed to ACE-Step (default: {DEFAULT_MAX_DURATION}). "
            "Demucs always processes the full file. "
            "Set to 0 to disable trimming (risks VRAM OOM on long tracks)."
        ),
    )

    # ── Demucs settings ───────────────────────────────────────────────────────
    parser.add_argument(
        "--demucs-model",
        type=str,
        default=DEFAULT_DEMUCS_MODEL,
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"],
        help=(
            "Demucs model to use for separation.\n"
            "  htdemucs    = fast, good quality          ← default\n"
            "  htdemucs_ft = fine-tuned, best quality, ~2× slower\n"
            "  mdx_extra   = alternative architecture\n"
        ),
    )
    parser.add_argument(
        "--demucs-output-dir",
        type=str,
        default=None,
        help=(
            "Where to write Demucs stems (default: auto temp dir, cleaned up after run). "
            "Pass a real path to keep the separated stems for inspection."
        ),
    )
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help=(
            "Skip Demucs and treat source_audio as already an instrumental stem. "
            "Useful if you ran Demucs separately or already have a clean stem."
        ),
    )

    # ── ACE-Step model ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="acestep-v15-sft",   # turbo has lower structure fidelity for cover
        help="DiT model variant (default: acestep-v15-sft — better than turbo for cover mode)",
    )
    parser.add_argument(
        "--steps",
        type=int_range(1, 100),
        default=None,
        help="Diffusion steps (default: 8 for turbo models, 50 otherwise)",
    )
    parser.add_argument(
        "--guidance",
        type=float_range(1.0, 20.0),
        default=7.0,
        help="Classifier-free guidance scale (default: 7.0)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default="gamma_cover.wav",
        help="Output filename (default: gamma_cover.wav)",
    )
    parser.add_argument(
        "--duration",
        type=int_range(DURATION_MIN, DURATION_MAX),
        default=60,
        help="Output duration in seconds (default: 60)",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 = random)",
    )
    parser.add_argument(
        "--bpm",
        type=int_range(BPM_MIN, BPM_MAX),
        default=None,
        help="Override BPM (leave unset to let the model infer from source)",
    )
    parser.add_argument(
        "--key-scale",
        type=str,
        default="",
        help="Override musical key (e.g. 'C major'). Leave empty to let model infer.",
    )
    parser.add_argument(
        "--infer-method",
        type=str,
        choices=["ode", "sde"],
        default="ode",
        help="Diffusion inference method (default: ode)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="Diffusion sampler (default: euler)",
    )
    parser.add_argument(
        "--download-source",
        type=str,
        choices=["huggingface", "modelscope", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int_range(1, 8),
        default=1,
        help="Number of variations to generate (default: 1)",
    )

    return parser.parse_args()
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_arguments()

    # ── Resolve key/time validation ───────────────────────────────────────────
    if args.key_scale:
        normalized = args.key_scale.strip()
        if normalized not in VALID_KEYSCALES:
            print(f"Error: Invalid --key-scale value: {args.key_scale!r}")
            raise SystemExit(1)

    # ── Resolve steps ─────────────────────────────────────────────────────────
    if args.steps is None:
        args.steps = TURBO_DEFAULT_STEPS if is_turbo_model(args.model) else NON_TURBO_DEFAULT_STEPS

    if is_turbo_model(args.model) and args.steps > TURBO_DEFAULT_STEPS:
        args.steps = TURBO_DEFAULT_STEPS

    # ── Output extension ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".wav", ".flac", ".mp3", ".opus", ".aac"}:
        args.output = str(out_path.with_suffix(".wav"))

    print("=" * 70)
    print("  GAMMA — Demucs-first Instrumental Cover Generator")
    print("=" * 70)
    print(f"  Source : {args.source_audio}")
    print(f"  Prompt : {args.prompt}")
    print(f"  Strength: {args.cover_strength}  |  Duration: {args.duration}s  |  Steps: {args.steps}")
    print(f"  Model  : {args.model}")
    print("=" * 70)

    total_start = time.time()
    demucs_tmp_created = False
    trimmed_tmp_created = False
    instrumental_path = args.source_audio

    # ── Step 1: Demucs separation ─────────────────────────────────────────────
    if args.skip_demucs:
        print("\n[demucs] Skipping separation (--skip-demucs set).")
        print(f"[demucs] Using source directly: {instrumental_path}")
    else:
        if args.demucs_output_dir is None:
            # Auto temp dir — we clean it up at the end
            demucs_tmp_dir = tempfile.mkdtemp(prefix="gamma_demucs_")
            demucs_tmp_created = True
        else:
            demucs_tmp_dir = args.demucs_output_dir
            demucs_tmp_created = False

        instrumental_path = separate_instrumental(
            src_path=args.source_audio,
            demucs_model=args.demucs_model,
            output_dir=demucs_tmp_dir,
        )

    # ── Step 2: Trim instrumental for VRAM budget ─────────────────────────────
    if args.max_duration and args.max_duration > 0:
        instrumental_path, trimmed_tmp_created = trim_audio(
            instrumental_path, args.max_duration
        )

    # ── Step 3: Load ACE-Step ─────────────────────────────────────────────────
    device = detect_device()
    project_root = os.path.abspath(os.environ.get("ACESTEP_PROJECT_ROOT", os.getcwd()))

    handler, model_load_time, success = initialize_model(
        args.model, device, project_root, args.download_source
    )
    if not success:
        raise SystemExit(1)

    # ── Step 4: Generate ──────────────────────────────────────────────────────
    # Empty lyrics with no language tag — critical for a true instrumental.
    # BETA had 'ar' hard-coded as the default vocal-language, which conditioned
    # the model toward Arabic singing even when lyrics were empty.
    # Here we pass an empty string so no vocal-language token is injected.
    lyrics = ""

    print(f"\n[generate] Prompt: {args.prompt}")
    print(f"[generate] Instrumental stem: {instrumental_path}")

    gc.collect()
    torch.cuda.empty_cache()

    gen_start = time.time()
    result = handler.generate_music(
        captions=args.prompt,
        lyrics=lyrics,
        bpm=args.bpm,
        key_scale=args.key_scale,
        time_signature="",          # let model infer from source
        vocal_language="",          # no vocal-language bias — key fix vs BETA
        audio_duration=float(args.duration),
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_random_seed=(args.seed == -1),
        seed=args.seed,
        batch_size=args.batch_size,
        task_type="cover",
        infer_method=args.infer_method,
        sampler_mode=args.sampler,
        src_audio=instrumental_path,
        audio_cover_strength=args.cover_strength,
    )
    gen_time = time.time() - gen_start

    # ── Step 5: Save output ───────────────────────────────────────────────────
    save_audio_outputs(result, args.output)

    # ── Cleanup temp files ────────────────────────────────────────────────────
    if trimmed_tmp_created:
        try:
            os.unlink(instrumental_path)
        except Exception:
            pass

    if demucs_tmp_created:
        try:
            shutil.rmtree(demucs_tmp_dir, ignore_errors=True)
            print(f"[cleanup] Removed temp Demucs dir: {demucs_tmp_dir}")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    if result.get("audios"):
        print("\nTiming Summary:")
        print(f"   Model loading:    {format_duration(model_load_time)}")
        print(f"   Music generation: {format_duration(gen_time)}")
        print(f"   Total elapsed:    {format_duration(time.time() - total_start)}")
        print(f"\n   Output: {args.output}")


if __name__ == "__main__":
    main()
