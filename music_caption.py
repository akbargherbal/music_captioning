"""
# Music Captioning with Music Flamingo (nvidia/music-flamingo-hf)
Optimised for Google Colab T4 (15 GB VRAM).

Instead of captioning every sequential chunk, the script extracts **four
landmark windows** that together summarise a track efficiently:

| # | Label      | Window                              |
|---|------------|-------------------------------------|
| 1 | `start`    | 0 → W                               |
| 2 | `early`    | W → 2W                              |
| 3 | `pre_mid`  | (mid − W) → mid                     |
| 4 | `post_mid` | mid → (mid + W)                     |

where `W` = `--window-seconds` (default 40 s) and `mid` = duration / 2.

Synthesis of chunk captions is intentionally left to a separate LLM step.

## Setup

```bash
pip install --upgrade pip
pip install --upgrade "git+https://github.com/huggingface/transformers" accelerate bitsandbytes
pip install torch torchaudio
```

## Usage

```bash
# Quick: one dense comma-separated summary per landmark chunk (default)
python music_caption.py --audio track.mp3

# Detailed: four focused tag-cluster prompts per chunk (genre, instruments, vocals, tempo/key)
python music_caption.py --audio track.mp3 --mode detailed

# Custom window size (default 40 s)
python music_caption.py --audio track.mp3 --window-seconds 30
```

## Output format

Captions are compact, comma-separated tag strings — no prose, no sentences.
Modelled on Suno.ai style descriptions. Example:

  "blues rock, electric guitar with slide technique, steady snare backbeat,
   walking bass line, male tenor with raspy delivery, 92 BPM, E minor, 4/4"

The resulting JSON is passed directly to a synthesis LLM (separate script)
which combines the landmark chunk captions into a single final style string.

## Output JSON Schema

```json
{
  "metadata": {
    "model": "nvidia/music-flamingo-hf",
    "generated_at": "2025-...",
    "audio_file": "/abs/path/track.mp3",
    "duration_seconds": 300.0,
    "window_seconds": 40,
    "total_chunks": 4,
    "mode": "quick"
  },
  "chunks": [
    {
      "chunk_index": 0,
      "label": "start",
      "start_seconds": 0,
      "end_seconds": 40,
      "captions": {
        "summary": "Khaleeji Arabic pop, prominent oud ..."
      }
    }
  ]
}
```
"""

# PYTORCH_ALLOC_CONF must be set before the very first `import torch`.
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import torch

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav"}


def _configure_logging(log_path: Path) -> None:
    """Attach a stdout handler and a file handler to the root logger."""
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    log.info(f"Logging to {log_path}")

# ---------------------------------------------------------------------------
# Prompt catalogue
# ---------------------------------------------------------------------------
#
# Output target: dense, comma-separated tags — no prose, no sentences.
# Modelled on Suno.ai style descriptions, e.g.:
#   "jazz fusion, Rhodes electric piano comping, fretless bass with slides,
#    brushed snare and ride cymbal, female alto with breathy delivery,
#    118 BPM in 7/8 time"
#
# Synthesis into a single final caption is handled downstream by a separate
# LLM step — these prompts only need to extract raw facts cleanly.
# ---------------------------------------------------------------------------

# quick: one prompt, one dense output per chunk — feeds the synthesis LLM
PROMPTS_QUICK = {
    "summary": (
        "Describe this music excerpt as a compact, comma-separated style tag string. "
        "Cover in order: genre and regional style, key instruments with their playing technique, "
        "percussion pattern and drum character, bass character, vocal style and ornamentation, "
        "tempo in BPM, musical key or maqam, and time signature. "
        "Use noun phrases only — no full sentences, no prose, no preamble. "
        "Example format: 'blues rock, electric guitar with slide technique, "
        "steady snare backbeat, walking bass line, male tenor with raspy delivery, "
        "92 BPM, E minor, 4/4'"
    ),
}

# detailed: four focused prompts — each returns a compact tag cluster
PROMPTS = {
    "genre_style": (
        "State the genre, subgenre, and regional style of this excerpt. "
        "Comma-separated noun phrases only — no sentences. "
        "Example: 'jazz, bebop, contemporary American, live recording feel'"
    ),
    "instruments": (
        "List every instrument audible in this excerpt with its playing technique or character. "
        "Comma-separated noun phrases only — no sentences. "
        "Example: 'electric guitar with clean fingerpicking, upright bass with arco bowing, "
        "brushed snare and ride cymbal, piano comping with sparse voicings'"
    ),
    "vocals": (
        "Describe the vocal style, voice type, delivery, and ornamentation in this excerpt. "
        "Comma-separated noun phrases only — no sentences. "
        "Example: 'female mezzo-soprano, legato phrasing, minimal ornamentation, "
        "occasional head voice breaks, restrained vibrato'"
    ),
    "tempo_key": (
        "State the tempo, musical key or maqam, and time signature of this excerpt. "
        "Comma-separated noun phrases only — no sentences. "
        "Example: '76 BPM, G major, 3/4 time'"
    ),
}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(dtype: str = "auto", flash_attn: bool = False, quantize: str = "4bit"):
    try:
        from transformers import (
            AudioFlamingo3ForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError:
        log.error(
            "AudioFlamingo3 not found. Install the latest Transformers:\n"
            "  pip install 'git+https://github.com/huggingface/transformers' accelerate"
        )
        sys.exit(1)

    model_id = "nvidia/music-flamingo-hf"
    log.info(f"Loading processor from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    resolved_dtype = (
        torch.float16
        if dtype == "auto"
        else {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
    )

    load_kwargs: dict = {
        "device_map": "auto",
        "max_memory": {0: "9GiB", "cpu": "48GiB"},
        "torch_dtype": resolved_dtype,
    }

    if flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        log.info("Flash Attention 2 enabled.")

    if quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        log.info("8-bit quantisation enabled.")
    elif quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        log.info("4-bit quantisation enabled (nf4 + double quant).")

    log.info(
        f"Loading model weights (dtype={resolved_dtype}) — this may take a minute ..."
    )
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    )
    model.eval()

    device = next(model.parameters()).device
    log.info(f"Model loaded on {device}.")
    return processor, model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def get_audio_duration(audio_path: str) -> Optional[float]:
    try:
        import torchaudio

        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return None


def select_landmark_chunks(
    audio_path: str,
    window_seconds: float = 40.0,
) -> list[dict]:
    """
    Extract four landmark windows from the track instead of processing every chunk.

    Given W = ``window_seconds`` and mid = duration / 2, the windows are:

    +---------+------------+-------------------------------+
    | Index   | Label      | Time range                    |
    +=========+============+===============================+
    | 0       | start      | 0 → W                         |
    +---------+------------+-------------------------------+
    | 1       | early      | W → 2W                        |
    +---------+------------+-------------------------------+
    | 2       | pre_mid    | (mid − W) → mid               |
    +---------+------------+-------------------------------+
    | 3       | post_mid   | mid → (mid + W)               |
    +---------+------------+-------------------------------+

    Windows that fall outside the track's duration are silently skipped.
    Temp WAV files are written to /tmp and deleted by the caller.

    Returns a list of dicts:
        [{"index": 0, "label": "start", "start": 0.0, "end": 40.0, "path": "..."}, ...]
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    total_frames = waveform.shape[1]
    duration = total_frames / sr
    mid = duration / 2.0
    w = window_seconds

    # Four landmark windows: (start_s, end_s, label)
    candidates: list[tuple[float, float, str]] = [
        (0.0, w, "start"),
        (w, 2.0 * w, "early"),
        (mid - w, mid, "pre_mid"),
        (mid, mid + w, "post_mid"),
    ]

    log.info(
        f"Track duration: {duration:.1f}s | mid-point: {mid:.1f}s | "
        f"window: {w:.0f}s → selecting {len(candidates)} landmark chunks"
    )

    chunks = []
    for idx, (start_s, end_s, label) in enumerate(candidates):
        # Clamp to valid range
        start_s = max(0.0, start_s)
        end_s = min(duration, end_s)

        if end_s <= start_s:
            log.warning(
                f"  Skipping landmark '{label}': window [{start_s:.1f}s, {end_s:.1f}s] is out of range."
            )
            continue

        start_frame = int(start_s * sr)
        end_frame = min(int(end_s * sr), total_frames)
        segment = waveform[:, start_frame:end_frame]

        tmp_path = f"/tmp/_mf_landmark_{idx}.wav"
        torchaudio.save(tmp_path, segment, sr)

        actual_end = round(end_frame / sr, 3)
        chunks.append(
            {
                "index": idx,
                "label": label,
                "start": round(start_s, 3),
                "end": actual_end,
                "path": tmp_path,
            }
        )
        log.info(f"  [{idx}] {label:10s}  {start_s:.1f}s → {actual_end:.1f}s")

    return chunks


# ---------------------------------------------------------------------------
# Caption generator (single chunk)
# ---------------------------------------------------------------------------


def caption_chunk(
    audio_path: str,
    processor,
    model,
    prompts: dict[str, str],
    max_new_tokens: int = 200,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> dict[str, str]:
    """Run all prompts against a single 30s audio chunk and return captions dict."""
    captions: dict[str, str] = {}
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    for label, prompt_text in prompts.items():
        log.info(f"    [{label}] ...")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        raw_inputs = output_ids = new_tokens = None
        try:
            gc.collect()
            torch.cuda.empty_cache()

            raw_inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = {
                k: (
                    v.to(device=device, dtype=model_dtype)
                    if torch.is_floating_point(v)
                    else v.to(device)
                )
                for k, v in raw_inputs.items()
            }

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generate_kwargs)

            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            caption = processor.batch_decode(new_tokens, skip_special_tokens=True)[
                0
            ].strip()
            captions[label] = caption

        except Exception as exc:
            log.warning(f"    x Failed [{label}]: {exc}")
            captions[label] = f"ERROR: {exc}"

        finally:
            del raw_inputs, output_ids, new_tokens
            gc.collect()
            torch.cuda.empty_cache()

    return captions


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def process_file(
    audio_path: str,
    processor,
    model,
    mode: str = "quick",
    max_new_tokens: int = 200,
    temperature: float = 0.3,
    top_p: float = 0.9,
    window_seconds: float = 40.0,
) -> dict:
    """Select the four landmark chunks and caption each one. Returns the full result dict."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    prompts = PROMPTS if mode == "detailed" else PROMPTS_QUICK
    duration = get_audio_duration(audio_path)
    chunks = select_landmark_chunks(audio_path, window_seconds=window_seconds)

    log.info(
        f"Processing: {Path(audio_path).name} | "
        f"mode={mode} | {len(chunks)} landmark chunk(s) x {len(prompts)} prompt(s)"
    )

    chunk_results = []
    for chunk in chunks:
        log.info(
            f"  Chunk {chunk['index'] + 1}/{len(chunks)} [{chunk['label']}] "
            f"({chunk['start']:.0f}s - {chunk['end']:.0f}s)"
        )
        try:
            captions = caption_chunk(
                audio_path=chunk["path"],
                processor=processor,
                model=model,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            # Always clean up temp chunk file regardless of success/failure
            try:
                os.remove(chunk["path"])
            except OSError:
                pass

        chunk_results.append(
            {
                "chunk_index": chunk["index"],
                "label": chunk["label"],
                "start_seconds": chunk["start"],
                "end_seconds": chunk["end"],
                "captions": captions,
            }
        )

    return {
        "metadata": {
            "model": "nvidia/music-flamingo-hf",
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "audio_file": str(Path(audio_path).resolve()),
            "duration_seconds": round(duration, 2) if duration else None,
            "window_seconds": window_seconds,
            "total_chunks": len(chunk_results),
            "mode": mode,
        },
        "chunks": chunk_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Chunk-and-caption every mp3/wav in DIR using nvidia/music-flamingo-hf.",
)
@click.option(
    "--dir", "audio_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing .mp3 / .wav files to process.",
)
@click.option(
    "--log-file",
    "log_file",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Path for the debug log file. "
        "Defaults to <dir>/caption_audio_<timestamp>.log"
    ),
)
@click.option(
    "--mode",
    type=click.Choice(["quick", "detailed"]),
    default="quick",
    show_default=True,
    help=(
        "quick → one dense summary per chunk. "
        "detailed → 4 focused tag-cluster prompts."
    ),
)
@click.option(
    "--window-seconds",
    type=float,
    default=40.0,
    show_default=True,
    metavar="S",
    help="Duration of each landmark window in seconds.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=120,
    show_default=True,
    metavar="N",
    help="Max new tokens per caption. Dense tag output rarely needs more.",
)
@click.option("--temperature", type=float, default=0.3, show_default=True, metavar="T")
@click.option("--top-p", type=float, default=0.9, show_default=True, metavar="P")
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--quantize",
    type=click.Choice(["none", "8bit", "4bit"]),
    default="4bit",
    show_default=True,
    help="Weight quantisation. Recommended: 4bit on T4.",
)
@click.option(
    "--flash-attn",
    is_flag=True,
    default=False,
    help="Enable Flash Attention 2 (Ampere+ only, NOT available on T4).",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    metavar="N",
    help="JSON indent width for output files.",
)
def main(
    audio_dir: Path,
    log_file: Optional[Path],
    mode: str,
    window_seconds: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    dtype: str,
    quantize: str,
    flash_attn: bool,
    indent: int,
) -> None:
    # ---- Logging setup ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_log = log_file or (audio_dir / f"caption_audio_{timestamp}.log")
    _configure_logging(resolved_log)

    # ---- Discover audio files ----
    audio_files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        log.error(f"No .mp3 or .wav files found in: {audio_dir}")
        sys.exit(1)

    log.info(f"Found {len(audio_files)} audio file(s) in {audio_dir}")

    # ---- Load model once ----
    processor, model = load_model(
        dtype=dtype, flash_attn=flash_attn, quantize=quantize
    )

    # ---- Process each file ----
    ok, failed = 0, 0
    for audio_path in audio_files:
        json_path = audio_path.with_suffix(".json")
        log.info(f"[{ok + failed + 1}/{len(audio_files)}] {audio_path.name} → {json_path.name}")
        try:
            result = process_file(
                audio_path=str(audio_path),
                processor=processor,
                model=model,
                mode=mode,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                window_seconds=window_seconds,
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=indent, ensure_ascii=False)

            log.info(f"  ✓ Saved {json_path}")

            # Preview to stdout
            print(f"\n{'='*60}")
            print(
                f"File : {audio_path.name}  ({result['metadata']['total_chunks']} landmark chunks)"
            )
            for chunk in result["chunks"]:
                print(
                    f"\n  -- [{chunk['label']}] Chunk {chunk['chunk_index']} "
                    f"({chunk['start_seconds']:.0f}s - {chunk['end_seconds']:.0f}s)"
                )
                for label, text in chunk["captions"].items():
                    print(f"  [{label.upper()}] {text}")

            ok += 1

        except Exception as exc:
            log.error(f"  ✗ Failed {audio_path.name}: {exc}")
            failed += 1

    log.info(f"\nDone — {ok} succeeded, {failed} failed. Log: {resolved_log}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
