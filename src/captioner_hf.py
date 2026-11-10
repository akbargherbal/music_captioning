"""Transformers backend for the ACE-Step captioner (4-bit, T4-safe).

This runs the same captioner model as the GGUF path, but through HuggingFace
``transformers`` + ``bitsandbytes`` instead of llama.cpp. It was adopted after
the user approved switching runtime: it avoids the multi-minute CUDA build and
uses the reference audio pipeline, which handles Qwen2.5-Omni audio natively.

T4 constraints (AGENTS.md): no bf16, no FlashAttention. We therefore ask for
``float16`` compute and ``sdpa`` attention. Note: the checkpoint config's own
``dtype: bfloat16`` currently wins for the non-quantized tensors, so the decoder
runs in bf16 despite the fp16 request (measured; see
``docs/CHUNK_LENGTH_EXPERIMENT.md``). Making fp16 take effect is an open task.

Heavy imports (torch / transformers / librosa) happen lazily inside methods so
the module can be imported, and its pure helpers tested, on a CPU-only runtime.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .captioner import (
    CaptionerError,
    _split_audio,
    audio_duration,
    plan_chunks,
    trim_silence_edges,
)
from .merge import merge_section_captions
from .tags import extract_tags, merge_tags, to_suno

PROMPT = (
    "*Task* Describe this audio in detail\n\n"
    "Do not mention timestamps, time markers, or durations (e.g. \"at 0:23\", "
    "\"after 30 seconds\", \"in the first minute\"). To describe when something "
    "happens, use general sequencing words such as \"opens with\", \"then\", "
    "\"later\", \"gradually\", \"eventually\", \"closes with\"."
)
# ~90 s keeps a 4.5-minute track at three windows (~8.5 GiB on a T4); 120 s is
# the hard cap, well under the measured ~190 s OOM ceiling.
DEFAULT_CHUNK_SECONDS = 90.0
DEFAULT_MAX_CHUNK_SECONDS = 120.0


@dataclass
class HFCaptioner:
    """Caption audio with the 4-bit ACE-Step captioner via transformers."""

    model_dir: Path
    device: str = "auto"
    max_new_tokens: int = 400
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS
    trim_silence: bool = True
    attn_implementation: str = "sdpa"
    load_in_4bit: bool = True
    _model: object = field(default=None, repr=False, compare=False)
    _processor: object = field(default=None, repr=False, compare=False)
    _last_run: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if not self.model_dir.exists():
            raise CaptionerError(f"model dir not found: {self.model_dir}")

    # -- loading ----------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import (
            Qwen2_5OmniConfig,
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        # The checkpoint ships its own quantization_config with bf16 compute,
        # which the T4 cannot do natively. Force fp16 compute instead.
        try:
            config = Qwen2_5OmniConfig.from_pretrained(str(self.model_dir))
        except Exception:
            config = None

        kwargs: dict[str, object] = {
            "device_map": self.device,
            "attn_implementation": self.attn_implementation,
        }
        if config is not None:
            kwargs["config"] = config
        if self.load_in_4bit:
            quant = getattr(config, "quantization_config", None) if config is not None else None
            if quant is not None:
                if isinstance(quant, dict):
                    quant["bnb_4bit_compute_dtype"] = "float16"
                else:
                    quant.bnb_4bit_compute_dtype = "float16"
            else:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_storage=torch.uint8,
                )
        else:
            kwargs["dtype"] = torch.float16

        self._processor = Qwen2_5OmniProcessor.from_pretrained(str(self.model_dir))
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(self.model_dir), **kwargs
        )
        self._model.eval()

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def build_conversation(prompt: str, audio_path: Path | str) -> list[dict]:
        """Conversation in the order the captioner card documents: text then audio."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": str(audio_path)},
                ],
            }
        ]

    @staticmethod
    def load_wav(path: Path | str) -> "object":
        """Load any audio file as 16 kHz mono float32 (Qwen2.5-Omni expects this)."""
        import librosa

        wav, _sr = librosa.load(str(path), sr=16000, mono=True)
        return wav.astype("float32")

    def caption_chunk(self, audio_path: Path | str) -> str:
        """Caption a single (short) audio file."""
        path = Path(audio_path)
        if not path.exists():
            raise CaptionerError(f"audio not found: {path}")
        self._load()
        import torch

        wav = self.load_wav(path)
        conversation = self.build_conversation(PROMPT, path)
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=text, audio=[wav], return_tensors="pt", padding=True
        )
        inputs = inputs.to(self._model.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(torch.float16)

        gen_kwargs: dict[str, object] = {
            "generation_mode": "text",  # captioner is text-only; skip the talker
            "thinker_max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        out = self._model.generate(**inputs, **gen_kwargs)

        gen = out[:, inputs["input_ids"].shape[1]:]
        caption = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        caption = caption.strip()
        if not caption:
            raise CaptionerError(f"empty caption from transformers on {path}")
        return caption

    def caption_audio(
        self, audio_path: Path | str, chunk_seconds: float | None = None
    ) -> list[str]:
        """Caption a full track in equal, silence-trimmed chunks.

        Leading/trailing silence is trimmed first (unless ``trim_silence`` is
        off), then the track is divided into the fewest equal chunks that fit
        ``max_chunk_seconds`` (see :func:`plan_chunks`), so there is no tiny
        final chunk. Returns one caption per chunk.

        Details of the last run (chunk count, seconds of silence removed) are
        left in :attr:`last_run` for callers that report them.
        """
        path = Path(audio_path)
        if not path.exists():
            raise CaptionerError(f"audio not found: {path}")
        target = self.chunk_seconds if chunk_seconds is None else chunk_seconds
        original = audio_duration(path)

        with tempfile.TemporaryDirectory(prefix="acestep_hf_chunks_") as tmp:
            workdir = Path(tmp)
            source, removed = (path, 0.0)
            if self.trim_silence:
                source, removed = trim_silence_edges(path, workdir)
            # Measure the trimmed file: a stream copy snaps to frame boundaries,
            # so it is not exactly ``original - removed``. Planning on the
            # computed value can emit a spurious trailing fragment.
            duration = audio_duration(source)
            if removed:
                removed = max(0.0, original - duration)
            count, segment = plan_chunks(duration, target, self.max_chunk_seconds)
            if count <= 1:
                captions = [self.caption_chunk(source)]
            else:
                chunks = _split_audio(source, workdir, segment)
                if not chunks:
                    raise CaptionerError(f"ffmpeg produced no chunks for {path}")
                captions = [self.caption_chunk(c) for c in chunks]

        self._last_run = {
            "original_duration_s": round(original, 3),
            "duration_s": round(duration, 3),
            "silence_trimmed_s": round(removed, 3),
            "chunk_count": len(captions),
            "segment_seconds": round(segment, 3) if count > 1 else round(duration, 3),
        }
        return captions

    @property
    def last_run(self) -> dict[str, float]:
        """Chunk count / trimmed seconds from the most recent ``caption_audio``."""
        return dict(self._last_run)

    def unload(self) -> None:
        """Drop the model so the next phase can have the T4's VRAM.

        The batch runs captioning for the whole album, then polishing for the
        whole album. Releasing the captioner before the rewrite model loads keeps
        the two models from sharing VRAM; the weights are re-read from disk for
        the next album.
        """
        self._model = None
        self._processor = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - CPU-only or no torch
            pass


def caption_and_tag(
    captioner: HFCaptioner,
    audio_path: Path | str,
    *,
    experimental_suno: bool = False,
) -> dict[str, object]:
    """Caption (chunked) and return the caption plus experimental term lists.

    ``suno_prompt`` is only added when ``experimental_suno=True`` and is marked
    ``suno_experimental``; the default output is caption + experimental tags.
    """
    captions = captioner.caption_audio(audio_path)
    merged = merge_tags([extract_tags(c) for c in captions])
    result: dict[str, object] = {
        "instruments": merged["instruments"],
        "vocals": merged["vocals"],
        "production": merged["production"],
        "caption": merge_section_captions(captions),
        "chunk_captions": captions,
        "tags_experimental": True,
    }
    if experimental_suno:
        result["suno_prompt"] = to_suno(merged)
        result["suno_experimental"] = True
    return result
