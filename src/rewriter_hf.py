"""Transformers backend for the stateless caption rewrite.

The rewrite model is a small text-only instruct LLM (default
``Qwen/Qwen3-4B-Instruct-2507``), separate from the audio captioner. It receives
a fixed system instruction plus the raw caption text and returns plain prose.

T4 (sm_75) constraints apply: fp16 compute, no bf16, no FlashAttention. The
default is **plain fp16** (`quantization="none"`): a 4B model is ~8 GiB and fits
alone on the T4 (the captioner is unloaded first), it loads without the slow
bitsandbytes quantization pass, and it avoids 8-bit dequantization overhead.
Use 4-bit only if you must share the GPU; 8-bit buys nothing on a T4.

Heavy imports are lazy, so the dataclass can be imported and inspected on a
CPU-only runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .rewrite import REWRITE_PROMPT_VERSION

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 384


class RewriterError(RuntimeError):
    """Raised when the rewrite model cannot be loaded or returns nothing."""


@dataclass
class HFRewriter:
    """Rewrite text with a small instruct model, one call at a time."""

    model_id: str = DEFAULT_MODEL_ID
    quantization: str = "none"  # "none" (fp16, default) | "4bit" | "8bit"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    device: str = "auto"
    model_dir: Path | None = None  # local snapshot; defaults to the HF cache
    _model: object = field(default=None, repr=False, compare=False)
    _tokenizer: object = field(default=None, repr=False, compare=False)
    _last_run: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.quantization not in {"8bit", "4bit", "none"}:
            raise ValueError(
                f"quantization must be 8bit, 4bit or none, got {self.quantization!r}"
            )
        if self.quantization == "none" and self.device == "auto":
            # Full fp16 weights: only sensible when the caller knows there is room.
            pass
        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir)

    @property
    def name(self) -> str:
        return self.model_id

    # -- loading ----------------------------------------------------------
    def _source(self) -> str:
        return str(self.model_dir) if self.model_dir is not None else self.model_id

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, object] = {"device_map": self.device}
        if self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "4bit":
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

        source = self._source()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except Exception as exc:  # pragma: no cover - requires weights
            raise RewriterError(
                f"could not load rewrite model {self.model_id}: {exc}"
            ) from exc
        self._model.eval()

    # -- generation -------------------------------------------------------
    def rewrite(self, messages: list[dict[str, str]]) -> str:
        """Generate one rewrite from a fresh message list (no history)."""
        if not messages:
            raise RewriterError("empty message list")
        self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        started = time.time()
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        if not text:
            raise RewriterError(
                f"rewrite model {self.model_id} returned an empty response"
            )

        self._last_run = {
            "model": self.model_id,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "quantization": self.quantization,
            "input_words": len(messages[-1]["content"].split()),
            "output_words": len(text.split()),
            "seconds": round(time.time() - started, 2),
        }
        return text

    @property
    def last_run(self) -> dict[str, object]:
        """Metadata from the most recent rewrite."""
        return dict(self._last_run)

    def unload(self) -> None:
        """Drop the rewrite model, freeing VRAM for the next caption pass."""
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - CPU-only or no torch
            pass
