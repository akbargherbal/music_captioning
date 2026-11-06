"""Pure-Python tests for the transformers backend. No GPU, model, or torch needed."""

from pathlib import Path

import pytest

from src.captioner_hf import PROMPT, HFCaptioner
from src.captioner import CaptionerError


def test_init_missing_model_dir(tmp_path: Path):
    with pytest.raises(CaptionerError):
        HFCaptioner(model_dir=tmp_path / "nope")


def test_build_conversation_text_then_audio(tmp_path: Path):
    audio = tmp_path / "song.mp3"
    audio.write_text("x")
    conv = HFCaptioner.build_conversation(PROMPT, audio)
    assert conv == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "audio": str(audio)},
            ],
        }
    ]


def test_prompt_matches_model_card():
    assert PROMPT == "*Task* Describe this audio in detail"


def test_missing_audio_raises_before_model_load(tmp_path: Path):
    # A model dir that exists avoids __post_init__ failing; the audio check
    # must fire before any heavy model loading.
    empty_model = tmp_path / "model"
    empty_model.mkdir()
    cap = HFCaptioner(model_dir=empty_model)
    with pytest.raises(CaptionerError):
        cap.caption_chunk(tmp_path / "missing.mp3")


def test_hf_chunk_defaults(tmp_path: Path):
    empty_model = tmp_path / "model"
    empty_model.mkdir()
    cap = HFCaptioner(model_dir=empty_model)
    assert cap.chunk_seconds == 90.0
    assert cap.max_chunk_seconds == 120.0
    assert cap.trim_silence is True
    assert cap.last_run == {}


def test_missing_audio_raises_from_caption_audio(tmp_path: Path):
    empty_model = tmp_path / "model"
    empty_model.mkdir()
    cap = HFCaptioner(model_dir=empty_model)
    with pytest.raises(CaptionerError):
        cap.caption_audio(tmp_path / "missing.mp3")
