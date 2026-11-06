"""Pure-Python tests for the captioner wrapper. No GPU or model required."""

from pathlib import Path

import pytest

from src.captioner import Captioner, CaptionerError


def test_init_requires_existing_paths(tmp_path: Path):
    with pytest.raises(CaptionerError):
        Captioner(
            llama_cli=tmp_path / "nope-cli",
            model=tmp_path / "nope.gguf",
            mmproj=tmp_path / "nope-mmproj.gguf",
        )


def test_missing_audio_raises(tmp_path: Path):
    # Create dummy files so only the audio check fails.
    cli = tmp_path / "llama-cli"
    cli.write_text("#!/bin/sh\n")
    model = tmp_path / "m.gguf"
    model.write_text("x")
    mmproj = tmp_path / "p.gguf"
    mmproj.write_text("x")
    cap = Captioner(llama_cli=cli, model=model, mmproj=mmproj)
    with pytest.raises(CaptionerError):
        cap.caption_chunk(tmp_path / "missing.mp3")


def test_clean_strips_llama_log_lines():
    raw = (
        "llama_model_loader: loaded meta data\n"
        "[  0.12] system_info: n_threads = 4\n"
        "The track opens with a warm acoustic guitar.\n"
        "main: decoding finished\n"
    )
    cleaned = Captioner._clean(raw)
    assert cleaned == "The track opens with a warm acoustic guitar."


def test_clean_strips_ansi_escapes():
    raw = "\x1b[1mreverb\x1b[0m and delay"
    assert Captioner._clean(raw) == "reverb and delay"


def test_clean_keeps_prose_that_mentions_noise():
    raw = "A lo-fi mix with tape hiss and vinyl crackle throughout."
    assert Captioner._clean(raw) == raw


def test_audio_duration_missing_file(tmp_path: Path):
    from src.captioner import audio_duration

    with pytest.raises(CaptionerError):
        audio_duration(tmp_path / "nope.wav")


def test_gguf_chunk_defaults(tmp_path: Path):
    cli = tmp_path / "llama-cli"
    cli.write_text("#!/bin/sh\n")
    model = tmp_path / "m.gguf"
    model.write_text("x")
    mmproj = tmp_path / "p.gguf"
    mmproj.write_text("x")
    cap = Captioner(llama_cli=cli, model=model, mmproj=mmproj)
    # The GGUF author validated one-minute clips, so this backend keeps 60 s.
    assert cap.chunk_seconds == 60.0
    assert cap.max_chunk_seconds == 90.0
    assert cap.trim_silence is True
    assert cap.last_run == {}
