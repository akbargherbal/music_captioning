"""Pure-Python tests for chunk planning and silence parsing. No GPU needed."""

import pytest

from src.captioner import CaptionerError, parse_silence, plan_chunks

HF_TARGET, HF_MAX = 90.0, 120.0


@pytest.mark.parametrize(
    ("duration", "count", "segment"),
    [
        (266.256, 3, 88.752),
        (259.224, 3, 86.408),     # the trimmed file from the validation track
        (240.1, 3, 80.0333),      # 240.1 / 3
        (200.2, 2, 100.1),
        (365.0, 4, 91.25),
        (480.0, 5, 96.0),
        (61.0, 1, 61.0),
        (134.0, 2, 67.0),         # rounded down to 1, then raised to fit 120 s
    ],
)
def test_plan_chunks_hf_defaults(duration, count, segment):
    got_count, got_segment = plan_chunks(duration, HF_TARGET, HF_MAX)
    assert got_count == count
    assert got_segment == pytest.approx(segment, abs=1e-3)


@pytest.mark.parametrize(
    "duration", [0.5, 10.0, 60.0, 61.0, 89.9, 90.0, 134.0, 200.2, 266.256,
                 317.0, 365.0, 480.0, 1000.0]
)
@pytest.mark.parametrize(("target", "max_chunk"), [(90.0, 120.0), (60.0, 90.0)])
def test_plan_chunks_invariants(duration, target, max_chunk):
    count, segment = plan_chunks(duration, target, max_chunk)
    assert count >= 1
    assert segment <= max_chunk
    assert count * segment == pytest.approx(duration, abs=1e-9)
    # Equal segments: there is never a short remainder chunk.
    assert segment >= min(10.0, duration)


def test_plan_chunks_gguf_defaults_have_no_tiny_tail():
    # The old 60 s slice produced 4 x 60 s + a 26.3 s tail.
    count, segment = plan_chunks(266.256, 60.0, 90.0)
    assert count == 4
    assert segment == pytest.approx(66.564)


def test_plan_chunks_rejects_bad_input():
    with pytest.raises(CaptionerError):
        plan_chunks(0.0, 90.0, 120.0)
    with pytest.raises(CaptionerError):
        plan_chunks(-1.0, 90.0, 120.0)
    with pytest.raises(CaptionerError):
        plan_chunks(100.0, 0.0, 120.0)
    with pytest.raises(CaptionerError):
        plan_chunks(100.0, 90.0, 0.0)
    with pytest.raises(CaptionerError):
        plan_chunks(100.0, 150.0, 120.0)


def test_parse_silence_leading_only():
    stderr = (
        "[silencedetect @ 0x557] silence_start: 0\n"
        "[silencedetect @ 0x557] silence_end: 1.5 | silence_duration: 1.5\n"
    )
    span = parse_silence(stderr, 100.0)
    assert span is not None
    assert span.start == pytest.approx(1.4)   # 1.5 s cut minus 0.1 s padding
    assert span.end == pytest.approx(100.0)
    assert span.removed == pytest.approx(1.4)


def test_parse_silence_trailing_only_matches_real_file():
    # Verbatim shape from ffmpeg on a track with a 7.1 s fade-to-silence tail.
    stderr = (
        "[silencedetect @ 0x599c3b26e7c0] silence_start: 259.089\n"
        "[silencedetect @ 0x599c3b26e7c0] silence_end: 266.217 | "
        "silence_duration: 7.12802\n"
    )
    span = parse_silence(stderr, 266.256)
    assert span is not None
    assert span.start == pytest.approx(0.0)
    assert span.end == pytest.approx(259.189)
    assert span.removed == pytest.approx(7.067, abs=1e-3)


def test_parse_silence_both_edges():
    stderr = (
        "silence_start: 0\n"
        "silence_end: 2.0 | silence_duration: 2.0\n"
        "silence_start: 98.0\n"
        "silence_end: 100.0 | silence_duration: 2.0\n"
    )
    span = parse_silence(stderr, 100.0)
    assert span is not None
    assert span.start == pytest.approx(1.9)
    assert span.end == pytest.approx(98.1)
    assert span.removed == pytest.approx(3.8)


def test_parse_silence_trailing_without_silence_end_line():
    span = parse_silence("silence_start: 95.0\n", 100.0)
    assert span is not None
    assert span.start == pytest.approx(0.0)
    assert span.end == pytest.approx(95.1)
    assert span.removed == pytest.approx(4.9)


def test_parse_silence_ignores_mid_song_silence():
    stderr = (
        "silence_start: 40.0\n"
        "silence_end: 42.0 | silence_duration: 2.0\n"
    )
    assert parse_silence(stderr, 100.0) is None


def test_parse_silence_no_silence():
    assert parse_silence("", 100.0) is None
    assert parse_silence("size=N/A time=00:00:10.00 bitrate=N/A speed=200x\n",
                         100.0) is None


def test_parse_silence_short_trailing_run_is_not_worth_trimming():
    stderr = "silence_start: 99.85\nsilence_end: 100.0 | silence_duration: 0.15\n"
    assert parse_silence(stderr, 100.0) is None


def test_parse_silence_fully_silent_file_is_left_alone():
    stderr = "silence_start: 0\nsilence_end: 100.0 | silence_duration: 100.0\n"
    assert parse_silence(stderr, 100.0) is None
