"""Batch output must carry the caption pipeline version and stay two-phase.

A missing or older version means the album was captioned by different chunking
or merge semantics, so the next batch run must not skip it. A version-3 album is
only done after the polish phase, and the raw chunk captions must survive so a
bad result can be traced to the captioner or to the rewrite model.
"""

from pathlib import Path

import pytest

from scripts.batch import process_album
from src.resume import (
    CAPTION_PIPELINE_VERSION,
    CAPTION_STAGE_CAPTIONED,
    CAPTION_STAGE_POLISHED,
)

GOOD = (
    "A piano track with a clean mix. The piano carries the tune, and the clean "
    "mix keeps the instrument forward throughout. The piano returns over the "
    "same clean mix, and the arrangement stays simple and focused from start to "
    "finish, with nothing else added to the texture at any point in the "
    "performance of the piece."
)


class _StubCaptioner:
    last_run = {"chunk_count": 2, "silence_trimmed_s": 1.5}

    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []

    def unload(self):
        self.events.append("unload-captioner")


class _StubRewriter:
    name = "stub-rewriter"

    def __init__(self, events: list[str] | None = None, output: str = GOOD):
        self.events = events if events is not None else []
        self.output = output
        self.calls: list[list[dict[str, str]]] = []

    def rewrite(self, messages):
        self.events.append("rewrite")
        self.calls.append(messages)
        return self.output

    def unload(self):
        self.events.append("unload-rewriter")


def _caption(chunk_captions=None):
    chunks = chunk_captions or ["A piano track with a clean mix."]
    return {
        "instruments": ["piano"],
        "vocals": [],
        "production": ["clean"],
        "caption": " ".join(chunks),
        "chunk_captions": chunks,
        "tags_experimental": True,
    }


@pytest.fixture
def album_entry(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    return {
        "artist": "ARTIST",
        "album": "Album",
        "source_dir": str(tmp_path),
        "tracks": [{"filename": "a.mp3", "uri": str(audio)}],
    }


def _patch_captioner(monkeypatch, caption):
    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr(
        "src.captioner_hf.caption_and_tag",
        lambda captioner, path, experimental_suno=False: caption,
    )


def test_process_album_writes_pipeline_version(tmp_path, album_entry, monkeypatch):
    _patch_captioner(monkeypatch, _caption())

    outputs = tmp_path / "outputs"
    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", outputs
    )

    assert album_obj["caption_pipeline_version"] == CAPTION_PIPELINE_VERSION
    written = outputs / "ARTIST" / "Album.json"
    assert written.exists()
    assert f'"caption_pipeline_version": {CAPTION_PIPELINE_VERSION}' in written.read_text()


def test_caption_only_stage_is_not_done(tmp_path, album_entry, monkeypatch):
    _patch_captioner(monkeypatch, _caption())
    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs"
    )
    assert album_obj["caption_stage"] == CAPTION_STAGE_CAPTIONED
    track = album_obj["tracks"][0]
    assert track["chunk_captions"] == ["A piano track with a clean mix."]
    assert track["caption_raw"] == "A piano track with a clean mix."
    assert track["rewrite_status"] == "disabled"


def test_all_captions_run_before_any_rewrite(tmp_path, monkeypatch):
    events: list[str] = []

    def fake_caption(captioner, path, experimental_suno=False):
        events.append("caption")
        return _caption(["A piano track with a clean mix.", "The piano returns."])

    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr("src.captioner_hf.caption_and_tag", fake_caption)

    entry = {
        "artist": "ARTIST",
        "album": "Album",
        "source_dir": str(tmp_path),
        "tracks": [
            {"filename": "a.mp3", "uri": str(tmp_path / "a.mp3")},
            {"filename": "b.mp3", "uri": str(tmp_path / "b.mp3")},
        ],
    }
    for name in ("a.mp3", "b.mp3"):
        (tmp_path / name).write_bytes(b"x")

    rewriter = _StubRewriter(events)
    album_obj = process_album(
        _StubCaptioner(events), entry, tmp_path / "work", tmp_path / "outputs",
        rewriter=rewriter,
        before_polish=lambda: events.append("before-polish"),
    )

    assert events == [
        "caption", "caption", "before-polish", "unload-captioner",
        "rewrite", "rewrite", "unload-rewriter",
    ]
    assert album_obj["caption_stage"] == CAPTION_STAGE_POLISHED
    assert album_obj["rewrite"]["ok"] == 2
    for track in album_obj["tracks"]:
        assert track["caption"] == GOOD
        assert track["rewrite_status"] == "ok"
        # The raw captioner output is kept for debugging.
        assert track["chunk_captions"]
        assert track["caption_raw"]


def test_rewrite_input_is_the_raw_text_only(tmp_path, album_entry, monkeypatch):
    chunks = [
        "A piano track with a clean mix.",
        "The piano returns over the same clean mix.",
    ]
    _patch_captioner(monkeypatch, _caption(chunks))
    rewriter = _StubRewriter()
    process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs",
        rewriter=rewriter,
    )
    messages = rewriter.calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "\n\n".join(chunks)
    assert "{" not in messages[1]["content"]


def test_checkpoint_fires_after_captioning_before_rewrite(
    tmp_path, album_entry, monkeypatch
):
    events: list[str] = []

    def fake_caption(captioner, path, experimental_suno=False):
        events.append("caption")
        return _caption(["A piano track with a clean mix.", "The piano returns."])

    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr("src.captioner_hf.caption_and_tag", fake_caption)

    def checkpoint(obj, stage):
        events.append(f"checkpoint:{stage}")
        # Raw captions must already be in the object and on disk.
        assert obj["tracks"][0]["chunk_captions"]

    process_album(
        _StubCaptioner(events), album_entry, tmp_path / "work",
        tmp_path / "outputs", rewriter=_StubRewriter(events),
        checkpoint=checkpoint,
    )
    assert events[0] == "caption"
    assert "checkpoint:captioned" in events
    assert events.index("checkpoint:captioned") < events.index("rewrite")
    assert events.index("checkpoint:captioned") < events.index("unload-captioner")


def test_single_chunk_track_is_not_rewritten(tmp_path, album_entry, monkeypatch):
    _patch_captioner(monkeypatch, _caption(["A piano track with a clean mix."]))
    rewriter = _StubRewriter()
    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs",
        rewriter=rewriter,
    )
    assert rewriter.calls == []
    assert album_obj["tracks"][0]["rewrite_status"] == "single_chunk"
    assert album_obj["caption_stage"] == CAPTION_STAGE_POLISHED


def test_rejected_rewrite_keeps_raw_caption(tmp_path, album_entry, monkeypatch):
    _patch_captioner(
        monkeypatch,
        _caption(["A piano track with a clean mix.", "The piano returns."]),
    )
    rewriter = _StubRewriter(output="A banjo track with a clean mix. " + GOOD)
    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs",
        rewriter=rewriter,
    )
    track = album_obj["tracks"][0]
    assert track["rewrite_status"] == "fallback"
    assert track["caption"] == track["caption_raw"]
    assert any("banjo" in r for r in track["rewrite_reasons"])
    assert album_obj["rewrite"]["fallback"] == 1


def test_rewriter_error_keeps_album_captioned_for_retry(
    tmp_path, album_entry, monkeypatch
):
    _patch_captioner(
        monkeypatch,
        _caption(["A piano track with a clean mix.", "The piano returns."]),
    )

    class _BrokenRewriter:
        name = "broken"

        def rewrite(self, messages):
            raise RuntimeError("CUDA out of memory")

        def unload(self):
            pass

    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs",
        rewriter=_BrokenRewriter(),
    )
    track = album_obj["tracks"][0]
    assert album_obj["caption_stage"] == CAPTION_STAGE_CAPTIONED
    assert track["rewrite_status"] == "error"
    assert track["caption"] == track["caption_raw"]
    assert track["chunk_captions"]
    assert album_obj["rewrite"]["error"] == 1


def test_captioned_album_is_reused_without_recaptioning(
    tmp_path, album_entry, monkeypatch
):
    def fail_if_called(captioner, path, experimental_suno=False):
        raise AssertionError("captioner must not run for a captioned album")

    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr("src.captioner_hf.caption_and_tag", fail_if_called)

    chunks = ["A piano track with a clean mix.", "The piano returns."]
    existing = {
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "caption_stage": CAPTION_STAGE_CAPTIONED,
        "status": "ok",
        "tracks": [{
            "file": "a.mp3",
            "duration_s": 10.0,
            "bitrate_kbps": 320,
            "chunk_count": 2,
            "silence_trimmed_s": 0.0,
            "caption_seconds": 1.0,
            "instruments": ["piano"],
            "vocals": [],
            "production": ["clean"],
            "chunk_captions": chunks,
            "caption_raw": " ".join(chunks),
            "caption": "",
            "rewrite_status": "pending",
            "rewrite_reasons": ["stale reason from a previous polish"],
            "tags_experimental": True,
        }],
        "failed_tracks": [],
        "skipped_tracks": [],
    }

    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", tmp_path / "outputs",
        existing=existing, retry_only_failed=True, rewriter=_StubRewriter(),
    )
    assert album_obj["tracks"][0]["chunk_captions"] == chunks
    assert album_obj["caption_stage"] == CAPTION_STAGE_POLISHED
    assert album_obj["tracks"][0]["rewrite_status"] == "ok"
    assert "rewrite_reasons" not in album_obj["tracks"][0]


def test_old_existing_output_is_not_reused_with_retry_failed(
    tmp_path, album_entry, monkeypatch
):
    calls: list[str] = []

    def fake_caption(captioner, path, experimental_suno=False):
        calls.append(str(path))
        return _caption()

    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr("src.captioner_hf.caption_and_tag", fake_caption)

    old = {
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION - 1,
        "status": "ok",
        "tracks": [{"file": "a.mp3", "caption_seconds": 1.0}],
        "failed_tracks": [],
        "skipped_tracks": [],
    }
    album_obj = process_album(
        _StubCaptioner(),
        album_entry,
        tmp_path / "work",
        tmp_path / "outputs",
        existing=old,
        retry_only_failed=True,
    )

    assert calls == [str(Path(album_entry["tracks"][0]["uri"]))]
    assert album_obj["caption_pipeline_version"] == CAPTION_PIPELINE_VERSION
