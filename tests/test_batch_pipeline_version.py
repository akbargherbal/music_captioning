"""Batch output must carry the caption pipeline version.

A missing or older version means the album was captioned by different chunking
or merge semantics, so the next batch run must not skip it.
"""

from pathlib import Path

import pytest

from scripts.batch import process_album
from src.resume import CAPTION_PIPELINE_VERSION


class _StubCaptioner:
    last_run = {"chunk_count": 3, "silence_trimmed_s": 1.5}


def _caption(chunk_captions=None):
    return {
        "instruments": ["piano"],
        "vocals": [],
        "production": ["clean"],
        "caption": "A piano track.",
        "chunk_captions": chunk_captions or ["A piano track."],
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


def test_process_album_writes_pipeline_version(tmp_path, album_entry, monkeypatch):
    monkeypatch.setattr(
        "src.captioner.audio_stream_info", lambda path: (10.0, 320.0)
    )
    monkeypatch.setattr(
        "src.captioner_hf.caption_and_tag",
        lambda captioner, path, experimental_suno=False: _caption(),
    )

    outputs = tmp_path / "outputs"
    album_obj = process_album(
        _StubCaptioner(), album_entry, tmp_path / "work", outputs
    )

    assert album_obj["caption_pipeline_version"] == CAPTION_PIPELINE_VERSION
    written = outputs / "ARTIST" / "Album.json"
    assert written.exists()
    assert f'"caption_pipeline_version": {CAPTION_PIPELINE_VERSION}' in written.read_text()


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
