"""``--repolish`` rebuilds albums from stored JSONs, so no audio is needed.

The polish phase only reads ``chunk_captions``; requiring the original audio
root just to discover which albums exist breaks for local roots from an earlier
session (the audio is gone, the JSON is in git).
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.batch import albums_from_outputs
from src.resume import CAPTION_PIPELINE_VERSION


def _album(artist: str, album: str, files: list[str], **extra) -> dict:
    data = {
        "artist": artist,
        "album": album,
        "source_dir": f"/content/{artist}",
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "tracks": [
            {"file": name, "chunk_captions": ["A piano track."]} for name in files
        ],
        "failed_tracks": [],
        "skipped_tracks": [],
    }
    data.update(extra)
    return data


def _write(outputs: Path, data: dict) -> None:
    path = outputs / data["artist"] / f"{data['album']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_entries_rebuilt_without_audio(tmp_path):
    outputs = tmp_path / "outputs"
    _write(outputs, _album("ARTIST", "Album", ["a.mp3", "b.mp3"]))

    albums = albums_from_outputs(outputs)

    assert list(albums) == ["ARTIST/Album"]
    entry = albums["ARTIST/Album"]
    assert entry["artist"] == "ARTIST"
    assert entry["album"] == "Album"
    assert entry["source_dir"] == "/content/ARTIST"
    assert entry["tracks"] == [
        {"filename": "a.mp3", "uri": ""},
        {"filename": "b.mp3", "uri": ""},
    ]


def test_artist_filter(tmp_path):
    outputs = tmp_path / "outputs"
    _write(outputs, _album("A1", "Album", ["a.mp3"]))
    _write(outputs, _album("A2", "Album", ["b.mp3"]))

    albums = albums_from_outputs(outputs, artists={"A2"})
    assert list(albums) == ["A2/Album"]


def test_non_album_json_is_ignored(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "ARTIST").mkdir(parents=True)
    (outputs / "ARTIST" / "broken.json").write_text("{not json", encoding="utf-8")
    (outputs / "ARTIST" / "empty.json").write_text("{}", encoding="utf-8")
    (outputs / "ARTIST" / "manifest.json").write_text(
        json.dumps({"version": 1, "albums": {}}), encoding="utf-8"
    )
    _write(outputs, _album("ARTIST", "Real", ["a.mp3"]))

    assert list(albums_from_outputs(outputs)) == ["ARTIST/Real"]


def test_old_pipeline_albums_are_not_repolished(tmp_path, capsys):
    """An older album has no chunk_captions; polishing it would blank it out."""
    outputs = tmp_path / "outputs"
    old = _album("ARTIST", "Old", ["a.mp3"])
    del old["caption_pipeline_version"]
    old["tracks"][0].pop("chunk_captions")
    _write(outputs, old)
    _write(outputs, _album("ARTIST", "New", ["b.mp3"]))

    albums = albums_from_outputs(outputs)

    assert list(albums) == ["ARTIST/New"]
    assert "need captioning, not repolishing" in capsys.readouterr().out


def test_failed_tracks_are_skipped_with_a_warning(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    _write(outputs, _album(
        "ARTIST", "Album", ["ok.mp3"],
        failed_tracks=[{"file": "bad.mp3", "error": "boom"}],
    ))

    albums = albums_from_outputs(outputs)

    assert [t["filename"] for t in albums["ARTIST/Album"]["tracks"]] == ["ok.mp3"]
    assert "skipping 1 failed track" in capsys.readouterr().out


# -- the polish phase must run without any captioner -------------------------

CLEAN = (
    "A piano track with a clean mix. The piano carries the tune while the clean "
    "mix keeps the instrument forward, and the piano returns over the same clean "
    "mix without adding anything else to the texture. The arrangement stays "
    "simple and focused from the first bar to the last, and the piano remains "
    "the only voice in the piece throughout the whole performance, with the "
    "clean mix holding it in place at every point of the track."
)


class _LongRewriter:
    name = "stub-gguf"

    def rewrite(self, messages):
        return CLEAN

    def unload(self):
        pass


def _stored_album(files: list[str]) -> dict:
    chunks = ["A piano track with a clean mix.", "The piano returns over the mix."]
    return {
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "caption_stage": "captioned",
        "status": "ok",
        "tracks": [
            {
                "file": name,
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
                "tags_experimental": True,
            }
            for name in files
        ],
        "failed_tracks": [],
        "skipped_tracks": [],
    }


def test_repolish_runs_without_a_captioner(tmp_path):
    from scripts.batch import process_album

    entry = {
        "artist": "ARTIST",
        "album": "Album",
        "source_dir": "/content/ARTIST",
        "tracks": [{"filename": "a.mp3", "uri": ""}],
    }
    album_obj = process_album(
        None, entry, tmp_path / "work", tmp_path / "outputs",
        existing=_stored_album(["a.mp3"]), retry_only_failed=True,
        rewriter=_LongRewriter(),
    )
    assert album_obj["caption_stage"] == "polished"
    assert album_obj["tracks"][0]["rewrite_status"] == "ok"


def test_repolish_reports_a_track_missing_from_storage(tmp_path):
    from scripts.batch import process_album

    entry = {
        "artist": "ARTIST",
        "album": "Album",
        "source_dir": "/content/ARTIST",
        "tracks": [
            {"filename": "a.mp3", "uri": ""},
            {"filename": "gone.mp3", "uri": ""},
        ],
    }
    album_obj = process_album(
        None, entry, tmp_path / "work", tmp_path / "outputs",
        existing=_stored_album(["a.mp3"]), retry_only_failed=True,
        rewriter=_LongRewriter(),
    )
    assert album_obj["status"] == "partial"
    assert "not in the stored captions" in album_obj["failed_tracks"][0]["error"]
