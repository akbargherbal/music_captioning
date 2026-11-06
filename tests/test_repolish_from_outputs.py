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
