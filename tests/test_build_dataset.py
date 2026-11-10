"""CPU-only tests for scripts/build_dataset.py assemble (no model, no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_dataset as bd

ABC = """X:1
T:test
M:4/4
L:1/8
Q:1/4=120
K:C
V: Vocal clef=treble
cccc cccc|cccc cccc|
V: Ins clef=treble
CDEF GABc|CDEF GABc|
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def tree(tmp_path: Path) -> dict:
    run_id = "20260101-000000_demo_run"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "song_123.mp3"
    audio.write_bytes(b"not really an mp3")

    abc_ref = tmp_path / "abc_ref"
    abc_ref.mkdir()
    (abc_ref / "song.abc").write_text(ABC, encoding="utf-8")

    transcripts = tmp_path / "transcripts" / "song_123"
    transcripts.mkdir(parents=True)
    (transcripts / "score.abc").write_text(ABC, encoding="utf-8")
    (transcripts / "key.lab").write_text("0.0 240.0 C\n", encoding="utf-8")
    (transcripts / "chord.lab").write_text("0.0 240.0 C\n", encoding="utf-8")
    (transcripts / "beat.lab").write_text(
        "\n".join(f"{i * 0.5:.3f} 1" for i in range(480)) + "\n",
        encoding="utf-8")
    (transcripts / "meta.json").write_text(json.dumps({
        "status": "ok", "duration_s": 240.0, "model": "m-a-p/SheetSage2",
        "model_revision": "deadbeef", "seconds": 40.0,
    }), encoding="utf-8")

    captions = tmp_path / "captions" / "arabmaqamrock"
    captions.mkdir(parents=True)
    album = {
        "artist": "arabmaqamrock",
        "album": "demo_run",
        "status": "ok",
        "caption_stage": "polished",
        "tracks": [{
            "file": "song_123.mp3",
            "duration_s": 240.0,
            "bitrate_kbps": 192,
            "chunk_count": 2,
            "silence_trimmed_s": 0.4,
            "caption_seconds": 30.0,
            "instruments": ["oud"],
            "vocals": [],
            "production": ["reverb"],
            "chunk_captions": ["a", "b"],
            "caption_raw": "raw text",
            "caption": "polished text",
            "rewrite_status": "ok",
            "rewrite_model": "gguf:test",
            "tags_experimental": True,
        }],
    }
    (captions / "demo_run.json").write_text(json.dumps(album), encoding="utf-8")

    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({
        "run_dir": f"/content/arabmaqamrock/out/{run_id}",
        "tracks": [{
            "name": "song", "seed": 123, "score_alias": "song",
            "abc_sha256": _sha(ABC.encode()), "abc_file": "scores/song.abc",
        }],
    }), encoding="utf-8")

    return {
        "run_id": run_id,
        "audio": audio_dir,
        "abc_ref": abc_ref,
        "transcripts": transcripts.parent,
        "captions": captions.parent,
        "run_manifest": run_manifest,
        "out": tmp_path / "package",
        "tmp": tmp_path,
    }


def _run(tree: dict) -> int:
    return bd.main([
        "assemble",
        "--run-manifest", str(tree["run_manifest"]),
        "--audio-dir", str(tree["audio"]),
        "--captions", str(tree["captions"]),
        "--abc-ref", str(tree["abc_ref"]),
        "--transcripts", str(tree["transcripts"]),
        "--out", str(tree["out"]),
        "--source-uri", "gs://bucket/run",
        "--dataset-name", "Demo",
        "--dataset-version", "v0.1",
    ])


def test_assemble_writes_package(tree: dict) -> None:
    assert _run(tree) == 0
    out = tree["out"]
    rows = [json.loads(line) for line in
            (out / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["track_id"] == f"{tree['run_id']}/song_123"
    assert row["caption"] == "polished text"
    assert row["abc_reference_sha256"] == _sha(ABC.encode())
    assert row["abc_transcription_parse"] == "ok"
    assert row["gs_uri"] == "gs://bucket/run/song_123.mp3"

    assert (out / "abc" / "song_123.abc").is_file()
    assert (out / "abc_transcription" / "song_123.abc").is_file()
    assert (out / "tracks.parquet").is_file()
    assert (out / "_SUCCESS").is_file()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["tracks"] == 1
    assert manifest["counts"]["with_abc_transcription"] == 1
    paths = {f["path"] for f in manifest["files"]}
    assert {"tracks.jsonl", "tracks.parquet", "abc/song_123.abc"} <= paths
    for entry in manifest["files"]:
        blob = (out / entry["path"]).read_bytes()
        assert _sha(blob) == entry["sha256"]
        assert len(blob) == entry["bytes"]

    per_track = json.loads(
        (out / "captions" / "song_123.json").read_text(encoding="utf-8"))
    assert per_track["caption"]["chunk_captions"] == ["a", "b"]
    assert per_track["abc_reference"]["matches_run_manifest"] is True


def test_assemble_rejects_reference_abc_mismatch(tree: dict) -> None:
    manifest = json.loads(tree["run_manifest"].read_text(encoding="utf-8"))
    manifest["tracks"][0]["abc_sha256"] = "0" * 64
    tree["run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(tree)


def test_assemble_records_missing_transcript_as_null(tree: dict) -> None:
    for path in tree["transcripts"].glob("*/*"):
        path.unlink()
    assert _run(tree) == 0
    row = json.loads(
        (tree["out"] / "tracks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["abc_transcription_path"] is None
    assert row["abc_transcription_flags"] == []
