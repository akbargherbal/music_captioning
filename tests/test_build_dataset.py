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


# --------------------------------------------------------------------------- #
# multi-run corpus (§6.3)
# --------------------------------------------------------------------------- #
def _multi_run_tree(tmp_path: Path, count: int) -> dict:
    """A watcher-style queue: one accepted take per name, spanning two runs.

    The manifest is ``{accepted: [...]}`` (no ``tracks``, no ``run_dir``), the
    shape ``yt500_sync.py`` writes.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    abc_ref = tmp_path / "abc_ref"
    abc_ref.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    captions = tmp_path / "captions" / "arabmaqamrock"
    captions.mkdir(parents=True)
    runs = ["20261110-083615_yt500inst", "20261110-120000_yt500inst-regen"]

    accepted, tracks = [], []
    for i in range(count):
        name = f"yt500-{i:04d}-vid{i}"
        run = runs[i % 2]
        stem = f"{name}_{1000 + i}"
        (audio_dir / f"{stem}.mp3").write_bytes(f"mp3-{i}".encode())
        (abc_ref / f"{name}.abc").write_text(ABC, encoding="utf-8")
        tracks.append({
            "file": f"{stem}.mp3",
            "duration_s": 250.0 + i,
            "bitrate_kbps": 192,
            "chunk_count": 2,
            "caption_seconds": 25.0,
            "instruments": ["oud"],
            "vocals": [],
            "production": ["reverb"],
            "chunk_captions": ["a", "b"],
            "caption_raw": f"raw {i}",
            "caption": f"polished {i}",
            "rewrite_status": "ok",
            "rewrite_model": "gguf:test",
            "tags_experimental": True,
        })
        accepted.append({
            "name": name,
            "run_id": run,
            "seed": 1000 + i,
            "take": 0,
            "file": f"{stem}.mp3",
            "stem": stem,
            "gs_uri": f"gs://bucket/runs/{run}/{stem}.mp3",
            "abc_sha256": _sha(ABC.encode()),
            "wav_duration_s": 250.0 + i,
            "nominal_seconds": 252.0 + i,
            "take_ratio": (250.0 + i) / (252.0 + i),
            "end_utc": "2026-11-10T08:38:44Z",
            "title": f"TITLE {i}",
            "source_artist": "Elissa",
            "country": "Lebanon",
            "tier": "Tier 1",
            "source_rank": i + 1,
            "source_duration_s": 258.0,
            "source_abc_sha256": _sha(b"source"),
            "generation_status": "ready",
        })

    # Two transcripts only: qc_track shells out to abc2midi, which is not a
    # Python dependency; row count must not depend on it.
    for row in accepted[:2]:
        tdir = transcripts / row["stem"]
        tdir.mkdir(parents=True)
        (tdir / "score.abc").write_text(ABC, encoding="utf-8")
        (tdir / "key.lab").write_text("0.0 240.0 C\n", encoding="utf-8")
        (tdir / "chord.lab").write_text("0.0 240.0 C\n", encoding="utf-8")
        (tdir / "beat.lab").write_text(
            "\n".join(f"{i * 0.5:.3f} 1" for i in range(480)) + "\n",
            encoding="utf-8")
        (tdir / "meta.json").write_text(json.dumps({
            "status": "ok", "duration_s": 240.0,
            "model": "m-a-p/SheetSage2", "model_revision": "deadbeef",
            "seconds": 40.0,
        }), encoding="utf-8")

    album_dir = captions
    (album_dir / "YT500_INSTRUMENTAL.json").write_text(json.dumps({
        "artist": "arabmaqamrock",
        "album": "YT500_INSTRUMENTAL",
        "status": "ok",
        "caption_stage": "polished",
        "tracks": tracks,
    }), encoding="utf-8")

    run_manifest = tmp_path / "manifest.json"
    run_manifest.write_text(json.dumps({
        "version": 1,
        "corpus_total": count,
        "counts": {"corpus_total": count, "accepted": count, "rejected": 1},
        "source_runs": [{"run_id": r, "takes": count // 2} for r in runs],
        "rejected": [{"name": "yt500-9999-short", "reasons": ["short_take"]}],
        "take_selection": {"rule": "latest end_utc", "superseded_takes": 3},
        "accepted": accepted,
    }), encoding="utf-8")

    return {
        "audio": audio_dir,
        "abc_ref": abc_ref,
        "transcripts": transcripts,
        "captions": tmp_path / "captions",
        "run_manifest": run_manifest,
        "out": tmp_path / "package",
        "tmp": tmp_path,
    }


def _run_multi(tree: dict) -> int:
    return bd.main([
        "assemble",
        "--run-manifest", str(tree["run_manifest"]),
        "--audio-dir", str(tree["audio"]),
        "--captions", str(tree["captions"]),
        "--abc-ref", str(tree["abc_ref"]),
        "--transcripts", str(tree["transcripts"]),
        "--out", str(tree["out"]),
        "--run-id", "YT500_INSTRUMENTAL",
        "--source-uri", "gs://bucket/runs",
        "--dataset-name", "YT500 Instrumental 489",
        "--dataset-version", "v0.1",
    ])


def test_assemble_multi_run_uses_per_track_run_id_and_gs_uri(
        tmp_path: Path) -> None:
    tree = _multi_run_tree(tmp_path, 4)
    assert _run_multi(tree) == 0
    rows = [json.loads(line) for line in
            (tree["out"] / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {r["run_id"] for r in rows} == {
        "20261110-083615_yt500inst", "20261110-120000_yt500inst-regen"}
    for row in rows:
        assert row["track_id"] == f"{row['run_id']}/{Path(row['audio_file']).stem}"
        assert row["gs_uri"] == (
            f"gs://bucket/runs/{row['run_id']}/{row['audio_file']}")
    per_track = json.loads(
        (tree["out"] / "captions" / f"{rows[0]['track_id'].split('/')[1]}.json")
        .read_text(encoding="utf-8"))
    assert per_track["run_id"] == rows[0]["run_id"]
    assert per_track["track_id"] == rows[0]["track_id"]


def test_assemble_multi_run_adds_the_corpus_columns(tmp_path: Path) -> None:
    tree = _multi_run_tree(tmp_path, 2)
    assert _run_multi(tree) == 0
    row = json.loads(
        (tree["out"] / "tracks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["title"] == "TITLE 0"
    assert row["source_artist"] == "Elissa"
    assert row["country"] == "Lebanon"
    assert row["tier"] == "Tier 1"
    assert row["source_rank"] == 1
    assert row["source_duration_s"] == 258.0
    assert row["generation_status"] == "ready"
    assert row["take_ratio"] == pytest.approx(250.0 / 252.0, abs=1e-6)
    assert row["instrumental"] is True


def test_assemble_multi_run_manifest_records_the_runs_and_rejects(
        tmp_path: Path) -> None:
    tree = _multi_run_tree(tmp_path, 4)
    assert _run_multi(tree) == 0
    manifest = json.loads(
        (tree["out"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["tracks"] == 4
    assert len(manifest["run_ids"]) == 2
    assert len(manifest["source_runs"]) == 2
    assert manifest["rejected"][0]["name"] == "yt500-9999-short"
    assert manifest["take_selection"]["superseded_takes"] == 3
    assert manifest["dataset"] == "YT500 Instrumental 489"
    card = (tree["out"] / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "one or more generated-instrumental" in card


def test_single_run_schema_keeps_the_reference_columns(tree: dict) -> None:
    assert _run(tree) == 0
    row = json.loads(
        (tree["out"] / "tracks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for column in bd.CORPUS_COLUMNS:
        assert column not in row


def test_assemble_writes_489_rows(tmp_path: Path) -> None:
    tree = _multi_run_tree(tmp_path, 489)
    assert _run_multi(tree) == 0
    out = tree["out"]
    lines = (out / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 489
    import pyarrow.parquet as pq

    assert pq.read_table(out / "tracks.parquet").num_rows == 489
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["tracks"] == 489
    assert (out / "_SUCCESS").is_file()
    for entry in manifest["files"]:
        blob = (out / entry["path"]).read_bytes()
        assert _sha(blob) == entry["sha256"]
