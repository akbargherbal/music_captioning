"""Tests for the frozen yt_top_500 manifest and the new batch guards.

No GPU, no GCS, no model: these exercise the pure functions and the
content-addressed download/restore guards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import abc_batch
import yt500_manifest


def _track(rank: int, artist: str, title: str, video_id: str,
           duration: float, size: int) -> dict:
    return {
        "track_id": f"{rank:016x}",
        "video_id": video_id,
        "rank": rank,
        "artist": artist,
        "title": title,
        "youtube_title": f"{artist} - {title}",
        "country": "Lebanon",
        "tier": "Tier 1",
        "duration_seconds": int(duration),
        "actual_duration": duration,
        "file_bytes": size,
        "sha256": f"{rank:064x}",
        "bitrate": 129400,
        "gcs_uri": f"gs://bucket/audio/{rank:04d}_{video_id}.m4a",
        "info_json_uri": f"gs://bucket/metadata/{rank:04d}_{video_id}.info.json",
    }


def _manifest() -> dict:
    specs = [
        (1, "A", "Same", "vid01", 100.0, 1_000_000),
        (2, "A", "Same", "vid02", 110.0, 1_100_000),
        (3, "B", "Other", "vid03", 120.0, 1_200_000),
        (4, "B", "Other", "vid04", 130.0, 1_300_000),
        (5, "C", "One", "vid05", 140.0, 1_400_000),
        (6, "D", "Two", "vid06", 150.0, 1_500_000),
        (7, "E", "Three", "vid07", 160.0, 1_600_000),
        (8, "F", "Four", "vid08", 170.0, 1_700_000),
        (9, "G", "Five", "vid09", 180.0, 1_800_000),
        (10, "H", "Six", "vid10", 190.0, 1_900_000),
        (11, "I", "Seven", "vid11", 200.0, 2_000_000),
        (12, "J", "Eight", "vid12", 210.0, 2_100_000),
    ]
    return {"tracks": [_track(*s) for s in specs]}


def test_build_rows_freezes_shards_and_duplicate_links(tmp_path):
    manifest = _manifest()
    rows, shards = yt500_manifest.build_rows(manifest, 3)
    assert len(rows) == 12
    assert len(shards) == 3
    assert sorted(len(v) for v in shards.values()) == [4, 4, 4]
    assert all(r["artist"] == "YT500" for r in rows)
    assert all(r["source_artist"] for r in rows)
    assert all(r["format"] == "m4a" for r in rows)
    assert all(r["file"].endswith(".m4a") for r in rows)

    out = tmp_path / "manifest.jsonl"
    yt500_manifest.write_outputs(
        rows, shards, manifest, "gs://bucket/manifest.json",
        "a" * 64, "b" * 64, out, "test", "gs://bucket")
    assert out.is_file()
    assert (tmp_path / "manifest.sha256").is_file()
    assert yt500_manifest.check_manifest(
        out, expected_tracks=12, expected_seconds=None,
        expected_bytes=None) == 0

    lines = [json.loads(line) for line in
             out.read_text(encoding="utf-8").splitlines() if line]
    assert lines[0]["record"] == "header"
    body = lines[1:]
    assert len(body) == 12
    groups = {r["duplicate_group"] for r in body if r["duplicate_group"]}
    assert len(groups) == 2
    assert sum(1 for r in body if r["duplicate_of"]) == 2
    assert sum(1 for r in body if r["canonical"]) == 10


def test_check_manifest_rejects_duplicate_track_ids(tmp_path):
    rows, shards = yt500_manifest.build_rows(_manifest(), 3)
    out = tmp_path / "manifest.jsonl"
    rows[1]["track_id"] = rows[0]["track_id"]
    yt500_manifest.write_outputs(
        rows, shards, _manifest(), "gs://bucket/manifest.json",
        "a" * 64, "b" * 64, out, "test", "gs://bucket")
    assert yt500_manifest.check_manifest(
        out, expected_tracks=12, expected_seconds=None,
        expected_bytes=None) == 1


def test_manifest_header_is_not_a_track_row(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps({"record": "header", "counts": {}}) + "\n" +
        json.dumps({"record": "track", "track_id": "a"}) + "\n" +
        json.dumps({"record": "track", "track_id": "b"}) + "\n",
        encoding="utf-8")
    rows = abc_batch.load_manifest(path)
    assert [r["track_id"] for r in rows] == ["a", "b"]


def test_manifest_sha256_guard(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text("hello\n", encoding="utf-8")
    expected = hashlib.sha256(b"hello\n").hexdigest()
    abc_batch.verify_manifest_sha256(path, expected)
    with pytest.raises(abc_batch.BatchError, match="manifest hash mismatch"):
        abc_batch.verify_manifest_sha256(path, "0" * 64)


def test_track_complete_requires_status_and_abc_hash(tmp_path):
    track = tmp_path / "0001_vid"
    track.mkdir()
    abc = b"X:1\nT:x\nK:C\nC D E F|\n"
    (track / "score.abc").write_bytes(abc)
    digest = hashlib.sha256(abc).hexdigest()
    (track / "meta.json").write_text(
        json.dumps({"status": "ok", "abc_sha256": digest}), encoding="utf-8")
    assert abc_batch.track_complete(track)
    (track / "score.abc").write_bytes(abc + b"G A B c|\n")
    assert not abc_batch.track_complete(track)


def test_download_track_retries_a_hash_mismatch(tmp_path, monkeypatch):
    good = b"audio-bytes"
    row = {
        "track_id": "YT500/shard-01/0001_vid",
        "file": "0001_vid.m4a",
        "source_uri": "gs://bucket/audio/0001_vid.m4a",
        "source_sha256": hashlib.sha256(good).hexdigest(),
    }
    calls = {"n": 0}

    def fake_download(row: dict, dest: Path, log: Path) -> None:
        calls["n"] += 1
        dest.write_bytes(b"corrupt" if calls["n"] == 1 else good)

    monkeypatch.setattr(abc_batch, "_download_once", fake_download)
    abc_batch.download_track(row, tmp_path, tmp_path / "download.log")
    assert (tmp_path / "0001_vid.m4a").read_bytes() == good
    assert calls["n"] == 2
