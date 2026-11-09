"""Tests for the standalone ``yt500_backup`` worker.

No GPU, no model, no real GCS: a fake ``gcloud storage`` is installed by
monkeypatching :func:`abc_batch.run`.  The assertions are about the contract
that matters — a complete track is archived and verified, an incomplete track
is never touched, a second pass is a no-op, and a conflicting remote object is
a hard stop rather than an overwrite.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

import abc_batch
import yt500_backup as backup

REAL_RUN = abc_batch.run


class FakeGCS:
    def __init__(self, root: Path):
        self.root = root
        self.copies: list[tuple[str, str]] = []
        self.calls: list[list[str]] = []

    def _path(self, uri: str) -> Path:
        assert uri.startswith("gs://"), uri
        return self.root / uri[len("gs://"):]

    def path(self, uri: str) -> Path:
        return self._path(uri)

    def __call__(self, cmd, *, cwd=None, log=None, timeout=None):
        self.calls.append(list(cmd))
        if cmd[0] != "gcloud":
            return REAL_RUN(cmd, cwd=cwd, log=log, timeout=timeout)
        rest = cmd[1:]
        assert rest[0] == "storage", cmd
        op, args = rest[1], rest[2:]
        if op == "cp":
            src, dst = args[-2], args[-1]
            if src.startswith("gs://"):
                src_path = self._path(src)
            else:
                src_path = Path(src)
            if dst.startswith("gs://"):
                dst_path = self._path(dst)
            else:
                dst_path = Path(dst)
            if not src_path.is_file():
                return subprocess.CompletedProcess(cmd, 1, "", "no such object")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_bytes(src_path.read_bytes())
            self.copies.append((src, dst))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if op == "ls":
            uri = args[-1]
            path = self._path(uri)
            if not path.is_file():
                return subprocess.CompletedProcess(cmd, 1, "", "not found")
            out = f"{path.stat().st_size}  2026-11-09T00:00:00Z  {uri}\n"
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if op == "hash":
            path = self._path(args[-1])
            if not path.is_file():
                return subprocess.CompletedProcess(cmd, 1, "", "not found")
            md5 = abc_batch.md5_b64(path)
            return subprocess.CompletedProcess(cmd, 0, f"md5_hash: {md5}\n", "")
        if op == "cat":
            path = self._path(args[-1])
            if not path.is_file():
                return subprocess.CompletedProcess(cmd, 1, "", "not found")
            return subprocess.CompletedProcess(
                cmd, 0, path.read_text(encoding="utf-8"), "")
        return subprocess.CompletedProcess(cmd, 1, "", f"unsupported {op}")


def _row(rank: int, video_id: str, shard: str = "shard-01") -> dict:
    stem = f"{rank:04d}_{video_id}"
    return {
        "record": "track",
        "track_id": f"YT500/{shard}/{stem}",
        "video_id": video_id,
        "source_rank": rank,
        "source_artist": "A",
        "title": f"T{rank}",
        "file": f"{stem}.m4a",
        "source_uri": f"gs://source/audio/{stem}.m4a",
        "source_bytes": 10,
        "source_sha256": "a" * 64,
        "duration_s": 100.0,
        "shard": shard,
        "output_stem": stem,
        "artist": "YT500",
        "album": shard,
        "status": "pending",
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"record": "header", "corpus": "test", "counts": {}})]
    lines += [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_track(work: Path, row: dict, abc: bytes = b"X:1\nT:x\nK:C\nC D E F|\n") -> Path:
    track_dir = (work / "out" / row["artist"] / row["album"]
                 / Path(row["file"]).stem)
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "score.abc").write_bytes(abc)
    (track_dir / "meta.json").write_text(json.dumps({
        "status": "ok",
        "abc_sha256": hashlib.sha256(abc).hexdigest(),
        "source_wav_sha256": "b" * 64,
        "duration_s": 100.0,
        "seconds": 12.0,
    }), encoding="utf-8")
    return track_dir


def _make_packed(outputs: Path, row: dict) -> None:
    packed = outputs / row["artist"] / "abc" / f"{row['album']}.json"
    packed.parent.mkdir(parents=True, exist_ok=True)
    packed.write_text(json.dumps({
        "artist": row["artist"], "album": row["album"], "status": "done",
        "tracks": [{"track_id": row["track_id"]}],
    }), encoding="utf-8")
    index = outputs / "YT500" / "abc_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"version": 1, "albums": {
        f"{row['artist']}/{row['album']}": {
            "path": f"{row['artist']}/abc/{row['album']}.json",
            "status": "done", "track_count": 1,
        }}}), encoding="utf-8")


def _archive_with_meta(path: Path, stem: str, abc_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / "stage"
    (tmp / stem).mkdir(parents=True, exist_ok=True)
    (tmp / stem / "meta.json").write_text(
        json.dumps({"status": "ok", "abc_sha256": abc_sha}), encoding="utf-8")
    (tmp / stem / "score.abc").write_text("X:1\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(tmp / stem, arcname=stem)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    gcs = FakeGCS(tmp_path / "fakegcs")
    monkeypatch.setattr(abc_batch, "run", gcs)
    monkeypatch.setattr(backup.batch, "run", gcs)
    monkeypatch.setattr(backup.shutil, "which", lambda _: "/usr/bin/gcloud")
    return gcs


def test_one_pass_uploads_only_complete_tracks(fake, tmp_path):
    outputs = tmp_path / "outputs"
    work = tmp_path / "work"
    manifest = outputs / "YT500" / "manifest.jsonl"
    good, pending = _row(1, "v1"), _row(2, "v2")
    _write_manifest(manifest, [good, pending])
    _make_track(work, good)
    _make_packed(outputs, good)

    rc = backup.main([
        "--manifest", str(manifest),
        "--outputs", str(outputs),
        "--work-dir", str(work),
        "--gcs-dest", "gs://fake/TRANSCRIPTS/YT500",
        "--once",
    ])
    assert rc == 0

    assert fake.path(
        "gs://fake/TRANSCRIPTS/YT500/YT500/shard-01/0001_v1.tar.gz").is_file()
    assert not fake.path(
        "gs://fake/TRANSCRIPTS/YT500/YT500/shard-01/0002_v2.tar.gz").exists()
    assert fake.path(
        "gs://fake/TRANSCRIPTS/YT500/YT500/abc/shard-01.json").is_file()
    assert fake.path("gs://fake/TRANSCRIPTS/YT500/_index.json").is_file()

    ledger = work / "backup" / "ledger.jsonl"
    entries = [json.loads(line) for line in
               ledger.read_text(encoding="utf-8").splitlines() if line]
    assert {e["kind"] for e in entries} == {"track", "packed"}
    status = json.loads((work / "backup" / "status.json").read_text())
    assert status["pending"] == 1
    assert status["shards"] == 1


def test_second_pass_is_a_no_op(fake, tmp_path):
    outputs = tmp_path / "outputs"
    work = tmp_path / "work"
    manifest = outputs / "YT500" / "manifest.jsonl"
    row = _row(1, "v1")
    _write_manifest(manifest, [row])
    _make_track(work, row)
    _make_packed(outputs, row)
    argv = [
        "--manifest", str(manifest),
        "--outputs", str(outputs),
        "--work-dir", str(work),
        "--gcs-dest", "gs://fake/TRANSCRIPTS/YT500",
        "--once",
    ]
    assert backup.main(argv) == 0
    fake.copies.clear()
    assert backup.main(argv) == 0
    assert fake.copies == []


def test_incomplete_track_is_refused(tmp_path):
    row = _row(1, "v1")
    work = tmp_path / "work"
    track_dir = work / "out" / row["artist"] / row["album"] / "0001_v1"
    track_dir.mkdir(parents=True)
    with pytest.raises(abc_batch.BatchError, match="incomplete track"):
        backup.ensure_track(
            row, track_dir, "gs://fake/TRANSCRIPTS/YT500",
            work / "archives", {}, work / "backup" / "ledger.jsonl",
            work / "backup" / "log.txt", False, False)


def test_conflicting_remote_transcript_is_a_hard_stop(fake, tmp_path):
    row = _row(1, "v1")
    work = tmp_path / "work"
    track_dir = _make_track(work, row)
    stem = Path(row["file"]).stem
    remote = fake.path(
        f"gs://fake/TRANSCRIPTS/YT500/YT500/shard-01/{stem}.tar.gz")
    _archive_with_meta(remote, stem, "f" * 64)

    with pytest.raises(abc_batch.BatchError, match="append-only"):
        backup.ensure_track(
            row, track_dir, "gs://fake/TRANSCRIPTS/YT500",
            work / "archives", {}, work / "backup" / "ledger.jsonl",
            work / "backup" / "log.txt", False, False)


def test_existing_matching_remote_is_adopted(fake, tmp_path):
    row = _row(1, "v1")
    work = tmp_path / "work"
    track_dir = _make_track(work, row)
    meta = json.loads((track_dir / "meta.json").read_text())
    stem = Path(row["file"]).stem
    remote = fake.path(
        f"gs://fake/TRANSCRIPTS/YT500/YT500/shard-01/{stem}.tar.gz")
    _archive_with_meta(remote, stem, meta["abc_sha256"])

    ledger_path = work / "backup" / "ledger.jsonl"
    action = backup.ensure_track(
        row, track_dir, "gs://fake/TRANSCRIPTS/YT500", work / "archives",
        {}, ledger_path, work / "backup" / "log.txt", False, False)
    assert action == "skipped"
    entries = [json.loads(line) for line in
               ledger_path.read_text(encoding="utf-8").splitlines() if line]
    assert entries[-1]["action"] == "adopted"


def test_packed_json_conflict_is_rejected(fake, tmp_path):
    row = _row(1, "v1")
    outputs = tmp_path / "outputs"
    packed = outputs / row["artist"] / "abc" / f"{row['album']}.json"
    packed.parent.mkdir(parents=True, exist_ok=True)
    packed.write_text('{"a": 1}\n', encoding="utf-8")
    remote = fake.path("gs://fake/TRANSCRIPTS/YT500/YT500/abc/shard-01.json")
    remote.parent.mkdir(parents=True, exist_ok=True)
    remote.write_text('{"a": 2, "bigger": true}\n', encoding="utf-8")

    with pytest.raises(abc_batch.BatchError, match="append-only"):
        backup.ensure_packed(
            row, packed, "gs://fake/TRANSCRIPTS/YT500", {},
            tmp_path / "ledger.jsonl", tmp_path / "log.txt", False, False)


def test_complete_requires_terminal_index_entries(tmp_path):
    outputs = tmp_path / "outputs"
    work = tmp_path / "work"
    row = _row(1, "v1")
    _write_manifest(outputs / "YT500" / "manifest.jsonl", [row])
    _make_track(work, row)
    backup_index = work / "backup" / "abc_index.json"
    backup_index.parent.mkdir(parents=True, exist_ok=True)
    backup_index.write_text(json.dumps({"version": 1, "albums": {}}))

    assert not backup.complete(
        [row], outputs, outputs / "YT500" / "abc_index.json", backup_index)

    _make_packed(outputs, row)
    backup_index.write_text(json.dumps({"version": 1, "albums": {
        "YT500/shard-01": {"status": "done"}}}))
    assert backup.complete(
        [row], outputs, outputs / "YT500" / "abc_index.json", backup_index)


def test_skip_backup_rejects_archive_flags(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(1, "v1")])
    (tmp_path / "transcribe.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(abc_batch.BatchError, match="skip-backup conflicts"):
        abc_batch.main([
            "--manifest", str(manifest),
            "--skip-backup", "--archive-every-track",
            "--amr-dir", str(tmp_path),
            "--work-dir", str(tmp_path / "work"),
        ])
