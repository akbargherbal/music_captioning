"""Download-policy tests for scripts/abc_batch.py.

One bad track must not cost the whole session. Only a systemic failure
(auth, permissions, no space) or an album with nothing usable is a hard stop.
"""

from pathlib import Path

import pytest

import abc_batch
from abc_batch import BatchError, album_pending, download_album


def _rows(*files: str) -> list[dict]:
    return [
        {
            "track_id": f"ELISSA/Album/{Path(f).stem}",
            "artist": "ELISSA",
            "album": "Album",
            "file": f,
            "source_uri": f"gs://bucket/DISCOGRAPHY/ELISSA/Album/{f}",
        }
        for f in files
    ]


def _fake_download(bad: set[str], *, error: str = "matched no objects"):
    def fake(row: dict, album_dir: Path, log: Path) -> None:
        if row["file"] in bad:
            raise BatchError(f"download failed: {row['source_uri']}: {error}")
        (album_dir / row["file"]).write_bytes(b"audio")
    return fake


def test_one_bad_track_is_skipped_and_the_rest_continue(tmp_path, monkeypatch):
    monkeypatch.setattr(abc_batch, "download_track",
                        _fake_download({"02.bad.mp3"}))
    available, failures = download_album(
        _rows("01.good.mp3", "02.bad.mp3", "03.good.mp3"),
        tmp_path / "album", tmp_path / "album.log")

    assert [r["file"] for r in available] == ["01.good.mp3", "03.good.mp3"]
    assert len(failures) == 1
    assert failures[0]["file"] == "02.bad.mp3"
    assert failures[0]["track_id"].endswith("/02.bad")
    assert "matched no objects" in failures[0]["error"]


def test_track_name_containing_403_is_not_systemic(tmp_path, monkeypatch):
    # The URI is part of the error message; "403.mp3" must not be read as an
    # HTTP 403 and abort the album.
    monkeypatch.setattr(abc_batch, "download_track",
                        _fake_download({"403.mp3"}))
    available, failures = download_album(
        _rows("403.mp3", "01.good.mp3"),
        tmp_path / "album", tmp_path / "album.log")
    assert [r["file"] for r in available] == ["01.good.mp3"]
    assert len(failures) == 1


def test_every_download_failing_is_a_hard_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(abc_batch, "download_track",
                        _fake_download({"01.mp3", "02.mp3"}))
    with pytest.raises(BatchError, match="no track could be downloaded"):
        download_album(_rows("01.mp3", "02.mp3"),
                       tmp_path / "album", tmp_path / "album.log")


@pytest.mark.parametrize("message", [
    "403 Forbidden",
    "401 Unauthorized",
    "invalid_grant",
    "No space left on device",
    "could not resolve host",
])
def test_systemic_error_aborts_immediately(tmp_path, monkeypatch, message):
    monkeypatch.setattr(abc_batch, "download_track",
                        _fake_download({"01.mp3"}, error=message))
    with pytest.raises(BatchError, match="systemic download failure"):
        download_album(_rows("01.mp3", "02.mp3"),
                       tmp_path / "album", tmp_path / "album.log")


def test_strict_downloads_restores_all_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(abc_batch, "download_track",
                        _fake_download({"02.mp3"}))
    with pytest.raises(BatchError, match="strict"):
        download_album(_rows("01.mp3", "02.mp3"),
                       tmp_path / "album", tmp_path / "album.log",
                       strict=True)


def test_failure_threshold_treated_as_systemic(tmp_path, monkeypatch):
    bad = {"01.mp3", "02.mp3", "03.mp3", "04.mp3"}
    monkeypatch.setattr(abc_batch, "download_track", _fake_download(bad))
    with pytest.raises(BatchError, match="treating as systemic"):
        download_album(_rows("01.mp3", "02.mp3", "03.mp3", "04.mp3", "05.mp3"),
                       tmp_path / "album", tmp_path / "album.log",
                       max_failures=3)


def test_partial_album_is_not_retried_by_default(tmp_path):
    index = {"albums": {"ELISSA/Album": {
        "status": "partial", "path": "ELISSA/abc/Album.json"}}}
    assert album_pending("ELISSA/Album", index, tmp_path,
                         force=False, retry_failed=False) is False
    assert album_pending("ELISSA/Album", index, tmp_path,
                         force=False, retry_failed=True) is True


def test_done_album_is_not_rerun_when_local_copy_is_gone(tmp_path):
    index = {"albums": {"ELISSA/Album": {
        "status": "done", "path": "ELISSA/abc/Album.json"}}}
    assert album_pending("ELISSA/Album", index, tmp_path,
                         force=False, retry_failed=False) is False


def test_qc_album_hard_stops_when_output_dir_is_missing(tmp_path):
    # A missing out_dir used to escape as a raw FileNotFoundError and crash
    # the batch instead of raising a readable BatchError.
    with pytest.raises(BatchError, match="no output directory"):
        abc_batch.qc_album(tmp_path / "missing", _rows("a.m4a"))


def test_relative_work_dir_is_resolved_against_the_caller_cwd(tmp_path, monkeypatch, capsys):
    # transcribe.sh runs with cwd=amr_dir, so the batch must hand it absolute
    # work-dir paths even when the user passes a relative --work-dir.
    (tmp_path / "amr").mkdir()
    (tmp_path / "amr" / "transcribe.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "m.jsonl").write_text(
        '{"track_id":"ELISSA/Album/a","artist":"ELISSA","album":"Album",'
        '"file":"a.m4a","source_uri":"gs://bucket/a.m4a","duration_s":10}\n',
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = abc_batch.main(["--manifest", "m.jsonl", "--work-dir", "work",
                         "--amr-dir", "amr", "--repo", "repo", "--list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"manifest: {tmp_path / 'm.jsonl'}" in out
