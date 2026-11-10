"""CPU-only tests for the YT500 GPU worker (no model, no GPU, no subprocess).

The worker must never start a model for work that is already done: on a second
pass it re-captions zero stems and re-polishes zero stems, and with nothing
pending it runs no subprocess at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import yt500_dataset as dataset
from scripts import yt500_learn as learn
from src.resume import CAPTION_PIPELINE_VERSION


def _track(name: str, *, status: str = "ok", caption: str = "a caption") -> dict:
    return {
        "file": name,
        "duration_s": 250.0,
        "chunk_captions": ["a", "b"],
        "caption_raw": "raw",
        "caption": caption,
        "rewrite_status": status,
        "instruments": ["oud"],
        "vocals": [],
        "production": ["reverb"],
        "tags_experimental": True,
    }


def _work(tmp_path: Path, *, tracks: list[dict] | None = None,
          transcript_meta: dict | None = None,
          accepted: int = 2) -> Path:
    work = tmp_path / "work"
    (work / "audio").mkdir(parents=True)
    rows = []
    for i in range(accepted):
        name = f"yt500-{i:04d}-v_{1000 + i}.mp3"
        (work / "audio" / name).write_bytes(b"mp3")
        rows.append({"name": f"yt500-{i:04d}-v", "run_id": "run-a",
                     "seed": 1000 + i, "take": 0, "file": name,
                     "gs_uri": f"gs://b/{name}", "abc_sha256": "a" * 64,
                     "take_ratio": 0.99, "stem": Path(name).stem})
    (work / "manifest.json").write_text(json.dumps({
        "version": 1, "corpus_total": accepted,
        "counts": {"corpus_total": accepted, "accepted": accepted,
                   "rejected": 0},
        "accepted": rows, "rejected": [],
    }), encoding="utf-8")

    if tracks is None:
        tracks = [_track(r["file"]) for r in rows]
    album = {
        "artist": dataset.ARTIST, "album": dataset.ALBUM,
        "status": "ok",
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "caption_stage": "polished",
        "tracks": tracks, "failed_tracks": [],
    }
    path = dataset.album_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(album), encoding="utf-8")

    meta = transcript_meta or {"status": "ok", "model": "m-a-p/SheetSage2",
                               "seconds": 40.0}
    for row in rows:
        tdir = dataset.transcripts_dir(work) / Path(row["file"]).stem
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "score.abc").write_text("X:1\nK:C\nCDEF|\n", encoding="utf-8")
        (tdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return work


def _args(work: Path, **overrides) -> argparse.Namespace:
    base = {
        "work_dir": work,
        "phases": ("caption", "transcribe"),
        "once": True,
        "min_batch": 12,
        "flush_minutes": 20.0,
        "max_batch": 48,
        "timeout_minutes": 0.0,
        "verify_schema": False,
        "transcribe_sh": Path("/tmp/transcribe.sh"),
        "dtype": "auto",
        "per_track_timeout": 1800.0,
        "hf_model": None,
        "rewrite_gguf": None,
        "llamacpp_server": None,
        "repolish": False,
        "no_rewrite": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_nothing_pending_starts_no_subprocess(tmp_path: Path,
                                              monkeypatch) -> None:
    work = _work(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(learn, "run_cmd",
                        lambda cmd, log, timeout: calls.append(cmd) or 0)

    ran = learn.learn_pass(_args(work), work, {}, set())

    assert ran == []
    assert calls == []
    assert learn.caption_pending(work) == []
    assert learn.transcribe_pending(work) == []


def test_a_new_take_makes_exactly_one_phase_pending(tmp_path: Path,
                                                   monkeypatch) -> None:
    work = _work(tmp_path)
    extra = "yt500-9999-new_5000.mp3"
    (work / "audio" / extra).write_bytes(b"mp3")
    manifest = json.loads((work / "manifest.json").read_text())
    manifest["accepted"].append({
        "name": "yt500-9999-new", "run_id": "run-b", "seed": 5000, "take": 0,
        "file": extra, "gs_uri": f"gs://b/{extra}", "abc_sha256": "b" * 64,
        "take_ratio": 0.97, "stem": Path(extra).stem})
    manifest["counts"]["accepted"] = len(manifest["accepted"])
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert learn.caption_pending(work) == [extra]
    assert learn.transcribe_pending(work) == [extra]

    shim = tmp_path / "transcribe.sh"
    shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(learn, "run_cmd",
                        lambda cmd, log, timeout: calls.append(cmd) or 0)
    ran = learn.learn_pass(_args(work, max_batch=1, transcribe_sh=shim),
                           work, {}, set())
    assert ran == ["caption", "transcribe"]
    assert "caption" in calls[0]
    assert "transcribe" in Path(calls[1][0]).name


def test_a_stale_transcript_is_pending_again(tmp_path: Path) -> None:
    work = _work(tmp_path, transcript_meta={"status": "parse_fail"})
    pending = learn.transcribe_pending(work)
    assert len(pending) == 2


def test_an_error_rewrite_is_pending_again(tmp_path: Path) -> None:
    work = _work(tmp_path, tracks=[_track("yt500-0000-v_1000.mp3", status="error"),
                                   _track("yt500-0001-v_1001.mp3")])
    assert learn.caption_pending(work) == ["yt500-0000-v_1000.mp3"]


def test_reused_transcript_with_a_hash_mismatch_is_pending(tmp_path: Path) -> None:
    work = _work(tmp_path)
    tdir = dataset.transcripts_dir(work) / "yt500-0000-v_1000"
    (tdir / "meta.json").write_text(json.dumps(
        {"status": "ok", "abc_sha256": "0" * 64}), encoding="utf-8")
    assert learn.transcribe_pending(work) == ["yt500-0000-v_1000.mp3"]


def test_verify_album_flags_an_empty_caption(tmp_path: Path) -> None:
    work = _work(tmp_path, tracks=[_track("yt500-0000-v_1000.mp3", status="ok", caption=""),
                                   _track("yt500-0001-v_1001.mp3")])
    problems = learn.verify_album(work)
    assert any("empty caption" in p for p in problems)


def test_verify_album_accepts_a_complete_album(tmp_path: Path) -> None:
    assert learn.verify_album(_work(tmp_path)) == []


def test_mirror_outputs_writes_the_album_and_jsonl(tmp_path: Path) -> None:
    work = _work(tmp_path)
    album = learn.mirror_outputs(work)
    assert album is not None
    assert (work / "caption_album.json").is_file()
    lines = (work / "captions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["run_id"] == "run-a"
    assert row["caption"] == "a caption"
    assert row["stem"].startswith("yt500-0000")


@pytest.mark.parametrize("pending, min_batch, last_age, expected", [
    (["a"], 12, 0, False),
    (["a"] * 12, 12, 0, True),
    (["a"], 12, 21 * 60, True),
    ([], 12, 999 * 60, False),
    (["a"], 0, 0, True),
])
def test_due_batches_small_arrivals(pending, min_batch, last_age, expected,
                                    tmp_path: Path) -> None:
    import time

    args = _args(tmp_path, once=False, min_batch=min_batch)
    state = {"caption": {"last_epoch": time.time() - last_age}}
    assert learn.due("caption", pending, args, state) is expected
