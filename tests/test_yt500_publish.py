"""CPU-only tests for the YT500 publisher (no GCS, no gcloud).

Two contracts matter here:

* only a transcript whose ``meta.json.status`` is ``ok`` and whose ``score.abc``
  matches ``meta.abc_sha256`` is uploaded, and
* nothing is ever deleted -- a re-run uploads only what changed, and a remote
  object that no local file claims is reported, never removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import yt500_dataset as dataset
from scripts import yt500_publish as publish

ABC = "X:1\nK:C\nCDEF|\n"


class FakeGcs:
    """Records uploads; exposes no delete operation at all."""

    def __init__(self) -> None:
        self.remote: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.rsyncs: list[str] = []

    def upload(self, local: Path, uri: str, *, verify: bool = True) -> dict:
        self.remote[uri] = Path(local).read_bytes()
        self.uploads.append(uri)
        return {"uri": uri, "bytes": len(self.remote[uri])}

    def rsync_up(self, local_dir: Path, uri: str) -> None:
        self.rsyncs.append(uri)
        for path in Path(local_dir).rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir)
                self.remote[f"{uri.rstrip('/')}/{rel}"] = path.read_bytes()

    def compare_dir(self, local_dir: Path, uri: str) -> dict:
        base = uri.rstrip("/") + "/"
        local = {f"{base}{p.relative_to(local_dir)}": p.read_bytes()
                 for p in Path(local_dir).rglob("*") if p.is_file()}
        missing = [k for k in local if k not in self.remote]
        mismatched = [{"path": k} for k, v in local.items()
                      if k in self.remote and self.remote[k] != v]
        extra = [k for k in self.remote if k.startswith(base) and k not in local]
        return {"missing": missing, "mismatched": mismatched, "extra": extra}


def _work(tmp_path: Path, *, bad_meta: bool = False,
          good_meta: bool = True) -> Path:
    work = tmp_path / "work"
    (work / "audio").mkdir(parents=True)
    rows = []
    for i in range(2):
        name = f"yt500-{i:04d}-v_{1000 + i}.mp3"
        (work / "audio" / name).write_bytes(b"mp3")
        rows.append({"name": f"yt500-{i:04d}-v", "run_id": "run-a",
                     "seed": 1000 + i, "file": name})
    (work / "manifest.json").write_text(json.dumps({
        "corpus_total": 2, "counts": {"corpus_total": 2, "accepted": 2},
        "accepted": rows, "rejected": []}), encoding="utf-8")
    (work / "rejected.json").write_text(json.dumps(
        {"rejected": []}), encoding="utf-8")

    for i, row in enumerate(rows):
        stem = Path(row["file"]).stem
        tdir = dataset.transcripts_dir(work) / stem
        tdir.mkdir(parents=True)
        (tdir / "score.abc").write_text(ABC, encoding="utf-8")
        good = good_meta if i == 0 else not bad_meta
        (tdir / "meta.json").write_text(json.dumps(
            {"status": "ok" if good else "parse_fail"}), encoding="utf-8")

    # The album is copied from here; publish only uploads a settled album.
    album = {"artist": dataset.ARTIST, "album": dataset.ALBUM, "tracks": []}
    dataset.caption_album_path(work).write_text(json.dumps(album),
                                               encoding="utf-8")
    dataset.captions_jsonl_path(work).write_text("{}\n", encoding="utf-8")
    return work


def _args(work: Path, **overrides) -> argparse.Namespace:
    base = {
        "work_dir": work,
        "gcs_work": "gs://bucket/GENERATION/YT500_INSTRUMENTAL/work",
        "gcs_dataset": "gs://bucket/DATASET/YT500_INSTRUMENTAL",
        "once": True,
        "watch": False,
        "interval": 60.0,
        "settle_seconds": 0.0,
        "reverify": False,
        "dry_run": False,
        "max_minutes": 0.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_only_ok_transcripts_are_uploaded(tmp_path: Path) -> None:
    work = _work(tmp_path, bad_meta=True)
    client = FakeGcs()
    args = _args(work)
    status = publish.publish_once(client, work_dir=work,
                                  gcs_work=args.gcs_work,
                                  gcs_dataset=args.gcs_dataset,
                                  ledger={}, args=args)

    uploaded = "\n".join(client.uploads)
    assert "yt500-0000-v_1000/score.abc" in uploaded
    assert "yt500-0001-v_1001/score.abc" not in uploaded
    assert status["errors"] == []
    assert status["counts"]["uploaded"] > 0


def test_a_second_pass_re_uploads_no_artifact(tmp_path: Path) -> None:
    work = _work(tmp_path)
    client = FakeGcs()
    args = _args(work)
    ledger = publish.load_ledger(work)
    publish.publish_once(client, work_dir=work, gcs_work=args.gcs_work,
                         gcs_dataset=args.gcs_dataset, ledger=ledger, args=args)
    first = len(client.uploads)

    publish.publish_once(client, work_dir=work, gcs_work=args.gcs_work,
                         gcs_dataset=args.gcs_dataset, ledger=ledger, args=args)

    def artifacts(uris):
        return [u for u in uris if "status_" not in u.rsplit("/", 1)[-1]]

    assert len(artifacts(client.uploads)) == first
    assert (work / "publish" / "ledger.jsonl").is_file()


def test_publishing_never_deletes_a_remote_object(tmp_path: Path) -> None:
    work = _work(tmp_path)
    client = FakeGcs()
    stray = f"{_args(work).gcs_work}/transcripts/old/score.abc"
    client.remote[stray] = b"stale"
    args = _args(work)

    publish.publish_once(client, work_dir=work, gcs_work=args.gcs_work,
                         gcs_dataset=args.gcs_dataset, ledger={}, args=args)

    assert client.remote[stray] == b"stale"
    assert not [name for name in dir(client) if "delete" in name.lower()]


def test_an_unsettled_album_is_not_uploaded(tmp_path: Path) -> None:
    work = _work(tmp_path)
    client = FakeGcs()
    args = _args(work, settle_seconds=3600.0)
    publish.publish_once(client, work_dir=work, gcs_work=args.gcs_work,
                         gcs_dataset=args.gcs_dataset, ledger={}, args=args)
    assert not any(uri.endswith("caption_album.json") for uri in client.uploads)


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    work = _work(tmp_path)
    client = FakeGcs()
    args = _args(work, dry_run=True)
    status = publish.publish_once(client, work_dir=work,
                                  gcs_work=args.gcs_work,
                                  gcs_dataset=args.gcs_dataset,
                                  ledger={}, args=args)
    assert client.uploads == []
    assert client.remote == {}
    assert status["counts"].get("dry-run", 0) > 0
    assert not (work / "status_publish.json").exists()


def test_the_package_is_uploaded_once_success_exists(tmp_path: Path) -> None:
    work = _work(tmp_path)
    package = dataset.package_dir(work)
    package.mkdir(parents=True)
    (package / "tracks.jsonl").write_text("{}\n", encoding="utf-8")
    (package / "_SUCCESS").write_text("2026-11-10T00:00:00Z\n",
                                      encoding="utf-8")
    client = FakeGcs()
    args = _args(work)

    status = publish.publish_once(client, work_dir=work,
                                  gcs_work=args.gcs_work,
                                  gcs_dataset=args.gcs_dataset,
                                  ledger={}, args=args)

    assert client.rsyncs == [args.gcs_dataset]
    assert status["package_uploaded"] is True
    assert status["counts"]["compare"] == {"missing": 0, "mismatched": 0,
                                           "extra": 0}
    assert status["errors"] == []


def test_a_package_mismatch_is_reported(tmp_path: Path) -> None:
    work = _work(tmp_path)
    package = dataset.package_dir(work)
    package.mkdir(parents=True)
    (package / "tracks.jsonl").write_text("{}\n", encoding="utf-8")
    (package / "_SUCCESS").write_text("2026-11-10T00:00:00Z\n",
                                      encoding="utf-8")

    class LyingGcs(FakeGcs):
        def compare_dir(self, local_dir, uri):
            return {"missing": ["tracks.jsonl"], "mismatched": [], "extra": []}

    lying = LyingGcs()
    args = _args(work)
    status = publish.publish_once(lying, work_dir=work,
                                  gcs_work=args.gcs_work,
                                  gcs_dataset=args.gcs_dataset,
                                  ledger={}, args=args)
    assert any("missing" in error for error in status["errors"])


def test_status_view_reads_the_work_state(tmp_path: Path) -> None:
    work = _work(tmp_path)
    text = dataset.render_status(work)
    assert "corpus         2" in text
    assert "accepted       2" in text
    assert "transcribed    2" in text
    assert "regen          0 / 0" in text


def test_regen_stats_count_a_recovered_name(tmp_path: Path) -> None:
    work = _work(tmp_path)
    (work / "rejected_history.json").write_text(json.dumps({
        "yt500-0000-v": {"reasons": ["short_take"]},
        "yt500-9999-gone": {"reasons": ["long_take"]},
    }), encoding="utf-8")
    stats = dataset.regen_stats(work, dataset.read_manifest(work))
    assert stats == {"ever_rejected": 2, "recovered": 1, "open": 1}
