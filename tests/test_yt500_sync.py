"""CPU-only tests for the YT500 instrumental watcher (no gcloud, no GPU).

Covers the §3.1 acceptance gate, take selection, and one sync pass against an
in-memory bucket: the rejected stem must never reach ``manifest.json`` or the
audio directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts import yt500_sync as sync

BUCKET = "gs://bucket"
RUNS = f"{BUCKET}/runs"
RUN_A = f"{RUNS}/20261110-083615_yt500inst"
RUN_B = f"{RUNS}/20261110-120000_yt500inst-regen"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _corpus(tmp_path: Path, tracks: list[dict]) -> Path:
    path = tmp_path / "corpus" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "generated_at": "2026-11-10T06:27:18Z",
        "counts": {"source": 495, "ready": len(tracks)},
        "tracks": tracks,
    }), encoding="utf-8")
    return path


def _track(name: str, nominal: float, abc: str, **extra) -> dict:
    row = {
        "name": name,
        "title": name.upper(),
        "source_artist": "Elissa",
        "country": "Lebanon",
        "tier": "Tier 1",
        "source_rank": 44,
        "duration_s": nominal - 2,
        "source_abc_sha256": _sha("source-" + name),
        "instrumental_abc_sha256": _sha(abc),
        "nominal_seconds": nominal,
        "generation_status": "ready",
    }
    row.update(extra)
    return row


def _take(run: str, name: str, seed: int, *, wav: float, abc: str,
          final: bool = True, marker: bool = True, mp3: bool = True,
          sidecar: bool = True, take: int = 0, end: str | None = None) -> dict:
    """One take as GCS would hold it (metadata text + an opaque MP3)."""
    stem = f"{name}_{seed}"
    files: dict[str, object] = {}
    manifest_track = {"idx": 1, "name": name, "take": take, "seed": seed,
                      "cap": 8008, "cot": "full",
                      "abc_sha256": _sha(abc),
                      "abc_file": f"scores/{name}.abc"}
    if final:
        files[f"{run}/batch_run_summary.json"] = json.dumps({"requests": 1})
    if sidecar:
        files[f"{run}/{stem}.json"] = json.dumps({
            "name": name, "take": take, "seed": seed, "cap": 8008,
            "abc_sha256": _sha(abc), "status": "ok", "exit": 0,
            "wav_duration_s": wav,
            "end_utc": end or "2026-11-10T08:38:44Z",
        })
    if marker:
        files[f"{run}/{stem}_time.txt"] = (
            'Command being timed: "audiocpp_cli --request-sequence"\n'
            "Exit status: 0\n")
    if mp3:
        files[f"{run}/{stem}.mp3"] = b"ID3-fake-mp3"
    return {"name": name, "seed": seed, "stem": stem, "run": run,
            "manifest_track": manifest_track, "files": files}


class FakeGcs:
    """In-memory stand-in for :class:`yt500_gcs.Gcs` (no subprocesses)."""

    def __init__(self, takes: list[dict], *, fail: set[str] | None = None):
        self.objects: dict[str, object] = {}
        by_run: dict[str, list[dict]] = {}
        for take in takes:
            self.objects.update(take["files"])
            by_run.setdefault(take["run"], []).append(take["manifest_track"])
        # One batch_manifest.json per run, listing every take in that run.
        for run, tracks in by_run.items():
            self.objects[f"{run}/batch_manifest.json"] = json.dumps(
                {"run_dir": run, "tracks": tracks})
        self.fail = fail or set()
        self.downloads: list[str] = []

    def copy_glob(self, patterns, dest_dir: Path) -> bool:
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        matched = False
        dest_dir.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            prefix, _, glob = pattern.rpartition("*")
            suffix = glob.strip("/")
            for uri, value in self.objects.items():
                name = uri.rsplit("/", 1)[-1]
                if not uri.startswith(prefix) or not isinstance(value, str):
                    continue
                if not name.endswith(suffix):
                    continue
                (dest_dir / name).write_text(value, encoding="utf-8")
                matched = True
        return matched

    def list_objects(self, pattern: str) -> dict[str, dict]:
        return {uri: {"size": len(v) if isinstance(v, str) else len(v),
                      "updated": "2026-11-10T10:00:00Z"}
                for uri, v in self.objects.items()
                if uri.startswith(pattern.split("*")[0])}

    def cat(self, uri: str, *, missing_ok: bool = False):
        value = self.objects.get(uri)
        return value if isinstance(value, str) else None

    def cat_json(self, uri: str, *, missing_ok: bool = True):
        text = self.cat(uri)
        if text is None:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None

    def exists(self, uri: str) -> bool:
        return uri in self.objects

    def download(self, uri: str, dest: Path) -> Path:
        if any(stem in uri for stem in self.fail):
            raise sync.gcsio.GcsError(f"synthetic failure for {uri}")
        if uri not in self.objects:
            raise sync.gcsio.GcsError(f"missing object: {uri}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ID3-fake-mp3")
        self.downloads.append(uri)
        return dest


def _args(work_dir: Path, **overrides) -> argparse.Namespace:
    base = {
        "runs_prefix": RUNS,
        "run_glob": "*_yt500inst*",
        "gcs_work": "",
        "work_dir": work_dir,
        "corpus_index": None,
        "min_ratio": 0.85,
        "max_ratio": 1.25,
        "limit": 0,
        "download_limit": 0,
        "dry_run": False,
        "allow_abc_mismatch": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# §3.1 gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wav, expected", [
    (0.83 * 300, "short_take"),
    (0.84 * 300, "short_take"),
    (0.85 * 300, None),
    (1.00 * 300, None),
    (1.25 * 300, None),
    (1.26 * 300, "long_take"),
])
def test_gate_ratio_band(wav: float, expected: str | None) -> None:
    abc = "X:1\nK:C\nCDEF|\n"
    take = sync.evaluate_take(
        name="yt500-0001-abc", run_id="r", seed=1, take=0,
        file="yt500-0001-abc_1.mp3", mp3_uri="gs://b/r/x.mp3",
        expected_abc_sha256=_sha(abc), abc_sha256=_sha(abc),
        marker_text="Exit status: 0", has_mp3=True,
        sidecar={"wav_duration_s": wav}, nominal_seconds=300,
        run_final=True,
    )
    assert take["reason"] == expected
    assert take["accepted"] is (expected is None)


def test_gate_rejects_abc_mismatch_first() -> None:
    take = sync.evaluate_take(
        name="n", run_id="r", seed=1, take=0, file="n_1.mp3",
        mp3_uri="gs://b/r/n_1.mp3", expected_abc_sha256=_sha("frozen"),
        abc_sha256=_sha("other"), marker_text="Exit status: 0",
        has_mp3=True, sidecar={"wav_duration_s": 300},
        nominal_seconds=300, run_final=True,
    )
    assert take["accepted"] is False
    assert take["pending"] is False
    assert take["reason"] == "abc_mismatch"


def test_gate_requires_the_exit_marker() -> None:
    take = sync.evaluate_take(
        name="n", run_id="r", seed=1, take=0, file="n_1.mp3",
        mp3_uri="gs://b/r/n_1.mp3", expected_abc_sha256=_sha("a"),
        abc_sha256=_sha("a"), marker_text=None, has_mp3=True,
        sidecar={"wav_duration_s": 300}, nominal_seconds=300,
        run_final=True,
    )
    assert take["reason"] == "no_marker"


def test_gate_marks_an_unfinished_run_pending_not_rejected() -> None:
    take = sync.evaluate_take(
        name="n", run_id="r", seed=1, take=0, file="n_1.mp3",
        mp3_uri="gs://b/r/n_1.mp3", expected_abc_sha256=_sha("a"),
        abc_sha256=_sha("a"), marker_text=None, has_mp3=False,
        sidecar=None, nominal_seconds=300, run_final=False,
    )
    assert take["accepted"] is False
    assert take["pending"] is True
    assert take["reason"] is None


def test_gate_needs_the_sidecar_duration() -> None:
    take = sync.evaluate_take(
        name="n", run_id="r", seed=1, take=0, file="n_1.mp3",
        mp3_uri="gs://b/r/n_1.mp3", expected_abc_sha256=_sha("a"),
        abc_sha256=_sha("a"), marker_text="Exit status: 0", has_mp3=True,
        sidecar={"name": "n"}, nominal_seconds=300, run_final=True,
    )
    assert take["reason"] == "missing_sidecar"


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_selection_takes_the_latest_accepted_take() -> None:
    old = {"name": "n", "run_id": "run-a", "seed": 1, "take": 0,
           "end_utc": "2026-11-10T08:00:00Z", "accepted": False,
           "pending": False, "reason": "short_take", "ratio": 0.32}
    new = {"name": "n", "run_id": "run-b", "seed": 99, "take": 1,
           "end_utc": "2026-11-10T12:00:00Z", "accepted": True,
           "pending": False, "reason": None, "ratio": 0.98}
    out = sync.select_takes([old, new])
    assert [t["run_id"] for t in out["accepted"]] == ["run-b"]
    assert out["rejected"] == []
    assert out["pending"] == []


def test_selection_prefers_the_newer_of_two_accepted_takes() -> None:
    first = {"name": "n", "run_id": "a", "seed": 1, "take": 0,
             "end_utc": "2026-11-10T08:00:00Z", "accepted": True,
             "pending": False, "reason": None, "ratio": 0.99}
    second = {"name": "n", "run_id": "b", "seed": 2, "take": 1,
              "end_utc": "2026-11-10T09:00:00Z", "accepted": True,
              "pending": False, "reason": None, "ratio": 0.97}
    out = sync.select_takes([first, second])
    assert out["accepted"][0]["run_id"] == "b"
    assert out["superseded_takes"] == 1


def test_selection_rejects_only_names_without_an_accepted_take() -> None:
    short = {"name": "bad", "run_id": "a", "seed": 1, "take": 0,
             "end_utc": "2026-11-10T08:00:00Z", "accepted": False,
             "pending": False, "reason": "short_take", "ratio": 0.32,
             "wav_duration_s": 96.0, "nominal_seconds": 300.0}
    out = sync.select_takes([short])
    assert out["accepted"] == []
    assert [r["name"] for r in out["rejected"]] == ["bad"]
    assert out["rejected"][0]["reasons"] == ["short_take"]


def test_pick_spread_covers_short_median_long() -> None:
    accepted = [{"name": f"n{i:02d}", "nominal_seconds": 150 + i * 10,
                 "end_utc": f"2026-11-10T0{i % 10}:00:00Z"}
                for i in range(30)]
    chosen = sync.pick_spread(accepted, 4)
    assert len(chosen) == 4
    lengths = sorted(t["nominal_seconds"] for t in chosen)
    assert lengths[0] == 150
    assert lengths[-1] == 440


# --------------------------------------------------------------------------- #
# one sync pass
# --------------------------------------------------------------------------- #
@pytest.fixture()
def bucket(tmp_path: Path) -> dict:
    good_abc = "X:1\nK:C\nCDEF GABc|\n"
    tracks = [
        _track("yt500-0001-good", 300.0, good_abc),
        _track("yt500-0002-short", 300.0, good_abc),
    ]
    takes = [
        _take(RUN_A, "yt500-0001-good", 111, wav=298.0, abc=good_abc),
        _take(RUN_A, "yt500-0002-short", 222, wav=96.0, abc=good_abc),
    ]
    return {"corpus": _corpus(tmp_path, tracks),
            "client": FakeGcs(takes), "takes": takes, "tmp": tmp_path}


def test_sync_pass_excludes_the_short_take_and_downloads_the_good_one(
        bucket: dict) -> None:
    work = bucket["tmp"] / "work"
    corpus = sync.load_corpus(bucket["corpus"])
    result = sync.sync_once(bucket["client"], args=_args(work), corpus=corpus,
                            loop=1)

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in manifest["accepted"]]
    assert names == ["yt500-0001-good"]
    assert manifest["counts"]["accepted"] == 1
    assert manifest["counts"]["rejected"] == 1
    assert manifest["counts"]["corpus_total"] == 2

    row = manifest["accepted"][0]
    assert row["take_ratio"] == pytest.approx(298.0 / 300.0, abs=1e-3)
    assert row["file"] == "yt500-0001-good_111.mp3"
    assert row["gs_uri"].endswith(f"{RUN_A}/yt500-0001-good_111.mp3")
    assert row["source_artist"] == "Elissa"
    assert row["title"] == "YT500-0001-GOOD"

    rejected = json.loads((work / "rejected.json").read_text(encoding="utf-8"))
    assert rejected["rejected"][0]["name"] == "yt500-0002-short"
    assert rejected["rejected"][0]["reasons"] == ["short_take"]

    audio = work / "audio"
    assert sorted(p.name for p in audio.iterdir()) == [
        "yt500-0001-good_111.mp3"]
    assert not list(audio.glob("*.part"))
    assert not (audio / "yt500-0002-short_222.mp3").exists()
    assert result["counts"]["accepted"] == 1

    status = json.loads((work / "status_sync.json").read_text(encoding="utf-8"))
    assert status["role"] == "sync"
    assert status["band"] == [0.85, 1.25]


def test_sync_pass_is_idempotent(bucket: dict) -> None:
    work = bucket["tmp"] / "work"
    corpus = sync.load_corpus(bucket["corpus"])
    sync.sync_once(bucket["client"], args=_args(work), corpus=corpus, loop=1)
    started = len(bucket["client"].downloads)
    sync.sync_once(bucket["client"], args=_args(work), corpus=corpus, loop=2)
    assert len(bucket["client"].downloads) == started


def test_sync_keeps_going_when_one_download_fails(bucket: dict) -> None:
    bucket["client"].fail = {"yt500-0001-good_111"}
    work = bucket["tmp"] / "work"
    corpus = sync.load_corpus(bucket["corpus"])
    result = sync.sync_once(bucket["client"], args=_args(work), corpus=corpus,
                            loop=1)
    assert result["downloads"]["failed"]
    assert not (work / "audio" / "yt500-0001-good_111.mp3").exists()
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["accepted"] == 1


def test_unfinished_run_keeps_a_name_pending_not_rejected(tmp_path: Path) -> None:
    abc = "X:1\nK:C\nCDEF|\n"
    corpus_path = _corpus(tmp_path, [_track("yt500-0003-wip", 300.0, abc)])
    take = _take(RUN_A, "yt500-0003-wip", 333, wav=0.0, abc=abc,
                 final=False, marker=False, mp3=False, sidecar=False)
    work = tmp_path / "work"
    corpus = sync.load_corpus(corpus_path)
    sync.sync_once(FakeGcs([take]), args=_args(work), corpus=corpus, loop=1)

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["pending"] == 1
    assert manifest["counts"]["rejected"] == 0
    rejected = json.loads((work / "rejected.json").read_text(encoding="utf-8"))
    assert rejected["rejected"] == []


def test_limit_queues_a_spread_and_keeps_the_total(bucket: dict) -> None:
    work = bucket["tmp"] / "work"
    corpus = sync.load_corpus(bucket["corpus"])
    sync.sync_once(bucket["client"], args=_args(work, limit=1), corpus=corpus,
                   loop=1)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["accepted_in_manifest"] == 1
    assert manifest["counts"]["accepted"] == 1
    assert manifest["take_selection"]["limit"] == 1


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    abc = "X:1\nK:C\nCDEF|\n"
    corpus_path = _corpus(tmp_path, [_track("yt500-0001-good", 300.0, abc)])
    take = _take(RUN_A, "yt500-0001-good", 111, wav=298.0, abc=abc)
    work = tmp_path / "work"
    corpus = sync.load_corpus(corpus_path)
    sync.sync_once(FakeGcs([take]), args=_args(work, dry_run=True),
                   corpus=corpus, loop=1)
    assert not (work / "manifest.json").exists()
    assert not (work / "audio").exists()


def test_abc_mismatch_is_a_hard_stop(tmp_path: Path) -> None:
    corpus_path = _corpus(tmp_path, [_track("yt500-0004-bad", 300.0,
                                            "X:1\nK:C\nCDEF|\n")])
    take = _take(RUN_A, "yt500-0004-bad", 444, wav=298.0,
                 abc="X:1\nK:G\nGABc|\n")
    work = tmp_path / "work"
    corpus = sync.load_corpus(corpus_path)
    with pytest.raises(sync.gcsio.GcsError, match="disagree"):
        sync.sync_once(FakeGcs([take]), args=_args(work), corpus=corpus,
                       loop=1)
    # Nothing was queued or downloaded.
    assert not (work / "audio").exists()
