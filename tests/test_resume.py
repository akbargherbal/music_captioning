"""Pure-Python tests for the resume/manifest helpers. No GPU, network, or model."""

import json
from pathlib import Path

from src.resume import (
    CAPTION_PIPELINE_VERSION,
    SkippedTrack,
    album_key,
    album_output_path,
    album_status,
    atomic_write_json,
    atomic_write_text,
    group_local_tracks,
    group_tracks,
    is_current_caption_output,
    is_done,
    load_manifest,
    parse_gs_track,
    pending_albums,
    run_tracks,
    skip_reason,
)

CURRENT_OUTPUT = {"caption_pipeline_version": CAPTION_PIPELINE_VERSION}

ROOT = "gs://bucket/DISCOGRAPHY"


def test_album_key_and_output_path(tmp_path: Path):
    assert album_key("ELISSA", "Album") == "ELISSA/Album"
    assert album_output_path(tmp_path, "ELISSA", "Album") == tmp_path / "ELISSA" / "Album.json"


def test_parse_gs_track_valid():
    uri = f"{ROOT}/ELISSA/Elissa - 2008 Ayami Beek/01. Ayami Bik.mp3"
    info = parse_gs_track(uri, ROOT)
    assert info is not None
    assert info["artist"] == "ELISSA"
    assert info["album"] == "Elissa - 2008 Ayami Beek"
    assert info["filename"] == "01. Ayami Bik.mp3"
    assert info["source_dir"] == f"{ROOT}/ELISSA/Elissa - 2008 Ayami Beek"


def test_parse_gs_track_rejects():
    assert parse_gs_track("gs://other/x/y/z.mp3", ROOT) is None
    assert parse_gs_track(f"{ROOT}/ELISSA/Album/cover.jpg", ROOT) is None
    assert parse_gs_track(f"{ROOT}/ELISSA/track.mp3", ROOT) is None


def test_group_tracks_groups_and_sorts():
    uris = [
        f"{ROOT}/ELISSA/Album B/02. Two.mp3",
        f"{ROOT}/ELISSA/Album B/01. One.mp3",
        f"{ROOT}/ELISSA/Album B/Covers/x.jpg",
        f"{ROOT}/FADL/Album A/01. Song.mp3",
    ]
    albums = group_tracks(uris, ROOT)
    assert set(albums) == {"ELISSA/Album B", "FADL/Album A"}
    assert [t["filename"] for t in albums["ELISSA/Album B"]["tracks"]] == [
        "01. One.mp3", "02. Two.mp3",
    ]


def test_atomic_write_text_creates_and_replaces(tmp_path: Path):
    target = tmp_path / "a" / "b.json"
    atomic_write_text(target, "first")
    assert target.read_text() == "first"
    atomic_write_text(target, "second")
    assert target.read_text() == "second"
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_json_roundtrip(tmp_path: Path):
    target = tmp_path / "m.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_load_manifest_missing_and_corrupt(tmp_path: Path):
    assert load_manifest(tmp_path / "nope.json")["albums"] == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_manifest(bad)["albums"] == {}


def test_is_done_requires_file_and_current_pipeline(tmp_path: Path):
    outputs = tmp_path / "outputs"
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    assert is_done("ELISSA/A", manifest, outputs) is False
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text("{}")
    # Pre-versioning output: the caption exists but must be redone.
    assert is_done("ELISSA/A", manifest, outputs) is False
    (outputs / "ELISSA" / "A.json").write_text(
        json.dumps(CURRENT_OUTPUT), encoding="utf-8"
    )
    assert is_done("ELISSA/A", manifest, outputs) is True
    assert is_done("ELISSA/missing", manifest, outputs) is False


def test_is_done_rejects_old_and_malformed_versions(tmp_path: Path):
    outputs = tmp_path / "outputs"
    (outputs / "ELISSA").mkdir(parents=True)
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    for bad in (
        {"caption_pipeline_version": CAPTION_PIPELINE_VERSION - 1},
        {"caption_pipeline_version": CAPTION_PIPELINE_VERSION + 1},
        {"caption_pipeline_version": "2"},
        {"caption_pipeline_version": None},
    ):
        (outputs / "ELISSA" / "A.json").write_text(
            json.dumps(bad), encoding="utf-8"
        )
        assert is_done("ELISSA/A", manifest, outputs) is False, bad
    (outputs / "ELISSA" / "A.json").write_text("{not json", encoding="utf-8")
    assert is_done("ELISSA/A", manifest, outputs) is False


def test_is_current_caption_output():
    assert is_current_caption_output(CURRENT_OUTPUT) is True
    assert is_current_caption_output({}) is False
    assert is_current_caption_output(None) is False
    assert is_current_caption_output([]) is False
    assert is_current_caption_output(
        {"caption_pipeline_version": CAPTION_PIPELINE_VERSION - 1}
    ) is False


def test_pending_albums_and_force(tmp_path: Path):
    outputs = tmp_path / "outputs"
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text(
        json.dumps(CURRENT_OUTPUT), encoding="utf-8"
    )
    albums = {
        "ELISSA/A": {"artist": "ELISSA", "album": "A", "tracks": []},
        "ELISSA/B": {"artist": "ELISSA", "album": "B", "tracks": []},
    }
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    assert pending_albums(albums, manifest, outputs) == ["ELISSA/B"]
    assert pending_albums(albums, manifest, outputs, force=True) == ["ELISSA/A", "ELISSA/B"]


def test_pending_albums_requeues_old_caption_version(tmp_path: Path):
    outputs = tmp_path / "outputs"
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text("{}", encoding="utf-8")
    albums = {"ELISSA/A": {"artist": "ELISSA", "album": "A", "tracks": []}}
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    assert pending_albums(albums, manifest, outputs) == ["ELISSA/A"]


def test_album_status_default_and_partial():
    manifest = {
        "albums": {
            "ELISSA/A": {"path": "ELISSA/A.json"},  # no status -> "ok"
            "ELISSA/B": {"path": "ELISSA/B.json", "status": "partial", "failed_count": 2},
        }
    }
    assert album_status("ELISSA/A", manifest) == "ok"
    assert album_status("ELISSA/B", manifest) == "partial"
    assert album_status("ELISSA/missing", manifest) is None


def test_pending_albums_retry_failed(tmp_path: Path):
    outputs = tmp_path / "outputs"
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text(json.dumps(CURRENT_OUTPUT))
    (outputs / "ELISSA" / "B.json").write_text(json.dumps(CURRENT_OUTPUT))
    albums = {
        "ELISSA/A": {"artist": "ELISSA", "album": "A", "tracks": []},
        "ELISSA/B": {"artist": "ELISSA", "album": "B", "tracks": []},
    }
    manifest = {"albums": {
        "ELISSA/A": {"path": "ELISSA/A.json", "status": "ok"},
        "ELISSA/B": {"path": "ELISSA/B.json", "status": "partial", "failed_count": 1},
    }}
    assert pending_albums(albums, manifest, outputs) == []
    assert pending_albums(albums, manifest, outputs, retry_failed=True) == ["ELISSA/B"]


def test_run_tracks_isolates_a_failure():
    tracks = [
        {"filename": "a.mp3", "uri": "a"},
        {"filename": "bad.mp3", "uri": "bad"},
        {"filename": "c.mp3", "uri": "c"},
    ]

    def caption_one(filename, uri):
        if filename == "bad.mp3":
            raise ValueError("boom")
        return {"file": filename, "caption_seconds": 1.0}

    results, failed, skipped = run_tracks(tracks, caption_one)
    assert [r["file"] for r in results] == ["a.mp3", "c.mp3"]
    assert failed == [{"file": "bad.mp3", "error": "ValueError: boom"}]
    assert skipped == []


def test_run_tracks_records_skipped_separately():
    tracks = [
        {"filename": "song.mp3", "uri": "song"},
        {"filename": "concert.mp3", "uri": "concert"},
    ]

    def caption_one(filename, uri):
        if filename == "concert.mp3":
            raise SkippedTrack("900s > 480s cap")
        return {"file": filename, "caption_seconds": 1.0}

    results, failed, skipped = run_tracks(tracks, caption_one)
    assert [r["file"] for r in results] == ["song.mp3"]
    assert failed == []  # a skip is not a failure
    assert skipped == [{"file": "concert.mp3", "reason": "900s > 480s cap"}]


def test_run_tracks_retry_only_failed_keeps_successes():
    existing = {
        "status": "partial",
        "tracks": [
            {"file": "a.mp3", "caption_seconds": 1.0},
            {"file": "c.mp3", "caption_seconds": 2.0},
        ],
        "failed_tracks": [{"file": "bad.mp3", "error": "x"}],
    }
    tracks = [{"filename": n, "uri": n} for n in ["a.mp3", "bad.mp3", "c.mp3"]]
    calls: list[str] = []

    def caption_one(filename, uri):
        calls.append(filename)
        return {"file": filename, "caption_seconds": 9.0}

    results, failed, _ = run_tracks(
        tracks, caption_one, existing=existing, retry_only_failed=True
    )
    assert calls == ["bad.mp3"]  # only the failed track is re-run
    assert failed == []
    assert [r["file"] for r in results] == ["a.mp3", "bad.mp3", "c.mp3"]
    assert results[0]["caption_seconds"] == 1.0  # prior success preserved
    assert results[1]["caption_seconds"] == 9.0  # retried track


def test_run_tracks_echo_reports_progress():
    msgs: list[str] = []
    run_tracks(
        [{"filename": "a.mp3", "uri": "a"}],
        lambda f, u: {"file": f, "caption_seconds": 3.0},
        echo=msgs.append,
    )
    assert msgs == ["a.mp3: 3.0s"]


def test_skip_reason_low_bitrate():
    assert skip_reason(200, 64, min_bitrate_kbps=128) == "64 kbps < 128 kbps"
    assert skip_reason(200, 96, min_bitrate_kbps=128) == "96 kbps < 128 kbps"
    assert skip_reason(200, 128, min_bitrate_kbps=128) is None
    assert skip_reason(200, 320, min_bitrate_kbps=128) is None


def test_skip_reason_long_track():
    assert skip_reason(600, 320, max_track_seconds=480) == "600s > 480s cap"
    assert skip_reason(480, 320, max_track_seconds=480) is None
    assert skip_reason(300, 320, max_track_seconds=480) is None


def test_skip_reason_unknown_values_do_not_skip():
    assert skip_reason(None, None, max_track_seconds=480, min_bitrate_kbps=128) is None
    # a known bad value still skips even if the other is unknown
    assert skip_reason(600, None, max_track_seconds=480) == "600s > 480s cap"
    assert skip_reason(None, 64, min_bitrate_kbps=128) == "64 kbps < 128 kbps"


def test_skip_reason_disabled_caps():
    assert skip_reason(999, 32) is None  # both caps default to 0 = off


def test_group_local_tracks_artist_dir(tmp_path: Path):
    # root itself is the artist dir: root/<album>/<file>
    root = tmp_path / "ELISSA"
    (root / "Album B").mkdir(parents=True)
    (root / "Album B" / "02. Two.mp3").write_bytes(b"x")
    (root / "Album B" / "01. One.mp3").write_bytes(b"x")
    (root / "Album B" / "cover.jpg").write_bytes(b"x")
    albums = group_local_tracks(root)
    assert set(albums) == {"ELISSA/Album B"}
    assert albums["ELISSA/Album B"]["artist"] == "ELISSA"
    assert [t["filename"] for t in albums["ELISSA/Album B"]["tracks"]] == [
        "01. One.mp3", "02. Two.mp3",
    ]


def test_group_local_tracks_artist_parent(tmp_path: Path):
    # root/<artist>/<album>/<file>
    root = tmp_path
    album = root / "FADL" / "Album A"
    album.mkdir(parents=True)
    (album / "01. Song.mp3").write_bytes(b"x")
    albums = group_local_tracks(root)
    assert set(albums) == {"FADL/Album A"}


def test_group_local_tracks_flat(tmp_path: Path):
    album = tmp_path / "Single"
    album.mkdir()
    (album / "song.mp3").write_bytes(b"x")
    albums = group_local_tracks(album)
    assert set(albums) == {"Single/Single"}
