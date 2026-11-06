"""CPU-only tests for ABC parsing, split integrity and dedupe (§9, §10.2)."""

import json
from pathlib import Path

from abcscore import parse_abc, qc_track, strip_chords

from build_abc_dataset import (
    assign_splits,
    build_rows,
    dedupe_groups,
    dedupe_keys,
)

SAMPLE_ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=101
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Em
% intro
V: Vocal
"Em"z8"Bm"c4d4e8|
V: Ins
Z|e2f2g2b2g2e2f6f2e4e8|
"""


def _row(track_id, artist, album, title, *, digest=None, dup=None,
         parse="ok"):
    return {
        "track_id": track_id,
        "artist": artist,
        "album": album,
        "title": title,
        "audio_sha256": digest,
        "duplicate_of": dup,
        "qc": {"parse": parse, "flags": []},
    }


def test_strip_chords_only_touches_music_lines():
    stripped = strip_chords(SAMPLE_ABC)
    assert 'name="Vocal Melody"' in stripped          # header quote survives
    assert '"Em"' not in stripped and '"Bm"' not in stripped
    assert "c4d4e8|" in stripped


def test_parse_abc_voices_notes_and_chords():
    parsed = parse_abc(SAMPLE_ABC)
    assert parsed.notes_vocal == 3
    assert parsed.notes_ins == 10
    assert parsed.has_chords
    assert parsed.key == "Em"
    assert parsed.tempo_bpm == 101.0
    assert parsed.measures == 2


def test_music_named_instrumental_is_not_flagged_vocal_sparse(tmp_path: Path):
    # ELISSA's "14.Music.mp3" has zero Vocal notes; that is an instrumental,
    # not a failed vocal transcription. A sung filename must still flag.
    ins_only = SAMPLE_ABC.split("V: Vocal\n")[0] + "V: Ins\nZ|e2f2g2b2|\n"
    (tmp_path / "score.abc").write_text(ins_only, encoding="utf-8")
    music = qc_track(tmp_path, 140.0, file="14.Music.mp3")
    sung = qc_track(tmp_path, 140.0, file="14.Song.mp3")
    assert music["instrumental"] is True
    assert "vocal_sparse" not in music["flags"]
    assert sung["instrumental"] is False
    assert "vocal_sparse" in sung["flags"]


def test_dedupe_by_hash_and_pointer_collapses_one_group():
    rows = [
        _row("A/alb/01-a", "A", "alb", "Song", digest="h1"),
        _row("B/alb/01-a", "B", "alb", "Song (remaster)", digest="h1"),
        _row("C/alb/01-a", "C", "alb", "Song", digest="h2", dup="A/alb/01-a"),
        _row("D/alb/01-d", "D", "alb", "Other", digest="h3"),
    ]
    groups = dedupe_groups(rows)
    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {
        "A/alb/01-a", "B/alb/01-a", "C/alb/01-a"}
    keys = dedupe_keys(rows)
    assert keys["D/alb/01-d"] != keys["A/alb/01-a"]


def test_split_keeps_albums_and_duplicate_groups_together():
    rows = [
        # train album A + compilation V share a hash
        _row("A/alb/01-a", "A", "Train Album", "Song", digest="h1"),
        _row("V/best/01-a", "V", "Val Album", "Song", digest="h1"),
        # train album B is linked to the val album only by duplicate_of
        _row("B/alb/01-a", "B", "Train Album 2", "Song", digest="h2",
             dup="V/best/01-a"),
        _row("B/alb/02-b", "B", "Train Album 2", "Other", digest="h3"),
        _row("Z/alb/01-z", "Z", "Untouched", "Solo", digest="h4"),
    ]
    splits = assign_splits(rows, val_albums={"V/Val Album"})

    # Whole duplicate component -> val; unconnected album stays train.
    assert {splits[t] for t in ("A/alb/01-a", "V/best/01-a", "B/alb/01-a")} == {"val"}
    assert splits["B/alb/02-b"] == "val"       # album-level, never track-level
    assert splits["Z/alb/01-z"] == "train"

    # No album straddles a split.
    by_album = {}
    for row in rows:
        by_album.setdefault(f"{row['artist']}/{row['album']}", set()).add(
            splits[row["track_id"]])
    assert all(len(v) == 1 for v in by_album.values())

    # No duplicate group straddles a split.
    for members in dedupe_groups(rows).values():
        assert len({splits[m] for m in members}) == 1


def test_quarantine_overrides_split_but_keeps_row():
    rows = [_row("A/alb/01-bad", "A", "alb", "Broken", digest="h", parse="fail")]
    assert assign_splits(rows)["A/alb/01-bad"] == "quarantine"


def _write_album(root: Path, artist: str, album: str, tracks: list[dict]) -> None:
    path = root / artist / "abc" / f"{album}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artist": artist, "album": album, "tracks": tracks,
    }), encoding="utf-8")


def _write_captions(root: Path, artist: str, album: str,
                    tracks: list[dict], skipped: list[str] | None = None) -> None:
    path = root / artist / f"{album}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artist": artist, "album": album, "tracks": tracks,
        "skipped_tracks": [{"file": f} for f in (skipped or [])],
    }), encoding="utf-8")


def test_build_rows_joins_caption_and_writes_null_for_skipped(tmp_path: Path):
    outputs = tmp_path / "outputs"
    _write_album(outputs, "A", "Alb", [
        {"track_id": "A/Alb/01-s", "file": "01. S.mp3", "title": "S",
         "abc": SAMPLE_ABC, "has_chords": True, "lyrics_eligible": True,
         "qc": {"parse": "ok", "flags": []}},
        {"track_id": "A/Alb/02-long", "file": "02. Long.mp3", "title": "Long",
         "abc": SAMPLE_ABC, "has_chords": True, "lyrics_eligible": True,
         "qc": {"parse": "ok", "flags": []}},
    ])
    _write_captions(outputs, "A", "Alb", [
        {"file": "01. S.mp3", "caption": "a caption",
         "instruments": ["oud"], "vocals": ["male vocal"],
         "production": ["reverb"]},
    ], skipped=["02. Long.mp3"])
    manifest = {
        "A/Alb/01-s": {"track_id": "A/Alb/01-s",
                       "source_uri": "gs://bucket/a.mp3"},
        "A/Alb/02-long": {"track_id": "A/Alb/02-long",
                          "source_uri": "gs://bucket/b.mp3"},
    }

    rows, stats = build_rows(outputs, manifest, val_albums=set())
    assert stats["rows_written"] == 2
    by_id = {r["track_id"]: r for r in rows}
    assert by_id["A/Alb/01-s"]["caption"] == "a caption"
    assert by_id["A/Alb/01-s"]["audio_uri"] == "gs://bucket/a.mp3"
    assert by_id["A/Alb/02-long"]["caption"] is None
    assert by_id["A/Alb/02-long"]["caption_status"] == "skipped_long"
    assert "caption_skipped_long" in by_id["A/Alb/02-long"]["qc_flags"]
    assert all(r["lyrics"] is None for r in rows)
