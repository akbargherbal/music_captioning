"""
test_caption.py — Pure-logic tests for caption.py
No GPU, no model, no Colab required. Run with: pytest test_caption.py -v

Covers:
  - build_prompt          (Task 2.3)
  - parse_job_file        (Task 5.1)
  - write_track_output    (Task 2.5)
  - write_summary         (Task 6.2)
  - discover_audio_files  (Task 4.1)
  - CLI entry-point validation via argparse
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make caption.py importable from the same directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from caption import (
    MAX_TOKENS_CEILING,
    MINIMAL_PROMPT,
    DEFAULT_PROMPT,
    SCRIPT_DEFAULTS,
    build_prompt,
    build_parser,
    discover_audio_files,
    parse_job_file,
    write_summary,
    write_track_output,
)


# ===========================================================================
# build_prompt — Task 2.3
# ===========================================================================

class TestBuildPrompt:
    def test_minimal_returns_locked_string(self):
        assert build_prompt("minimal") == MINIMAL_PROMPT

    def test_default_returns_locked_string(self):
        assert build_prompt("default") == DEFAULT_PROMPT

    def test_custom_returns_supplied_string(self):
        p = build_prompt("custom", custom_prompt="List instruments only.")
        assert p == "List instruments only."

    def test_custom_without_prompt_raises(self):
        with pytest.raises(ValueError, match="custom-prompt"):
            build_prompt("custom")

    def test_custom_with_empty_string_raises(self):
        with pytest.raises(ValueError):
            build_prompt("custom", custom_prompt="")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt mode"):
            build_prompt("arabic")


# ===========================================================================
# parse_job_file — Task 5.1
# ===========================================================================

def _write_job(tmp_path, payload):
    """Helper: write a job JSON to a temp file and return its path."""
    p = tmp_path / "job.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


class TestParseJobFile:

    # --- Error conditions ---

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_job_file(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            parse_job_file(str(p))

    def test_empty_tracks_raises(self, tmp_path):
        job = _write_job(tmp_path, {"globals": {}, "tracks": []})
        with pytest.raises(ValueError, match="no tracks"):
            parse_job_file(job)

    def test_missing_path_key_raises_with_index(self, tmp_path):
        job = _write_job(tmp_path, {
            "globals": {},
            "tracks": [{"prompt_mode": "minimal"}],  # no 'path'
        })
        with pytest.raises(ValueError, match="index 0"):
            parse_job_file(job)

    def test_second_track_missing_path_raises_correct_index(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [
                {"path": "track_01.mp3"},
                {"prompt_mode": "default"},   # index 1, no 'path'
            ]
        })
        with pytest.raises(ValueError, match="index 1"):
            parse_job_file(job)

    # --- Override precedence ---

    def test_script_defaults_applied_when_globals_absent(self, tmp_path):
        job = _write_job(tmp_path, {"tracks": [{"path": "a.mp3"}]})
        tracks = parse_job_file(job)
        assert tracks[0]["prompt_mode"]    == SCRIPT_DEFAULTS["prompt_mode"]
        assert tracks[0]["max_new_tokens"] == SCRIPT_DEFAULTS["max_new_tokens"]
        assert tracks[0]["output_dir"]     == SCRIPT_DEFAULTS["output_dir"]
        assert tracks[0]["precision"]      == SCRIPT_DEFAULTS["precision"]

    def test_globals_override_script_defaults(self, tmp_path):
        job = _write_job(tmp_path, {
            "globals": {"prompt_mode": "default", "max_new_tokens": 128},
            "tracks":  [{"path": "a.mp3"}],
        })
        tracks = parse_job_file(job)
        assert tracks[0]["prompt_mode"]    == "default"
        assert tracks[0]["max_new_tokens"] == 128

    def test_per_track_overrides_globals(self, tmp_path):
        job = _write_job(tmp_path, {
            "globals": {"prompt_mode": "default", "max_new_tokens": 128},
            "tracks": [
                {"path": "a.mp3"},                                    # inherits globals
                {"path": "b.mp3", "prompt_mode": "minimal",          # overrides prompt
                 "max_new_tokens": 64},                               # overrides tokens
            ],
        })
        tracks = parse_job_file(job)
        # First track inherits globals
        assert tracks[0]["prompt_mode"]    == "default"
        assert tracks[0]["max_new_tokens"] == 128
        # Second track overrides both
        assert tracks[1]["prompt_mode"]    == "minimal"
        assert tracks[1]["max_new_tokens"] == 64

    def test_per_track_custom_prompt_preserved(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [{
                "path": "a.mp3",
                "prompt_mode":   "custom",
                "custom_prompt": "List drums only.",
            }]
        })
        tracks = parse_job_file(job)
        assert tracks[0]["custom_prompt"] == "List drums only."

    # --- Tags ---

    def test_tags_present_in_resolved_track(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [{"path": "a.mp3", "tags": {"qasida": "Mu'allaqa", "part": 1}}]
        })
        tracks = parse_job_file(job)
        assert tracks[0]["tags"] == {"qasida": "Mu'allaqa", "part": 1}

    def test_missing_tags_defaults_to_empty_dict(self, tmp_path):
        job = _write_job(tmp_path, {"tracks": [{"path": "a.mp3"}]})
        tracks = parse_job_file(job)
        assert tracks[0]["tags"] == {}

    def test_invalid_tags_type_defaults_to_empty_dict(self, tmp_path):
        # tags: "string" is not a dict — should not crash, should default to {}
        job = _write_job(tmp_path, {
            "tracks": [{"path": "a.mp3", "tags": "not-a-dict"}]
        })
        tracks = parse_job_file(job)
        assert tracks[0]["tags"] == {}

    # --- max_new_tokens clamping ---

    def test_max_new_tokens_clamped_at_ceiling(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [{"path": "a.mp3", "max_new_tokens": 9999}]
        })
        tracks = parse_job_file(job)
        assert tracks[0]["max_new_tokens"] == MAX_TOKENS_CEILING

    def test_max_new_tokens_at_ceiling_is_not_clamped(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [{"path": "a.mp3", "max_new_tokens": MAX_TOKENS_CEILING}]
        })
        tracks = parse_job_file(job)
        assert tracks[0]["max_new_tokens"] == MAX_TOKENS_CEILING

    # --- Minimum valid job (from phased plan spec) ---

    def test_minimum_valid_job_runs_without_error(self, tmp_path):
        """The exact minimal job from the Phase 5 spec must parse cleanly."""
        job = _write_job(tmp_path, {
            "globals": {"max_new_tokens": 256, "output_dir": "./captions/"},
            "tracks":  [{"path": "track_01.mp3"}],
        })
        tracks = parse_job_file(job)
        assert len(tracks) == 1
        assert tracks[0]["path"] == "track_01.mp3"

    # --- Return shape ---

    def test_all_required_keys_present(self, tmp_path):
        job = _write_job(tmp_path, {"tracks": [{"path": "a.mp3"}]})
        tracks = parse_job_file(job)
        required = {"path", "model_id", "precision", "max_new_tokens",
                    "prompt_mode", "custom_prompt", "output_dir", "tags"}
        assert required.issubset(tracks[0].keys())

    def test_multiple_tracks_returns_all(self, tmp_path):
        job = _write_job(tmp_path, {
            "tracks": [
                {"path": "a.mp3"},
                {"path": "b.wav"},
                {"path": "c.flac"},
            ]
        })
        tracks = parse_job_file(job)
        assert len(tracks) == 3
        assert [t["path"] for t in tracks] == ["a.mp3", "b.wav", "c.flac"]


# ===========================================================================
# write_track_output — Task 2.5
# ===========================================================================

def _ok_result():
    return {
        "raw_output":          "Vocals, oud, darbuka. Tempo ~72 BPM.",
        "processing_time_sec": 10.1,
        "status":              "ok",
        "error":               None,
    }

def _error_result():
    return {
        "raw_output":          None,
        "processing_time_sec": None,
        "status":              "error",
        "error":               "CUDA OOM",
    }


class TestWriteTrackOutput:

    def test_creates_output_dir_if_absent(self, tmp_path):
        out_dir = str(tmp_path / "deep" / "nested")
        write_track_output(
            result=_ok_result(), audio_path="track.mp3",
            prompt_used="...", prompt_mode="minimal",
            tags={}, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=out_dir,
        )
        assert os.path.isdir(out_dir)

    def test_output_filename_is_stem_plus_caption_json(self, tmp_path):
        path = write_track_output(
            result=_ok_result(), audio_path="/audio/my_track.mp3",
            prompt_used="...", prompt_mode="minimal",
            tags={}, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        assert Path(path).name == "my_track.caption.json"

    def test_all_nine_schema_fields_present(self, tmp_path):
        path = write_track_output(
            result=_ok_result(), audio_path="track.mp3",
            prompt_used="my prompt", prompt_mode="minimal",
            tags={"part": 1}, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for field in ("file", "model_id", "prompt_mode", "prompt_used",
                      "raw_output", "processing_time_sec", "tags", "status", "error"):
            assert field in data, f"Missing field: {field}"

    def test_tags_written_to_json(self, tmp_path):
        tags = {"qasida": "Mu'allaqa", "part": 2}
        path = write_track_output(
            result=_ok_result(), audio_path="t.mp3",
            prompt_used="...", prompt_mode="minimal",
            tags=tags, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        data = json.load(open(path, encoding="utf-8"))
        assert data["tags"] == tags

    def test_none_tags_written_as_empty_dict(self, tmp_path):
        path = write_track_output(
            result=_ok_result(), audio_path="t.mp3",
            prompt_used="...", prompt_mode="minimal",
            tags=None, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        data = json.load(open(path, encoding="utf-8"))
        assert data["tags"] == {}

    def test_error_result_written_correctly(self, tmp_path):
        path = write_track_output(
            result=_error_result(), audio_path="t.mp3",
            prompt_used="...", prompt_mode="minimal",
            tags={}, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        data = json.load(open(path, encoding="utf-8"))
        assert data["status"] == "error"
        assert data["raw_output"] is None
        assert data["error"] == "CUDA OOM"

    def test_existing_file_is_overwritten_silently(self, tmp_path):
        kwargs = dict(
            audio_path="t.mp3", prompt_used="...", prompt_mode="minimal",
            tags={}, model_id="nvidia/music-flamingo-2601-hf",
            output_dir=str(tmp_path),
        )
        write_track_output(result=_ok_result(), **kwargs)
        write_track_output(result=_error_result(), **kwargs)  # should not raise
        data = json.load(open(tmp_path / "t.caption.json", encoding="utf-8"))
        assert data["status"] == "error"


# ===========================================================================
# write_summary — Task 6.2
# ===========================================================================

def _make_records(n_ok, n_fail):
    records = []
    for i in range(n_ok):
        records.append({
            "file": f"ok_{i}.mp3", "status": "ok",
            "processing_time_sec": 10.0, "error": None,
            "output_file": f"./captions/ok_{i}.caption.json", "tags": {},
        })
    for i in range(n_fail):
        records.append({
            "file": f"fail_{i}.mp3", "status": "error",
            "processing_time_sec": None, "error": "CUDA OOM",
            "output_file": None, "tags": {},
        })
    return records


class TestWriteSummary:

    def test_creates_results_summary_json(self, tmp_path):
        write_summary(_make_records(2, 0), str(tmp_path))
        assert (tmp_path / "results_summary.json").exists()

    def test_counts_are_correct(self, tmp_path):
        write_summary(_make_records(3, 1), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert data["total_tracks"]  == 4
        assert data["success_count"] == 3
        assert data["fail_count"]    == 1

    def test_all_top_level_fields_present(self, tmp_path):
        write_summary(_make_records(1, 0), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        for field in ("timestamp", "total_tracks", "success_count", "fail_count", "results"):
            assert field in data, f"Missing field: {field}"

    def test_results_list_length_matches_input(self, tmp_path):
        records = _make_records(2, 2)
        write_summary(records, str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert len(data["results"]) == 4

    def test_all_failures_no_successes(self, tmp_path):
        write_summary(_make_records(0, 3), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert data["success_count"] == 0
        assert data["fail_count"]    == 3

    def test_all_successes_no_failures(self, tmp_path):
        write_summary(_make_records(5, 0), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert data["fail_count"] == 0

    def test_creates_output_dir_if_absent(self, tmp_path):
        out_dir = str(tmp_path / "new_dir")
        write_summary(_make_records(1, 0), out_dir)
        assert os.path.isdir(out_dir)

    def test_timestamp_is_iso8601(self, tmp_path):
        write_summary(_make_records(1, 0), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        from datetime import datetime, timezone
        # Should parse without raising
        datetime.fromisoformat(data["timestamp"])

    def test_single_record_summary(self, tmp_path):
        """write_summary is called for --file mode too — must handle a list of one."""
        write_summary(_make_records(1, 0), str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert data["total_tracks"] == 1

    def test_unicode_filenames_survive_round_trip(self, tmp_path):
        records = [{
            "file": "قصيدة_01.mp3", "status": "ok",
            "processing_time_sec": 12.0, "error": None,
            "output_file": "./captions/قصيدة_01.caption.json", "tags": {},
        }]
        write_summary(records, str(tmp_path))
        data = json.load(open(tmp_path / "results_summary.json", encoding="utf-8"))
        assert data["results"][0]["file"] == "قصيدة_01.mp3"


# ===========================================================================
# discover_audio_files — Task 4.1
# ===========================================================================

def _make_files(tmp_path, names):
    """Create empty files under tmp_path and return their paths."""
    for name in names:
        (tmp_path / name).write_bytes(b"")


class TestDiscoverAudioFiles:

    def test_finds_mp3_wav_flac(self, tmp_path):
        _make_files(tmp_path, ["a.mp3", "b.wav", "c.flac"])
        found = discover_audio_files(str(tmp_path))
        assert len(found) == 3

    def test_skips_non_audio_extensions(self, tmp_path):
        _make_files(tmp_path, ["track.mp3", "notes.txt", "README.md", ".DS_Store"])
        found = discover_audio_files(str(tmp_path))
        assert len(found) == 1
        assert found[0].endswith("track.mp3")

    def test_case_insensitive_extension_matching(self, tmp_path):
        _make_files(tmp_path, ["A.MP3", "B.Wav", "C.FLAC"])
        found = discover_audio_files(str(tmp_path))
        assert len(found) == 3

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_files(tmp_path, ["root.mp3"])
        _make_files(sub, ["nested.wav"])
        found = discover_audio_files(str(tmp_path))
        assert len(found) == 2

    def test_returns_sorted_absolute_paths(self, tmp_path):
        _make_files(tmp_path, ["c.mp3", "a.mp3", "b.wav"])
        found = discover_audio_files(str(tmp_path))
        assert found == sorted(found)
        assert all(os.path.isabs(p) for p in found)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        found = discover_audio_files(str(tmp_path))
        assert found == []

    def test_nonexistent_directory_calls_sys_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            discover_audio_files(str(tmp_path / "does_not_exist"))


# ===========================================================================
# CLI argument parser — entry-point validation
# ===========================================================================

class TestCLIParser:
    """Test the argparse layer and shared pre-flight logic in main()."""

    def _parse(self, argv):
        return build_parser().parse_args(argv)

    def test_file_mode_parses(self):
        args = self._parse(["--file", "track.mp3"])
        assert args.file == "track.mp3"
        assert args.dir  is None
        assert args.job  is None

    def test_dir_mode_parses(self):
        args = self._parse(["--dir", "./tracks/"])
        assert args.dir == "./tracks/"

    def test_job_mode_parses(self):
        args = self._parse(["--job", "batch.json"])
        assert args.job == "batch.json"

    def test_defaults_are_correct(self):
        args = self._parse(["--file", "t.mp3"])
        assert args.max_tokens  == 256
        assert args.prompt_mode == "minimal"
        assert args.precision   == "bf16"
        assert args.output_dir  == "./captions/"
        assert args.custom_prompt is None

    def test_invalid_prompt_mode_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--file", "t.mp3", "--prompt-mode", "arabic"])

    def test_invalid_precision_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--file", "t.mp3", "--precision", "fp32"])

    def test_custom_flags_accepted(self):
        args = self._parse([
            "--file", "t.mp3",
            "--prompt-mode", "custom",
            "--custom-prompt", "Describe tempo only.",
            "--max-tokens", "128",
            "--precision", "4bit",
            "--output-dir", "/tmp/out/",
        ])
        assert args.prompt_mode   == "custom"
        assert args.custom_prompt == "Describe tempo only."
        assert args.max_tokens    == 128
        assert args.precision     == "4bit"
        assert args.output_dir    == "/tmp/out/"
