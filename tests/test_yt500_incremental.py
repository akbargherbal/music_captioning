"""CPU-only tests for the incremental caption/polish pass.

The YT500 corpus grows while it is being captioned, so one album is written
many times: a pass must caption only the new tracks and re-polish only the
tracks that are still pending. Re-running the rewriter for tracks that are
already finished costs 15-25 minutes per pass and changes nothing.
"""

from __future__ import annotations

from scripts.batch import polish_album, rewrite_done
from scripts.build_dataset import already_polished
from src.resume import (
    CAPTION_PIPELINE_VERSION,
    CAPTION_STAGE_CAPTIONED,
    CAPTION_STAGE_POLISHED,
)

CHUNKS = [
    "A warm oud line opens the piece over a slow percussion bed.",
    "The oud repeats while a frame drum keeps a steady pulse underneath.",
]


class CountingRewriter:
    name = "fake-rewriter"

    def __init__(self) -> None:
        self.calls = 0

    def rewrite(self, messages):  # noqa: ANN001 - protocol shape
        self.calls += 1
        return ("A warm oud line opens the piece over a slow percussion bed. "
                "The oud repeats while a frame drum keeps a steady pulse "
                "underneath, with no vocal and no dynamic change.")


def _track(name: str, status: str = "pending", caption: str = "") -> dict:
    return {
        "file": f"{name}.mp3",
        "chunk_captions": list(CHUNKS),
        "caption_raw": " ".join(CHUNKS),
        "caption": caption,
        "rewrite_status": status,
        "duration_s": 300.0,
    }


def _album(tracks: list[dict]) -> dict:
    return {
        "artist": "arabmaqamrock",
        "album": "YT500_INSTRUMENTAL",
        "caption_pipeline_version": CAPTION_PIPELINE_VERSION,
        "caption_stage": CAPTION_STAGE_POLISHED,
        "tracks": tracks,
        "failed_tracks": [],
    }


def test_rewrite_done_needs_a_status_and_a_caption() -> None:
    assert rewrite_done(_track("a", "ok", "text")) is True
    assert rewrite_done(_track("a", "fallback", "text")) is True
    assert rewrite_done(_track("a", "single_chunk", "text")) is True
    assert rewrite_done(_track("a", "ok", "   ")) is False
    assert rewrite_done(_track("a", "pending", "text")) is False
    assert rewrite_done(_track("a", "error", "text")) is False


def test_only_pending_keeps_finished_tracks_and_rewrites_one() -> None:
    album = _album([_track("done", "ok", "kept text"),
                    _track("new", "pending", "")])
    rewriter = CountingRewriter()

    out = polish_album(album, rewriter, only_pending=True)

    assert rewriter.calls == 1
    assert out["tracks"][0]["caption"] == "kept text"
    assert out["tracks"][0]["rewrite_status"] == "ok"
    assert out["tracks"][1]["caption"]
    assert out["rewrite"]["reused"] == 1
    assert out["caption_stage"] == CAPTION_STAGE_POLISHED


def test_a_second_pass_re_polishes_zero_tracks() -> None:
    album = _album([_track("done", "ok", "kept text"),
                    _track("new", "pending", "")])
    rewriter = CountingRewriter()
    first = polish_album(album, rewriter, only_pending=True)
    assert rewriter.calls == 1

    second = polish_album(first, rewriter, only_pending=True)

    assert rewriter.calls == 1
    assert second["rewrite"]["reused"] == 2
    assert second["tracks"][1]["rewrite_status"] in ("ok", "fallback")


def test_full_polish_still_rewrites_everything() -> None:
    album = _album([_track("done", "ok", "kept text"),
                    _track("new", "pending", "")])
    rewriter = CountingRewriter()
    polish_album(album, rewriter, only_pending=False)
    assert rewriter.calls == 2


def test_disabled_rewrite_does_not_clobber_a_polished_track() -> None:
    album = _album([_track("done", "ok", "kept text")])
    out = polish_album(album, None, only_pending=True)
    assert out["tracks"][0]["caption"] == "kept text"
    assert out["tracks"][0]["rewrite_status"] == "ok"
    assert out["rewrite"]["reused"] == 1
    # The stage drops so the next normal pass re-polishes the rest.
    assert out["caption_stage"] == CAPTION_STAGE_CAPTIONED


# --------------------------------------------------------------------------- #
# the "already polished" short-circuit
# --------------------------------------------------------------------------- #
def test_already_polished_accepts_a_complete_album() -> None:
    album = _album([_track("a", "ok", "text")])
    assert already_polished(album, 1) is True


def test_already_polished_retries_when_a_track_failed() -> None:
    album = _album([_track("a", "ok", "text")])
    album["failed_tracks"] = [{"file": "b.mp3", "error": "CUDA OOM"}]
    assert already_polished(album, 1) is False


def test_already_polished_retries_when_the_corpus_grew() -> None:
    album = _album([_track("a", "ok", "text")])
    assert already_polished(album, 2) is False


def test_already_polished_rejects_a_caption_only_album() -> None:
    album = _album([_track("a", "pending", "")])
    album["caption_stage"] = CAPTION_STAGE_CAPTIONED
    assert already_polished(album, 1) is False


def test_already_polished_rejects_an_old_pipeline() -> None:
    album = _album([_track("a", "ok", "text")])
    album["caption_pipeline_version"] = CAPTION_PIPELINE_VERSION + 1
    assert already_polished(album, 1) is False
