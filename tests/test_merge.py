"""Pure-Python tests for the deterministic caption merge. No model needed."""

from src.merge import (
    is_coda_claim,
    is_whole_song_opener,
    merge_section_captions,
    split_sentences,
)

OPENER = (
    "An energetic Arabic pop track driven by a powerful beat. "
    "The vocal is expressive and warm."
)


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four") == [
        "One.", "Two!", "Three?", "Four"
    ]
    assert split_sentences("  ") == []
    assert split_sentences("") == []
    assert split_sentences("No terminator here") == ["No terminator here"]


def test_opener_detection():
    assert is_whole_song_opener("An energetic Arabic pop track driven by strings.")
    assert is_whole_song_opener("The song builds slowly.")
    assert is_whole_song_opener("This piece is a ballad.")
    assert not is_whole_song_opener("A piano carries the melody.")
    assert not is_whole_song_opener("Strings swell under the vocal.")


def test_coda_detection():
    for sentence in (
        "The song concludes abruptly with a tape-stop sound effect.",
        "The track ends on a final chord.",
        "The outro fades out.",
        "It closes with a long silence.",
        "There is a brief silence.",
    ):
        assert is_coda_claim(sentence), sentence
    assert not is_coda_claim("A piano carries the melody.")
    assert not is_coda_claim("The weekend arrives in the second verse.")


def test_one_framing_only_and_false_coda_dropped():
    sections = [
        OPENER + " A piano carries the melody. "
        "The song concludes abruptly with a tape-stop sound effect.",
        OPENER + " A guitar solo appears late. "
        "The track fades out with a long silence.",
    ]
    merged = merge_section_captions(sections)

    assert merged.count("An energetic Arabic pop track") == 1
    assert "tape-stop" not in merged
    assert "fades out" in merged  # the real ending is kept
    assert merged.startswith(OPENER)
    assert merged.endswith("The track fades out with a long silence.")
    # Section order is preserved.
    assert merged.index("piano") < merged.index("guitar")


def test_middle_coda_dropped_final_coda_kept():
    sections = [
        "An Arabic pop track with a slow groove. The piano leads. "
        "The track ends suddenly.",
        "Strings answer the vocal. The song concludes with a fade.",
        "A guitar counters the strings. The outro fades out slowly.",
    ]
    merged = merge_section_captions(sections)

    assert "ends suddenly" not in merged
    assert "concludes with a fade" not in merged
    assert "outro fades out slowly" in merged
    assert merged.count("An Arabic pop track") == 1


def test_duplicate_sentence_removed_across_sections():
    sections = [
        "An Arabic pop track with a slow groove. The mix is warm and wide.",
        "The mix is warm and wide. A guitar solo appears late.",
    ]
    merged = merge_section_captions(sections)
    assert merged.count("warm and wide") == 1
    assert "guitar solo" in merged


def test_duplicate_kept_when_it_adds_a_new_instrument():
    # 0.75 Jaccard: the same sentence plus one new instrument. A plain
    # near-duplicate at this score is dropped (see the control below), so the
    # raised threshold is what keeps it.
    sections = [
        "An Arabic pop track with a slow groove. The mix is warm and wide.",
        "The mix is warm and wide with oud.",
    ]
    merged = merge_section_captions(sections)
    assert "oud" in merged

    control = [
        "An Arabic pop track with a slow groove. The mix is warm and wide.",
        "The mix is warm and wide today.",
    ]
    assert "today" not in merge_section_captions(control)


def test_single_section_unchanged():
    caption = (
        "An Arabic pop track with a slow groove. The piano leads. "
        "The piano leads. The track ends on a final chord."
    )
    assert merge_section_captions([caption]) == caption


def test_empty_input():
    assert merge_section_captions([]) == ""
    assert merge_section_captions([""]) == ""
    assert merge_section_captions(["   ", "\n"]) == ""


def test_section_that_becomes_empty():
    sections = [
        "An Arabic pop track with a slow groove. The piano leads.",
        "An Arabic pop track with a slow groove.",
    ]
    merged = merge_section_captions(sections)
    assert merged == "An Arabic pop track with a slow groove. The piano leads."


def test_dropped_audit_sink():
    dropped: list[str] = []
    sections = [
        "An Arabic pop track with a slow groove. The track ends suddenly.",
        "An Arabic pop track with a slow groove. A guitar enters. "
        "An Arabic pop track with a slow groove.",
    ]
    merged = merge_section_captions(sections, dropped=dropped)
    assert merged
    assert len(dropped) >= 2
    assert any("non-final coda" in entry for entry in dropped)
    assert any("repeat framing" in entry for entry in dropped)


class _StubCaptioner:
    """Stands in for a real backend so the wiring can be tested on CPU."""

    def __init__(self, captions):
        self._captions = captions

    def caption_audio(self, audio_path):
        return self._captions


SECTIONS = [
    "An Arabic pop track with a slow groove. The piano leads. "
    "The track ends suddenly.",
    "An Arabic pop track with a slow groove. A guitar enters late.",
]


def test_both_backends_store_the_merged_caption(tmp_path):
    from src.captioner import caption_and_tag as gguf_caption_and_tag
    from src.captioner_hf import caption_and_tag as hf_caption_and_tag

    for func in (gguf_caption_and_tag, hf_caption_and_tag):
        result = func(_StubCaptioner(SECTIONS), tmp_path / "song.mp3")
        assert result["caption"] == merge_section_captions(SECTIONS)
        assert "ends suddenly" not in result["caption"]
        assert result["chunk_captions"] == SECTIONS  # kept for debugging
        assert result["tags_experimental"] is True
