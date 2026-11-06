"""Pure-Python tests for lexicon tag extraction. No GPU or model required."""

from pathlib import Path

import pytest

from src.tags import (
    CATEGORIES,
    build_output,
    extract_tags,
    load_lexicon,
    merge_tags,
    to_suno,
)

FIXTURES = Path(__file__).parent / "fixtures" / "captions"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_lexicon_has_all_categories():
    lex = load_lexicon()
    assert set(lex) == set(CATEGORIES)
    for cat in CATEGORIES:
        assert lex[cat], f"empty category: {cat}"


def test_empty_caption_yields_no_tags():
    tags = extract_tags("")
    assert tags == {cat: [] for cat in CATEGORIES}
    assert to_suno(tags) == ""
    assert extract_tags("   \n  ") == {cat: [] for cat in CATEGORIES}


def test_none_caption_is_safe():
    assert extract_tags(None) == {cat: [] for cat in CATEGORIES}  # type: ignore[arg-type]


def test_repeated_terms_are_deduped():
    tags = extract_tags("reverb, more reverb, and even more REVERB everywhere")
    assert tags["production"] == ["reverb"]


def test_case_insensitive():
    tags = extract_tags("A huge REVERB and a Distorted SYNTH")
    assert "reverb" in tags["production"]
    assert "distorted" in tags["production"]
    assert "synth" in tags["instruments"]


def test_multiword_term_wins_over_substring():
    tags = extract_tags("A gentle acoustic guitar arpeggio.")
    assert tags["instruments"] == ["acoustic guitar"]
    assert "guitar" not in tags["instruments"]


def test_bass_guitar_not_reduced_to_bass():
    tags = extract_tags("A funky bass guitar groove.")
    assert "bass guitar" in tags["instruments"]
    assert "bass" not in tags["instruments"]


def test_bare_bass_matches():
    tags = extract_tags("A deep, rumbling bass anchors the low end.")
    assert "bass" in tags["instruments"]


def test_substring_terms_do_not_match_inside_longer_words():
    # "guitarist" / "basses" must not trigger "guitar" / "bass".
    tags = extract_tags("The guitarist and the basses played together.")
    assert "guitar" not in tags["instruments"]
    assert "bass" not in tags["instruments"]


def test_specific_vocal_term_wins():
    tags = extract_tags("Soft female vocals over a warm pad.")
    assert tags["vocals"] == ["female vocals"]
    assert "vocals" not in tags["vocals"]
    assert "pad" in tags["instruments"]


def test_order_preserved_by_first_appearance():
    tags = extract_tags("First a piano, then a guitar, then a violin.")
    assert tags["instruments"] == ["piano", "guitar", "violin"]


def test_to_suno_joins_comma_separated():
    tags = {"instruments": ["guitar"], "vocals": ["male vocals"], "production": ["reverb"]}
    assert to_suno(tags) == "guitar, male vocals, reverb"


def test_build_output_shape():
    out = build_output("A warm acoustic guitar with plate reverb.")
    assert set(out) == {
        "instruments", "vocals", "production", "caption", "tags_experimental"
    }
    assert out["caption"] == "A warm acoustic guitar with plate reverb."
    assert "acoustic guitar" in out["instruments"]
    assert "plate reverb" in out["production"]
    assert out["tags_experimental"] is True
    assert "suno_prompt" not in out


def test_suno_prompt_is_opt_in_and_marked_experimental():
    out = build_output(
        "A warm acoustic guitar with plate reverb.", experimental_suno=True
    )
    assert out["suno_experimental"] is True
    assert isinstance(out["suno_prompt"], str)
    assert "acoustic guitar" in out["suno_prompt"]


def test_merge_tags_order_preserving_dedupe():
    a = {"instruments": ["guitar", "piano"], "vocals": [], "production": ["reverb"]}
    b = {"instruments": ["piano", "drums"], "vocals": ["choir"], "production": []}
    merged = merge_tags([a, b])
    assert merged["instruments"] == ["guitar", "piano", "drums"]
    assert merged["vocals"] == ["choir"]
    assert merged["production"] == ["reverb"]


def test_folk_fixture():
    tags = extract_tags(read_fixture("folk_guitar.txt"))
    assert "acoustic guitar" in tags["instruments"]
    assert "guitar" not in tags["instruments"]
    assert "upright bass" in tags["instruments"]
    assert "drums" in tags["instruments"]
    assert "male vocal" in tags["vocals"]
    assert "harmonies" in tags["vocals"] or "harmony vocals" in tags["vocals"]
    assert "plate reverb" in tags["production"]
    assert "warm" in tags["production"]


def test_electronic_fixture():
    tags = extract_tags(read_fixture("electronic.txt"))
    assert "808" in tags["instruments"]
    assert "drum machine" in tags["instruments"]
    assert "synth bass" in tags["instruments"] or "synth pad" in tags["instruments"]
    assert "female vocals" in tags["vocals"]
    assert "autotune" in tags["vocals"]
    assert "sidechain compression" in tags["production"]
    assert "tape saturation" in tags["production"]
    assert "lo-fi" in tags["production"]


def test_custom_lexicon_override():
    lex = {"instruments": ["kazoo"], "vocals": [], "production": []}
    tags = extract_tags("A solo kazoo, plus a guitar.", lex)
    assert tags["instruments"] == ["kazoo"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
