"""Pure-Python tests for the stateless caption rewrite. No model or GPU needed."""

from src.rewrite import (
    REWRITE_PROMPT_VERSION,
    REWRITE_SYSTEM_PROMPT,
    build_messages,
    build_source_text,
    check_rewrite,
    invented_terms,
    rewrite_caption,
)
from src.rewriter_hf import DEFAULT_MODEL_ID, HFRewriter

SECTIONS = [
    (
        "An upbeat Arabic pop track built on a driving drum machine beat and a "
        "warm synth bass. A passionate male lead vocal carries the melody. Lush "
        "synth pads and a bright oud line fill the chorus. The production is "
        "clean and modern with a touch of reverb."
    ),
    (
        "The arrangement keeps the same driving drum machine beat. A male lead "
        "vocal repeats the chorus hook. A bright oud line answers the vocal. "
        "The mix stays clean and the reverb sits behind the voice."
    ),
    (
        "A final chorus repeats the hook over the drum machine beat and synth "
        "bass. The oud line and synth pads return. The track fades out with a "
        "long reverb tail."
    ),
]

GOOD = (
    "An upbeat Arabic pop track built on a driving drum machine beat, a warm "
    "synth bass, lush synth pads and a bright oud line. A passionate male lead "
    "vocal carries the melody and repeats the chorus hook, answered by the oud. "
    "The production is clean and modern, with a touch of reverb behind the "
    "voice. The track fades out with a long reverb tail."
)


class FakeRewriter:
    name = "fake"

    def __init__(self, output: str = GOOD):
        self.output = output
        self.calls: list[list[dict[str, str]]] = []

    def rewrite(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.output


def test_source_text_is_plain_prose():
    text = build_source_text(SECTIONS)
    assert text.startswith("An upbeat Arabic pop track")
    assert "\n\n" in text
    for marker in ("{", "}", '"instruments"', "chunk", "JSON"):
        assert marker not in text


def test_user_message_is_only_the_text():
    text = build_source_text(SECTIONS)
    messages = build_messages(text)
    assert messages[0] == {"role": "system", "content": REWRITE_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": text}
    assert len(messages) == 2


def test_prompt_version_is_recorded():
    result = rewrite_caption(SECTIONS, FakeRewriter())
    assert result.prompt_version == REWRITE_PROMPT_VERSION
    assert REWRITE_PROMPT_VERSION >= 1


def test_rewrite_is_stateless():
    fake = FakeRewriter()
    first = rewrite_caption(SECTIONS, fake)
    second = rewrite_caption(SECTIONS, fake)
    assert first.ok and second.ok
    assert fake.calls[0] == fake.calls[1]
    assert all(m["role"] in {"system", "user"} for m in fake.calls[1])
    assert not any("assistant" in m["role"] for m in fake.calls[1])


def test_accepts_faithful_rewrite():
    result = rewrite_caption(SECTIONS, FakeRewriter())
    assert result.ok, result.reasons
    assert result.text == GOOD
    assert result.source_words > result.output_words


def test_rejects_invented_instrument():
    fake = FakeRewriter(GOOD.replace("answered by the oud", "answered by a sitar"))
    result = rewrite_caption(SECTIONS, fake)
    assert not result.ok
    assert any("sitar" in r for r in result.reasons)
    assert result.text == " ".join(SECTIONS)


def test_rejects_ending_claim_when_final_section_has_none():
    sections = [
        SECTIONS[0],
        SECTIONS[1],
        "A final chorus repeats the hook over the drum machine beat and synth bass.",
    ]
    result = rewrite_caption(sections, FakeRewriter())
    assert not result.ok
    assert any("ending claim" in r for r in result.reasons)


def test_allows_ending_claim_from_final_section():
    test_accepts_faithful_rewrite()


def test_rejects_markup():
    fake = FakeRewriter(
        "Here is the description:\n\n- An upbeat Arabic pop track.\n- A male "
        "lead vocal.\n- A clean production with reverb.\n- The track fades "
        "out with a long reverb tail, after a final chorus over the drum "
        "machine beat and synth bass, plus lush synth pads and a bright oud line."
    )
    reasons = check_rewrite(SECTIONS, fake.output)
    assert reasons


def test_rejects_json_output():
    reasons = check_rewrite(SECTIONS, '{"caption": "an Arabic pop track"}')
    assert reasons


def test_rejects_too_short():
    reasons = check_rewrite(SECTIONS, "An Arabic pop track.")
    assert any("too short" in r for r in reasons)


def test_rejects_unchanged_text():
    reasons = check_rewrite(SECTIONS, build_source_text(SECTIONS))
    assert any("unchanged" in r for r in reasons)


def test_allows_ending_claim_when_final_section_says_fading_out():
    sections = [
        SECTIONS[0],
        SECTIONS[1],
        "A final chorus repeats the hook, then the groove keeps going before "
        "fading out on the drum machine beat.",
    ]
    reasons = check_rewrite(sections, GOOD)
    assert reasons == []


def test_production_adjectives_are_not_invented_facts():
    # "melodic", "textures" and "reverb-drenched" are in the production
    # lexicon, but they are ordinary prose, not claims about the recording.
    assert invented_terms("a clean mix with reverb", "a melodic texture, "
                          "reverb-drenched and warm") == []
    reasons = check_rewrite(
        SECTIONS,
        GOOD.replace("a touch of reverb behind the voice",
                     "a reverb-drenched, melodic texture behind the voice"),
    )
    assert reasons == []


def test_invented_terms_ignores_words_present_in_source():
    assert (
        invented_terms(
            "A clean mix with reverb and an oud.",
            "The mix is clean with reverb over an oud.",
        )
        == []
    )
    assert invented_terms("A clean mix with reverb.", "A clean mix with a banjo.") == [
        ("instruments", "banjo")
    ]


def test_rewriter_error_falls_back_to_source():
    class Broken:
        name = "broken"

        def rewrite(self, messages):
            raise RuntimeError("CUDA out of memory")

    result = rewrite_caption(SECTIONS, Broken())
    assert not result.ok
    assert result.text == " ".join(SECTIONS)
    assert "CUDA out of memory" in result.reasons[0]


def test_no_rewriter_returns_source_untouched():
    result = rewrite_caption(SECTIONS, None)
    assert not result.ok
    assert result.text == " ".join(SECTIONS)
    assert "no rewriter" in result.reasons[0]


def test_hf_rewriter_defaults():
    r = HFRewriter()
    assert r.name == DEFAULT_MODEL_ID
    assert r.quantization == "none"
    assert r.max_new_tokens == 384
    assert HFRewriter(quantization="4bit").quantization == "4bit"


def test_rewrite_demo_loads_chunk_captions(tmp_path):
    import json

    import pytest

    from scripts.rewrite_demo import load_sections

    payload = tmp_path / "track.json"
    payload.write_text(
        json.dumps({"chunk_captions": ["One track.", "Two track."]}),
        encoding="utf-8",
    )
    assert load_sections(payload) == ["One track.", "Two track."]

    text = tmp_path / "track.txt"
    text.write_text("First part.\n\nSecond part.\n", encoding="utf-8")
    assert load_sections(text) == ["First part.", "Second part."]

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_sections(empty)


def test_scrub_removes_a_time_reference():
    from src.rewrite import scrub_timestamps, timestamp_hits

    text = "A low pad hums while a beat emerges at 0:23, adding groove."
    cleaned = scrub_timestamps(text)
    assert "0:23" not in cleaned
    assert cleaned.startswith("A low pad hums while a beat emerges later,")
    assert timestamp_hits(cleaned) == []


def test_scrub_sentence_initial_is_capitalised():
    from src.rewrite import scrub_timestamps

    assert scrub_timestamps("At 0:34, an oud joins.") == "Then, an oud joins."


def test_scrub_leaves_clean_text_byte_identical():
    from src.rewrite import scrub_timestamps

    text = "Tempo 120 bpm in 4/4,  with a warm bass."
    assert scrub_timestamps(text) == text


def test_check_rewrite_rejects_a_timestamp():
    from src.rewrite import check_rewrite

    source = ["the track opens quietly and builds " + "word " * 60]
    bad = "at 0:23 the drums enter " + "word " * 60
    assert any("timestamp" in r for r in check_rewrite(source, bad))


def test_polish_album_scrubs_a_fallback_caption():
    """A rejected rewrite keeps raw text; the scrub still runs on it."""
    from scripts.batch import polish_album
    from src.rewrite import timestamp_hits

    class Echo:
        name = "echo"

        def rewrite(self, messages):
            return messages[-1]["content"]

    body = "the beat enters at 0:23 and " + "groove " * 60
    album = {"tracks": [{"file": "a.mp3", "caption_raw": body,
                         "chunk_captions": [body, "a second section " * 10]}]}
    out = polish_album(album, Echo())
    track = out["tracks"][0]
    assert timestamp_hits(track["caption"]) == []
    assert track["timestamp_cleanup"] == "ok"
    assert track["caption_pre_cleanup"] == body


def test_decades_are_not_timestamps():
    """'a classic 80s drum machine' is an era, not a position in the track."""
    from src.rewrite import timestamp_hits

    for text in ("a classic 80s drum machine", "sounds from the 90s",
                 "a 2000s-era synth", "1980s gated snare"):
        assert timestamp_hits(text) == [], text
    # real references still fire
    assert timestamp_hits("at 1:40 it shifts") == ["1:40"]
    assert timestamp_hits("after 30 seconds") == ["30 seconds"]
