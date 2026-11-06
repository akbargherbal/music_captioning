"""Deterministic merge of per-chunk captions into one description.

Every chunk is captioned independently, and the model answers each one with a
complete, self-contained description: a framing sentence ("An energetic Arabic
pop track driven by ..."), body sentences, and a coda ("... concludes abruptly
with ..."). Blindly joining those repeats the framing once per chunk and asserts
a false ending at every mid-song cut.

This module removes that structurally, with no extra model call and no
randomness:

1. keep only the first section's framing sentence;
2. drop coda claims from every section except the last;
3. drop near-duplicate sentences that an earlier section already said
   (Jaccard >= 0.6 on lowercase content words);
4. keep reading order and join with single spaces.

Everything here is pure string processing, so it is testable on CPU with no
model present. Sentences are only compared *across* sections, so a single-
section caption is returned unchanged apart from whitespace normalisation.
"""

from __future__ import annotations

import re

# Sentence boundary: punctuation followed by whitespace. Keeps abbreviations
# inside a sentence together (they have no following whitespace + capital).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A section's first sentence is a whole-song framing when it opens with an
# article and names the track itself.
_WHOLE_SONG_OPENER = re.compile(
    r"^(?:An?|The|This)\b.*\b(?:track|song|piece|ballad)\b",
    re.IGNORECASE,
)

# Claims about how the audio stops. Only the final section may make one.
# The verb forms require a track/song subject ("the track ends", "it closes"),
# so body phrasing such as "a close-miked vocal" or "the phrase ends on a
# suspended note" is not mistaken for an ending claim.
_CODA_CLAIM = re.compile(
    r"\b(?:"
    r"(?:track|song|piece|arrangement|audio|performance|it)\s+"
    r"(?:conclud(?:e|es|ed)|ends?|fades?\s*out|closes?"
    r"|comes?\s+to\s+an?\s+(?:abrupt\s+)?end"
    r"|draws?\s+to\s+an?\s+(?:abrupt\s+)?close)\b"
    r"|(?:concluding|closing|final)\s+(?:section|bars?|chord|moments?|seconds?)\b"
    r"|(?:track|song|piece)'s\s+final\b"
    r"|outro|fade[-\s]?out|tape[-\s]?stop|silence"
    r")\b",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """
    a an the and or but with without of in on at to for from by as is are was
    were be been being it its this that these those there here then than so
    very much more most some any all both each few own other such no nor not
    only same too can will would should could may might must just now still
    into over under again further once during while which who whom what when
    where why how
    """.split()
)

# Whole-sentence similarity thresholds. Sentences that introduce an instrument
# no earlier section mentioned are only dropped when they are near-identical,
# so the merge never loses an instrument to a loose comparison.
DUPLICATE_THRESHOLD = 0.6
NEW_INSTRUMENT_THRESHOLD = 0.85


def split_sentences(caption: str) -> list[str]:
    """Split ``caption`` into sentences, dropping empties."""
    text = " ".join((caption or "").split())
    if not text:
        return []
    return [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def is_whole_song_opener(sentence: str) -> bool:
    """True when ``sentence`` frames the whole track ("An ... pop track ...")."""
    return bool(_WHOLE_SONG_OPENER.match(sentence.strip()))


def is_coda_claim(sentence: str) -> bool:
    """True when ``sentence`` claims the audio is ending."""
    return bool(_CODA_CLAIM.search(sentence))


def _content_words(sentence: str) -> frozenset[str]:
    """Lowercase content words of ``sentence`` (stopwords removed)."""
    return frozenset(
        token for token in _TOKEN.findall(sentence.lower())
        if token not in _STOPWORDS
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_INSTRUMENT_TERMS: tuple[str, ...] | None = None


def _instrument_terms() -> tuple[str, ...]:
    """Instrument lexicon terms, loaded once per process."""
    global _INSTRUMENT_TERMS
    if _INSTRUMENT_TERMS is None:
        from .tags import load_lexicon

        _INSTRUMENT_TERMS = tuple(load_lexicon()["instruments"])
    return _INSTRUMENT_TERMS


def _instruments(sentence: str) -> frozenset[str]:
    low = sentence.lower()
    return frozenset(term for term in _instrument_terms() if term in low)


def merge_section_captions(
    captions: list[str] | tuple[str, ...],
    *,
    dropped: list[str] | None = None,
) -> str:
    """Merge per-section captions into one description.

    ``dropped`` is an optional sink for an audit trail: every removed sentence
    is appended as ``"[section N] reason: sentence"``. Pass a list to inspect
    what the merge discarded; nothing is printed by default.
    """
    sections = [split_sentences(c) for c in captions]
    sections = [s for s in sections if s]
    if not sections:
        return ""

    last_index = len(sections) - 1
    kept: list[str] = []
    # (section index, content words) for cross-section duplicate detection.
    kept_index: list[tuple[int, frozenset[str]]] = []
    kept_instruments: set[str] = set()

    def record(section: int, reason: str, sentence: str) -> None:
        if dropped is not None:
            dropped.append(f"[section {section}] {reason}: {sentence}")

    for index, sentences in enumerate(sections):
        for position, sentence in enumerate(sentences):
            if position == 0 and index > 0 and is_whole_song_opener(sentence):
                record(index, "repeat framing", sentence)
                continue
            if index < last_index and is_coda_claim(sentence):
                record(index, "non-final coda", sentence)
                continue

            words = _content_words(sentence)
            instruments = _instruments(sentence)
            threshold = (
                NEW_INSTRUMENT_THRESHOLD
                if instruments - kept_instruments
                else DUPLICATE_THRESHOLD
            )
            duplicate = any(
                other_index < index and _jaccard(words, other_words) >= threshold
                for other_index, other_words in kept_index
            )
            if duplicate:
                record(index, "near-duplicate", sentence)
                continue

            kept.append(sentence)
            kept_index.append((index, words))
            kept_instruments |= instruments

    return " ".join(kept)
