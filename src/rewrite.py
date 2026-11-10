"""Stateless plain-text rewrite of repeated per-chunk captions.

The captioner answers every audio window as if it were a whole song, so the
concatenated chunk captions repeat the framing, repeat descriptions of the same
instrumentation, and contain filler. Instead of deleting sentences with string
similarity heuristics, we hand the raw text to a small instruct model and ask it
to tidy it up:

    system: <fixed instruction>
    user:   <the caption text, nothing else>

No JSON, no field names, no per-chunk metadata reach the model, and every call
is built from scratch (no conversation history), so the rewrite is stateless and
reproducible from its inputs.

The model is not trusted blindly. A deterministic *verification* pass (not a
rewrite) rejects an output that invents a lexicon term, claims an ending the
final source section does not claim, changes the length wildly, or returns
markup. On failure we keep the untouched source text and report why, so nothing
is silently dropped.

Pure helpers only: importing this module needs no torch and no GPU, so the
checks and prompt shape are unit-testable on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .merge import is_coda_claim, split_sentences
from .tags import extract_tags

# Bump when the instruction or the input layout changes: the rewrite output is
# part of the caption, so stale results must not be reused (see resume.py).
REWRITE_PROMPT_VERSION = 2

# Fixed instruction. Kept short on purpose: the user message is only the text.
REWRITE_SYSTEM_PROMPT = (
    "The text below describes one music track. It was written in several "
    "passes, so it repeats itself and contains filler. Rewrite it as one "
    "clean, natural description of the track: keep every distinct detail, "
    "remove the repetition and padding, and add nothing that is not in the "
    "text. Output only the rewritten description, with no headings, lists, "
    "or commentary. "
    "Never include timestamps or time references in the output. If the input "
    "chunk captions contain any, drop the exact time and express the ordering "
    "with general language (\"early on\", \"soon after\", \"later\", \"toward "
    "the end\") while keeping the order of events the same."
)

# Time references must never reach the final caption. Chunk captions answer a
# ~90 s window, so a timestamp in them is chunk-relative and becomes a lie once
# the chunks are merged into one track description. The patterns are
# deliberately broad: a false positive costs one rewrite retry, a false
# negative publishes a wrong claim.
TIMESTAMP_PATTERNS = (
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*-?\s*"
        r"(?:s|sec|secs|seconds?|min|mins|minutes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:first|last|final)\s+(?:few\s+)?(?:seconds?|minutes?)\b",
        re.IGNORECASE,
    ),
)

# Two extra attempts after the first: the guard must not accept a wrong caption,
# but one retry often loses the time reference without changing the content.
TIMESTAMP_RETRIES = 2

# Sanity bounds for a rewrite. A 200-300 word source should land well inside.
MIN_WORDS = 50
MAX_WORDS = 400

_WORD = re.compile(r"[a-z0-9']+")
_MARKUP = re.compile(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+")
_COMMENTARY = re.compile(
    r"\b(?:here(?:'s| is)|as an ai|rewritten|rewrite|summary:|below is)\b",
    re.IGNORECASE,
)

# "before fading out", "a fading beat": the final section does describe an
# ending, even when it uses the progressive form the coda regex does not match.
_ENDING_HINT = re.compile(
    r"\b(?:fad(?:e|es|ed|ing)|fade[- ]?out|outro)\b", re.IGNORECASE
)

# Facts worth verifying. Production adjectives live in the lexicon too
# ("melodic", "textures", "warm"), but they are stylistic wording, not claims
# about the recording; rejecting them made 3 of 24 tracks fall back needlessly.
VERIFIED_CATEGORIES = ("instruments", "vocals")


class Rewriter(Protocol):
    """Anything that turns chat messages into plain text."""

    name: str

    def rewrite(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of one rewrite attempt."""

    text: str
    status: str  # "ok" | "fallback"
    reasons: tuple[str, ...] = ()
    source_words: int = 0
    output_words: int = 0
    rewriter: str = ""
    prompt_version: int = REWRITE_PROMPT_VERSION

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def timestamp_hits(text: str) -> list[str]:
    """Every time reference found in ``text``, in order (empty = clean)."""
    hits: list[str] = []
    for pattern in TIMESTAMP_PATTERNS:
        hits.extend(match.group(0) for match in pattern.finditer(text or ""))
    return hits


# Deterministic, sentence-local rewording of a time reference. Each rule keeps
# the clause around it and its position, and has a sentence-initial and a
# mid-sentence form so "At 0:23, X" -> "Then, X" but "emerges at 0:23" ->
# "emerges later" (never "emerges Then").
_SCRUB_RULES = (
    (re.compile(r"at the \d+(?:\.\d+)?[- ]?(?:second|sec|minute|min)s?"
                r"[- ]?(?:mark)?", re.IGNORECASE), "Then", "later"),
    (re.compile(r"around the \d+(?:\.\d+)?[- ]?(?:second|sec|minute|min)s?"
                r"[- ]?(?:mark)?", re.IGNORECASE), "At one point",
     "at one point"),
    (re.compile(r"at \d{1,2}:\d{2}(?::\d{2})?(?:\s*mark)?", re.IGNORECASE),
     "Then", "later"),
    (re.compile(r"in the first (?:few )?(?:seconds?|minutes?)",
                re.IGNORECASE), "Early on", "early on"),
    # preposition-led forms first, or "in the last few seconds" becomes
    # "in toward the end"
    (re.compile(r"(?:in|for) the last (?:few )?(?:seconds?|minutes?)",
                re.IGNORECASE), "Toward the end", "toward the end"),
    (re.compile(r"(?:the )?last (?:few )?(?:seconds?|minutes?)",
                re.IGNORECASE), "Toward the end", "toward the end"),
    (re.compile(r"(?:the )?first (?:few )?(?:seconds?|minutes?)",
                re.IGNORECASE), "Early on", "early on"),
    (re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\b"), "a point later",
     "a point later"),
    (re.compile(r"\d+(?:\.\d+)?[- ]?(?:second|sec|minute|min)s?\b",
                re.IGNORECASE), "a moment", "a moment"),
)


def _scrub_sentence(sentence: str) -> str:
    out = sentence
    for pattern, at_start, mid_sentence in _SCRUB_RULES:
        out = pattern.sub(
            lambda m, a=at_start, b=mid_sentence: a if m.start() == 0 else b,
            out,
        )
    return re.sub(r"\s{2,}", " ", out)


def scrub_timestamps(text: str) -> str:
    """Replace time references with general sequencing language.

    Deterministic and sentence-local so it can run on every caption path
    (rewritten, fallback, single-chunk), where the model guard cannot: it never
    rewords anything but the time expression itself. ``text`` is returned
    unchanged when it has no hit, so a clean caption keeps its exact bytes.
    """
    if not text or not timestamp_hits(text):
        return text
    sentences = [
        _scrub_sentence(s) if timestamp_hits(s) else s
        for s in split_sentences(text)
    ]
    return " ".join(s.strip() for s in sentences).strip()


def build_source_text(sections: Sequence[str]) -> str:
    """Join chunk captions into the plain text handed to the model."""
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def build_messages(text: str) -> list[dict[str, str]]:
    """A fresh system+user conversation: the user message is the text alone."""
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _tokens(text: str) -> set[str]:
    return {_stem(tok) for tok in _WORD.findall((text or "").lower())}


def invented_terms(source: str, output: str) -> list[tuple[str, str]]:
    """Instrument/vocal terms in ``output`` whose words never appear in ``source``.

    Production terms are deliberately not checked: the lexicon includes
    adjectives ("melodic", "warm", "textures") and hyphenated variants
    ("reverb-drenched") that are ordinary prose, not invented facts.
    """
    source_tokens = _tokens(source)
    found = extract_tags(output)
    invented: list[tuple[str, str]] = []
    for category in VERIFIED_CATEGORIES:
        for term in found[category]:
            words = _tokens(term)
            if words and not words <= source_tokens:
                invented.append((category, term))
    return invented


def check_rewrite(sections: Sequence[str], output: str) -> list[str]:
    """Reasons to reject ``output``; empty means it passed."""
    text = (output or "").strip()
    if not text:
        return ["empty output"]

    reasons: list[str] = []
    words = text.split()
    if len(words) < MIN_WORDS:
        reasons.append(f"too short ({len(words)} words)")
    elif len(words) > MAX_WORDS:
        reasons.append(f"too long ({len(words)} words)")

    source = build_source_text(sections)
    if " ".join(text.split()) == " ".join(source.split()):
        reasons.append("unchanged source text")

    if text.lstrip().startswith(("{", "[")) or _MARKUP.search(text) or "#" in text:
        reasons.append("markup instead of prose")
    if _COMMENTARY.search(text):
        reasons.append("commentary instead of the description")

    hits = timestamp_hits(text)
    if hits:
        listed = ", ".join(dict.fromkeys(hits))
        reasons.append(f"timestamp in output: {listed}")

    invented = invented_terms(source, text)
    if invented:
        listed = ", ".join(f"{term} ({cat})" for cat, term in invented)
        reasons.append(f"invented terms: {listed}")

    # A coda is only allowed when the last source section actually describes
    # an ending: mid-song cuts are not the end of the track.
    if is_coda_claim(text):
        last_section = split_sentences(sections[-1]) if sections else []
        if not any(
            is_coda_claim(s) or _ENDING_HINT.search(s) for s in last_section
        ):
            reasons.append("ending claim not supported by the final section")

    return reasons


def rewrite_caption(
    sections: Sequence[str],
    rewriter: Rewriter | None,
    *,
    fallback: str | None = None,
) -> RewriteResult:
    """Rewrite concatenated chunk captions into one description.

    Stateless: one fresh message list per call. On any failure the untouched
    source text is returned with ``status="fallback"`` and the reasons, so the
    caller can log it and nothing is lost. ``fallback`` lets the caller reuse the
    deterministic merge already stored as ``caption_raw``.
    """
    source = build_source_text(sections)
    source_words = len(source.split())
    fallback_text = (
        fallback
        if fallback is not None
        else " ".join(s.strip() for s in sections if s and s.strip())
    )
    if not source:
        return RewriteResult("", "fallback", ("empty source",), 0, 0)
    if rewriter is None:
        return RewriteResult(
            fallback_text, "fallback", ("no rewriter configured",), source_words,
            len(fallback_text.split()),
        )

    name = getattr(rewriter, "name", type(rewriter).__name__)
    attempts = 0
    while True:
        attempts += 1
        try:
            output = rewriter.rewrite(build_messages(source))
        except Exception as exc:  # keep one bad track from killing a batch
            return RewriteResult(
                fallback_text,
                "fallback",
                (f"rewriter error: {type(exc).__name__}: {exc}",),
                source_words,
                len(fallback_text.split()),
                name,
            )

        output = (output or "").strip()
        reasons = check_rewrite(sections, output)
        timed = any(r.startswith("timestamp in output") for r in reasons)
        # Only the timestamp guard retries: it is the one failure this session
        # refuses to publish, and a retry is cheap next to re-captioning.
        if not timed or attempts > TIMESTAMP_RETRIES:
            break

    if timed:
        # Still timed after the retries: never report this as "ok". Keep the
        # raw caption (it is stored anyway) and flag the entry for review.
        return RewriteResult(
            fallback_text,
            "failed",
            tuple(reasons) + (f"timestamp guard: gave up after {attempts} "
                              "attempts",),
            source_words,
            len(output.split()),
            name,
        )
    if reasons:
        return RewriteResult(
            fallback_text, "fallback", tuple(reasons), source_words,
            len(output.split()), name,
        )
    return RewriteResult(
        output, "ok", (), source_words, len(output.split()), name
    )
