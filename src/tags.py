"""Experimental, deterministic lexicon-based tag extraction.

No hosted LLM is used here: matching is plain ``\\b``-delimited, case-insensitive
substring matching against ``lexicon.json``. Longer/more specific terms win when
spans overlap (e.g. "acoustic guitar" is kept, a bare "guitar" is not emitted).

**Status: experimental.** The extracted terms are a lossy, auditable view of the
caption, not a finished music-generation prompt. The known gaps (no genre/mood,
plural and phrase misses, chunk contradictions, no negation) are documented in
``docs/TAG_AUDIT.md``. ``to_suno()`` is kept for experiments only and must be
opted into explicitly (``build_output(..., experimental_suno=True)``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CATEGORIES = ("instruments", "vocals", "production")
DEFAULT_LEXICON_PATH = Path(__file__).with_name("lexicon.json")

# Marks every output that contains the deterministic term lists. Consumers
# should treat these lists as experimental until docs/TAG_AUDIT.md is resolved.
TAGS_EXPERIMENTAL = True


def load_lexicon(path: Path | str | None = None) -> dict[str, list[str]]:
    """Load the lexicon JSON, guaranteeing every category exists."""
    lex_path = Path(path) if path is not None else DEFAULT_LEXICON_PATH
    data = json.loads(lex_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"lexicon must be a JSON object, got {type(data).__name__}")
    return {
        cat: list(dict.fromkeys(data.get(cat, []))) for cat in CATEGORIES
    }


def _compile(lexicon: dict[str, list[str]]) -> list[tuple[str, str, re.Pattern[str]]]:
    entries: list[tuple[str, str, re.Pattern[str]]] = []
    for cat in CATEGORIES:
        for term in lexicon.get(cat, []):
            # Join words with \s+ so multi-word terms still match across line
            # wraps in a caption, then anchor with word boundaries.
            body = r"\s+".join(re.escape(tok) for tok in term.split())
            pattern = re.compile(r"\b" + body + r"\b", re.IGNORECASE)
            entries.append((cat, term, pattern))
    return entries


def extract_tags(
    caption: str, lexicon: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Return ``{category: [tags]}`` found in ``caption``.

    Tags are de-duplicated and ordered by first appearance in the caption.
    Overlapping matches are resolved so the most specific term wins.
    """
    lex = lexicon if lexicon is not None else load_lexicon()
    text = caption or ""
    result: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}

    # (start, end, category, term) for every candidate match.
    candidates: list[tuple[int, int, str, str]] = []
    for cat, term, pattern in _compile(lex):
        for m in pattern.finditer(text):
            candidates.append((m.start(), m.end(), cat, term))

    # Prefer longer spans (more specific), then earlier ones; skip overlaps.
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    chosen: list[tuple[int, int, str, str]] = []
    for cand in candidates:
        start, end = cand[0], cand[1]
        if any(not (end <= c[0] or start >= c[1]) for c in chosen):
            continue
        chosen.append(cand)

    chosen.sort(key=lambda c: c[0])  # back to reading order
    seen: set[tuple[str, str]] = set()
    for _start, _end, cat, term in chosen:
        key = (cat, term)
        if key in seen:
            continue
        seen.add(key)
        result[cat].append(term)
    return result


def to_suno(tags: dict[str, list[str]]) -> str:
    """Join all tags into a comma-separated string.

    **Experimental.** This is not a supported Suno prompt: it omits genre and
    mood, can contain synonym duplicates, and can carry contradictory chunk
    tags. Only call it behind an explicit opt-in (see ``build_output``).
    """
    parts: list[str] = []
    for cat in CATEGORIES:
        parts.extend(tags.get(cat, []))
    return ", ".join(parts)


def merge_tags(tag_dicts: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Merge per-chunk tag dicts with order-preserving de-duplication."""
    merged: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for tags in tag_dicts:
        for cat in CATEGORIES:
            for term in tags.get(cat, []):
                if term not in merged[cat]:
                    merged[cat].append(term)
    return merged


def build_output(
    caption: str,
    lexicon: dict[str, list[str]] | None = None,
    *,
    experimental_suno: bool = False,
) -> dict[str, object]:
    """Full result: experimental term lists plus the original caption.

    ``suno_prompt`` is **opt-in** and marked ``suno_experimental``. It is not
    part of the default output because the deterministic string is not yet a
    reliable generation prompt (see ``docs/TAG_AUDIT.md``).
    """
    tags = extract_tags(caption, lexicon)
    out: dict[str, object] = {
        "instruments": tags["instruments"],
        "vocals": tags["vocals"],
        "production": tags["production"],
        "caption": (caption or "").strip(),
        "tags_experimental": TAGS_EXPERIMENTAL,
    }
    if experimental_suno:
        out["suno_prompt"] = to_suno(tags)
        out["suno_experimental"] = True
    return out
