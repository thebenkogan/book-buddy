"""Natural-language reading position -> flattened TOC ordinal.

Two safety rules drive this module:

1. **Progress is an ordinal, never a chapter number.** In the Jewish War
   ``CHAPTER 1.`` occurs seven times and 22 of the 33 bare chapter numbers
   appear in more than one book, so "chapter 3" does not identify anything.
2. **Never silently guess when ambiguous.** If the top two candidates score
   within ``AMBIGUITY_MARGIN`` of each other, ``ambiguous=True`` is returned and
   the caller is expected to ask the user.

The model is never allowed to free-form a section. It is shown a numbered array
and must return an INTEGER INDEX into that array, so it cannot invent a chapter
that does not exist.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.bookbuddy import chat, extract_json
from src.bookbuddy.book import Section

logger = logging.getLogger(__name__)

#: If the top-2 lexical scores are within this many points (0-100), the request
#: is ambiguous and we refuse to pick.
AMBIGUITY_MARGIN = 12.0

#: Scores below this are not a match at all.
MIN_CONFIDENCE = 30.0

#: Confidence assigned when the model returns a valid index from the array.
MODEL_TRUST = 85.0

#: Below this top score the lexical layer has no real signal at all — every
#: section scores similarly because none of them match. That is different from
#: a genuine tie between two plausible answers, so in that case the model's
#: in-range answer is accepted instead of prompting the reader.
LEXICAL_SIGNAL_FLOOR = 60.0

#: Ordinals move forward only; a backwards jump of more than this many sections
#: requires confirmation rather than being applied silently.
MAX_SILENT_REWIND = 0

_RESOLVE_SYSTEM = (
    "You map a reader's description of where they are in a book to one entry "
    "from a numbered table of contents. You must answer with a JSON object "
    "containing 'index' (an integer copied from the array you were shown, or "
    "null if nothing matches) and 'evidence' (a short quote or paraphrase from "
    "the book itself that supports the match). You may not invent an index."
)

_RESOLVE_PROMPT = """\
Book: {title}

Table of contents (choose one index; the array order is authoritative):

{toc}

The reader says:
"{user_text}"

Return ONLY:
{{"index": <integer or null>, "evidence": "<short supporting quote from the book>"}}
"""


class Candidate(BaseModel):
    """One scored resolution candidate."""

    ordinal: int
    label: str
    parent: str = ""
    title: str = ""
    score: float

    @property
    def display(self) -> str:
        if self.title:
            return f"{self.label} — {self.title}"
        return self.label


class ResolvedPosition(BaseModel):
    """Result of :func:`resolve_position`."""

    ordinal: Optional[int] = None
    confidence: float = 0.0
    ambiguous: bool = False
    evidence: str = ""
    candidates: List[Candidate] = Field(default_factory=list)
    reason: str = ""
    clamped_from: Optional[int] = None
    needs_confirmation: bool = False

    @property
    def resolved(self) -> bool:
        return self.ordinal is not None and not self.ambiguous


# --------------------------------------------------------------------------- #
# Lexical scoring
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "i",
    "im",
    "ive",
    "read",
    "reading",
    "up",
    "to",
    "just",
    "finished",
    "finishedreading",
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "and",
    "through",
    "chapter",
    "book",
    "part",
    "section",
    "end",
    "about",
    "where",
    "me",
    "my",
    "am",
    "got",
    "get",
    "currently",
    "now",
    "still",
    "but",
    "with",
    "for",
    "it",
    "this",
    "that",
}

_ROMAN_WORDS = {
    "one": "I",
    "two": "II",
    "three": "III",
    "four": "IV",
    "five": "V",
    "six": "VI",
    "seven": "VII",
    "eight": "VIII",
    "nine": "IX",
    "ten": "X",
    "first": "I",
    "second": "II",
    "third": "III",
    "fourth": "IV",
    "fifth": "V",
    "sixth": "VI",
    "seventh": "VII",
    "last": "LAST",
    "final": "LAST",
}

_BOOK_RE = re.compile(
    r"\bbook\s+([0-9]{1,2}|[ivxlc]{1,7}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_CHAPTER_RE = re.compile(
    r"\b(?:chapter|chap\.?|ch\.)\s+([0-9]{1,3}|[ivxlc]{1,7}|one|two|three|four|five|six|seven|eight|nine|ten|last|final)\b"
)


def _int_to_roman(value: int) -> Optional[str]:
    if not 1 <= value <= 3999:
        return None
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out: List[str] = []
    for number, numeral in numerals:
        while value >= number:
            out.append(numeral)
            value -= number
    return "".join(out)


def _normalize_chapter(token: str) -> Optional[str]:
    """Chapter numbers converge on ARABIC; book numbers converge on ROMAN.

    Section labels read "BOOK IV. / CHAPTER 12." — books are numbered in roman
    and chapters in arabic — so the two normalizers must not be the same
    function.
    """
    lowered = token.strip().lower()
    if not lowered:
        return None
    if lowered in _ROMAN_WORDS:
        mapped = _ROMAN_WORDS[lowered]
        if mapped == "LAST":
            return mapped
        return _roman_to_int_str(mapped)
    if lowered.isdigit():
        return str(int(lowered))
    if re.fullmatch(r"[ivxlc]+", lowered):
        return _roman_to_int_str(lowered.upper())
    return token.strip()


def _roman_to_int_str(roman: str) -> Optional[str]:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not roman or any(ch not in values for ch in roman):
        return None
    total = 0
    previous = 0
    for char in reversed(roman):
        value = values[char]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return str(total)


def _normalize_roman(token: str) -> Optional[str]:
    """Return the roman/arabic form of a spoken number token.

    "4" -> "IV", "four" -> "IV", "iv" -> "IV". Section labels are stored with
    roman book numbers, so both spellings have to converge on one form.
    """
    lowered = token.strip().lower()
    if not lowered:
        return None
    if lowered in _ROMAN_WORDS:
        return _ROMAN_WORDS[lowered]
    if lowered.isdigit():
        return _int_to_roman(int(lowered)) or token.strip()
    if re.fullmatch(r"[ivxlc]+", lowered):
        return lowered.upper()
    return token.strip()


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in _STOPWORDS]


def has_structural_mention(user_text: str) -> bool:
    """True if the reader named a book and/or chapter number.

    When they did, the deterministic parse is authoritative and the model is
    not allowed to overrule it. Live testing: for "I just finished book 4" the
    lexical answer (the BOOK IV. heading) was right and the model picked
    BOOK IV. / CHAPTER 11. and even cited BOOK V as its evidence. Numbers are
    parseable; only prose descriptions should go to the model.
    """
    user_norm = re.sub(r"\s+", " ", user_text.lower()).strip()
    return bool(_BOOK_RE.findall(user_norm) or _CHAPTER_RE.findall(user_norm))


def candidate_text(section: Section) -> str:
    """The string a section is matched against."""
    parts = [section.label]
    if section.parent:
        parts.append(section.parent)
    if section.title:
        parts.append(section.title)
    return " ".join(parts)


def score_sections(toc: Sequence[Section], user_text: str) -> List[Candidate]:
    """Score every section against ``user_text``, best first.

    Scoring has two layers:

    * **Structural layer** — when the user names a book and/or chapter
      ("book 2 chapter 3", "finished book four"), the sections that mention
      exactly those numbers are selected. This layer dominates fuzzy matching,
      which handles structural references badly. A bare chapter number that
      matches more than one section is scored identically for every match, so
      the caller sees a zero gap and asks the user rather than guessing.
    * **Fuzzy layer** — rapidfuzz similarity over the label/parent/title, which
      is what handles "the bit where X happens".
    """
    if not toc:
        return []
    user_norm = re.sub(r"\s+", " ", user_text.lower()).strip()
    user_tokens = _tokens(user_text)

    book_mentions = [
        b for b in (_normalize_roman(m) for m in _BOOK_RE.findall(user_norm)) if b
    ]
    chapter_mentions = [
        c for c in (_normalize_chapter(m) for m in _CHAPTER_RE.findall(user_norm)) if c
    ]

    def mentions_books(section: Section) -> bool:
        # A word boundary is mandatory: "BOOK I" is a substring of "BOOK II",
        # and matching on the substring sends every book-II chapter to book I.
        haystack = f"{section.label} {section.parent}".upper()
        return any(
            re.search(rf"BOOK\s+{re.escape(book)}\b", haystack)
            for book in book_mentions
        )

    def mentions_chapters(section: Section) -> bool:
        return mentions_chapters_of(section, chapter_mentions)

    scored: List[Candidate] = []
    for section in toc:
        text = candidate_text(section)
        fuzzy = fuzz.token_set_ratio(user_norm, text.lower())
        fuzzy = max(fuzzy, fuzz.partial_ratio(user_norm, text.lower()))
        if user_tokens:
            fuzzy = max(
                fuzzy,
                fuzz.token_sort_ratio(" ".join(user_tokens), " ".join(_tokens(text))),
            )
        score = float(fuzzy)

        if "LAST" in chapter_mentions and section is toc[-1]:
            score = max(score, 93.0)
        elif (
            chapter_mentions
            and mentions_chapters(section)
            and (not book_mentions or mentions_books(section))
        ):
            # All chapter mentions matched. If this chapter number occurs in
            # more than one book we deliberately assign the SAME score to
            # every match, producing an ambiguous result instead of a
            # silent, arbitrary choice.
            # Count matches within the named book when one was given;
            # across the whole toc when the user said only "chapter 3".
            occurrences = sum(
                1
                for other in toc
                if mentions_chapters_of(other, chapter_mentions)
                and (not book_mentions or mentions_books(other))
            )
            score = 95.0 if occurrences == 1 else 70.0
        elif chapter_mentions:
            # A specific chapter was named and this is not it, whatever book
            # it belongs to; keep it below the true match so the enclosing book
            # section cannot win the comparison.
            score = min(score, 55.0)
        elif book_mentions and mentions_books(section):
            # A bare "book 4" points at the book itself. Assigned rather than
            # max()'d, so that a fuzzy score on a chapter of that book cannot
            # tie with the book heading and force a pointless question.
            score = 95.0 if section.is_book else 75.0
        elif book_mentions or chapter_mentions:
            # The user gave structural information that this section does not
            # satisfy; penalise it below the fuzzy floor.
            score = min(score, 40.0)
        elif user_text.strip():
            score = score * 0.95

        scored.append(
            Candidate(
                ordinal=section.ordinal,
                label=section.label,
                parent=section.parent,
                title=section.title,
                score=round(score, 2),
            )
        )

    scored.sort(key=lambda c: (-c.score, c.ordinal))
    return scored


def mentions_chapters_of(section: Section, chapters: Sequence[str]) -> bool:
    """True if ``section``'s label mentions any of ``chapters``.

    Both spellings are accepted because real books are inconsistent: the Jewish
    War labels most chapters "CHAPTER 5." but a few "CHAPTER V.".
    """
    haystack = section.label.upper()
    for chapter in chapters:
        forms = [chapter]
        try:
            roman = _int_to_roman(int(chapter))
        except (TypeError, ValueError):
            roman = None
        if roman:
            forms.append(roman)
        for form in forms:
            if re.search(rf"CHAPTER\s+{re.escape(form)}\b", haystack):
                return True
    return False


def top_gap(scored: Sequence[Candidate]) -> Optional[float]:
    """Difference between the best and second-best score, or None."""
    if not scored:
        return None
    if len(scored) == 1:
        return 100.0
    return round(float(scored[0].score - scored[1].score), 2)


def format_toc(toc: Sequence[Section], limit: int = 0) -> str:
    """Render the TOC as a numbered array for the model. Index == ordinal."""
    lines: List[str] = []
    for section in toc:
        suffix = f" — {section.title}" if section.title else ""
        lines.append(f"[{section.ordinal}] {section.label}{suffix}")
    if limit and len(lines) > limit:
        head = lines[:limit]
        head.append(f"... ({len(lines) - limit} further sections omitted)")
        return "\n".join(head)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _parse_model_response(
    response_text: str, toc: Sequence[Section]
) -> Tuple[Optional[int], str]:
    """Extract a validated (index, evidence) pair from a model response."""
    payload = extract_json(response_text)
    index: Optional[int] = None
    evidence = ""
    if isinstance(payload, dict):
        evidence = str(payload.get("evidence") or "").strip()
        raw_index = payload.get("index", payload.get("ordinal"))
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = None
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first = payload[0]
        evidence = str(first.get("evidence") or "").strip()
        try:
            index = int(first.get("index"))
        except (TypeError, ValueError):
            index = None
    if index is None:
        return None, evidence
    # Constrained selection: reject any index not actually in the array.
    if not any(section.ordinal == index for section in toc):
        logger.warning("model returned out-of-range index %s; rejecting", index)
        return None, evidence
    return index, evidence


def resolve_position(
    toc: Sequence[Section],
    user_text: str,
    title: str = "",
    client: Optional[Callable[[str, str], str]] = None,
    current_ordinal: Optional[int] = None,
    shortlist: int = 30,
    use_model: bool = True,
) -> ResolvedPosition:
    """Resolve "I read up to chapter 3" to an ordinal over ``toc``.

    Returns a :class:`ResolvedPosition`. ``ambiguous=True`` means the caller
    must ask the user; ``ordinal`` is then left at the best candidate but is
    NOT considered resolved.

    ``current_ordinal`` enables the forward-only rule: a resolved position
    behind the current one is reported with ``needs_confirmation=True`` rather
    than applied.

    ``shortlist`` bounds how far down the lexical ranking the model's index may
    sit and still be honoured. The Jewish War has 117 sections and a vague
    description can legitimately point at any of them, so the default is
    deliberately generous; rejecting a correct answer as "out of range" is
    worse than accepting a merely surprising one.
    """
    if not toc:
        return ResolvedPosition(reason="toc is empty")

    scored = score_sections(toc, user_text)
    if not scored:
        return ResolvedPosition(reason="no candidates")

    gap = top_gap(scored)
    best = scored[0]
    confidence = best.score
    # The ambiguity verdict is computed from the LEXICAL top-2 gap and is never
    # overridden by the model. "chapter 3" ties between Book I and Book II; a
    # model that confidently picks one of them has still guessed, and the spec
    # is explicit that we must ask instead.
    lexical_gap = gap
    lexical_top_two = scored[:2]
    evidence = ""
    evidence_ordinal: Optional[int] = None

    # An injected client is always used; use_model=False only disables the
    # real network client (used by the deterministic test suite).
    model_enabled = bool(user_text.strip()) and (client is not None or use_model)
    if model_enabled:
        prompt = _RESOLVE_PROMPT.format(
            title=title or "unknown",
            toc=format_toc(toc),
            user_text=user_text,
        )
        response_text: Optional[str] = None
        try:
            if client is not None:
                response_text = client(prompt, _RESOLVE_SYSTEM)
            else:
                response_text = chat(
                    prompt,
                    system=_RESOLVE_SYSTEM,
                    temperature=0.0,
                    max_tokens=800,
                    reasoning_effort="low",
                )["content"]
        except Exception:
            logger.exception("position resolution call failed; using lexical only")
        if response_text:
            index, evidence = _parse_model_response(response_text, toc)
            if index is not None:
                evidence_ordinal = index

    # The model may only choose from the lexical shortlist; it can never
    # introduce a section the shortlist does not contain. When it returns a
    # valid index we defer to it: it saw the whole numbered array and the
    # reader's own words, which is exactly the disambiguation that the
    # ambiguity check exists to ask for. An index outside the shortlist is
    # discarded and the lexical answer stands.
    model_decided = False
    if evidence_ordinal is not None:
        if has_structural_mention(user_text):
            logger.info(
                "reader named a book/chapter number; keeping the deterministic "
                "parse (ordinal %s) and ignoring the model's %s",
                best.ordinal,
                evidence_ordinal,
            )
        else:
            rank = next(
                (
                    position
                    for position, candidate in enumerate(scored)
                    if candidate.ordinal == evidence_ordinal
                ),
                None,
            )
            if rank is not None and rank < shortlist:
                scored.insert(0, scored.pop(rank))
                best = scored[0]
                confidence = max(confidence, MODEL_TRUST)
                gap = top_gap(scored)
                model_decided = True
            else:
                logger.warning(
                    "model index %s is outside the shortlist; ignoring it",
                    evidence_ordinal,
                )

    ambiguous = False
    reason = ""
    if confidence < MIN_CONFIDENCE:
        ambiguous = True
        reason = f"no section scored above the confidence floor ({MIN_CONFIDENCE:.0f})"
    elif (
        lexical_gap is not None
        and lexical_gap < AMBIGUITY_MARGIN
        and not (model_decided and lexical_top_two[0].score < LEXICAL_SIGNAL_FLOOR)
    ):
        ambiguous = True
        first, second = lexical_top_two
        reason = (
            f"ambiguous: [{first.ordinal}] {first.label} scored "
            f"{first.score:.1f} but [{second.ordinal}] {second.label} "
            f"scored {second.score:.1f} (gap {lexical_gap:.1f} < "
            f"{AMBIGUITY_MARGIN:.1f})"
        ) + (
            f"; model chose [{best.ordinal}] but the ambiguity stands"
            if model_decided
            else ""
        )

    position = ResolvedPosition(
        ordinal=best.ordinal,
        confidence=round(confidence, 2),
        ambiguous=ambiguous,
        evidence=evidence,
        candidates=scored[:5],
        reason=reason,
    )

    if current_ordinal is not None:
        position = apply_forward_rule(position, current_ordinal)
    return position


def apply_forward_rule(
    position: ResolvedPosition, current_ordinal: int
) -> ResolvedPosition:
    """Progress only moves forward; a rewind is surfaced, not applied.

    ``MAX_SILENT_REWIND`` is honoured here rather than merely declared: it is
    the number of sections a reader may go backwards without being asked to
    confirm. At 0 (the default, and the only tested value) every rewind is
    surfaced, which is what the previous hardcoded ``ordinal >= current_ordinal``
    test did -- but it did so by ignoring the constant, so the documented
    threshold and the enforced one were only ever accidentally in agreement.
    """
    if position.ordinal is None:
        return position
    rewind = current_ordinal - position.ordinal
    if rewind <= MAX_SILENT_REWIND:
        return position
    return position.model_copy(
        update={
            "clamped_from": position.ordinal,
            "ordinal": current_ordinal,
            "needs_confirmation": True,
            "reason": (
                f"resolved ordinal {position.ordinal} is behind current "
                f"progress {current_ordinal}; held at {current_ordinal} "
                "pending confirmation"
            ),
        }
    )


def format_ambiguity_question(position: ResolvedPosition) -> str:
    """The question to put to the user when progress is ambiguous."""
    if len(position.candidates) < 2:
        return (
            "I couldn't work out where you are in this book — could you tell "
            "me which section you last finished?"
        )
    first, second = position.candidates[0], position.candidates[1]
    return (
        "I'm not sure where you are — two sections match what you said:\n"
        f"  {first.ordinal}. {first.display}\n"
        f"  {second.ordinal}. {second.display}\n"
        "Which one?"
    )


def summarize_toc(toc: Sequence[Section], limit: int = 20) -> str:
    """Short human-readable TOC listing for conversational UIs."""
    return format_toc(toc, limit=limit)


__all__ = [
    "AMBIGUITY_MARGIN",
    "MIN_CONFIDENCE",
    "Candidate",
    "ResolvedPosition",
    "apply_forward_rule",
    "candidate_text",
    "format_ambiguity_question",
    "format_toc",
    "resolve_position",
    "score_sections",
    "summarize_toc",
    "top_gap",
]
