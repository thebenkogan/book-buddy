"""LLM-assisted structural sectionization.

Why this is not one deterministic script
-----------------------------------------
Gutenberg wraps prose at a fixed width (~70 chars), so a wrapped prose line is
*also* a short standalone line. Measured on real books, a line-anchored regex
with a "followed by prose" guard yields 213 correct chapter headings and 1,329
false positives in the Jewish War, and 357 / 12,784 in Les Mis. No threshold
fixes this: the signal a heading carries (it is short and alone) is the same
signal a wrapped sentence carries.

So the split is:

1. ``propose_sections`` - deterministic, cheap, HIGH RECALL. Narrow the book
   to a few hundred candidate boundaries. Wrong answers here are recoverable.
2. ``adjudicate_sections`` - the LLM decides which candidates are real section
   starts and names them. Constrained selection: the model returns INTEGER
   INDICES into the candidate array it was shown, so it cannot invent an
   offset. Wrong answers here are NOT recoverable, which is why it is the only
   stage allowed to decide, and why stage 3 exists.
3. ``verify_toc`` - deterministic, never raises, returns problems. A wrong
   offset does not crash; it silently produces confident wrong answers, so the
   refusal to answer has to be explicit.

Determinism is kept wherever it is trustworthy (offsets are computed once, by
code, and are never round-tripped through the model) and handed to the model
only where it is not (is this line a heading, and what is it called).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.bookbuddy import chat, extract_json
from src.bookbuddy.book import (
    Book,
    Section,
    contents_sample,
    has_prose_after,
    iter_lines,
    measure_wrap_width,
    preview_at,
    span_has_prose,
    verify_book_structure,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stage 1: deterministic proposal (high recall)
# --------------------------------------------------------------------------- #

#: A line a structural heading might plausibly sit on.
SHORT_LINE_MAX = 90

#: Vocabulary that marks a structural heading regardless of the book's
#: numbering convention. This is a *prior*, not a rule - the LLM adjudicates.
STRUCTURAL_WORDS = re.compile(
    r"(?i)\b("
    r"chapter|book|part|volume|section|canto|act|scene|letter|episode|"
    r"booklet|appendix|preface|contents|prologue|epilogue"
    r")\b"
)

#: Numbered heading, in any convention seen in practice:
#: "CHAPTER 12.", "CHAPTER XII", "Chapter 3. Title", "BOOK FIRST--A JUST MAN",
#: "VOLUME I", "PART II".
NUMBERED_HEADING = re.compile(
    r"(?i)^(?P<word>chapter|book|part|volume|section|canto|act|scene|letter)\b"
    r"[\s:.,;-]*"
    r"(?P<num>[0-9]{1,3}|[ivxlc]{1,7}|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
)

#: A heading is not a heading if it is merely a line of prose.
SENTENCE_END = (".", "!", "?", '"', "”", "'", ":", ";")


def propose_sections(
    book: Book, max_candidates: int = 600
) -> List[Dict[str, Any]]:
    """Deterministically propose candidate section boundaries.

    Returns dicts ``{index, offset, line, prior}`` sorted by offset, where
    ``prior`` is 'strong' (numbered/structural vocabulary), 'weak' (short
    standalone line) or None (rejected).

    Recall matters more than precision here: a missed boundary is a section the
    reader can never address, whereas a spurious candidate is just noise the
    LLM discards.
    """
    wrap = measure_wrap_width(book)
    proposals: List[Dict[str, Any]] = []

    previous_nonblank: Optional[str] = None
    for line_start, stripped in iter_lines(book):
        # iter_lines anchors on the first non-whitespace character, so this is
        # the offset of the heading's first real character even when the book
        # indents it (Monte Cristo: " Chapter 2. Father and Son").
        offset = line_start
        line_end = line_start + len(stripped)
        prior: Optional[str] = None

        if 2 <= len(stripped) <= SHORT_LINE_MAX:
            numbered = bool(NUMBERED_HEADING.match(stripped))
            structural = bool(STRUCTURAL_WORDS.search(stripped))
            # A line at the wrap width is prose that happens to break there.
            at_wrap_width = len(stripped) >= wrap - 4
            begins_paragraph = (
                previous_nonblank is not None
                and previous_nonblank.endswith(SENTENCE_END)
            )
            if numbered:
                prior = "strong"
            elif structural and begins_paragraph and not at_wrap_width:
                prior = "strong"
            elif begins_paragraph and not at_wrap_width and not stripped.endswith(
                (",", ";", "—")
            ):
                prior = "weak"
            elif at_wrap_width:
                prior = None

        if prior and has_prose_after(book, line_end):
            proposals.append(
                {
                    "offset": offset,
                    "line": stripped,
                    "prior": prior,
                    "preview": preview_at(book, line_end),
                }
            )
        previous_nonblank = stripped

    # Keep every strong candidate; fill the budget with the best weak ones.
    strong = [p for p in proposals if p["prior"] == "strong"]
    weak = [p for p in proposals if p["prior"] == "weak"]
    if len(strong) + len(weak) <= max_candidates:
        kept = proposals
    else:
        budget = max(max_candidates - len(strong), 0)
        kept = strong + weak[:budget]
    kept.sort(key=lambda item: item["offset"])

    kept = _drop_toc_cluster(book, kept)
    for index, item in enumerate(kept, start=1):
        item["index"] = index
    logger.info(
        "propose_sections(%s): wrap=%d, strong=%d, weak=%d, kept=%d",
        book.book_id,
        wrap,
        len(strong),
        len(weak),
        len(kept),
    )
    return kept


def _drop_toc_cluster(
    book: Book, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Remove candidates that sit inside the book's own contents listing.

    A table of contents is a dense run of heading-shaped lines with almost no
    prose between them. Real sections are separated by running text. In all
    three test books the false positives cluster at tiny offsets ("CHAPTER 13."
    at offset 359 in the Jewish War is a contents line, not a section) and
    dropping them here means the LLM is never asked about them.

    Deterministic on purpose: this is the exact bug that produced a 0.08%
    slice of the Jewish War, so it must not depend on a model being careful.
    """
    if not candidates:
        return candidates

    gap_limit = 200
    kept: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []

    def flush() -> None:
        if len(run) >= 4:
            logger.debug("dropping %d table-of-contents candidates", len(run))
            return
        kept.extend(run)

    previous_offset: Optional[int] = None
    for candidate in candidates:
        if previous_offset is not None and candidate["offset"] - previous_offset < gap_limit:
            # Adjacent heading-shaped lines with NO running prose between them
            # are contents entries. A real section always has body text in the
            # gap: BOOK I. and CHAPTER 1. in the Jewish War are only 184
            # characters apart, but the volume's title line sits between them.
            if not span_has_prose(book, previous_offset, candidate["offset"]):
                run.append(candidate)
            else:
                flush()
                run = [candidate]
        else:
            flush()
            run = [candidate]
        previous_offset = candidate["offset"]
    flush()

    # The front matter and the contents listing live before the real text
    # begins. Locating that boundary is what stops "CHAPTER 13." at offset 359
    # from becoming a section; no gap heuristic can do it, because front matter
    # is prose-dense and looks like body text.
    kept = [c for c in kept if c["offset"] >= _body_start(book, kept)]
    return kept


def _body_start(book: Book, candidates: List[Dict[str, Any]]) -> int:
    """Offset where the real text begins, ignoring the contents listing.

    A contents listing is a tight run of candidates packed far closer together
    than the book's own rhythm; the body starts at the first candidate after
    the LAST such run. Falls back to the first prose-rich candidate so a book
    with no listing still works.

    The listing run is a run of consecutive SMALL GAPS, so the boundary has to
    be derived by walking gaps and landing on a candidate's offset. The
    previous version did ``listing_end = tight[-1]`` and then compared
    ``candidate["offset"] > listing_end`` -- a gap LENGTH against an ABSOLUTE
    OFFSET. Those are different units, so the comparison was meaningless: on
    the Jewish War the tight gaps are ~19 chars and the candidate offsets start
    in the thousands, so ``offset > 19`` held for every candidate and the
    listing branch was indistinguishable from the fallback. The log line it
    wrote ("listing run ending %d") printed a gap where it claimed an offset.
    """
    if not candidates:
        return 0

    gaps = [
        candidates[i + 1]["offset"] - candidates[i]["offset"]
        for i in range(len(candidates) - 1)
    ]
    listing_end: Optional[int] = None
    if gaps:
        median = sorted(gaps)[len(gaps) // 2]
        if median > 0:
            tight_threshold = max(median // 4, 1)
            # Longest run of CONSECUTIVE tight gaps. gaps[i] spans
            # candidates[i] and candidates[i+1], so a run i in [start, end]
            # covers candidates[start .. end+1], and the body begins at the
            # next candidate along.
            best_len = 0
            best_end = -1
            run_start = None
            for i, gap in enumerate(gaps):
                if gap < tight_threshold:
                    if run_start is None:
                        run_start = i
                    if i - run_start + 1 > best_len:
                        best_len = i - run_start + 1
                        best_end = i
                else:
                    run_start = None
            if best_len >= 5 and best_end + 2 < len(candidates):
                listing_end = candidates[best_end + 2]["offset"]

    if listing_end is not None:
        for candidate in candidates:
            if candidate["offset"] >= listing_end and span_has_prose(
                book, candidate["offset"], candidate["offset"] + 6000, need=4
            ):
                logger.debug(
                    "body starts at %d (after listing run ending at %d)",
                    candidate["offset"],
                    listing_end,
                )
                return candidate["offset"]

    for candidate in candidates:
        if span_has_prose(book, candidate["offset"], candidate["offset"] + 6000, need=4):
            return candidate["offset"]
    return candidates[0]["offset"]


# --------------------------------------------------------------------------- #
# Stage 2: LLM adjudication (constrained selection)
# --------------------------------------------------------------------------- #

ADJUDICATE_SYSTEM = (
    "You identify the real section boundaries of a classic book. You are shown "
    "a numbered list of candidate lines, each with the opening words that "
    "follow it. Decide which candidates are genuine section headings and give "
    "each a short title. Wrapped prose lines are NOT headings. A contents "
    "listing is NOT a heading. Output one line per accepted heading, in the "
    "order given, formatted as '<index>|<short title>', and nothing else. "
    "Never invent an index. Never summarise plot events that happen later in "
    "the book."
)

ADJUDICATE_PROMPT = """\
Book: {title}

{context}

Candidate lines (choose which are genuine section headings):

{candidates}

Output one line per heading you accept, in the order shown, each formatted as:
<index>|<short title, 3-12 words>
and nothing else.
"""


def adjudicate_sections(
    book: Book,
    candidates: Sequence[Dict[str, Any]],
    client: Optional[Any] = None,
) -> List[Section]:
    """Ask the model which candidates are real sections, then build the TOC.

    Offsets come from stage 1 and are never round-tripped through the model.
    The model chooses *which* offsets are boundaries and what to call them.
    """
    if not candidates:
        return []

    context = _context_note(contents_sample(book))
    lines = [
        f"[{c['index']}] {c['line']}\n    {c['preview'][:240]}"
        for c in candidates
    ]
    prompt = ADJUDICATE_PROMPT.format(
        title=book.title,
        context=context,
        candidates="\n".join(lines),
    )

    response_text: Optional[str] = None
    try:
        if client is not None:
            response_text = client(prompt, ADJUDICATE_SYSTEM)
        else:
            response_text = chat(
                prompt,
                system=ADJUDICATE_SYSTEM,
                temperature=0.0,
                max_tokens=16000,
                reasoning_effort="low",
            )["content"]
    except Exception:
        logger.exception(
            "adjudication call failed for %s; falling back to strong priors",
            book.book_id,
        )

    titles = _parse_adjudication(response_text or "", candidates)
    return _sections_from(candidates, titles)


def _context_note(sample: str, limit: int = 1200) -> str:
    """Tell the adjudicator where the contents listing begins.

    Without this the model has no way to tell "CHAPTER 4. Conspiracy" in the
    listing from the real one, and would happily accept both - which is exactly
    the failure this pipeline exists to prevent.
    """
    if not sample:
        return "No table of contents was found in this text."
    return (
        "A table-of-contents listing begins here; lines that merely repeat it "
        f"are not headings:\n{sample[:limit]}"
    )


def _parse_adjudication(
    response_text: str, candidates: Sequence[Dict[str, Any]]
) -> Dict[int, str]:
    """Parse ``index|title`` lines, falling back to a JSON array."""
    by_index = {c["index"]: c for c in candidates}
    titles: Dict[int, str] = {}

    for raw_line in response_text.splitlines():
        line = raw_line.strip().lstrip("-* \t").strip()
        if not line or "|" not in line:
            continue
        head, _, tail = line.partition("|")
        digits = re.match(r"[\[\(]?\s*(\d+)", head)
        if not digits:
            continue
        index = int(digits.group(1))
        title = tail.strip().strip('",').strip()
        if index in by_index and title:
            titles.setdefault(index, title)

    if titles:
        return titles

    payload = extract_json(response_text)
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("index")
            index = int(raw) if isinstance(raw, (int, str)) and str(raw).strip(
                "-"
            ).isdigit() else None
            title = str(entry.get("title") or "").strip()
            if index is not None and index in by_index and title:
                titles.setdefault(index, title)
    return titles


def _sections_from(
    candidates: Sequence[Dict[str, Any]], titles: Dict[int, str]
) -> List[Section]:
    """Build a flattened TOC from accepted candidates.

    Ordinals are 1..N over the accepted list, assigned here and never by the
    model. A candidate the model did not name is dropped; a candidate it named
    but which is absent is impossible (it can only echo indices shown).
    """
    accepted = [c for c in candidates if c["index"] in titles]
    accepted.sort(key=lambda c: c["offset"])

    sections: List[Section] = []
    parent = ""
    for ordinal, candidate in enumerate(accepted, start=1):
        line = str(candidate["line"])
        label, kind = _classify(line)
        if kind == "container" and not label:
            label = line
        if kind == "container":
            parent = line
            full_label = line
        else:
            full_label = f"{parent} / {line}" if parent else line
        sections.append(
            Section(
                ordinal=ordinal,
                label=full_label,
                parent=parent if kind != "container" else "",
                offset=int(candidate["offset"]),
                title=titles.get(candidate["index"], ""),
            )
        )
    return sections


CONTAINER_RE = re.compile(
    r"(?i)^(?P<word>book|volume|part|canto)\s*"
    r"[0-9ivxlc]{0,7}\s*(?:[-—:.].*)?$"
)


def _classify(line: str) -> Tuple[str, str]:
    """Return ``(label, kind)`` where kind is 'container' or 'unit'."""
    stripped = line.strip()
    if CONTAINER_RE.match(stripped) and not re.match(
        r"(?i)^(chapter|section|act|scene|letter|episode)\b", stripped
    ):
        return stripped, "container"
    return stripped, "unit"