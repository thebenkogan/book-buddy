"""Book loading, TOC construction/verification, and the spoiler gate.

The single most important rule in this module: :func:`text_up_to` is the ONLY
function permitted to slice ``Book.text`` into a prompt. ``tests/test_book.py``
asserts mechanically that no other module does.

Design notes
------------
* Gutenberg boilerplate is stripped with regexes anchored on the
  ``*** START/END OF THE PROJECT GUTENBERG EBOOK ... ***`` markers.
* Structural headings (``BOOK IV.`` / ``CHAPTER 12.``) are only accepted when
  they stand alone on their own line **and** are followed by real prose. A naive
  ``text.find("BOOK IV.")`` matches the TABLE OF CONTENTS at byte ~1110 and
  yields a 0.08% slice of the book — a silent, confident, wrong answer. There is
  a regression test pinning that exact bug.
* Ordinals are 1..N over the FLATTENED section list. In the Jewish War
  ``CHAPTER 1.`` occurs 7 times, so a bare chapter number is meaningless.
* :func:`verify_toc` never raises. It returns a list of problems, because a
  wrong offset does not crash — it silently produces confident wrong answers,
  which is the worst failure mode this tool has.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.bookbuddy import cache_dir, chat, extract_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Gutenberg stripping
# --------------------------------------------------------------------------- #

GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.DOTALL,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.DOTALL,
)

# --------------------------------------------------------------------------- #
# Structural headings
# --------------------------------------------------------------------------- #

#: A heading must occupy its own line with nothing else on it. Leading
#: whitespace is NOT allowed: Gutenberg table-of-contents lines are indented
#: (" BOOK I."), real headings are flush left.
# Numbering conventions seen in real Gutenberg texts, all measured:
#   jewish_war  "BOOK I." / "CHAPTER 12."   117 headings
#   pnp         "Chapter I."                61 headings  (mixed case + "]")
#   war_peace   "BOOK ONE: 1805" / "CHAPTER I"   380 headings (no trailing period)
#   sherlock    "I." / "II. THE RED-HEADED LEAGUE"  12 headings (bare roman)
#
# Built by string concatenation on purpose. Written as an rf-string, the
# quantifier in "[^\n]{0,70}" rendered as the literal capture group "(0, 70)",
# which silently disabled the whole "roman + title" branch and made Sherlock
# look undetectable -- a failure that reads as "the book is unusual" rather
# than "the regex is broken".
ROMAN_RE = r"[IVXLCDM]{1,9}"
ARABIC_RE = r"[0-9]{1,4}"
WORD_NUM_RE = (
    r"ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|"
    r"FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY"
)
SECTION_NUM_RE = (
    r"(?:" + ARABIC_RE + r"|" + ROMAN_RE + r"|(?:" + WORD_NUM_RE + r"))"
)

# Four failure categories found by benchmarking 24 unseen Gutenberg books
# (see tests/test_heading_detector.py). Three are regex-shaped and fixed here:
#   title on the same line   "Chapter I. Into the Primitive"   (call of the wild)
#   other structural words   "Canto 1", "PART ONE"             (divine comedy, ti)
#   bare roman, no keyword   "XII"                            (treasure island)
# The fourth -- all-caps headings with NO numbering, e.g. Grimms' 136 lines
# like "THE GOLDEN BIRD" -- is deliberately NOT attempted. A regex cannot tell a
# story title from an all-caps sentence; that judgement needs a model, and
# pretending otherwise is how a TOC ends up with 136 fabricated sections.
_HEADING_KEYWORD = (
    r"BOOK|Book|CHAPTER|Chapter|PART|Part|Canto|CANTO|Act|ACT|Scene|SCENE|"
    r"Volume|VOLUME|Section|SECTION|Lesson|LESSON|Letter|LETTER"
)
HEADING_RE = re.compile(
    r"(?m)^(?P<label>"
    r"(?:" + _HEADING_KEYWORD + r")\s+(?:" + SECTION_NUM_RE + r")"
    r"(?::[ \t]*[0-9]{4}|\.[ \t]*)?"
    r"|(?:" + _HEADING_KEYWORD + r")\s+(?:" + SECTION_NUM_RE + r")\.[ \t]+"  # title tail
    r"[A-Z0-9][^\n]{0,70}"
    r"|" + ROMAN_RE + r"\.[ \t]+[A-Z0-9][^\n]{0,70}"
    r"|" + ROMAN_RE + r"[ \t]*\.[ \t]*"
    r")[ \t]*\]?[ \t]*$"
)
#: A bare roman numeral alone on its line ("XII"). Needed for Treasure Island,
#: whose chapter headings carry no keyword at all -- but it is the most
#: dangerous shape in the file, because a lone "XII" also appears as a page
#: number or a list marker. Only accepted when the line stands alone between
#: blank lines, which is why this is a separate regex checked per line.
BARE_ROMAN_RE = re.compile(r"^(?P<label>" + ROMAN_RE + r")[ \t]*$")

#: Same, but tolerating indentation. Used only to *detect* (and then reject)
#: table-of-contents clusters, never to produce an offset.
INDENTED_HEADING_RE = re.compile(
    r"(?m)^[ \t]+(?P<label>"
    r"(?:BOOK|Book|CHAPTER|Chapter)\s+(?:" + SECTION_NUM_RE + r")"
    r"(?::[ \t]*[0-9]{4}|\.[ \t]*)?"
    r")" r"[ \t]*$"
)

#: Prose: an ordinary wrapped line of running text. Deliberately permissive —
#: we are trying to distinguish "a paragraph of English" from "another line of
#: a contents listing", and real book text contains every punctuation mark.
PROSE_LINE_RE = re.compile(r"^[^\s].{39,}$")

#: How far past a candidate heading we look for real prose before deciding the
#: candidate is a table-of-contents entry rather than a heading.
PROSE_WINDOW = 1500
PROSE_LINES_REQUIRED = 2
PROSE_MIN_LEN = 40

#: How far past an offset we look for the heading line when verifying a TOC.
OFFSET_HEADING_TOLERANCE = 200


class Book(BaseModel):
    """A loaded book. ``text`` is post-Gutenberg-strip and never sliced
    outside :func:`text_up_to`."""

    book_id: str
    title: str
    text: str
    path: Optional[str] = None
    raw_len: int = 0
class Section(BaseModel):
    """One flattened TOC entry.

    ``ordinal`` is the position in the flattened 1..N list and is the ONLY
    representation of reading progress. ``label`` is the human label
    ("Book IV, Chapter 3"), ``parent`` the enclosing book heading, and
    ``offset`` a character index into ``Book.text``.
    """

    ordinal: int
    label: str
    parent: str = ""
    offset: int
    title: str = ""

    @property
    def is_book(self) -> bool:
        """True only for a book heading, not for a chapter *inside* a book.

        Note this cannot be ``label.startswith("BOOK")``: a chapter's label is
        "BOOK II. / CHAPTER 1.", which starts with BOOK. Getting this wrong
        makes "book 2" a four-way tie between the book and its chapters.
        """
        return bool(
            re.fullmatch(
                r"BOOK\s+(?:[0-9]{1,3}|[IVXLC]{1,7})\.?", self.label.strip().upper()
            )
        )


class TocProblem(BaseModel):
    """A single verification failure. Rendered to a plain string by
    :func:`verify_toc` for caller convenience."""

    ordinal: Optional[int] = None
    code: str
    detail: str

    def __str__(self) -> str:
        where = f"ordinal={self.ordinal}" if self.ordinal is not None else "toc"
        return f"[{self.code}] {where}: {self.detail}"


def strip_gutenberg(raw: str) -> str:
    """Return the book body with Gutenberg header/footer removed.

    Raises ``ValueError`` if the markers are missing — a book we cannot strip is
    a book whose offsets we cannot trust.
    """
    start = GUTENBERG_START.search(raw)
    if not start:
        raise ValueError("no Gutenberg START marker found")
    end = GUTENBERG_END.search(raw)
    if not end:
        raise ValueError("no Gutenberg END marker found")
    if end.start() < start.end():
        raise ValueError("Gutenberg END marker precedes START marker")
    return raw[start.end() : end.start()].strip("\n")


def title_from_markers(raw: str) -> str:
    """Best-effort title extraction from the START marker line."""
    match = GUTENBERG_START.search(raw)
    if not match:
        return ""
    line = match.group(0)
    inner = re.search(r"EBOOK\s+(.*?)\s*\*\*\*", line, re.DOTALL)
    if not inner:
        return ""
    return re.sub(r"\s+", " ", inner.group(1)).strip()


def load_book(path: str, book_id: Optional[str] = None) -> Book:
    """Load a Gutenberg plain-text file and strip its boilerplate."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
    except OSError:
        logger.exception("could not read book file %s", path)
        raise
    text = strip_gutenberg(raw)
    title = title_from_markers(raw) or os.path.splitext(os.path.basename(path))[0]
    resolved_id = book_id or os.path.splitext(os.path.basename(path))[0]
    return Book(
        book_id=resolved_id,
        title=title,
        text=text,
        path=os.path.abspath(path),
        raw_len=len(raw),
    )


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #


def _has_prose_near(
    text: str, offset: int, window: int = PROSE_WINDOW, need: int = PROSE_LINES_REQUIRED
) -> bool:
    """True if real running prose follows ``offset`` within the window.

    A table-of-contents entry is always followed by more short heading-like
    lines; a real heading is followed by paragraphs.
    """
    window_text = text[offset : offset + window]
    found = 0
    for line in window_text.split("\n"):
        stripped = line.strip()
        if len(stripped) >= PROSE_MIN_LEN and PROSE_LINE_RE.match(stripped):
            found += 1
            if found >= need:
                return True
    return False


#: Below this many sections per 100k characters, treat the deterministic result
#: as suspect. Measured over 24 unseen Gutenberg books: books with real chapter
#: structure land between roughly 1.5 and 40 sections per 100k characters,
#: while the ones that defeat the regex (Grimms, Moby Dick, Modest Proposal)
#: land at zero. 0.5 sits below every book that genuinely has chapters.
SECTIONS_PER_100K_FLOOR = 0.5


def toc_needs_model(sections, text_len: int, audit_issues: int = 0):
    """Should a model adjudicate this TOC, or is the deterministic one good?

    Returns ``(needs_model, reason)``. The point is that this decision is made
    by a rule rather than by whoever happens to be looking, because the failure
    mode is silent: a book whose chapters were never found still produces a
    confident, clean-looking TOC. Measured on 24 unseen Gutenberg books the
    deterministic path alone passed on 17 and failed on 7 -- and the failures
    are precisely the books that look healthiest by section count.

    Two signals, because either alone gives false alarms:

    1. nothing was found, or the audit objected;
    2. section density is below what any genuinely chaptered book shows.
    """
    count = len(sections)
    if count == 0:
        return True, ("no headings matched at all -- either an unrecognised "
                      "convention or a book with no chapters")
    if audit_issues:
        # Accept an int or the issue objects themselves; callers pass whichever
        # they have, and iterating an int is the kind of mistake that only
        # shows up on the books that need the model.
        try:
            codes = sorted({getattr(i, "code", "?") for i in audit_issues})
        except TypeError:
            codes = []
        return True, f"audit reported {audit_issues} issue(s): {codes}"
    density = count / max(text_len, 1) * 100_000
    if density < SECTIONS_PER_100K_FLOOR:
        return True, (f"only {count} sections in {text_len:,} chars "
                      f"({density:.2f} per 100k, below the "
                      f"{SECTIONS_PER_100K_FLOOR} floor that every chaptered "
                      f"book clears) -- chapters are probably being missed")
    return False, f"{count} sections at {density:.2f} per 100k chars, audit clean"


def find_headings(text: str) -> List[Dict[str, Any]]:
    """Locate structural headings in ``text``.

    Returns dicts with ``label``, ``offset`` and ``kind`` (``"book"`` or
    ``"chapter"``), sorted by offset.

    Two independent guards keep table-of-contents entries out:

    1. the regex requires the heading alone on its line, flush left (TOC lines
       are indented);
    2. the candidate must be followed by real prose within
       ``PROSE_WINDOW`` characters.

    Guard 2 is what makes this safe for books whose TOC is flush left.
    """
    headings: List[Dict[str, Any]] = []
    tailed: List[Dict[str, Any]] = []
    for match in HEADING_RE.finditer(text):
        label = " ".join(match.group("label").split())
        offset = match.start()
        if not _has_prose_near(text, match.end()):
            continue
        if any(h["offset"] == offset for h in headings):
            continue
        entry = {
            "label": label,
            "offset": offset,
            "kind": "book" if label.upper().startswith("BOOK") else "chapter",
        }
        headings.append(entry)
        line_end = text.find(chr(10), offset)
        line = text[offset:line_end if line_end > 0 else None]
        if _HEADING_TAIL_RE.match(line):
            tailed.append(entry)

    # A heading that carries its title on the same line is exactly what a
    # table-of-contents entry looks like, so this branch cannot use the
    # "alone on its line" guard that keeps the plain shape safe. Dracula is
    # the measured case: its contents lists 27 entries ("CHAPTER I.
    # Jonathan Harker's Journal") inside ~1000 characters, and without this
    # the TOC came back as 54 sections -- the 27 real chapters plus the 27
    # listing entries. Only dense runs among tailed headings are dropped:
    # real chapters never sit 40 characters apart.
    dropped = {id(h) for h in _drop_dense_runs(tailed)}
    if dropped:
        headings = [h for h in headings if id(h) not in dropped]

    # Bare romans need line context, so that pass walks lines rather than
    # regex-matching the whole text -- and it runs ONLY when the book's own
    # keyword convention came up nearly empty. Measured across 24 unseen
    # books: enabling it unconditionally took Dracula from 27 correct sections
    # to 54 with 26 spurious ones, while Treasure Island, whose chapters are
    # bare numerals under only six "PART ONE" headings, needs it. The rule
    # "bare roman wins only where keywords nearly fail" separates those two.
    if len(headings) < BARE_ROMAN_MAX_KEYWORD_HEADINGS:
        _add_bare_roman_headings(text, headings)

    headings.sort(key=lambda item: item["offset"])
    return headings


#: The bare-roman branch runs only when the keyword convention found fewer
#: headings than this, i.e. the book has effectively no keyword headings.
#: Measured across 24 unseen books: Treasure Island has six ("PART ONE".."PART
#: SIX") for 34 chapters and needs the branch; Dracula has 27 real ones and must
#: not get it, or it goes from 27 correct sections to 54 with 26 spurious. An
#: earlier version scaled this against the bare-roman count instead, which did
#: not separate them -- Dracula also contains many scattered numerals.
BARE_ROMAN_MAX_KEYWORD_HEADINGS = 8

#: The shape of a heading that carries its title on the same line.
_HEADING_TAIL_RE = re.compile(
    r"^(?P<label>(?:BOOK|Book|CHAPTER|Chapter|PART|Part|Canto|CANTO|Act|ACT|"
    r"Scene|SCENE|Volume|VOLUME|Section|SECTION|Lesson|LESSON|Letter|LETTER)"
    r"\s+(?:[0-9]{1,4}|[IVXLCDM]{1,9}|[0-9]{1,4})\.[ \t]+\S)"
)


def _drop_dense_runs(entries, gap: int = 200, run_length: int = 4):
    """Entries belonging to a contents listing: ``run_length``+ in a row, each
    within ``gap`` characters of the last."""
    dropped = []
    run = []
    for entry in entries:
        if run and entry["offset"] - run[-1]["offset"] < gap:
            run.append(entry)
            continue
        if len(run) >= run_length:
            dropped.extend(run)
        run = [entry]
    if len(run) >= run_length:
        dropped.extend(run)
    return dropped


def _add_bare_roman_headings(text: str, headings: List[Dict[str, Any]]) -> None:
    """Add headings that are a lone roman numeral ("XII"), for Treasure Island.

    The most dangerous shape in any book: a bare "XII" is also how a page
    number or a list marker appears, so two conditions have to hold -- the
    numeral alone on its line, and real prose following it. Requiring blank
    lines on BOTH sides was measured to reject all 34 of Treasure Island's
    chapters: the blank line is above the numeral and the prose begins
    immediately below it. Blank-above alone is the weaker of the two signals,
    which is why this stays a separate branch the audit re-checks.
    """
    lines = text.split("\n")
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1

    for i, line in enumerate(lines):
        if line != line.strip() or not BARE_ROMAN_RE.match(line):
            continue
        if i == 0 or i == len(lines) - 1:
            continue
        start = starts[i] + len(line) + 1
        # A heading opens a block: require a blank line above it, or the very
        # start of the text.
        if i > 0 and lines[i - 1].strip():
            continue
        if not _has_prose_near(text, start):
            continue
        headings.append(
            {
                "label": line.strip(),
                "offset": starts[i],
                "kind": "chapter",
            }
        )


def _is_title_shaped(line: str) -> bool:
    """True when a line reads as a heading title rather than a sentence.

    Measured, the discriminator is the share of WORDS that begin with a
    capital, not the share of capital letters. The Jewish War sets its titles in
    title case ("Concerning The Successors Of Judas, Who Were Jonathan And
    Simon"): every word is capitalised, but the function words still contain
    lowercase letters, so a character-level test rejects the real titles.
    Measured values -- word-initial capitals 1.00 on real titles against
    0.17-0.38 on body prose; character-level lowercase 0.76-0.81 against
    0.91-0.97, which separates but with far less margin. 0.7 is the midpoint.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'\u2019]*", line)
    if len(words) < 3:
        return False
    return sum(1 for w in words if w[0].isupper()) / len(words) >= 0.7


def derive_title(text: str, offset: int, max_chars: int = 170) -> str:
    """The descriptive title for a heading, taken from the text that follows.

    Measured: on the Jewish War this returns a title for 117 of 117 headings,
    with no model call. A heading is usually a bare label ("CHAPTER 28.") whose
    title wraps onto the next one or two lines, so the title is already in the
    book -- reading it is cheaper and more faithful than asking a model to
    paraphrase it, and it cannot invent a chapter that is not there.

    The following text is only accepted when it is actually title-SHAPED, not
    merely the next line. Measured across four books: the Jewish War titles its
    chapters in full capitals ("CHAPTER 28." / "HOW ANTIPATER IS HATED OF ALL
    MEN"), so requiring capitals recovers 117/117 real titles. Pride and
    Prejudice, War and Peace and Sherlock leave their chapters untitled, and an
    unchecked rule invents titles out of the opening sentence of body prose
    ("Anna Pavlovna's drawing room was gradually filling"). An empty title is
    honest; a fabricated one is a spoiler risk in the UI, so the capital test
    is what separates the two cases.

    Returns "" when the heading carries no following title text.
    """
    start = offset + len(text[offset : text.find("\n", offset)].lstrip("\n"))
    if start < 0 or start >= len(text):
        return ""

    pieces: List[str] = []
    for line in text[start : start + max_chars + 200].split("\n")[:6]:
        stripped = line.strip()
        if not stripped:
            continue
        # The next heading means this one had no title of its own.
        if HEADING_RE.match(line) or INDENTED_HEADING_RE.match(line):
            break
        if re.match(r"^(BOOK|Chapter|CHAPTER)\b", stripped, re.I):
            break
        if re.match(r"^[0-9]{1,3}\.?\s", stripped):
            break          # narrative numbering, not a title
        if PROSE_LINE_RE.match(stripped) and len(pieces) >= 2:
            break          # already into the body prose
        if not _is_title_shaped(stripped):
            break          # untitled chapter: the next line is body prose
        pieces.append(stripped)
        if sum(len(p) for p in pieces) > max_chars:
            break
    return " ".join(pieces).strip(" .,;")


def toc_region_end(text: str, headings: Optional[Sequence[str]] = None) -> int:
    """Character offset where the table of contents ends.

    Defined as the offset of the first structural heading. Everything before it
    is front matter plus the TOC listing; no section offset may land there.

    ``headings`` lets a caller supply the heading lines a TOC was built from.
    Without it we fall back to the strict ``BOOK IV.`` / ``CHAPTER 12.`` shape,
    which only some books use - so for a book like Les Mis that fallback finds
    nothing and would (wrongly) declare the entire text to be contents. When no
    shape is recognised, return 0 and let the per-offset heading check do the
    work instead of condemning every offset in the book.
    """
    if headings:
        return first_heading_offset(text, headings) or 0

    strict = find_headings(text)
    if strict:
        return int(strict[0]["offset"])

    # No recognisable structural heading: do not assert a region. Returning
    # len(text) here would mark every section as "inside the contents", which
    # is a rejection without evidence.
    return 0


def first_heading_offset(text: str, headings: Sequence[str]) -> Optional[int]:
    """Offset of the first line matching any heading in ``headings``."""
    wanted = {_loose_match_key(h) for h in headings if h}
    if not wanted:
        return None
    for match in re.finditer(r"(?m)^(\S.{0,120}?)[ \t]*$", text):
        if _loose_match_key(match.group(1).strip()) in wanted:
            return match.start()
    return None


def _loose_match_key(value: str) -> str:
    """Normalise a heading line so a label and its text can be compared.

    Brackets matter here and their absence was a real bug: Pride and Prejudice
    writes its first chapter heading as "Chapter I.]" while the TOC label is
    "Chapter I.". Without stripping brackets the two keys differ, so
    ``first_heading_offset`` skipped the genuine first chapter, reported
    Chapter II as the first structural heading, and then flagged the real
    Chapter I as "inside the contents region" -- a chapter that verify_toc
    correctly refused to accept.
    """
    value = value.upper()
    for char in "—–-_.,:;'\"[]()":
        value = value.replace(char, " ")
    return " ".join(value.split())


def heading_line_at(text: str, offset: int) -> str:
    """The line ``offset`` sits on, or the nearest standalone line after it.

    The line an offset is INSIDE takes precedence over any line that starts
    later. Some books indent their headings (" Chapter 2. Father and Son"), so
    an offset can legitimately land in that indent -- one character before the
    heading begins. Scanning forward from there skipped the heading entirely
    and returned the first flush-left line below it, which is prose. On the
    indented fixture that returned 'He turned towards the ship.' for an offset
    that is 1 char into " Chapter 2. Father and Son".

    The old ordering made that fallback unreachable in practice: the forward
    scan runs first and virtually always finds some later line, so the
    containing-line branch only fired on the last line of the book. The
    containing line is also the *nearer* line, so preferring it is the correct
    precedence and not a loosening -- a mid-chapter offset still lands on a
    prose line, which is what keeps it from reading as a heading.

    The branch also has to apply at ``line_start == offset``, not only when the
    offset is strictly inside the line. An indented heading's own offset points
    AT its leading space, so ``line_start < offset`` was False and the fallback
    did not run for the precise case it exists to handle; the forward scan then
    skipped the indent and returned the prose line below. Hence ``<=``, and
    hence the strip(): for an indented heading this returns the heading with its
    indent removed rather than stepping past it.
    """
    line_start = text.rfind("\n", 0, offset) + 1
    if 0 <= offset < len(text) and line_start <= offset:
        end_of_line = text.find("\n", line_start)
        if end_of_line < 0:
            end_of_line = len(text)
        candidate = text[line_start:end_of_line]
        if candidate.strip():
            return candidate.strip()

    tail = text[offset : offset + OFFSET_HEADING_TOLERANCE]
    for match in re.finditer(r"(?m)^(\S.*)$", tail):
        return match.group(1).strip()
    return ""


def nearest_heading_line(text: str, offset: int) -> str:
    """Diagnostic helper: what short line actually sits near ``offset``."""
    start = max(0, offset - OFFSET_HEADING_TOLERANCE)
    for match in re.finditer(r"(?m)^(\S.{1,90}?)[ \t]*$", text[start : offset + 200]):
        return match.group(1).strip()
    return ""


def looks_like_heading(
    text: str,
    offset: int,
    heading_line: Optional[str] = None,
    tolerated: Sequence[str] = (),
) -> bool:
    """True if a plausible heading line sits at (or just before) ``offset``.

    Two ways to satisfy this:

    * the strict ``BOOK IV.`` / ``CHAPTER 12.`` shape, which is what the
      original books use; or
    * ``offset`` lands on a short standalone line, optionally confirmed
      against ``heading_line`` or an entry in ``tolerated``.

    The second path exists because real books do not share a convention: Les
    Mis writes "CHAPTER I—M. MYRIEL" and Monte Cristo writes "Chapter 1.
    Father and Son". Verification must not encode one book's dialect, or every
    other book is rejected as malformed.

    The strict path accepts a heading found AT OR BEFORE ``offset`` only.
    The previous test was ``<= offset + OFFSET_HEADING_TOLERANCE``, which is
    vacuously true for every match in ``window`` -- ``window`` is sliced to end
    at exactly that offset. So any offset landing up to 200 chars AFTER a real
    heading verified clean, which is the "mid-chapter offset" case the check
    exists to reject. A heading ahead of the offset is evidence the offset is
    in the PREVIOUS section's body, not that it is a section start.
    """
    start = max(0, offset - OFFSET_HEADING_TOLERANCE)
    window = text[start : offset + OFFSET_HEADING_TOLERANCE]
    for match in HEADING_RE.finditer(window):
        if start + match.start() <= offset:
            return True
    for match in INDENTED_HEADING_RE.finditer(window):
        if start + match.start() <= offset:
            return True

    # Generic path: when the caller supplies the heading it expects here, the
    # line at the offset must actually match it. Requiring the match is what
    # keeps this a real check - accepting any short standalone line would let a
    # mid-chapter offset masquerade as a section boundary.
    line = heading_line_at(text, offset)
    if heading_line:
        wanted = heading_line.split("/")[-1].strip()
        if not wanted or not line:
            return False
        if _loose_match(wanted, line):
            return True
        # A heading line can wrap onto the next line; accept when the tail of
        # the expected heading starts the line at the offset.
        return _loose_match(wanted.split(" ")[0], line.split(" ")[0])

    for accepted in tolerated:
        if accepted and line and _loose_match(
            accepted.split("/")[-1].strip(), line
        ):
            return True

    if line and len(line) <= 120:
        start_of_line = text[offset : offset + len(line)]
        if start_of_line.strip() == line and _has_prose_near(
            text, offset + len(line), window=400, need=1
        ):
            return True
    return False


def _loose_match(wanted: str, actual: str) -> bool:
    """Compare two heading strings ignoring case, dashes and spacing."""
    def norm(value: str) -> str:
        value = value.upper()
        for char in "—–-_.,:;'\"":
            value = value.replace(char, " ")
        return " ".join(value.split())

    return norm(wanted) == norm(actual)


# --------------------------------------------------------------------------- #
# TOC building (one LLM call, cached to disk)
# --------------------------------------------------------------------------- #

_TOC_SYSTEM = (
    "You label the structural sections of a classic book. You are given a "
    "numbered list of section headings already located in the text. For each "
    "section output ONE line of the form '<index>|<short title>'. The title "
    "must be 3-12 words taken from the opening of that section. Output one "
    "line per input section, in the same order, and nothing else — no "
    "preamble, no commentary, no code fences. Never write about plot events "
    "that happen later in the book."
)

#: The Jewish War has 117 sections; at ~25 tokens per title the free default
#: model needs well over the 4k default ceiling and truncates mid-array.
TOC_MAX_TOKENS = 16000

_TOC_PROMPT = """\
The following sections were located in "{title}".

{sections}

Output exactly {count} lines, one per section above, in the same order, each
formatted as:
<index>|<short title>
and nothing else.
"""


def iter_lines(book: Book):
    """Yield ``(offset, content)`` for every non-blank line of ``book``.

    Anchors the offset on the FIRST NON-WHITESPACE CHARACTER of the line.
    That matters because some books indent their headings — Monte Cristo
    writes " Chapter 2. Father and Son" — and an offset pointing at the indent
    is a position where no line begins, which makes heading lookup skip the
    heading and land on the prose below it.

    The audited accessor for STRUCTURAL analysis. Callers get offsets and the
    line at each one rather than a slice of ``Book.text``, so no module outside
    this file holds book text it could accidentally paste into a prompt.
    Anything derived from here that must reach a model goes through
    :func:`preview_at`.
    """
    for match in re.finditer(r"(?m)^(.*)$", book.text):
        raw = match.group(0)
        stripped = raw.strip()
        if stripped:
            yield match.start() + (len(raw) - len(raw.lstrip())), stripped


def preview_at(book: Book, offset: int, limit: int = 320) -> str:
    """Opening words of the section that starts at ``offset``.

    The audited accessor for PROMPT-FACING structural text. Every character a
    structural scan sends to a model passes through here, which makes "what did
    we send, and from where?" a question this function can answer.
    """
    segment = book.text[offset : offset + limit]
    return re.sub(r"\s+", " ", segment).strip()


def measure_wrap_width(book: Book, sample: int = 400) -> int:
    """Estimate the prose wrap width in characters.

    The mode of line length in the middle of the body is the wrap width. Lines
    at or above it are almost certainly wrapped prose rather than headings -
    which is exactly why a length heuristic can propose candidates but cannot
    decide among them.
    """
    total = len(book.text)
    start = total // 5
    counts: Dict[int, int] = {}
    for _offset, content in iter_lines(_SliceView(book, start, start + sample * 100)):
        if 30 <= len(content) <= 110:
            counts[len(content)] = counts.get(len(content), 0) + 1
    if not counts:
        return 72
    return max(counts, key=lambda k: (counts[k], k))


class _SliceView:
    """A read-only window onto a ``Book`` used for offline measurement.

    Only :func:`measure_wrap_width` uses it, and it never leaves this module:
    the width is a single integer that carries no book text with it.
    """

    def __init__(self, book: Book, start: int, end: int) -> None:
        self.text = book.text[start:end]


def has_prose_after(
    book: Book, offset: int, window: int = PROSE_WINDOW, need: int = PROSE_LINES_REQUIRED
) -> bool:
    """Public, audited form of the "is prose after this offset?" check."""
    return _has_prose_near(book.text, offset, window=window, need=need)


SPAN_PROSE_RE = re.compile(r"(?m)^\S.{39,}$")


def span_has_prose(book: Book, start: int, end: int, need: int = 2) -> bool:
    """True if ``need`` lines of running prose sit between ``start`` and ``end``.

    Lets a caller tell a contents-listing run (heading lines stacked with
    nothing between them) from a genuine short section, without ever holding
    the intervening text itself.
    """
    span = book.text[start:end]
    found = 0
    for line in span.split("\n"):
        stripped = line.strip()
        if len(stripped) >= PROSE_MIN_LEN and SPAN_PROSE_RE.match(stripped):
            found += 1
            if found >= need:
                return True
    return False


def contents_sample(book: Book, limit: int = 1200) -> str:
    """Sample of the book's own contents listing, for the adjudicator prompt.

    Returning this is deliberate: the adjudicator has to be shown where the
    listing starts so it does not accept the listing as a section.
    """
    match = re.search(r"(?im)^[ \t]*(?:table of )?contents[ \t]*$", book.text)
    if not match:
        return ""
    start = match.start()
    return re.sub(r"\s+", " ", book.text[start : start + limit]).strip()


def verify_book_structure(sections: Sequence[Section], book: Book) -> List[str]:
    """Audited form of :func:`verify_toc` for structural callers."""
    return verify_toc(sections, book.text)


def _section_preview(text: str, offset: int, limit: int = 700) -> str:
    end = min(len(text), offset + limit)
    return re.sub(r"\s+", " ", text[offset:end]).strip()


def _deterministic_sections(headings: Sequence[Dict[str, Any]]) -> List[Section]:
    """Offline fallback labelling: the raw heading text plus its book parent."""
    sections: List[Section] = []
    parent = ""
    for index, heading in enumerate(headings, start=1):
        label = str(heading["label"])
        if heading["kind"] == "book":
            parent = label
            full_label = label
        else:
            full_label = f"{parent} / {label}" if parent else label
        sections.append(
            Section(
                ordinal=index,
                label=full_label,
                parent=parent,
                offset=int(heading["offset"]),
                title="",
            )
        )
    return sections


def _merge_llm_sections(sections: List[Section], response_text: str) -> List[Section]:
    """Apply model-supplied titles onto detected sections.

    The model may only contribute *titles*; offsets, ordinals and parents all
    come from the deterministic detector, so a hallucinated index cannot move a
    section and a wrong parent cannot break :func:`verify_toc`.

    Two response formats are accepted. The preferred one is a compact
    ``index|title`` line per section: the free default model truncates a
    117-entry JSON array at its token ceiling, which silently costs us every
    title. The JSON array is still handled for models that prefer it.
    """
    titles = _parse_titles(response_text)
    if not titles:
        raise ValueError("no usable titles in TOC response")
    merged: List[Section] = []
    for section in sections:
        title = titles.get(section.ordinal, "")
        merged.append(section.model_copy(update={"title": title}))
    return merged


def _parse_titles(response_text: str) -> Dict[int, str]:
    """Parse ``index|title`` lines, falling back to a JSON array."""
    titles: Dict[int, str] = {}
    for raw_line in (response_text or "").splitlines():
        line = raw_line.strip().lstrip("-* \t").strip()
        if not line or "|" not in line:
            continue
        index_text, _, title = line.partition("|")
        try:
            index = int(index_text.strip().strip("[](){}.").strip())
        except ValueError:
            continue
        cleaned = title.strip().strip('",').strip()
        if index > 0 and cleaned:
            titles.setdefault(index, cleaned)

    if titles:
        return titles

    payload = extract_json(response_text or "")
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            title = str(entry.get("title") or "").strip()
            if index > 0 and title:
                titles.setdefault(index, title)
    return titles


def build_toc(
    book: Book,
    client: Optional[Callable[[str, str], str]] = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> List[Section]:
    """Build (or load) the TOC for ``book``.

    Structural offsets are always produced by :func:`find_headings` — the LLM
    only supplies human labels. One LLM call per book; the result is cached at
    ``cache/{book_id}_toc.json``.

    ``client`` is an injectable ``(prompt, system) -> response_text`` callable
    for tests. When omitted, the real OpenRouter client is used. If the call
    fails, we fall back to deterministic labels and log the exception: a
    missing title is a cosmetic loss, whereas raising would block the reader.
    """
    cache_path = os.path.join(cache_dir(), f"{book.book_id}_toc.json")
    if use_cache and not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            sections = [Section.model_validate(item) for item in payload]
            if sections:
                return sections
        except (OSError, ValueError, TypeError):
            logger.warning("discarding corrupt TOC cache %s", cache_path)

    headings = find_headings(book.text)
    sections = _deterministic_sections(headings)
    if not sections:
        logger.warning("no structural headings found in %s", book.book_id)
        return []

    preview_lines: List[str] = []
    for index, heading in enumerate(headings, start=1):
        preview = _section_preview(book.text, int(heading["offset"]))
        preview_lines.append(f"[{index}] {heading['label']}\n    {preview[:400]}")
    prompt = _TOC_PROMPT.format(
        title=book.title, sections="\n".join(preview_lines), count=len(headings)
    )

    response_text: Optional[str] = None
    try:
        if client is not None:
            response_text = client(prompt, _TOC_SYSTEM)
        else:
            response_text = chat(
                prompt,
                system=_TOC_SYSTEM,
                temperature=0.0,
                max_tokens=TOC_MAX_TOKENS,
                reasoning_effort="low",
            )["content"]
    except Exception:
        logger.exception(
            "TOC labelling call failed for %s; using deterministic labels",
            book.book_id,
        )

    if response_text:
        try:
            sections = _merge_llm_sections(sections, response_text)
        except ValueError:
            logger.exception("unusable TOC response shape for %s", book.book_id)

    if use_cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(
                    [section.model_dump() for section in sections],
                    handle,
                    indent=2,
                )
        except OSError:
            logger.exception("could not write TOC cache %s", cache_path)
    return sections
# --------------------------------------------------------------------------- #
# Verification (MANDATORY)
# --------------------------------------------------------------------------- #


def verify_toc(toc: Sequence[Section], text: str) -> List[str]:
    """Return a list of problems with ``toc``. Empty list means verified.

    Checks performed:

    * ordinals are exactly 1..N, ascending, no duplicates;
    * offsets strictly ascending, non-negative, inside the text;
    * no offset falls at or before the end of the table-of-contents region;
    * every offset lands on/just before a plausible heading line;
    * labels are non-empty and parents refer to sections that exist;
    * the final section does not start past the end of the text.
    """
    problems: List[TocProblem] = []

    if not toc:
        return ["[empty] toc: table of contents is empty"]

    region_end = toc_region_end(text, [s.label for s in toc])

    # Ordinals must be exactly 1..N.
    for position, section in enumerate(toc, start=1):
        if section.ordinal != position:
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="ordinal-sequence",
                    detail=(
                        f"expected ordinal {position} in flattened list, "
                        f"got {section.ordinal}"
                    ),
                )
            )

    # Offsets must be strictly ascending and in range.
    previous: Optional[int] = None
    for section in toc:
        if section.offset < 0:
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="negative-offset",
                    detail=f"offset {section.offset} is negative",
                )
            )
        elif section.offset > len(text):
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="offset-out-of-range",
                    detail=(f"offset {section.offset} exceeds text length {len(text)}"),
                )
            )
        if section.offset < region_end:
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="inside-toc-region",
                    detail=(
                        f"offset {section.offset} is at or before the first "
                        f"structural heading ({region_end}); this is the table "
                        "of contents, not a section start"
                    ),
                )
            )
        if previous is not None and section.offset <= previous:
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="offset-not-ascending",
                    detail=(
                        f"offset {section.offset} does not advance past "
                        f"previous offset {previous}"
                    ),
                )
            )
        if not looks_like_heading(text, section.offset, heading_line=section.label):
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="offset-not-a-heading",
                    detail=(
                        f"offset {section.offset} does not land on a "
                        f"structural heading line (nearest: "
                        f"{nearest_heading_line(text, section.offset)!r})"
                    ),
                )
            )
        if not section.label.strip():
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="empty-label",
                    detail="label is empty",
                )
            )
        previous = section.offset

    known_labels = {section.label.strip() for section in toc}
    for section in toc:
        parent = section.parent.strip()
        if parent and parent not in known_labels:
            problems.append(
                TocProblem(
                    ordinal=section.ordinal,
                    code="unknown-parent",
                    detail=f"parent {parent!r} is not a section label in this toc",
                )
            )

    return [str(problem) for problem in problems]


def toc_coverage(toc: Sequence[Section], text: str) -> Dict[str, Any]:
    """Coverage stats for a TOC, useful for demos and diagnostics."""
    if not toc:
        return {"sections": 0, "covered_chars": 0, "text_chars": len(text)}
    start = toc[0].offset
    covered = len(text) - start
    return {
        "sections": len(toc),
        "first_offset": start,
        "covered_chars": covered,
        "text_chars": len(text),
        "coverage_pct": round(100.0 * covered / max(1, len(text)), 2),
    }


# --------------------------------------------------------------------------- #
# THE SPOILER GATE
# --------------------------------------------------------------------------- #


def section_index(toc: Sequence[Section], ordinal: int) -> int:
    """Return the list index for ``ordinal``, raising if it does not exist."""
    for index, section in enumerate(toc):
        if section.ordinal == ordinal:
            return index
    raise ValueError(f"ordinal {ordinal} is not in this toc (1..{len(toc)})")


def text_up_to(book: Book, toc: Sequence[Section], ordinal: int) -> str:
    """Return the portion of ``book`` the reader has finished, and nothing more.

    THE ONLY function in this codebase permitted to slice ``Book.text`` into
    anything that reaches a prompt. Text from the start of section
    ``ordinal + 1`` onward is withheld, which is what makes the tool
    spoiler-safe: the model physically cannot see beyond the read portion,
    regardless of what it knows about the book from pretraining.

    The returned slice begins at the first section (front matter and the
    table of contents are dropped) and ends where section ``ordinal + 1``
    begins. ``ordinal == len(toc)`` yields the whole body.
    """
    if not toc:
        raise ValueError("cannot slice text without a verified toc")
    index = section_index(toc, ordinal)
    start = toc[0].offset
    if index + 1 < len(toc):
        end = toc[index + 1].offset
    else:
        end = len(book.text)
    if start < 0 or end > len(book.text) or end < start:
        logger.error(
            "refusing to slice %s at [%s:%s] (len=%s)",
            book.book_id,
            start,
            end,
            len(book.text),
        )
        raise ValueError("toc offsets are inconsistent with book text")
    return book.text[start:end]


def section_text(book: Book, toc: Sequence[Section], ordinal: int) -> str:
    """Text of a single section (used by the map-reduce summarizer)."""
    index = section_index(toc, ordinal)
    start = toc[index].offset
    end = toc[index + 1].offset if index + 1 < len(toc) else len(book.text)
    return book.text[start:end]


def count_tokens(text: str) -> int:
    """Token count via tiktoken, falling back to a char heuristic."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:  # pragma: no cover - network/dep failure fallback
        logger.warning("tiktoken unavailable; estimating tokens from characters")
        return len(text) // 4
