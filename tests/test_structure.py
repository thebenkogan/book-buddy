"""End-to-end structure tests: propose -> adjudicate -> verify, fully offline.

These run against the one real book in ``data/`` (``jewish_war.txt``). The
adjudicator is a deterministic fake client, so nothing here touches the
network; what is being tested is the part of the pipeline that is code, not
model: candidate offsets, label construction, and whether verification agrees
with them.

An earlier version of this file also covered Monte Cristo and Les Mis, which
are indented-heading and em-dash-dialect books respectively. Those texts have
been deleted, so the properties they pinned now live against synthetic
fixtures below: the indented-heading case is reproduced exactly (one leading
space) rather than being dropped, because it is a real regression class and
the fix in ``iter_lines`` / ``heading_line_at`` still needs a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.bookbuddy.book import (
    Book,
    heading_line_at,
    load_book,
    looks_like_heading,
    verify_book_structure,
)
from src.bookbuddy.structure import propose_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
JEWISH_WAR = REPO_ROOT / "data" / "jewish_war.txt"

#: A line beginning with a quotation mark and ending in terminal punctuation is
#: spoken dialogue from the novel's body, not a structural heading.
DIALOGUE_RE = re.compile(r'^[“"][^“”"]{10,}["”]$')

#: The Jewish War's heading dialect: flush left, arabic chapters.
JEWISH_WAR_HEADING_RE = re.compile(r"(?:BOOK|CHAPTER)\s+(?:[IVXLC]+|[0-9]+)\.")


def _roman_or_int(token: str) -> int:
    """Chapter numbers in this book are mostly arabic with a few roman ones."""
    if token.isdigit():
        return int(token)
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    previous = 0
    for char in reversed(token.upper()):
        value = values[char]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total


def structural_client(prompt: str, system: str) -> str:
    """A deterministic stand-in for the adjudication model.

    Accepts exactly the candidates whose line starts with a structural word,
    which is the convention this book uses, and echoes them back as
    ``index|title``. No network, no randomness, same result every run.
    """
    accepted = []
    for match in re.finditer(r"^\[(\d+)\] (.*)$", prompt, re.MULTILINE):
        index = int(match.group(1))
        line = match.group(2).strip()
        if re.match(r"(?i)^(chapter|book|part|volume)\b", line):
            accepted.append(f"{index}|{line}")
    return "\n".join(accepted)


def run_pipeline(path: Path, book_id: str):
    """propose -> adjudicate with the fake client -> verify. Returns all three."""
    from src.bookbuddy.structure import adjudicate_sections

    book = load_book(str(path), book_id=book_id)
    candidates = propose_sections(book)
    sections = adjudicate_sections(book, candidates, client=structural_client)
    problems = verify_book_structure(sections, book)
    return book, sections, problems


def assert_well_formed(sections, problems):
    assert problems == [], f"{len(problems)} problems, first: {problems[0]}"
    assert sections, "pipeline produced no sections"
    assert [s.ordinal for s in sections] == list(range(1, len(sections) + 1))
    offsets = [s.offset for s in sections]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets), "duplicate offsets"
    for section in sections:
        assert section.label.strip(), f"empty label at ordinal {section.ordinal}"


@pytest.fixture(scope="module")
def jewish_war_result():
    if not JEWISH_WAR.exists():
        pytest.skip("jewish_war.txt not present")
    return run_pipeline(JEWISH_WAR, "jewish_war")


# --------------------------------------------------------------------------- #
# Jewish War: "CHAPTER 4.", flush left, arabic chapter numbers
# --------------------------------------------------------------------------- #


def test_pipeline_verifies_clean(jewish_war_result):
    _, sections, problems = jewish_war_result
    assert_well_formed(sections, problems)


def test_offsets_land_on_heading_lines(jewish_war_result):
    """Every accepted offset must actually point at its own heading."""
    book, sections, _ = jewish_war_result
    for section in sections:
        actual = heading_line_at(book.text, section.offset)
        assert actual == section.label.split(" / ")[-1], (
            f"ordinal {section.ordinal}: label {section.label!r} but text at "
            f"offset is {actual!r}"
        )


def test_no_accepted_dialogue_lines(jewish_war_result):
    """Regression guard for the model accepting prose as a section.

    A run where quoted dialogue ("A sort of book, written upon strips of
    cloth.") was accepted as a section is the failure this asserts against.
    """
    _, sections, _ = jewish_war_result
    for section in sections:
        assert not DIALOGUE_RE.match(section.label), (
            f"ordinal {section.ordinal} accepted a dialogue line: "
            f"{section.label!r}"
        )


def test_section_count_is_plausible(jewish_war_result):
    _, sections, _ = jewish_war_result
    assert 100 <= len(sections) <= 130, f"{len(sections)} sections"


def test_every_heading_shape_is_recognised(jewish_war_result):
    """Accepted labels are structural, never mid-sentence prose.

    Scoped to the DETERMINISTIC headings. The adjudicator stage is a model
    decision by design, and the fake client used here accepts anything
    beginning with a structural word -- including a line of translator
    footnote residue ("Part I. p. 207. But of this younger Antiochus, ...").
    Catching that is ``audit_structure``'s job
    (``footnote_block_section``), and it is pinned in test_rag.py; asserting it
    here would be asserting that stage 2 does stage 5's work.
    """
    from src.bookbuddy.book import build_toc

    book, _sections, _ = jewish_war_result
    for section in build_toc(book, client=lambda p, s: "", use_cache=False):
        tail = section.label.split(" / ")[-1]
        assert JEWISH_WAR_HEADING_RE.match(tail), (
            f"non-structural label accepted: {section.label!r}"
        )


def test_chapters_are_numbered_in_order_within_each_book(jewish_war_result):
    """CHAPTER n. under BOOK m. must be strictly increasing in n.

    Catches the ordinal-drift class of bug: if a section is dropped or
    duplicated by the merge, the chapter numbering under a book stops being
    monotone and every reader position past that point is wrong. Asserted on
    the deterministic scan, which is the layer that owns structure.
    """
    from src.bookbuddy.book import build_toc

    book, _sections, _ = jewish_war_result
    sections = build_toc(book, client=lambda p, s: "", use_cache=False)
    current_book = ""
    last_chapter = 0
    for section in sections:
        label = section.label
        if label.startswith("BOOK ") and " / " not in label:
            current_book = label
            last_chapter = 0
            continue
        match = re.search(r"CHAPTER\s+([0-9IVXLC]+)\.", label)
        assert match, f"no chapter number in {label!r}"
        number = _roman_or_int(match.group(1))
        assert number > last_chapter, (
            f"chapter {number} follows {last_chapter} under {current_book!r} "
            f"at ordinal {section.ordinal}; a section was lost or duplicated"
        )
        last_chapter = number


def test_every_book_opens_with_its_first_chapter(jewish_war_result):
    """Each BOOK heading must be followed by its CHAPTER 1.

    The sharpest regression guard for the merge bug. "BOOK I." and
    "BOOK I. / CHAPTER 1." are ~180 chars apart, inside the merge tolerance,
    so both tolerance-based passes in ``merge_with_deterministic`` could treat
    one as covering the other -- which silently deleted every CHAPTER 1 in the
    book and shifted all later ordinals.
    """
    from src.bookbuddy.book import build_toc
    from src.bookbuddy.onboarding import merge_with_deterministic

    book, _sections, _ = jewish_war_result
    det = build_toc(book, client=lambda p, s: "", use_cache=False)

    def first_chapters(sections):
        found = []
        for index, section in enumerate(sections):
            if section.label.startswith("BOOK ") and " / " not in section.label:
                nxt = sections[index + 1].label if index + 1 < len(sections) else ""
                found.append((section.label, nxt))
        return found

    det_pairs = first_chapters(det)
    assert len(det_pairs) >= 7, f"expected 7 books, found {len(det_pairs)}"
    for book_label, next_label in det_pairs:
        assert re.search(r"CHAPTER\s+(?:1|I)\.", next_label), (
            f"{book_label!r} is not followed by its first chapter "
            f"(got {next_label!r})"
        )

    # And the merge must preserve that property.
    merged = merge_with_deterministic(
        [s.model_copy(update={"title": f"T{s.ordinal}"}) for s in det], book
    )
    for book_label, next_label in first_chapters(merged):
        assert re.search(r"CHAPTER\s+(?:1|I)\.", next_label), (
            f"merge broke {book_label!r} -> {next_label!r}"
        )


def test_no_section_is_dropped_between_deterministic_scan_and_merge(jewish_war_result):
    """The deterministic scan's headings must all survive the merge.

    Regression guard for the nearest-vs-first bug in
    ``onboarding.merge_with_deterministic``: "BOOK I." sits ~184 chars above
    "BOOK I. / CHAPTER 1.", inside the merge tolerance, so a first-match lookup
    mapped every CHAPTER 1 onto its BOOK heading and the dedupe pass then
    deleted it. Seven sections vanished and every later ordinal shifted by one.
    """
    from src.bookbuddy.book import build_toc
    from src.bookbuddy.onboarding import merge_with_deterministic

    book, _sections, _ = jewish_war_result
    det = build_toc(book, client=lambda p, s: "", use_cache=False)
    merged = merge_with_deterministic(
        [s.model_copy(update={"title": f"T{s.ordinal}"}) for s in det], book
    )
    missing = sorted({int(s.offset) for s in det} - {int(s.offset) for s in merged})
    assert missing == [], (
        f"{len(missing)} deterministic headings missing after merge: {missing[:5]}"
    )


def test_merge_of_the_deterministic_list_with_itself_is_the_identity():
    """Merging must not delete sections, even on a perfect model response.

    The sharpest form of the regression above: if the model's offsets are all
    correct and all its labels are structural, then merge has nothing to fix
    and must return the same list. Any shrinkage is data loss.
    """
    from src.bookbuddy.book import build_toc
    from src.bookbuddy.onboarding import merge_with_deterministic

    if not JEWISH_WAR.exists():
        pytest.skip("jewish_war.txt not present")
    book = load_book(str(JEWISH_WAR), book_id="jewish_war")
    det = build_toc(book, client=lambda p, s: "", use_cache=False)
    titled = [s.model_copy(update={"title": f"T{s.ordinal}"}) for s in det]

    merged = merge_with_deterministic(titled, book)

    assert len(merged) == len(titled), (
        f"merge deleted sections: {len(titled)} -> {len(merged)}"
    )
    assert [int(s.offset) for s in merged] == [int(s.offset) for s in titled]
    assert [s.label for s in merged] == [s.label for s in titled]
    assert [s.title for s in merged] == [s.title for s in titled]
    assert [int(s.ordinal) for s in merged] == list(range(1, len(merged) + 1))


# --------------------------------------------------------------------------- #
# Indented headings -- the Monte Cristo dialect, reproduced synthetically
# --------------------------------------------------------------------------- #


def test_heading_line_at_handles_an_indented_heading():
    """Pins the indented-heading fix now that the source book is gone.

    Monte Cristo's headings were `` Chapter 2. Father and Son`` -- one leading
    space -- so an offset taken at the start of the LINE points into the
    indent, where no line begins, and heading lookup lands on the prose below.
    ``heading_line_at`` therefore falls back to the line the offset is inside.
    """
    text = (
        " Chapter 1. Marseilles-The Arrival\n\n"
        "The count looked out at the sign of the Notte.\n\n"
        " Chapter 2. Father and Son\n\nHe turned towards the ship.\n"
    )
    line_start = text.index(" Chapter 2.")
    assert text[line_start] == " ", "premise: the heading carries a leading space"
    # An offset inside the indent still resolves to the heading line.
    assert heading_line_at(text, line_start) == "Chapter 2. Father and Son"
    assert heading_line_at(text, line_start + 1) == "Chapter 2. Father and Son"


def test_iter_lines_anchors_on_first_non_whitespace_character():
    from src.bookbuddy.book import iter_lines

    if not JEWISH_WAR.exists():
        pytest.skip("jewish_war.txt not present")
    book = load_book(str(JEWISH_WAR), book_id="jewish_war")
    # Every yielded offset must be the first non-blank character of its line.
    for offset, content in iter_lines(book):
        assert not book.text[offset].isspace()
        assert content == book.text[offset : offset + len(content)]


# --------------------------------------------------------------------------- #
# The fix must not weaken verification
# --------------------------------------------------------------------------- #


def test_mid_chapter_offset_still_rejected():
    """An offset in the middle of a paragraph is still not a heading.

    ``heading_line_at`` falls back to the line an offset is inside. That must
    not turn arbitrary mid-prose positions into sections.
    """
    text = (
        " Chapter 1. Marseilles-The Arrival\n\n"
        + (
            "The count looked out at the sign of the Notte, and the crowd\n"
            "gathered in the contre-jour of the gas-light behind him. "
        )
        * 6
        + "\n Chapter 2. Father and Son\n\nHe turned again towards the ship.\n"
    )
    mid = text.index("the crowd")
    assert not looks_like_heading(text, mid, heading_line="Chapter 3. The Catalans")


def test_offset_before_a_heading_is_not_a_section_start():
    """A heading found AFTER the offset must not validate that offset.

    Regression for a vacuous guard in ``looks_like_heading``. The strict path
    tested ``match_start <= offset + OFFSET_HEADING_TOLERANCE``, but the window
    it searched was already sliced to end at exactly that value, so the test
    held for every match in the window. An offset landing anywhere in the 200
    chars before a real heading therefore verified clean -- so an offset in
    the middle of the preceding section's prose, just ahead of the next
    chapter, was accepted as that chapter's start.
    """
    prose = "prose line long enough to count as running text in this book here\n" * 60
    text = prose + "\nCHAPTER 2.\n\n" + ("more prose of a similar generous length\n" * 60)
    heading_at = text.index("CHAPTER 2.")
    before = heading_at - 60

    # The heading is AHEAD of this offset, so the offset is in the previous
    # section's body, not at a section start.
    assert not looks_like_heading(text, before, heading_line="CHAPTER 2.")
    # The heading's own offset must still validate.
    assert looks_like_heading(text, heading_at, heading_line="CHAPTER 2.")


def test_verify_accepts_a_correct_toc_and_rejects_a_shifted_one():
    """A passing audit must mean the checks ran, not that they were skipped.

    Uses a synthetic book whose text really does contain the headings, so a
    clean verdict is earned rather than vacuous, and then shifts every offset
    into the prose to confirm the same list is rejected.
    """
    from src.bookbuddy.book import Section

    prose = "prose line long enough to count as running text in this book here\n" * 60
    text = "CHAPTER 1.\n\n" + prose + "\nCHAPTER 2.\n\n" + prose
    book = Book(book_id="t", title="t", text=text)
    first = text.index("CHAPTER 1.")
    second = text.index("CHAPTER 2.")

    good = [
        Section(ordinal=1, label="CHAPTER 1.", offset=first),
        Section(ordinal=2, label="CHAPTER 2.", offset=second),
    ]
    assert verify_book_structure(good, book) == []

    # Same labels, offsets walked 300 chars into the preceding prose.
    shifted = [
        Section(ordinal=1, label="CHAPTER 1.", offset=first + 300),
        Section(ordinal=2, label="CHAPTER 2.", offset=second + 300),
    ]
    problems = verify_book_structure(shifted, book)
    assert problems != [], "a TOC pointing into the middle of prose verified clean"
    assert all("offset-not-a-heading" in p for p in problems), problems
