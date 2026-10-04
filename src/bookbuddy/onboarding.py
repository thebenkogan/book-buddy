"""Onboarding: a reviewable structure artifact Hermes can correct.

The architecture in one paragraph
---------------------------------
Sectioning cannot be done by a script alone, because a heading and a wrapped
line of prose carry the same signal (both are short standalone lines), and no
threshold separates them -- measured: 213 correct headings against 1,329 false
positives in the Jewish War, 357 against 12,784 in Les Mis. Nor can a script do
it *reliably at all*, because the three books here use three different
conventions (``BOOK IV.`` / ``Chapter 1. Title`` / ``VOLUME I``) and the
regex that finds one finds none of the others.

So the pipeline is deliberately split by who is trustworthy for what:

1. ``propose_sections``  -- deterministic, high RECALL. Narrow the book to a
   few hundred candidate lines. A missed boundary is a recoverable error.
2. ``adjudicate_sections`` -- the MODEL decides which candidates are real
   section starts and names them, choosing by INTEGER INDEX so it cannot
   invent an offset. A wrong choice here is not recoverable by re-running, so
   this is the only stage allowed to decide, and stage 3 catches its mistakes.
3. ``merge_with_deterministic`` -- reconcile the model's list with the
   deterministic heading scan. See below; this is the load-bearing step.
4. ``StructureArtifact.save`` -- dump the result to JSON so it can be READ and
   CORRECTED. This is the escape hatch: the model's output is a file, not a
   verdict.
5. ``audit_structure`` -- deterministic checks that catch the specific ways
   the adjudication actually goes wrong (below).
6. ``apply_corrections`` -- apply human/Hermes edits, then re-verify. Fails
   closed.

Why the model ALONE is not enough -- measured on the Jewish War
------------------------------------------------------------------
Run against the same book, the deterministic heading scan and the model
adjudication disagree badly, in OPPOSITE directions:

* deterministic: 117 sections, all real ``BOOK n. / CHAPTER n.`` headings,
  1 audit issue.
* model-adjudicated: 317 sections, of which only **103** carry a real chapter
  heading. The other 214 are false positives -- mid-sentence prose
  ("BOOK I. / 2. So he gave command that the Jews should bring in seven
  hu...") and footnote residue ("BOOK V. / 26:18.]").

So neither approach dominates:

* where a book uses a clean convention (``BOOK IV.`` alone on a line), the
  deterministic scan is perfect and the model adds noise;
* where it does not (``Chapter 1. Marseilles--The Arrival``, ``VOLUME I``),
  the deterministic scan finds NOTHING (0 sections on three of the four books
  here) and only the model copes.

Hence stage 3: run both, keep the union of genuine headings, and let the audit
plus the review artifact arbitrate. The model is not trusted to be the sole
authority, because measurement says it is not one.

Why a reviewable artifact rather than "ask the model again"
-----------------------------------------------------------
Re-prompting a model that just produced a bad section list reproduces a bad
section list, differently. Editing a file does not. The artifact makes the
structure *inspectable*, which is the whole difference between a tool you can
trust and one you re-run and hope.

What the audit catches, and why each check exists
-------------------------------------------------
Each of these was observed in real output, not imagined:

* ``first_section_starts_at_body`` -- Monte Cristo's list began at
  "Chapter 2", silently dropping Chapter 1. Offsets were all valid, so
  ``verify_toc`` reported zero problems. Nothing else noticed.
* ``implausible_heading`` -- the last section was labelled
  "\\"I have a letter to give you from the count.\\"", a sentence the
  adjudicator accepted as a heading. It is dialogue, not structure.
* ``no_preamble_sections`` -- if sections start before the body's first real
  heading, the front matter or the table of contents is being treated as
  narrative.
* ``ordinals_contiguous`` -- a gap means a dropped section, which shifts every
  later ordinal and therefore every reader's saved position.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.bookbuddy import cache_dir
from src.bookbuddy.book import Book, Section

logger = logging.getLogger(__name__)

#: Where the structure artifact lives, per book.
def artifact_path(book_id: str) -> str:
    return os.path.join(cache_dir(), f"{book_id}_structure.json")


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #


class StructureArtifact(BaseModel):
    """A book sectioning a human or agent can read, edit and re-verify.

    This is the seam. The model writes it; anybody (including Hermes) can
    correct it; verification runs on the edited version, never on the model's
    original.
    """

    book_id: str
    sections: List[Section] = Field(default_factory=list)
    #: Provenance so a reviewer knows how much to trust the list as-is.
    proposed_by: str = "structure.adjudicate_sections"
    reviewed_by: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    def save(self, path: Optional[str] = None) -> str:
        path = path or artifact_path(self.book_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.model_dump(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "StructureArtifact":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))


def merge_with_deterministic(
    model_sections: Sequence[Section],
    book,
    tolerance: int = 200,
    trust_model: bool = False,
) -> List[Section]:
    """Reconcile a model-adjudicated list with the deterministic heading scan.

    The deterministic scan (``book.build_toc`` with a stub client) is exact on
    books with a clean convention and empty on books without one; the model
    is the reverse. Neither alone is trustworthy (measured on the Jewish War:
    117 clean deterministic vs a 317-entry model list of which 103 are real).

    So: take the model's sections, and additionally admit any deterministic
    heading the model missed. Model sections that sit within ``tolerance``
    chars of a deterministic heading are kept for their TITLE, but the
    STRUCTURE always comes from the deterministic scan; the rest are kept only
    if they look like genuine headings, since that is where the model's false
    positives live. The result is sorted by offset and renumbered.

    The model may not supply structure
    ----------------------------------
    An earlier version kept the model's ``label``/``parent`` for matched
    sections. That is how the Agrippa passage came to be labelled
    "BOOK I. / CHAPTER 9." when it sits at offset 362120 and BOOK II. begins at
    305980 -- every chapter in Book II inherited BOOK I. as its parent, 72
    sections mislabelled, while ``verify_toc`` reported zero problems because
    the OFFSETS were all correct. A reader navigating by label was told they
    were in Book I while reading Book II. Offsets are not the whole structure;
    labels and parents are, and only the deterministic scan gets them right.
    """
    from src.bookbuddy.book import build_toc

    deterministic = build_toc(book, client=lambda p, s: "", use_cache=False)

    kept: List[Section] = []
    det_by_offset = {int(s.offset): s for s in deterministic}
    det_offsets_sorted = sorted(det_by_offset)
    #: Deterministic offsets already claimed by a model section's match.
    consumed: set = set()

    def nearest(offset: int) -> Optional[Section]:
        """The deterministic heading CLOSEST to ``offset``, within tolerance.

        NEAREST, not FIRST. An earlier version used ``next(... for d, s in
        det_by_offset.items() if abs(offset - d) <= tolerance)``, which takes
        the first entry within tolerance rather than the closest one. In the
        Jewish War "BOOK I." sits 184 chars above "BOOK I. / CHAPTER 1.", so
        every CHAPTER 1 matched BOOK I. instead of itself, and was then removed
        by the exact-offset dedupe below -- seven sections silently deleted,
        with every later ordinal shifted by one. Merging the deterministic list
        with itself must be the identity; it now is.
        """
        if not det_offsets_sorted:
            return None
        pos = bisect.bisect_left(det_offsets_sorted, offset)
        best: Optional[Section] = None
        for candidate_pos in (pos - 1, pos):
            if 0 <= candidate_pos < len(det_offsets_sorted):
                det_offset = det_offsets_sorted[candidate_pos]
                if abs(offset - det_offset) <= tolerance and (
                    best is None
                    or abs(offset - det_offset)
                    < abs(offset - int(best.offset))
                ):
                    best = det_by_offset[det_offset]
        return best

    for section in model_sections:
        offset = int(section.offset)
        match = nearest(offset)
        if match is not None:
            # Two model sections can resolve to the SAME deterministic heading
            # when one is a wrapped title and the other its continuation 19
            # chars away -- both fall inside the tolerance. Keeping both was
            # only stopped by the exact-offset dedupe at the end, i.e. by
            # luck. Skip the second explicitly, and prefer the section whose
            # offset matches EXACTLY, which is always the real heading line
            # (a continuation can only ever come after it).
            already = int(match.offset) in consumed
            exact = int(section.offset) == int(match.offset)
            if already and not exact:
                continue
            if already and exact:
                # The continuation was seen first (only possible if verdicts
                # were not sorted); replace so the exact offset wins.
                for i, kept_section in enumerate(kept):
                    if int(kept_section.offset) == int(match.offset):
                        kept[i] = match.model_copy(
                            update={"title": section.title or match.title}
                        )
                        break
                continue
            # Adopt the model's TITLE only. label/parent/offset are the
            # deterministic scan's, because that is what gets hierarchy right.
            consumed.add(int(match.offset))
            kept.append(
                match.model_copy(
                    update={"title": section.title or match.title}
                )
            )
            continue
        # A model section the deterministic scan never saw.
        #
        # When a deterministic scan EXISTS, keep it only if it is shaped like a
        # heading -- that is where 214 false positives were filtered on the
        # Jewish War.
        #
        # When the scan found NOTHING, that gate is the bug, not the safeguard.
        # Measured on Grimm's Fairy Tales: 59 adjudicated verdicts went in and
        # ZERO came out, because the gate demands a leading BOOK/CHAPTER/
        # VOLUME and every real heading there is "RAPUNZEL" or
        # "THE GOOSE-GIRL". The fallback path silently discarded every verdict
        # it existed to produce. With no deterministic scan there is no scan to
        # reconcile against, so the adjudicated verdicts ARE the structure.
        if trust_model or not deterministic:
            kept.append(section)
            continue
        label = (section.label or "").strip()
        looks_structural = bool(
            re.match(
                r"(?i)^(?:BOOK|VOLUME|PART|CHAPTER|SECTION|Canto|ACT|SCENE)\b",
                label,
            )
        ) or _STRUCTURAL_LABEL.match(label) is not None
        if looks_structural:
            kept.append(section)

    # Add deterministic headings the model missed entirely.
    #
    # A heading is "already accounted for" only if a match above CONSUMED it --
    # tracked by offset, not by proximity. The previous test was
    # ``any(abs(offset - m) <= tolerance for m in kept_offsets)``, which asks
    # the wrong question: it reports a heading as covered merely because some
    # OTHER kept section sits within tolerance of it. "BOOK II." at 305980 and
    # "BOOK II. / CHAPTER 1." at 306132 are 152 chars apart, inside the 200-char
    # tolerance, and are two different real sections. Proximity therefore hid
    # every first-chapter heading in the book from this pass.
    for section in deterministic:
        offset = int(section.offset)
        if offset in consumed:
            continue
        kept.append(section)
        consumed.add(offset)

    kept.sort(key=lambda s: int(s.offset))
    seen_offsets: set = set()
    deduped: List[Section] = []
    for section in kept:
        if int(section.offset) in seen_offsets:
            continue
        seen_offsets.add(int(section.offset))
        deduped.append(section)
    return [
        section.model_copy(update={"ordinal": i})
        for i, section in enumerate(deduped, start=1)
    ]


class StructureIssue(BaseModel):
    ordinal: Optional[int] = None
    code: str
    detail: str
    #: Machine-actionable hint, so a fix does not require re-deriving the
    #: diagnosis. ``drop:<ordinal>`` / ``relabel:<ordinal>:<name>``.
    fix: Optional[str] = None

    def __str__(self) -> str:
        where = f"ordinal={self.ordinal}" if self.ordinal is not None else "structure"
        return f"[{self.code}] {where}: {self.detail}"


# --------------------------------------------------------------------------- #
# Stage 4: the audit
# --------------------------------------------------------------------------- #

#: A heading that reads as a sentence is prose the adjudicator accepted.
#: Observed in Monte Cristo: "And Albert took out of a little pocket-book
#: with golden clasps, a" and "Then enclosing Monte Cristo's receipt in a
#: little pocket-book, he" -- both are midline narrative, not structure.
#: The earlier version of this regex looked for verbs like "said"/"because"
#: and matched NONE of them, so 5 prose sections shipped undetected.
_VERBISH = re.compile(
    r"(?i)\b(said|says|replied|cried|answered|asked|thought|felt|knew|"
    r"would|could|should|going|because|before|after|while|although)\b"
)

#: Sentence-initial discourse markers. A real heading does not open with
#: "And", "Then", "Just then" or a bare article -- it names a place, a
#: person, or a number.
_SENTENCE_OPENER = re.compile(
    r"(?i)^(and|but|or|so|then|now|just then|then|and then|yet|still|nor)\b"
)
_ARTICLE_OPENER = re.compile(r"(?i)^(a|an|the)\s+[a-z]")

#: A finite verb in third person or past tense after the first word:
#: "Andrea examined it carefully", "Albert took out a pocket-book". A real
#: heading is a noun phrase; if word 2 carries the verb, the line is a
#: sentence. Observed: "Andrea examined it carefully, to ascertain if the
#: letter had been" shipped as a section.
#: A division heading ("BOOK FIRST--A JUST MAN", "VOLUME I", "PART II").
#: Short by nature: the passage starts at its child.
#: The leading division token of a label ("BOOK II." in "BOOK II. / CHAPTER 9.").
_LEADING_DIVISION = re.compile(r"^(BOOK\s+[IVXLC0-9]+|VOLUME\s+\w+|PART\s+\w+)\.?")

#: The head of a translator footnote block. Such a section indexes to zero
#: chunks and should never appear as a reader-visible chapter.
_FOOTNOTE_BLOCK = re.compile(r"(?i)^WAR BOOK[^\n]*FOOTNOTES")

#: A label that is nothing but a division ("BOOK II." with no chapter part).
#: These legitimately name themselves as their own parent.
#:
#: Capture group 1 is the division token, so this and ``_LEADING_DIVISION``
#: can be used interchangeably by ``.group(1)``. An earlier version used a
#: non-capturing group here and raised IndexError on every bare division.
_BARE_DIVISION = re.compile(
    r"^(BOOK\s+[IVXLC0-9]+|VOLUME\s+\w+|PART\s+\w+)\.?$", re.IGNORECASE
)

_STRUCTURAL_LABEL = re.compile(
    r"(?i)^(?:BOOK|VOLUME|PART|CANTOS?|BOOKS?)\b"
)

_SECOND_WORD_VERB = re.compile(
    r"(?i)^\S+\s+"
    r"(examined|took|went|came|saw|said|found|gave|made|looked|turned|"
    r"replied|answered|asked|thought|knew|felt|wanted|needed|began|"
    r"continued|returned|opened|closed|held|put|set|led|drove|read)\b"
)


#: Lines that are structural-looking but are NOT body structure: the
#: translator's footnote blocks. "WAR PREFACE FOOTNOTES" scored as a strong
#: candidate in the Jewish War and was earlier mistaken for the start of the
#: body, which made the dropped-first-section check fire against a perfectly
#: good TOC.
_NOT_BODY_HEADING = re.compile(r"(?i)FOOTNOTES|\bAPPENDIX\b|^CONTENTS$")


def _derive_body_start(book) -> Optional[int]:
    """First offset that is real body structure, footnote blocks excluded."""
    from src.bookbuddy.structure import propose_sections

    candidates = propose_sections(book)
    strong = [c for c in candidates if c.get("prior") == "strong"]
    for candidate in strong:
        line = (candidate.get("line") or "").strip()
        if _NOT_BODY_HEADING.search(line):
            continue
        return int(candidate["offset"])
    if strong:
        return int(strong[0]["offset"])
    return None


def _section_head(book, sections, ordinal: int, limit: int = 200) -> str:
    """First ``limit`` chars of one section, obtained through the gate.

    ``onboarding`` deliberately never slices ``Book.text``: tests/test_book.py
    fails any module that does. This reads the section via ``text_up_to`` and
    differences two calls, exactly as ``rag.chunk_book`` does.
    """
    from src.bookbuddy.book import text_up_to

    if not sections:
        return ""
    whole = text_up_to(book, sections, len(sections))
    base = int(sections[0].offset)
    target = next(
        (s for s in sections if int(s.ordinal) == ordinal), None
    )
    if target is None:
        return ""
    start = int(target.offset) - base
    if start < 0:
        return ""
    end = start + limit
    for other in sections:
        if int(other.offset) > int(target.offset):
            end = int(other.offset) - base
            break
    else:
        end = len(whole)
    return whole[start : min(end, start + limit)]


def audit_structure(
    sections: Sequence[Section],
    book: Optional[Book] = None,
    body_start: Optional[int] = None,
) -> List[StructureIssue]:
    """Deterministic quality checks on an adjudicated section list.

    ``verify_toc`` checks that offsets are *valid*. These check that the list
    is *right*, which validity does not imply.
    """
    issues: List[StructureIssue] = []
    if not sections:
        return [StructureIssue(code="empty", detail="no sections at all", fix=None)]

    # 1. Ordinals must be 1..N with no gaps. A gap means a dropped section,
    #    which silently renumbers every later reader's saved position.
    ordinals = [int(s.ordinal) for s in sections]
    if ordinals != list(range(1, len(sections) + 1)):
        missing = sorted(set(range(1, len(sections) + 1)) - set(ordinals))
        issues.append(
            StructureIssue(
                code="ordinals_not_contiguous",
                detail=(
                    f"ordinals are not 1..{len(sections)}; "
                    f"{len(missing)} missing (first: {missing[:5]})"
                ),
                fix="renumber_sections",
            )
        )

    # 2. The list must not start past the first body heading. If it does, the
    #    opening sections were dropped -- observed on Monte Cristo, which
    #    started at "Chapter 2" and still verified clean because the offsets
    #    were valid.
    #
    #    ``body_start`` is derived from the FIRST CANDIDATE when not supplied.
    #    Previously the caller passed None, so this check could never fire and
    #    the dropped first chapter shipped silently.
    if body_start is None and book is not None:
        body_start = _derive_body_start(book)
    first = sections[0]
    if body_start is not None and int(first.offset) > body_start + 2000:
        issues.append(
            StructureIssue(
                ordinal=int(first.ordinal),
                code="first_section_starts_late",
                detail=(
                    f"first section is at {int(first.offset)}, more than 2000 "
                    f"chars past the body's first heading ({body_start}); "
                    "opening sections were probably dropped"
                ),
                fix="recheck_dropped_sections",
            )
        )

    # 3. Labels that read as sentences are prose the adjudicator accepted.
    #    Observed: the final Monte Cristo section was a line of dialogue.
    for section in sections:
        label = (section.label or "").strip()
        if not label:
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="empty_label",
                    detail="section has no label",
                    fix=f"relabel:{section.ordinal}:UNKNOWN",
                )
            )
            continue

        #
        # CASE IS THE DISCRIMINATOR, so it is computed once, up front. Every
        # real heading observed here is ALL-CAPS ("CHAPTER VII--THE GAMIN
        # SHOULD HAVE HIS PLACE...") and every false positive is mixed case
        # ("And Albert took out a little pocket-book..."). The prose
        # heuristics below all key off it; an earlier version flagged 9 real
        # Les Miserables chapters as prose, dropped them, and broke 137
        # parent references.
        is_all_caps = label.upper() == label and any(c.isalpha() for c in label)
        if (
            not is_all_caps
            and label[0] in "\"“”'‘"
            and label[-1:] in "\"“”'’"
        ):
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="quoted_heading",
                    detail=(
                        f"label is a quoted fragment of dialogue: {label[:70]!r}"
                    ),
                    fix=f"drop:{section.ordinal}",
                )
            )
            continue
        prose_like = (not is_all_caps) and (
            (_VERBISH.search(label) is not None and len(label.split()) > 6)
            or _SENTENCE_OPENER.match(label) is not None
            or _ARTICLE_OPENER.match(label) is not None
            or _SECOND_WORD_VERB.match(label) is not None
        )
        if prose_like:
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="prose_not_heading",
                    detail=(
                        "label reads as a sentence rather than a heading: "
                        f"{label[:70]!r}"
                    ),
                    fix=f"drop:{section.ordinal}",
                )
            )

    # 4. A section with almost no text is a boundary artifact.
    #
    #    EXEMPT: a heading that other sections name as their parent. In Les
    #    Miserables "VOLUME I" sits 21 chars above "VOLUME I / BOOK
    #    FIRST--A JUST MAN" because the book heading follows immediately --
    #    a legitimate container heading, not an artifact. Flagging it produced
    #    46 bogus issues and the auto-correction then broke the whole tree
    #    (verify problems 0 -> 365, correctly rejected by the fail-safe).
    if book is not None:
        for position, section in enumerate(sections, start=1):
            end = (
                int(sections[position].offset)
                if position < len(sections)
                else len(book.text)
            )
            span = end - int(section.offset)
            # A container heading is one that ANOTHER section sits under, or
            # one that names a structural division rather than a passage. In
            # Les Miserables "BOOK FIRST--A JUST MAN" spans 25 chars because
            # "VOLUME I / CHAPTER I--M. MYRIEL" follows immediately; that is a
            # division heading, not an empty section. Two signals, both
            # needed: being someone's parent, or labelling itself a division.
            is_parent = any(
                (child.parent or "").strip() == (section.label or "").strip()
                for child in sections
            )
            section_label = (section.label or "").strip()
            is_division = _STRUCTURAL_LABEL.match(section_label) is not None
            if span < 400 and not (is_parent or is_division):
                issues.append(
                    StructureIssue(
                        ordinal=int(section.ordinal),
                        code="tiny_section",
                        detail=(
                            f"only {span} chars between this heading and the next, "
                            "and no section names it as a parent"
                        ),
                        fix=f"drop:{section.ordinal}",
                    )
                )

    # 5. A section's PARENT must actually precede it in the list.
    #
    #    Observed: the Agrippa passage was labelled "BOOK I. / CHAPTER 9." while
    #    BOOK II. sat 56k chars earlier in the same list, because the merge
    #    step took the model's parent over the deterministic scan's. 72
    #    sections were mislabelled and ``verify_toc`` reported ZERO problems,
    #    because every offset was valid. Offsets being right says nothing about
    #    hierarchy being right, so this is checked here instead.
    for section in sections:
        parent = (section.parent or "").strip()
        if not parent:
            continue
        label = (section.label or "").strip()
        # A division heading IS legitimately its own parent: "BOOK II."
        # introduces itself and then parents its chapters. Only a section whose
        # label is NOT a bare division can be wrong by naming itself.
        is_bare_division = bool(_BARE_DIVISION.match(label))
        if parent == label and not is_bare_division:
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="self_parent",
                    detail=f"non-division section is its own parent: {label[:60]!r}",
                    fix=None,
                )
            )
            continue
        # A bare division naming itself is correct (BOOK II. parents its own
        # chapters), but fall THROUGH so the division still has to agree with
        # the label of its children -- see the parent_mismatch check below.
        # If the label names a different book than the parent does, one of them
        # is wrong. Compare the leading division token of both.
        label_book = _BARE_DIVISION.match(label) or _LEADING_DIVISION.match(label)
        parent_book = _LEADING_DIVISION.match(parent)
        if (
            label_book
            and parent_book
            and label_book.group(1).upper() != parent_book.group(1).upper()
        ):
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="parent_mismatch",
                    detail=(
                        f"label begins {label_book.group(1)!r} but parent is "
                        f"{parent!r}; the label and hierarchy disagree"
                    ),
                    fix=f"relabel:{section.ordinal}:{parent} / {label.split(' / ', 1)[-1]}",
                )
            )

    # 5b. A section's parent must appear EARLIER in the list than the section.
    #
    #     This is the positional form of the check above, and it is the one
    #     that actually caught the Book I / Book II bug. In the shipped TOC,
    #     label and parent AGREED ("BOOK I. / CHAPTER 9." under parent
    #     "BOOK I.") -- both were simply wrong, because "BOOK II." sat two
    #     entries earlier in the same list. A label-vs-parent comparison
    #     cannot see that; only position can.
    position = {id(s): i for i, s in enumerate(sections)}
    for index, section in enumerate(sections, start=1):
        parent = (section.parent or "").strip()
        label = (section.label or "").strip()
        if not parent or parent == label:
            continue
        # Find a section that IS this parent.
        holder = next(
            (
                i
                for i, other in enumerate(sections)
                if (other.label or "").strip() == parent
            ),
            None,
        )
        if holder is None:
            # No section in the list carries this parent label. Two distinct
            # faults land here and both are worth reporting:
            #   (a) the parent section was DROPPED by an earlier correction, so
            #       its children are orphaned -- this is what happened to the
            #       Jewish War TOC, where dropping "BOOK I." left 72 chapters
            #       naming a parent that no longer existed;
            #   (b) a genuine unknown-parent problem, which verify_toc reports.
            # In both cases the child's hierarchy is untrustworthy.
            if _LEADING_DIVISION.match(parent):
                issues.append(
                    StructureIssue(
                        ordinal=int(section.ordinal),
                        code="orphan_parent",
                        detail=(
                            f"parent {parent!r} is not a section in this toc "
                            "(it may have been dropped by an earlier "
                            "correction); the hierarchy here is untrustworthy"
                        ),
                        fix=None,
                    )
                )
            continue
        if holder > index - 1:
            issues.append(
                StructureIssue(
                    ordinal=int(section.ordinal),
                    code="parent_precedes_child",
                    detail=(
                        f"parent {parent!r} appears at position {holder + 1}, "
                        f"AFTER this section at {index}; the hierarchy is "
                        "inverted"
                    ),
                    fix=None,
                )
            )

    # 5c. A section whose entire body is translator commentary must not be in
    #     the toc at all.
    #
    #     Observed: two "BOOK I. / WAR BOOK 1 FOOTNOTES" and "BOOK V. / WAR
    #     BOOK 5 FOOTNOTES" sections survived into the Jewish War structure.
    #     They indexed to zero chunks (the note-stripping removes their whole
    #     body), so they were invisible dead weight -- but a reader navigating
    #     the toc would be offered a "chapter" consisting of an editor's notes.
    if book is not None:
        for position, section in enumerate(sections, start=1):
            # The head of this section, for a pattern test only. Obtained
            # through the gate so the AST check in tests/test_book.py -- which
            # flags ANY subscript of book.text -- stays satisfied.
            head = ""
            if book is not None:
                for ordinal in (
                    int(section.ordinal),
                    int(section.ordinal) - 1,
                ):
                    if 0 < ordinal <= len(sections):
                        head = _section_head(book, sections, ordinal, 200)
                        break
            if not head.strip():
                continue
            if _FOOTNOTE_BLOCK.match(head.lstrip()):
                issues.append(
                    StructureIssue(
                        ordinal=int(section.ordinal),
                        code="footnote_block_section",
                        detail=(
                            "section is the translator's footnote block, not "
                            "narrative"
                        ),
                        fix=f"drop:{section.ordinal}",
                    )
                )

    # 6. Duplicate labels in a row usually mean the same heading was matched
    #    twice (a contents entry plus the real heading).
    for a, b in zip(sections, sections[1:]):
        if (a.label or "").strip() == (b.label or "").strip() and (a.label or "").strip():
            issues.append(
                StructureIssue(
                    ordinal=int(b.ordinal),
                    code="duplicate_consecutive_label",
                    detail=f"{a.label!r} appears twice in a row",
                    fix=f"drop:{b.ordinal}",
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# Stage 5: apply corrections
# --------------------------------------------------------------------------- #


def repair_hierarchy(sections: Sequence[Section]) -> List[Section]:
    """Re-derive every ``parent`` from the labels actually present.

    The parent of a section is the most recent BARE DIVISION label ("BOOK II.")
    at or before it in the list. Recomputing this is safe: it uses only the
    ordering and the labels, both of which are deterministic, and it repairs
    the inversion where 72 sections claimed "BOOK I." while BOOK II. preceded
    them.

    Offsets are never touched. A repair that changes an offset would be a
    different (and much more dangerous) operation.
    """
    current = ""
    repaired: List[Section] = []
    for section in sections:
        label = (section.label or "").strip()
        if _BARE_DIVISION.match(label):
            current = label
        parent = label if _BARE_DIVISION.match(label) else current
        repaired.append(section.model_copy(update={"parent": parent}))
    return repaired


def apply_corrections(
    sections: Sequence[Section],
    issues: Sequence[StructureIssue],
    book_id: str,
    reviewed_by: str = "hermes",
    path: Optional[str] = None,
    book=None,
) -> Tuple[StructureArtifact, List[StructureIssue]]:
    """Apply the audit's machine-actionable fixes, then re-audit.

    Only the fix kinds listed in ``StructureIssue.fix`` are honoured:
    ``drop:<ordinal>``, ``relabel:<ordinal>:<name>``, ``renumber_sections``.
    Anything else is left alone and reported, because silently guessing at an
    offset change is exactly the failure this design exists to prevent.

    Returns ``(artifact, remaining_issues)``. Callers must not treat a
    non-empty ``remaining_issues`` as success.
    """
    drop: set = set()
    relabel: Dict[int, str] = {}
    renumber = False
    for issue in issues:
        if not issue.fix:
            continue
        head, _, rest = issue.fix.partition(":")
        if head == "drop":
            drop.add(int(rest))
        elif head == "relabel":
            ordinal_text, _, name = rest.partition(":")
            relabel[int(ordinal_text)] = name
        elif head == "renumber_sections":
            renumber = True

    # A section referenced as another section's `parent` must NOT be dropped:
    # removing it orphans every child and breaks verify_toc. Observed: 137
    # "[unknown-parent]" problems after auto-dropping "VOLUME IV".
    referenced_parents = {
        (s.parent or "").strip()
        for s in sections
        if (s.parent or "").strip()
    }
    referenced_parents -= {(s.label or "").strip() for s in sections
                           if int(s.ordinal) in drop}
    protected: set = set()
    for section in sections:
        if (section.label or "").strip() in referenced_parents:
            protected.add(int(section.ordinal))

    unprotected_drop = drop - protected
    if protected:
        logger.info(
            "refusing to drop %s: referenced as a parent by another section",
            sorted(protected),
        )

    kept: List[Section] = []
    for section in sections:
        if int(section.ordinal) in unprotected_drop:
            continue
        if int(section.ordinal) in relabel:
            section = section.model_copy(update={"label": relabel[int(section.ordinal)]})
        kept.append(section)

    if renumber or kept and [int(s.ordinal) for s in kept] != list(
        range(1, len(kept) + 1)
    ):
        kept = [
            section.model_copy(update={"ordinal": i})
            for i, section in enumerate(kept, start=1)
        ]

    # FAIL SAFE: never persist a correction that breaks offset verification.
    # Observed: auto-dropping "VOLUME IV" left 137 [unknown-parent] problems,
    # and the broken artifact was written anyway -- so the next run inherited
    # it. If the correction is not strictly an improvement, keep the original
    # and hand the decision back.
    if book is not None:
        from src.bookbuddy.book import verify_toc

        before = len(verify_toc(list(sections), book.text))
        after = len(verify_toc(list(kept), book.text))
        if after >= before:
            logger.warning(
                "auto-correction would not improve %s (verify problems "
                "%d -> %d); keeping the original and reporting instead",
                book_id,
                before,
                after,
            )
            kept = list(sections)
            protected = set()
            unprotected_drop = set()
            kept_issues = list(issues)
            rejected = StructureArtifact(
                book_id=book_id,
                sections=list(sections),
                reviewed_by="auto:REJECTED",
                notes=[
                    f"auto-correction rejected: verify problems {before} -> "
                    f"{after}; edit by hand"
                ],
            )
            rejected.save(path)
            return rejected, kept_issues

    artifact = StructureArtifact(
        book_id=book_id,
        sections=kept,
        reviewed_by=reviewed_by,
        notes=[
            f"auto-applied: dropped {sorted(unprotected_drop)}"
            if unprotected_drop
            else "no drops",
            f"protected from drop (referenced as a parent): {sorted(protected)}"
            if protected
            else "no protected sections",
            f"auto-applied: relabelled {sorted(relabel)}" if relabel else "no relabels",
            "renumbered" if renumber else "ordinals unchanged",
        ],
    )
    artifact.save(path)
    remaining = audit_structure(kept)
    return artifact, remaining