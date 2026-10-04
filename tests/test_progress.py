"""Tests for natural-language reading-position resolution.

Deterministic: no network. The model is either disabled with ``use_model=False``
or supplied as a fake ``(prompt, system) -> text`` callable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from src.bookbuddy.book import Section
from src.bookbuddy.progress import (
    AMBIGUITY_MARGIN,
    MIN_CONFIDENCE,
    Candidate,
    ResolvedPosition,
    apply_forward_rule,
    candidate_text,
    format_ambiguity_question,
    format_toc,
    resolve_position,
    score_sections,
    summarize_toc,
    top_gap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JEWISH_WAR = REPO_ROOT / "data" / "jewish_war.txt"


@pytest.fixture(scope="module")
def toc() -> List[Section]:
    """A flattened 2-book, 6-chapter TOC with ordinals 1..8."""
    sections: List[Section] = []
    plan = [
        ("BOOK I.", "", "Antiochus Epiphanes takes Jerusalem"),
        ("BOOK I. / CHAPTER 1.", "BOOK I.", "The city is taken"),
        ("BOOK I. / CHAPTER 2.", "BOOK I.", "The temple is pillaged"),
        ("BOOK I. / CHAPTER 3.", "BOOK I.", "The Jews revolt"),
        ("BOOK II.", "", "Vespasian and Titus"),
        ("BOOK II. / CHAPTER 1.", "BOOK II.", "The revolt spreads"),
        ("BOOK II. / CHAPTER 2.", "BOOK II.", "The siege of Jerusalem"),
        ("BOOK II. / CHAPTER 3.", "BOOK II.", "The Temple is destroyed"),
    ]
    for ordinal, (label, parent, title) in enumerate(plan, start=1):
        sections.append(
            Section(
                ordinal=ordinal,
                label=label,
                parent=parent,
                offset=ordinal * 1000,
                title=title,
            )
        )
    return sections


def _client_returning(index, evidence: str = "because"):
    def client(prompt: str, system: str) -> str:
        return json.dumps({"index": index, "evidence": evidence})

    return client


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_score_sections_returns_every_section_sorted(toc):
    scored = score_sections(toc, "I finished chapter 2 of book 1")
    assert len(scored) == len(toc)
    scores = [c.score for c in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_sections_empty_toc():
    assert score_sections([], "anything") == []
    assert top_gap([]) is None


def test_book_mention_is_detected(toc):
    scored = score_sections(toc, "I just finished book 2")
    assert scored[0].ordinal == 5
    assert "BOOK II." in scored[0].label


def test_book_and_chapter_mention_is_more_specific(toc):
    scored = score_sections(toc, "I read up to book 2 chapter 3")
    assert scored[0].ordinal == 8


def test_word_numbers_are_understood(toc):
    scored = score_sections(toc, "I finished book two chapter three")
    assert scored[0].ordinal == 8


def test_top_gap_of_single_candidate():
    assert top_gap([Candidate(ordinal=1, label="X", score=50)]) == 100.0


def test_top_gap_measured():
    scored = [
        Candidate(ordinal=1, label="A", score=90.0),
        Candidate(ordinal=2, label="B", score=80.0),
    ]
    assert top_gap(scored) == 10.0


def test_candidate_text_includes_parent_and_title(toc):
    text = candidate_text(toc[1])
    assert "BOOK I." in text
    assert "The city is taken" in text


def test_candidate_display(toc):
    candidate = Candidate(
        ordinal=2, label="BOOK I. / CHAPTER 1.", title="The city is taken", score=90
    )
    assert candidate.display == "BOOK I. / CHAPTER 1. — The city is taken"
    plain = Candidate(ordinal=2, label="BOOK I. / CHAPTER 1.", score=90)
    assert plain.display == "BOOK I. / CHAPTER 1."


# --------------------------------------------------------------------------- #
# Ambiguity: never silently guess
# --------------------------------------------------------------------------- #


def test_bare_chapter_number_in_two_books_is_ambiguous():
    """'chapter 3' matches ordinal 4 (book 1) and ordinal 8 (book 2)."""
    sections = [
        Section(ordinal=1, label="BOOK I. / CHAPTER 1.", parent="BOOK I.", offset=1000),
        Section(ordinal=2, label="BOOK I. / CHAPTER 3.", parent="BOOK I.", offset=2000),
        Section(
            ordinal=3, label="BOOK II. / CHAPTER 1.", parent="BOOK II.", offset=3000
        ),
        Section(
            ordinal=4, label="BOOK II. / CHAPTER 3.", parent="BOOK II.", offset=4000
        ),
    ]
    scored = score_sections(sections, "i finished chapter 3")
    assert len({c.ordinal for c in scored[:2]}) == 2
    result = resolve_position(sections, "i finished chapter 3", use_model=False)
    assert result.ambiguous is True
    assert "ambiguous" in result.reason
    assert result.resolved is False


def test_unrecognisable_input_is_not_resolved(toc):
    result = resolve_position(toc, "zzzz qqqq wwww", use_model=False)
    assert result.ambiguous is True
    assert result.confidence < MIN_CONFIDENCE
    assert result.resolved is False


def test_empty_toc_is_not_resolved():
    result = resolve_position([], "chapter 3", use_model=False)
    assert result.ordinal is None
    assert result.reason


def test_empty_user_text_returns_empty_candidates(toc):
    result = resolve_position(toc, "", use_model=False)
    assert result.candidates
    assert result.ordinal is not None


def test_ambiguity_question_lists_both_candidates():
    sections = [
        Section(ordinal=1, label="BOOK I. / CHAPTER 3.", parent="BOOK I.", offset=1000),
        Section(
            ordinal=2, label="BOOK II. / CHAPTER 3.", parent="BOOK II.", offset=2000
        ),
    ]
    result = resolve_position(sections, "chapter 3", use_model=False)
    question = format_ambiguity_question(result)
    assert "not sure" in question
    assert "1." in question and "2." in question


def test_ambiguity_question_with_one_candidate():
    question = format_ambiguity_question(ResolvedPosition())
    assert "could you tell me" in question


def test_confidence_is_bounded(toc):
    result = resolve_position(toc, "book 2 chapter 3 temple destroyed", use_model=False)
    assert 0.0 <= result.confidence <= 100.0


# --------------------------------------------------------------------------- #
# Constrained selection: the model cannot invent a chapter
# --------------------------------------------------------------------------- #


def test_model_index_is_honoured(toc):
    result = resolve_position(
        toc,
        "I read up to chapter 2 of book 2",
        client=_client_returning(7, "The siege of Jerusalem"),
    )
    assert result.ordinal == 7
    assert result.evidence == "The siege of Jerusalem"


def test_model_cannot_invent_an_index_outside_the_toc(toc):
    result = resolve_position(
        toc,
        "some vague description",
        client=_client_returning(9999, "invented"),
    )
    assert result.ordinal in {s.ordinal for s in toc}
    assert result.ordinal != 9999


def test_model_returning_null_falls_back_to_lexical(toc):
    def client(prompt: str, system: str) -> str:
        return json.dumps({"index": None, "evidence": ""})

    result = resolve_position(toc, "book 2 chapter 1", client=client)
    assert result.ordinal == 6


def test_model_garbage_falls_back_to_lexical(toc):
    def client(prompt: str, system: str) -> str:
        return "I cannot determine that."

    result = resolve_position(toc, "book 2 chapter 1", client=client)
    assert result.ordinal == 6


def test_model_prompt_shows_a_numbered_array(toc):
    captured = {}

    def client(prompt: str, system: str) -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return json.dumps({"index": 6, "evidence": "x"})

    resolve_position(toc, "book 2 chapter 1", client=client)
    prompt = captured["prompt"]
    for section in toc:
        assert f"[{section.ordinal}]" in prompt
    assert "integer" in captured["system"].lower()
    assert "INDEX" in prompt or "index" in prompt


def test_model_failure_does_not_crash(toc):
    def boom(prompt: str, system: str) -> str:
        raise RuntimeError("network down")

    result = resolve_position(toc, "book 2 chapter 1", client=boom)
    assert result.ordinal == 6


def test_model_response_may_promote_a_lexical_shortlist_entry(toc):
    """For a prose description the model can pick a different section than the
    lexical top, as long as it is in the shortlist."""
    utterance = "the countryside and the fields"
    lexical_top = resolve_position(toc, utterance, use_model=False)
    assert lexical_top.candidates[0].score < 60.0
    target = 3 if lexical_top.ordinal != 3 else 4
    assert target != lexical_top.ordinal

    def client(prompt: str, system: str) -> str:
        return json.dumps({"index": target, "evidence": "a description"})

    result = resolve_position(toc, utterance, client=client)
    assert result.ordinal == target
    assert result.candidates[0].ordinal == target
    assert result.ambiguous is False


def test_model_index_is_rejected_when_far_outside_the_shortlist(toc):
    """A 30-deep shortlist over a huge toc still bounds what the model can do."""
    sections = [
        Section(ordinal=i, label=f"CHAPTER {i}.", offset=i * 100, title="")
        for i in range(1, 201)
    ]
    result = resolve_position(
        sections,
        "a vague memory of something",
        client=_client_returning(180, "way down the list"),
        shortlist=5,
    )
    assert result.ordinal != 180
    assert result.ordinal <= 200


def test_model_cannot_overrule_a_deterministic_number_parse(toc):
    """Live-tested regression: for "I just finished book 4" the model picked
    BOOK IV. / CHAPTER 11. and cited BOOK V as its evidence. Numbers are
    parseable, so the deterministic parse wins and the model is advisory only.
    """
    result = resolve_position(
        toc, "book 2 chapter 1", client=_client_returning(5, "the book opening")
    )
    assert result.ordinal == 6
    assert result.ambiguous is False


def test_model_cannot_suppress_a_lexical_ambiguity(toc):
    """Spec: never silently guess when the top-2 candidates are close. A model
    that confidently picks one of the tied sections has still guessed."""
    result = resolve_position(
        toc,
        "chapter 3",
        client=_client_returning(4, "BOOK I. / CHAPTER 3."),
    )
    assert result.ambiguous is True
    assert result.resolved is False
    assert result.ordinal in {4, 8}
    assert "gap 0.0" in result.reason


def test_evidence_is_returned(toc):
    result = resolve_position(
        toc,
        "book 2",
        client=_client_returning(6, '"the revolt spreads"'),
    )
    assert "revolt" in result.evidence


def test_bare_book_mention_picks_the_book_section(toc):
    result = resolve_position(
        toc,
        "i just finished book 2",
        client=_client_returning(5, "Vespasian"),
    )
    assert result.ordinal == 5
    # Lexically the book heading must clearly outrank its own chapters,
    # otherwise "book 2" is a tie between BOOK II. and BOOK II. / CHAPTER 1.
    lexical = resolve_position(toc, "book 2", use_model=False)
    assert lexical.ordinal == 5
    assert lexical.ambiguous is False


def test_model_decides_when_the_lexical_layer_has_no_signal(toc):
    """A vague description where every section scores alike is not a genuine
    tie — it is an absence of signal. The model's in-range answer is taken."""
    vague = "I'm somewhere in the middle of the fighting"
    lexical = resolve_position(toc, vague, use_model=False)
    assert lexical.ambiguous is True
    assert lexical.candidates[0].score < 60.0

    result = resolve_position(toc, vague, client=_client_returning(7, "the siege"))
    assert result.ordinal == 7
    assert result.ambiguous is False


# --------------------------------------------------------------------------- #
# Forward-only progress
# --------------------------------------------------------------------------- #


def test_forward_progress_is_applied(toc):
    result = resolve_position(
        toc, "book 2 chapter 2", current_ordinal=2, use_model=False
    )
    assert result.ordinal == 7
    assert result.needs_confirmation is False


def test_rewind_is_clamped_and_flagged(toc):
    result = resolve_position(
        toc, "book 1 chapter 1", current_ordinal=7, use_model=False
    )
    assert result.ordinal == 7
    assert result.clamped_from == 2
    assert result.needs_confirmation is True
    assert "behind current progress" in result.reason


def test_apply_forward_rule_is_a_noop_when_not_rewinding():
    position = ResolvedPosition(ordinal=5, confidence=90.0)
    same = apply_forward_rule(position, 3)
    assert same.ordinal == 5
    assert same.needs_confirmation is False


def test_apply_forward_rule_handles_unresolved():
    position = ResolvedPosition(ordinal=None)
    same = apply_forward_rule(position, 4)
    assert same.ordinal is None


def test_progress_cannot_be_moved_backwards_by_a_model(toc):
    """Even if the model confidently says chapter 1, we hold the position."""
    result = resolve_position(
        toc,
        "i went back to the beginning",
        client=_client_returning(2, "the city is taken"),
        current_ordinal=8,
    )
    assert result.ordinal == 8
    assert result.needs_confirmation is True


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_format_toc_uses_ordinals_as_indices(toc):
    rendered = format_toc(toc)
    lines = rendered.split("\n")
    assert len(lines) == len(toc)
    assert lines[0].startswith("[1] BOOK I.")
    assert lines[-1].startswith("[8] ")


def test_format_toc_limit(toc):
    rendered = format_toc(toc, limit=3)
    lines = rendered.split("\n")
    assert len(lines) == 4
    assert "further sections omitted" in lines[-1]


def test_summarize_toc_matches_format_toc(toc):
    assert summarize_toc(toc, limit=4) == format_toc(toc, limit=4)


# --------------------------------------------------------------------------- #
# Real-book TOC (deterministic build, no network)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_toc() -> List[Section]:
    from src.bookbuddy.book import build_toc, load_book, verify_toc

    book = load_book(str(JEWISH_WAR), book_id="jewish_war")
    toc = build_toc(book, client=lambda p, s: "[]", use_cache=False)
    assert verify_toc(toc, book.text) == []
    return toc


def test_real_book_resolution_of_a_named_book(real_toc):
    result = resolve_position(
        real_toc, "I finished book 4", title="The Jewish War", use_model=False
    )
    assert result.ordinal is not None
    assert "BOOK IV." in format_toc([real_toc[result.ordinal - 1]])


def test_real_book_bare_chapter_is_ambiguous(real_toc):
    result = resolve_position(real_toc, "chapter 3", use_model=False)
    assert result.ambiguous is True
    assert result.resolved is False


def test_real_book_book_and_chapter_is_resolved(real_toc):
    result = resolve_position(
        real_toc, "I read up to book 2 chapter 5", use_model=False
    )
    assert result.ambiguous is False
    assert result.resolved is True
    section = real_toc[result.ordinal - 1]
    assert "BOOK II." in section.parent or "BOOK II." in section.label


def test_real_book_resolved_position_is_within_bounds(real_toc):
    result = resolve_position(real_toc, "book 7 chapter 2", use_model=False)
    assert 1 <= result.ordinal <= len(real_toc)
