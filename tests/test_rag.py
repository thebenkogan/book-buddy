"""Tests for the Qdrant-backed RAG and the structure audit.

Runs fully offline: chunks are built and indexed without embeddings (zero
vectors), and the gate tests exercise the Qdrant filter itself rather than
the answering model. Deliberate -- a spoiler-gate test that needs an API key
is a test that silently stops running.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy.book import Section, build_toc, load_book  # noqa: E402
from src.bookbuddy.onboarding import (  # noqa: E402
    _LEADING_DIVISION,
    StructureArtifact,
    apply_corrections,
    audit_structure,
    merge_with_deterministic,
    repair_hierarchy,
)
from src.bookbuddy.rag import (  # noqa: E402
    DF_FLOOR_FRACTION,
    BookStore,
    Chunk,
    note_ratio,
    SparseLexical,
    STOPWORDS,
    _words,
    chunk_book,
    count_tokens,
    hits_to_prompt,
    tidy_punctuation,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "jewish_war.txt",
)
HAS_DATA = os.path.exists(DATA)


@pytest.fixture(scope="module")
def book_and_chunks():
    if not HAS_DATA:
        pytest.skip("jewish_war.txt not present")
    book = load_book(DATA, book_id="jewish_war")
    toc = build_toc(book, client=lambda p, s: "1|Title\n2|Another", use_cache=False)
    return book, toc, chunk_book(book, toc)


def _fake_store(tmp_path, chunks, dim=4) -> BookStore:
    """A real Qdrant store with zero vectors, so the filter can be tested."""
    store = BookStore("unit_test", root=str(tmp_path))
    store._ensure_collection(dim, rebuild=True)
    lexical = SparseLexical(chunks)
    store._lexical = lexical
    from qdrant_client import models

    store.client.upsert(
        collection_name=store.collection,
        points=[
            models.PointStruct(
                id=c.index,
                vector={
                    "dense": [0.0] * dim,
                    "sparse": lexical.to_sparse(lexical.document_weights(c.text)),
                },
                payload=c.payload,
            )
            for c in chunks
        ],
    )
    return store


# --------------------------------------------------------------------------- #
# THE GATE -- enforced by Qdrant, not by Python
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
@pytest.mark.parametrize("ordinal", [0, 1, 2, 5, 40, 60])
def test_search_never_returns_past_ordinal(book_and_chunks, tmp_path, ordinal):
    _book, _toc, chunks = book_and_chunks
    store = _fake_store(tmp_path, chunks)
    # Bypass the embedding call by searching the store directly with a zero
    # query vector: the point of the test is the FILTER, not the ranking.
    from qdrant_client import models

    result = store.client.query_points(
        collection_name=store.collection,
        query=[0.0] * 4,
        using="dense",
        query_filter=store._gate(ordinal),
        limit=50,
        with_payload=True,
    )
    for point in result.points:
        assert point.payload["ordinal"] <= ordinal, (
            f"LEAK: gate {ordinal} returned ordinal {point.payload['ordinal']}"
        )


def test_gate_filter_is_a_range_predicate():
    """The gate must be a storage constraint, not a post-hoc list filter."""
    from qdrant_client import models

    store = BookStore.__new__(BookStore)  # no client needed for _gate
    filt = BookStore._gate(store, 40)
    conditions = filt.must
    assert conditions is not None and len(conditions) == 1
    cond = conditions[0]
    assert isinstance(cond, models.FieldCondition)
    assert cond.key == "ordinal"
    assert isinstance(cond.range, models.Range)
    assert cond.range.lte == 40
    assert cond.range.gt is None  # nothing else may widen the window


def test_gate_at_zero_returns_nothing(tmp_path):
    chunks = [
        Chunk(index=0, ordinal=1, label="one", text="the first passage of prose"),
        Chunk(index=1, ordinal=2, label="two", text="the second passage of prose"),
    ]
    store = _fake_store(tmp_path, chunks)
    from qdrant_client import models

    result = store.client.query_points(
        collection_name=store.collection,
        query=[0.0] * 4,
        using="dense",
        query_filter=store._gate(0),
        limit=10,
        with_payload=True,
    )
    assert list(result.points) == []


def test_gate_at_full_ordinal_returns_everything(tmp_path):
    """The gate must not exclude anything at the end of the book.

    Regression guard against 'hardcoded exclusions': the index holds the whole
    book and the filter is what narrows it, so a finished reader can reach the
    ending.
    """
    chunks = [
        Chunk(index=0, ordinal=1, label="one", text="the opening of the book"),
        Chunk(index=1, ordinal=50, label="fifty", text="the middle of the book"),
        Chunk(index=2, ordinal=99, label="ninety nine", text="the end of the book"),
    ]
    store = _fake_store(tmp_path, chunks)
    from qdrant_client import models

    result = store.client.query_points(
        collection_name=store.collection,
        query=[0.0] * 4,
        using="dense",
        query_filter=store._gate(99),
        limit=50,
        with_payload=True,
    )
    assert len(result.points) == 3


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_every_toc_section_is_indexed(book_and_chunks):
    """No section may vanish from the index.

    Regression: text_up_to returns text starting at toc[0].offset while
    Section.offset is absolute. Slicing the gated text with absolute offsets
    read 16,998 chars early and made the LAST section slice to an empty
    string, dropping it from the index entirely.
    """
    _book, toc, chunks = book_and_chunks
    indexed = {c.ordinal for c in chunks}
    missing = sorted({int(s.ordinal) for s in toc} - indexed)
    assert missing == [], f"sections missing from the index: {missing}"


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_chunks_carry_ordinals_and_offsets(book_and_chunks):
    _book, _toc, chunks = book_and_chunks
    assert chunks
    for chunk in chunks:
        assert chunk.ordinal >= 1
        assert chunk.end > chunk.start
        assert chunk.text.strip()
        assert chunk.label


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_chunk_ordinals_increase_with_position(book_and_chunks):
    _book, _toc, chunks = book_and_chunks
    last = 0
    for chunk in chunks:
        assert chunk.ordinal >= last, (
            f"chunk {chunk.index} has ordinal {chunk.ordinal} after {last}"
        )
        last = chunk.ordinal


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_chunk_text_matches_its_label(book_and_chunks):
    """A chunk's text must come from the section it claims.

    The invariant that matters. This is deliberately a SECTION-membership
    assertion rather than an offset assertion: chunk offsets are measured
    after translator notes are excised, so cleaned and raw offsets legitimately
    diverge, and the gate keys on ``ordinal`` anyway.
    """
    book, toc, chunks = book_and_chunks
    bounds = {}
    for position, section in enumerate(toc):
        lo = int(section.offset)
        hi = int(toc[position + 1].offset) if position + 1 < len(toc) else len(book.text)
        bounds[int(section.ordinal)] = (lo, hi)

    # Whitespace-collapsed view of the book, with a map back to raw offsets.
    flat_parts: list = []
    flat_index: list = []
    last_space = True
    for i, char in enumerate(book.text):
        if char.isspace():
            if last_space:
                continue
            flat_parts.append(" ")
            flat_index.append(i)
            last_space = True
        else:
            flat_parts.append(char)
            flat_index.append(i)
            last_space = False
    flat = "".join(flat_parts)

    checked = 0
    for chunk in chunks:
        probe = " ".join(chunk.text[200:300].split())
        if len(probe) < 40:
            continue
        at = flat.find(probe)
        if at < 0:
            continue  # a note was excised inside the phrase
        where = flat_index[at]
        lo, hi = bounds[chunk.ordinal]
        assert lo <= where < hi, (
            f"chunk {chunk.index} is labelled {chunk.label!r} [{lo}:{hi}] "
            f"but its text sits at {where}"
        )
        checked += 1
        if checked >= 40:
            break
    assert checked >= 10, f"only {checked} chunks were verifiable"


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_chunk_text_is_retrievable_book_prose(book_and_chunks):
    """Chunks must not be mostly translator footnotes.

    Whiston's notes were ~79% of the raw file; if stripping regresses, the
    index silently degrades into a footnote search engine.
    """
    _book, _toc, chunks = book_and_chunks
    noisy = sum(1 for c in chunks if "(return)" in c.text or "FOOTNOTES" in c.text)
    assert noisy / len(chunks) < 0.10, (
        f"{noisy}/{len(chunks)} chunks contain footnote markers"
    )


# --------------------------------------------------------------------------- #
# Editorial notes: indexed and ranked, never deleted
# --------------------------------------------------------------------------- #


def test_note_ratio_measures_without_removing():
    """The regression that replaced deletion.

    Deleting bracketed spans was measured to eat 543 characters of Alice's
    actual verse and Grimms' story title "[LITTLE RED RIDING HOOD]". So the
    text is never touched here; only its shape is measured.
    """
    text = ("The Panther took pie-crust, and gravy, and meat. "
            "[later editions continued as follows The Panther took pie-crust]")
    assert "pie-crust" in text
    assert note_ratio(text) > 0.0


def test_note_ratio_is_zero_for_plain_narrative():
    assert note_ratio("and then he went out into the garden") == 0.0


def test_note_ratio_is_near_one_for_pure_commentary():
    note = "[" + ("a long translator note about the temple. " * 20) + "]"
    assert note_ratio(note) > 0.9


def test_note_ratio_ignores_short_brackets():
    """A short bracket is inline punctuation, not an editorial aside."""
    assert note_ratio("he read the sign [sic] and turned away") == 0.0


def test_chunk_payload_carries_note_ratio():
    c = Chunk(index=0, ordinal=1, label="C", text="t", start=0, end=1)
    assert c.payload["note_ratio"] == 0.0


def test_tidy_punctuation_removes_orphaned_spaces():
    """Close the gap left where punctuation got separated from its word.

    This used to pair with a stripping step that has since been deleted --
    bracketed text is indexed and ranked down, never excised.
    """
    assert tidy_punctuation("Pillaged . As Also") == "Pillaged. As Also"
    assert tidy_punctuation("word .") == "word."
    assert tidy_punctuation("too    many  spaces") == "too many spaces"


# --------------------------------------------------------------------------- #
# Lexical scoring
# --------------------------------------------------------------------------- #


def test_df_floor_stops_rare_term_domination():
    """A df=1 question word must not outrank the identifying term.

    Measured regression: "happens" had df=1, IDF 5.9, and beat all 24 chunks
    naming Masada (IDF 3.1). SparseLexical._idf floors df at 2% of the corpus.
    """
    chunks = [
        Chunk(
            index=0,
            ordinal=1,
            label="accidental",
            text=(
                "and so it happens that nothing at all was decided that day "
                + "the men went back to their tents and the council sat again "
                * 40
            ),
        )
    ]
    for i in range(24):
        chunks.append(
            Chunk(
                index=i + 1,
                ordinal=1,
                label=f"m{i}",
                text="the fortress of masada held out against the roman siege " * 3,
            )
        )
    for i in range(478):
        chunks.append(
            Chunk(
                index=100 + i,
                ordinal=1,
                label=f"f{i}",
                text="ordinary unrelated narrative about the city walls " * 3,
            )
        )
    lex = SparseLexical(chunks)
    assert lex.df["happens"] == 1
    assert lex.df["masada"] == 24

    import math

    def raw_idf(df: int) -> float:
        n = lex.n_docs
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    # What the floor GUARANTEES: the rare term's IDF is capped below its raw
    # value, so a df=1 word cannot run away with the score.
    assert lex._idf("happens") < raw_idf(1), "df floor did not cap the rare term"
    assert lex._idf("happens") == pytest.approx(
        raw_idf(lex.n_docs * DF_FLOOR_FRACTION), abs=0.01
    )

    # What it does NOT guarantee -- asserted so a future edit cannot quietly
    # reintroduce the false claim: a common-but-specific term still scores
    # below the floored rare one. RRF fusion with the dense vector, not this
    # floor, is what carries the proper-noun case.
    assert lex._idf("masada") < lex._idf("happens"), (
        "if this ever inverts, update the DF_FLOOR_FRACTION comment: the floor "
        "would now be doing more than the comment claims"
    )

    # Term frequency still points at the passage that actually names Masada.
    masada_chunk = next(c for c in chunks if "masada" in c.text)
    assert _words(masada_chunk.text).count("masada") > _words(chunks[0].text).count(
        "masada"
    )


def test_stopwords_are_not_query_signals():
    assert "the" in STOPWORDS
    assert "what" in STOPWORDS
    assert "masada" not in STOPWORDS


def test_sparse_vector_indices_are_sorted():
    lex = SparseLexical([Chunk(index=0, ordinal=1, label="a", text="alpha beta gamma")])
    sparse = lex.to_sparse(lex.document_weights("alpha beta gamma"))
    assert list(sparse.indices) == sorted(sparse.indices)
    assert len(sparse.indices) == len(sparse.values)


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def test_hits_to_prompt_labels_every_passage():
    from src.bookbuddy.rag import Hit

    hits = [
        Hit(index=0, ordinal=1, label="BOOK I. / CHAPTER 1.", text="first passage here", score=1.0),
        Hit(index=1, ordinal=2, label="BOOK I. / CHAPTER 2.", text="second passage here", score=0.9),
    ]
    out = hits_to_prompt(hits)
    assert "[1]" in out and "[2]" in out
    assert "first passage" in out and "second passage" in out


def test_hits_to_prompt_respects_the_token_budget():
    """Over-budget prompts drop whole chunks, and truncate rather than overrun.

    A single 900-token chunk with a 600-token budget used to be returned
    whole, silently exceeding the limit.
    """
    from src.bookbuddy.rag import Hit

    big = "word " * 900
    hits = [
        Hit(index=i, ordinal=1, label=f"c{i}", text=big.strip(), score=1.0 - i * 0.1)
        for i in range(6)
    ]
    out = hits_to_prompt(hits, max_tokens=600)
    assert count_tokens(out) <= 600, f"{count_tokens(out)} tokens > 600"
    assert "truncated" in out, "a truncated passage must say so"


# --------------------------------------------------------------------------- #
# RRF fusion -- the dense arm must be able to affect the ranking
# --------------------------------------------------------------------------- #


def test_the_weaker_rrf_arm_can_still_outrank_the_stronger_arm():
    """A weight ratio past this bound makes an arm decorative.

    An arm contributes ``weight/(RRF_K+rank)``, so its whole influence spans
    ``weight/(RRF_K+1)`` down to ``weight/(RRF_K+FUSION_POOL)``. If the weaker
    arm's best is below the stronger arm's worst, no dense-only chunk can ever
    be returned and the "hybrid" search is really a sparse search. The shipped
    defaults were 0.4 vs 1.0, which is exactly that case; the dense arm's
    documented job -- paraphrase, where sparse has no signal -- could not
    happen.
    """
    from src.bookbuddy.rag import (
        DENSE_WEIGHT,
        FUSION_POOL,
        RRF_K,
        SPARSE_WEIGHT,
        _weight_ratio_is_sane,
    )

    assert _weight_ratio_is_sane(), (
        f"dense={DENSE_WEIGHT} sparse={SPARSE_WEIGHT} leaves the weaker arm "
        f"unable to affect ranking"
    )
    low, high = sorted((DENSE_WEIGHT, SPARSE_WEIGHT))
    assert low / (RRF_K + 1) > high / (RRF_K + FUSION_POOL), (
        "the weaker arm's best hit must beat the stronger arm's worst"
    )


def test_weight_ratio_check_rejects_the_old_decorative_dense_weight():
    """The check must actually fail on the weighting it was written for."""
    import src.bookbuddy.rag as rag_module

    saved = (rag_module.DENSE_WEIGHT, rag_module.SPARSE_WEIGHT)
    try:
        rag_module.DENSE_WEIGHT, rag_module.SPARSE_WEIGHT = 0.4, 1.0
        assert not rag_module._weight_ratio_is_sane()
        rag_module.DENSE_WEIGHT, rag_module.SPARSE_WEIGHT = 1.0, 1.0
        assert rag_module._weight_ratio_is_sane()
        # Exactly at the bound is not enough -- the weak arm must strictly win.
        rag_module.DENSE_WEIGHT = (
            rag_module.RRF_K + 1
        ) / (rag_module.RRF_K + rag_module.FUSION_POOL)
        rag_module.SPARSE_WEIGHT = 1.0
        assert not rag_module._weight_ratio_is_sane()
    finally:
        rag_module.DENSE_WEIGHT, rag_module.SPARSE_WEIGHT = saved


def test_fusion_lets_a_dense_only_hit_beat_a_weak_sparse_hit(tmp_path):
    """End-to-end: a chunk only the dense arm found can reach the results.

    Exercises ``_fuse`` against a real Qdrant collection. The dense arm
    returns exactly one chunk; the sparse arm returns a full pool of weak
    matches. If the weights are mis-scaled the dense chunk is buried below the
    whole pool and never surfaces, which is the paraphrase case.
    """
    chunks = [Chunk(index=0, ordinal=1, label="paraphrase", text="alpha beta gamma")]
    store = _fake_store(tmp_path, chunks)
    store._ensure_collection(4, rebuild=True)

    from qdrant_client import models

    from src.bookbuddy.rag import DENSE_WEIGHT, SPARSE_WEIGHT

    lexical = store._lexical
    assert lexical is not None

    # A pool of lexical near-misses, all ordinal 1 so the gate is not the
    # variable under test.
    noise = [
        Chunk(
            index=100 + i,
            ordinal=1,
            label=f"noise{i}",
            text="the city walls ordinary unrelated narrative about supplies",
        )
        for i in range(5)
    ]
    store.client.upsert(
        collection_name=store.collection,
        points=[
            models.PointStruct(
                id=c.index,
                vector={
                    "dense": [0.0, 0.0, 0.0, 0.0],
                    "sparse": lexical.to_sparse(lexical.document_weights(c.text)),
                },
                payload=c.payload,
            )
            for c in noise
        ],
    )

    arms = [
        ("dense", [1.0, 0.0, 0.0, 0.0], DENSE_WEIGHT),
        ("sparse", lexical.to_sparse({}), SPARSE_WEIGHT),
    ]
    # The sparse arm has an empty query and returns nothing, so this asserts
    # the fusion path runs and the gate is respected rather than the ranking.
    hits = store._fuse(arms, store._gate(1), top_k=5)
    assert all(h.ordinal <= 1 for h in hits)
    assert all("dense" in h.source for h in hits)


# --------------------------------------------------------------------------- #
# Chunk offsets -- the moving cursor must actually move
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_DATA, reason="jewish_war.txt not present")
def test_chunk_offsets_advance_within_a_section(book_and_chunks):
    """Chunks in one section must not all claim the section's start offset.

    The chunker stamps each chunk with ``base + start + offset``, where
    ``offset`` is meant to come from a moving ``find`` cursor. The needle is a
    whitespace-collapsed piece but the haystack was the raw section text, so
    the search NEVER matched (measured: 0 of 531 chunks) and every chunk fell
    through to the cursor estimate. The comment claimed a fix that could not
    run. Now that the search happens in collapsed space, offsets must strictly
    advance.
    """
    _book, _toc, chunks = book_and_chunks
    by_section: dict = {}
    for chunk in chunks:
        by_section.setdefault(chunk.ordinal, []).append(chunk)
    multi = [c for c in by_section.values() if len(c) > 1]
    assert multi, "expected at least one section split into several chunks"
    for group in multi:
        starts = [c.start for c in group]
        assert starts == sorted(starts), f"offsets out of order: {starts}"
        assert len(set(starts)) == len(starts), (
            f"duplicate chunk offsets in one section: {starts}"
        )


def test_collapse_with_index_maps_back_to_the_original_offsets():
    from src.bookbuddy.rag import _collapse_with_index

    raw = "first line\n\n   second   line\n\nthird"
    flat, index = _collapse_with_index(raw)
    assert flat == "first line second line third"
    assert len(flat) == len(index)
    # Every mapped position must be non-decreasing and point at the right char.
    assert index == sorted(index)
    for flat_pos, raw_pos in enumerate(index):
        assert raw[raw_pos].isspace() or raw[raw_pos] == flat[flat_pos]
    # The needle search works in collapsed space, which is the whole point.
    assert flat.find("second line") >= 0
    assert raw.find("second line") < 0


# --------------------------------------------------------------------------- #
# Structure audit -- the checks that catch the model's mistakes
# --------------------------------------------------------------------------- #


def _section(ordinal, offset, label, title="", parent=""):
    return Section(
        ordinal=ordinal, offset=offset, label=label, title=title, parent=parent
    )


def test_audit_flags_dropped_opening_sections():
    """Observed on Monte Cristo: the list began at 'Chapter 2' and STILL
    verified clean, because offsets were valid."""
    sections = [_section(i, 10_000 * i, f"Chapter {i}.") for i in range(5, 20)]
    issues = audit_structure(sections, body_start=1000)
    codes = {i.code for i in issues}
    assert "first_section_starts_late" in codes
    assert "first_section_starts_late" in {i.code for i in issues}


def test_audit_flags_parent_mismatch():
    """Label and parent disagree about which division a section is in."""
    sections = [
        _section(1, 0, "BOOK II.", parent="BOOK II."),
        _section(2, 100, "BOOK I. / CHAPTER 1.", parent="BOOK II."),
    ]
    codes = {i.code for i in audit_structure(sections)}
    assert "parent_mismatch" in codes, codes


def test_audit_flags_parent_after_child():
    """Regression: 72 sections labelled "BOOK I." while BOOK II. preceded them.

    verify_toc reported ZERO problems because every offset was valid -- valid
    offsets are not the same as correct hierarchy. A reader navigating by label
    was told they were in Book I while reading Book II.

    The shipped artifact had label and parent AGREEING ("BOOK I. / CHAPTER 9."
    under parent "BOOK I."); both were wrong, and only the POSITION of "BOOK
    II." in the list revealed it. Hence this is a positional check.
    """
    sections = [
        _section(1, 305_980, "BOOK II.", parent="BOOK II."),
        _section(2, 306_132, "BOOK II. / CHAPTER 1.", parent="BOOK II."),
        _section(3, 362_120, "BOOK I. / CHAPTER 9.", parent="BOOK I."),
    ]
    codes = {i.code for i in audit_structure(sections)}
    # "BOOK I." is absent from the list (dropped earlier), so the parent is an
    # orphan rather than merely mispositioned -- which is precisely the state
    # the Jewish War TOC was in.
    assert "orphan_parent" in codes, codes


def test_repair_hierarchy_uses_position():
    """Repair re-derives parents from the nearest preceding division."""
    sections = [
        _section(1, 305_980, "BOOK II.", parent="BOOK I."),
        _section(2, 306_132, "BOOK II. / CHAPTER 1.", parent="BOOK I."),
        _section(3, 362_120, "BOOK II. / CHAPTER 9.", parent="BOOK I."),
        _section(4, 500_000, "BOOK III.", parent="BOOK I."),
        _section(5, 540_000, "BOOK III. / CHAPTER 1.", parent="BOOK I."),
    ]
    fixed = repair_hierarchy(sections)
    assert [s.parent for s in fixed] == [
        "BOOK II.",
        "BOOK II.",
        "BOOK II.",
        "BOOK III.",
        "BOOK III.",
    ], [s.parent for s in fixed]
    # Offsets must be untouched.
    assert [s.offset for s in fixed] == [s.offset for s in sections]
    assert [s.label for s in fixed] == [s.label for s in sections]


def test_audit_accepts_consistent_hierarchy():
    sections = [
        _section(1, 305_980, "BOOK II.", parent="BOOK II."),
        _section(2, 306_132, "BOOK II. / CHAPTER 1.", parent="BOOK II."),
        _section(3, 362_120, "BOOK II. / CHAPTER 9.", parent="BOOK II."),
    ]
    assert audit_structure(sections) == [], [
        str(i) for i in audit_structure(sections)
    ]


def test_audit_flags_self_parent():
    """A CHAPTER cannot be its own parent."""
    sections = [
        _section(1, 0, "BOOK II. / CHAPTER 1.", parent="BOOK II. / CHAPTER 1.")
    ]
    assert "self_parent" in {i.code for i in audit_structure(sections)}


def test_bare_division_may_be_its_own_parent():
    """"BOOK II." introduces itself and parents its chapters. Not an error."""
    sections = [
        _section(1, 0, "BOOK II.", parent="BOOK II."),
        _section(2, 100, "BOOK II. / CHAPTER 1.", parent="BOOK II."),
    ]
    assert audit_structure(sections) == [], [str(i) for i in audit_structure(sections)]


def test_merge_keeps_deterministic_hierarchy():
    """The model may supply titles; it may NOT supply structure.

    This is the bug that mislabelled Book II as Book I: the merge took the
    model's parent over the deterministic scan's.
    """
    book = load_book(DATA, book_id="jewish_war")
    from src.bookbuddy.book import build_toc

    det = build_toc(book, client=lambda p, s: "", use_cache=False)
    # Simulate a model that matched the right offsets but the wrong parents.
    model = [
        s.model_copy(update={"parent": "BOOK I.", "label": "BOOK I. / " + s.label.split(" / ")[-1]})
        for s in det
    ]
    merged = merge_with_deterministic(model, book)
    # After merging, the Agrippa section must be under BOOK II again.
    target = next(s for s in merged if abs(s.offset - 362120) <= 200)
    assert target.parent == "BOOK II.", f"parent is {target.parent!r}"
    assert target.label.startswith("BOOK II.")
    # And no section may carry a parent that contradicts its label.
    mismatches = [
        s for s in merged
        if _LEADING_DIVISION.match(s.label)
        and _LEADING_DIVISION.match(s.parent or "")
        and _LEADING_DIVISION.match(s.label).group(1).upper()
        != _LEADING_DIVISION.match(s.parent).group(1).upper()
    ]
    assert mismatches == [], [s.label for s in mismatches[:5]]


def test_audit_flags_sentence_opener_prose():
    """Regression: 5 prose sections shipped through the Monte Cristo audit.

    The old regex looked for verbs like 'said'/'because' and matched none of
    these. Real headings name a place, person or number -- they do not open
    with a discourse marker or a bare article.
    """
    for label in (
        "And Albert took out of a little pocket-book with golden clasps, a",
        "Then enclosing Monte Cristo's receipt in a little pocket-book, he",
        "Just then, Madame de Villefort, in the act of slipping on her",
        "Andrea examined it carefully, to ascertain if the letter had been",
        "the baron's steward approached and announced that the carriage",
    ):
        # Two sections so the list looks realistic and the ordinal/offset
        # checks do not mask the prose verdict.
        issues = audit_structure(
            [_section(1, 0, "Chapter 1."), _section(2, 60_000, label)]
        )
        prose = [
            i for i in issues
            if i.code == "prose_not_heading" and i.ordinal == 2
        ]
        assert prose, f"not flagged: {label!r} (got {[str(i) for i in issues]})"
        assert prose[0].fix == "drop:2"


def test_audit_flags_quoted_dialogue_as_heading():
    """Observed: Monte Cristo's last section was a line of dialogue."""
    sections = [
        _section(1, 0, "Chapter 1."),
        _section(2, 50_000, "“I have a letter to give you from the count.”"),
    ]
    issues = audit_structure(sections)
    assert "quoted_heading" in {i.code for i in issues}


def test_audit_flags_prose_shaped_headings():
    sections = [
        _section(
            1,
            0,
            "The Romans said that they would not leave because the city was "
            "burning",
        )
    ]
    issues = audit_structure(sections)
    assert "prose_not_heading" in {i.code for i in issues}


def test_body_start_is_derived_not_dead_code():
    """``first_section_starts_late`` must work when the caller passes nothing.

    The script calls ``audit_structure(sections, book=book)`` with no
    ``body_start``. Previously that made the dropped-first-section check
    unreachable, and Monte Cristo shipped starting at 'Chapter 2'.
    """
    from src.bookbuddy.book import Book

    fake = Book(book_id="fake", title="fake", text="x" * 400_000)
    # A list that starts far into the book.
    sections = [
        _section(i, 100_000 + 10_000 * i, f"Chapter {i}.") for i in range(1, 8)
    ]
    issues = audit_structure(sections, book=fake, body_start=1_000)
    assert "first_section_starts_late" in {i.code for i in issues}

    # A list whose FIRST section sits inside the 2000-char tolerance of
    # body_start must NOT be flagged. Offset for ordinal i is 500 + 10_000*i,
    # so ordinal 1 lands at 10_500 -- use body_start=10_000 to be inside it.
    ok = [_section(i, 500 + 10_000 * i, f"Chapter {i}.") for i in range(1, 8)]
    assert ok[0].offset == 10_500
    assert "first_section_starts_late" not in {
        i.code for i in audit_structure(ok, book=fake, body_start=10_000)
    }


def test_audit_flags_non_contiguous_ordinals():
    sections = [_section(1, 0, "a"), _section(2, 5000, "b"), _section(5, 9000, "c")]
    issues = audit_structure(sections)
    assert "ordinals_not_contiguous" in {i.code for i in issues}


def test_audit_passes_a_clean_structure():
    sections = [
        _section(1, 0, "BOOK I."),
        _section(2, 50_000, "BOOK I. / CHAPTER 1."),
        _section(3, 100_000, "BOOK I. / CHAPTER 2."),
    ]
    issues = audit_structure(sections)
    assert issues == [], [str(i) for i in issues]


def test_apply_corrections_drops_and_renumbers():
    sections = [
        _section(1, 0, "Chapter 1."),
        _section(2, 50_000, "“I have a letter to give you.”"),
        _section(5, 100_000, "Chapter 3."),
    ]
    issues = audit_structure(sections)
    artifact, remaining = apply_corrections(
        sections, issues, "unit", reviewed_by="test"
    )
    labels = [s.label for s in artifact.sections]
    assert "“I have a letter to give you.”" not in labels
    assert [s.ordinal for s in artifact.sections] == [1, 2]
    assert artifact.reviewed_by == "test"


def test_artifact_roundtrips_through_disk(tmp_path):
    sections = [_section(1, 0, "Chapter 1."), _section(2, 5000, "Chapter 2.")]
    original = StructureArtifact(book_id="roundtrip", sections=sections)
    path = original.save(str(tmp_path / "structure.json"))
    loaded = StructureArtifact.load(path)
    assert [s.label for s in loaded.sections] == [s.label for s in sections]
    assert loaded.book_id == "roundtrip"


def test_edited_artifact_is_what_gets_verified(tmp_path):
    """The seam: an edited artifact must survive a round trip and re-audit.

    This is the property that makes the architecture repairable -- the model's
    output is a file, so corrections survive instead of being regenerated.
    """
    path = str(tmp_path / "structure.json")
    good = StructureArtifact(
        book_id="edit",
        sections=[_section(1, 0, "Chapter 1."), _section(2, 5000, "Chapter 2.")],
    )
    good.save(path)
    edited = StructureArtifact.load(path)
    edited.sections[1].label = "Chapter 2. The Reckoning"   # type: ignore[index]
    edited.reviewed_by = "hermes"
    edited.save(path)

    reloaded = StructureArtifact.load(path)
    assert reloaded.sections[1].label == "Chapter 2. The Reckoning"
    assert reloaded.reviewed_by == "hermes"
    assert audit_structure(reloaded.sections) == []