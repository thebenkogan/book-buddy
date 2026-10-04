#!/usr/bin/env python
"""Build the index for an already-adjudicated book, and prove the gate.

    set -a && . ~/.hermes/.env && set +a
    uv run python scripts/onboard_book.py data/<file>.txt

NO MODEL CALLS except the embeddings themselves. The section list must already
exist at ``cache/{book_id}_structure.json``; this script verifies it, builds
the Qdrant index, and proves the spoiler gate. If the TOC is missing it says so
and points at the two-step flow instead of quietly building one.

Building a TOC is Hermes's job, split across subagents:
    scripts/toc_candidates.py data/<file> --slice N --of M   # candidates
    (delegate a slice per subagent, collect index|title verdicts)
    scripts/toc_apply.py data/<file> --verdicts verdicts.json# merge + verify
    scripts/onboard_book.py data/<file>                      # index + gate
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy import DEFAULT_MODEL, get_model  # noqa: E402
from src.bookbuddy.answer import answer_question  # noqa: E402
from src.bookbuddy.book import (  # noqa: E402
    Section,
    count_tokens,
    load_book,
    text_up_to,
    toc_needs_model,
    verify_toc,
)
from src.bookbuddy.onboarding import (  # noqa: E402
    StructureArtifact,
    apply_corrections,
    artifact_path,
    audit_structure,
    repair_hierarchy,
)
from src.bookbuddy.rag import BookStore  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Chosen to exercise both retrieval failure modes we actually hit: a proper
#: noun (fails if lexical scoring is broken) and a paraphrase (fails if dense
#: scoring is broken). The ordinal is clamped to the book's section count.
SAMPLE_QUERIES = [
    ("Who is Ananus?", 60),
    ("What happens at Masada?", 10**9),
    ("the famine in Jerusalem", 10**9),
]


#: Audit codes that mean "the structure is wrong, do not index".
BLOCKING = {
    "empty",
    "orphan_parent",
    "parent_mismatch",
    "parent_precedes_child",
    "self_parent",
    "first_section_starts_late",
    "prose_not_heading",
    "quoted_heading",
    "tiny_section",
    "ordinals_not_contiguous",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Gutenberg .txt file")
    parser.add_argument("--book-id", default=None)
    parser.add_argument("--refresh", action="store_true", help="force re-embed")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the Qdrant index even if one exists")
    parser.add_argument(
        "--restructure",
        action="store_true",
        help="ignore an existing artifact and re-run the model",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    path = args.path if os.path.isabs(args.path) else os.path.join(REPO, args.path)
    book_id = args.book_id or os.path.splitext(os.path.basename(path))[0]
    art_path = artifact_path(book_id)

    print(f"model: {args.model or get_model()} (default {DEFAULT_MODEL})")
    print(f"file:  {path}")

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    t0 = time.time()
    book = load_book(path, book_id=book_id)
    print(
        f"\n1. loaded {book.title[:60]!r}\n"
        f"   {len(book.text):,} chars, ~{count_tokens(book.text):,} tokens "
        f"({time.time() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------ #
    # 2. Structure: propose -> adjudicate -> artifact
    # ------------------------------------------------------------------ #
    print("\n2. structure")
    if args.restructure:
        # Deterministic-only rebuild. Measured on four real books this is
        # EXACT for jewish_war (117/117) and pnp (61/61), 375/380 for
        # war_peace, and over-detects on sherlock's bare-roman contents
        # listing. It needs no model call at all, so it is the default.
        # If audit_structure objects below, THAT is the signal to fall back
        # to the subagent path in skills/book-buddy/onboard-book/SKILL.md.
        from src.bookbuddy.book import derive_title, find_headings

        headings = find_headings(book.text)
        sections = [
            Section(
                ordinal=i,
                label=h["label"],
                title=derive_title(book.text, int(h["offset"])),
                offset=int(h["offset"]),
                parent="",
            )
            for i, h in enumerate(headings, 1)
        ]
        sections = repair_hierarchy(sections)

        # Route by rule, not by whoever is looking. Measured on 24 unseen
        # Gutenberg books the deterministic path alone passed on 17, and the
        # 7 failures produced TOCs that looked perfectly healthy -- which is
        # exactly why this cannot be a judgement call made after the fact.
        needs_model, reason = toc_needs_model(sections, len(book.text))
        if needs_model:
            print(f"   ROUTER: NEEDS A MODEL -- {reason}")
            print("   Fall back to the subagent path:")
            print(f"     uv run python scripts/toc_candidates.py {args.path} --slice 1 --of 8")
        else:
            print(f"   ROUTER: deterministic is good -- {reason}")

        artifact = StructureArtifact(
            book_id=book_id,
            sections=sections,
            proposed_by="book.find_headings+derive_title (deterministic)",
            reviewed_by="deterministic",
        )
        artifact.save(art_path)
        titled = sum(1 for s in sections if s.title.strip())
        print(f"   deterministic rebuild: {len(sections)} sections, "
              f"{titled} with titles read from the text")
        print(f"   wrote {art_path}")

    elif os.path.exists(art_path):
        artifact = StructureArtifact.load(art_path)
        print(f"   using existing TOC {art_path} ({len(artifact.sections)} sections)")
        if artifact.reviewed_by:
            print(f"   adjudicated by: {artifact.reviewed_by}")
    else:
        print("   NO TOC for this book. This script does not invent one --")
        print("   build it with Hermes first:")
        print(f"     uv run python scripts/toc_candidates.py {args.path} --slice 1 --of 4")
        print("     (delegate each slice to a subagent, collect index|title verdicts)")
        print(f"     uv run python scripts/toc_apply.py {args.path} --verdicts verdicts.json")
        return 1

    sections = list(artifact.sections)
    if not sections:
        print("   NO SECTIONS. Confirm the file is a Gutenberg text with real "
              "chapter headings; see skills/book-buddy/onboard-book/SKILL.md")
        return 1

    # Repair the hierarchy before auditing. The parent of a section is the
    # most recent bare division label at or before it, recomputed from
    # position alone. This is what put Book II's chapters back under Book II:
    # an earlier merge step let the model's parent win, and 72 sections were
    # labelled "BOOK I." while BOOK II. sat two entries above them -- with
    # every offset valid, so verify_toc reported zero problems.
    before_parents = [s.parent for s in sections]
    sections = repair_hierarchy(sections)
    repaired = sum(
        1 for a, b in zip(before_parents, [s.parent for s in sections]) if a != b
    )
    if repaired:
        print(f"   repaired parent on {repaired} section(s) from list position")
        artifact = artifact.model_copy(update={"sections": sections})
        artifact.save(art_path)

    kinds: dict = {}
    for section in sections:
        key = section.label.split()[0].upper() if section.label.split() else "?"
        kinds[key] = kinds.get(key, 0) + 1
    print(f"   label kinds: {kinds}")
    print(f"   first 2: {[s.label for s in sections[:2]]}")
    print(f"   last 2:  {[s.label for s in sections[-2:]]}")

    # ------------------------------------------------------------------ #
    # 3. Verify offsets (validity)
    # ------------------------------------------------------------------ #
    problems = verify_toc(sections, book.text)
    print(f"\n3. verify_toc: {len(problems)} problems (offsets must be valid)")
    for problem in problems[:10]:
        print(f"   {problem}")
    if problems:
        print("   STOP: a wrong offset does not crash, it silently produces "
              "confident wrong answers.")
        return 1

    # ------------------------------------------------------------------ #
    # 4. Audit quality (correctness) -- what verify_toc cannot see
    # ------------------------------------------------------------------ #
    issues = audit_structure(sections, book=book)
    print(f"\n4. structure audit: {len(issues)} issue(s)")
    for issue in issues[:20]:
        print(f"   {issue}" + (f"   fix={issue.fix}" if issue.fix else ""))

    if issues:
        artifact, remaining = apply_corrections(
            sections,
            issues,
            book_id,
            reviewed_by="auto:apply_corrections",
            book=book,
        )
        sections = list(artifact.sections)
        print(
            f"   auto-corrected -> {len(sections)} sections, "
            f"{len(remaining)} issue(s) remaining"
        )
        for issue in remaining[:20]:
            print(f"   REMAINING {issue}")
        blocking = [i for i in remaining if i.code in BLOCKING]
        if blocking:
            print(
                "\n   STRUCTURE IS WRONG. Do NOT index: the audit still flags "
                f"{sorted({i.code for i in blocking})}."
            )
            print(f"   Edit {art_path}, then re-run. Machine-actionable fixes are "
                  "in each issue's `fix` field; anything else needs a decision.")
            return 1

    problems = verify_toc(sections, book.text)
    if problems:
        print(f"\n5. verify_toc after corrections: {len(problems)} problems")
        for problem in problems[:10]:
            print(f"   {problem}")
        return 1
    print("\n5. verify_toc after corrections: 0 problems")

    # ------------------------------------------------------------------ #
    # 6. Index into Qdrant (dense + sparse)
    # ------------------------------------------------------------------ #
    print("\n6. index (Qdrant: hybrid dense + BM25 sparse)")
    store = BookStore(book_id)
    t0 = time.time()
    # --force matters: build() reuses an existing collection when its
    # dimension matches, which silently leaves a stale ordinal layout behind a
    # changed TOC. That is not a safe default after a structure edit.
    n_chunks = store.build(
        book, sections, refresh=args.refresh or args.force, progress=True
    )
    if not n_chunks:
        print("   no chunks built - aborting")
        return 1
    print(
        f"   {n_chunks} chunks in {time.time() - t0:.1f}s "
        f"(collection {store.collection})"
    )

    # ------------------------------------------------------------------ #
    # 7. Prove the gate
    # ------------------------------------------------------------------ #
    print("\n7. spoiler gate (a Qdrant range filter, not a Python check)")
    last = len(sections)
    for ordinal in (1, max(2, last // 3), last):
        hits = store.search("What happens at the end?", ordinal, top_k=6)
        max_ord = max((h.ordinal for h in hits), default=0)
        ok = all(h.ordinal <= ordinal for h in hits)
        print(
            f"   ordinal={ordinal:>5} -> {len(hits)} hits, max ordinal "
            f"{max_ord}, all within position: {ok}"
        )
        if not ok:
            print("   GATE FAILED - aborting")
            return 1
    zero = store.search("anything", 0, top_k=6)
    print(f"   ordinal=      0 -> {len(zero)} hits (expect 0)")
    if zero:
        print("   GATE FAILED at ordinal 0 - aborting")
        return 1
    print("   gate holds: the store cannot return a chunk past the reader's "
          "position")

    # ------------------------------------------------------------------ #
    # 8. Retrieval spot-checks
    # ------------------------------------------------------------------ #
    print("\n8. retrieval spot-checks")
    queries = [(q, min(o, last)) for q, o in SAMPLE_QUERIES]
    for question, ordinal in queries:
        hits = store.search(question, ordinal, top_k=3)
        print(f"\n   Q: {question!r} (reader at {ordinal}/{last})")
        if not hits:
            print("      (nothing readable at that position)")
            continue
        # Show the passage AROUND the query term, not the first 120 chars.
        # Showing only the head of a chunk made a correct retrieval look
        # wrong: "Who is Ananus?" returned BOOK IV. / CHAPTER 3 -- the chapter
        # titled "Concerning The Zealots And The High Priest Ananus" -- whose
        # opening words are about other business.
        terms = [w for w in question.lower().split() if len(w) > 3]
        for hit in hits:
            flat = " ".join(hit.text.split())
            low = flat.lower()
            at = min(
                (low.find(t) for t in terms if low.find(t) >= 0),
                default=0,
            )
            start = max(0, at - 60)
            preview = flat[start : start + 200]
            print(
                f"      {hit.score:.4f} ord={hit.ordinal} "
                f"{hit.label[:30]:32} {preview!r}"
            )

    # ------------------------------------------------------------------ #
    # 9. Answer end to end
    # ------------------------------------------------------------------ #
    if not args.skip_answer:
        print("\n9. end-to-end answers (citations required)")
        for question, ordinal in queries[:2]:
            result = answer_question(question, store, ordinal, model=args.model)
            print(f"\n   Q: {question!r} @ {ordinal}")
            print(f"   tokens in={result.tokens_in} out={result.tokens_out}")
            print("   A: " + " ".join(result.answer[:600].split()))
            used = result.citations_used()
            print(f"   cited: {used or 'NONE - answering from memory'}")
            if not used:
                print("   WARNING: no citations; the model ignored the context.")

    whole = text_up_to(book, sections, len(sections))
    third = max(1, len(sections) // 3)
    portion = text_up_to(book, sections, third)
    print(
        f"\nDone. {book_id}: {len(sections)} sections, {n_chunks} chunks, "
        f"{len(whole):,} chars.\n"
        f"Ordinal {third} = {100 * len(portion) / len(whole):.1f}% of the text "
        "(section sizes are uneven)."
    )
    print(f"Structure artifact (editable): {art_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())