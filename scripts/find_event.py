#!/usr/bin/env python
"""Resolve a remembered event to section ordinals — evidence, not a verdict.

    uv run python scripts/find_event.py jewish_war "the emperor's statue set up in the temple"
    uv run python scripts/find_event.py jewish_war "the part where the ship sinks" --window 15
    uv run python scripts/find_event.py jewish_war "BOOK III. / CHAPTER 5."        # a label
    uv run python scripts/find_event.py jewish_war "..." --no-semantic             # titles only

Ben sometimes names his position by what happened, not by chapter ("I read up to
where the statue went into the temple"). This prints the candidates for that
description and nothing else: ordinals, labels and TITLES — never passage text,
so nothing here can spoil a position.

Three independent signals, deliberately NOT fused into one score:

  1. LABEL MATCH  — the query is literally a section label ("BOOK III. / CHAPTER 5.")
  2. TITLE MATCH  — token overlap with the section titles (deterministic, no model)
  2b. TEXT MATCH  — token overlap with the section TEXT (deterministic, gated),
                    printed as ordinals, labels and the matched words only
  3. SEMANTIC     — embedding search, GATED at position + --window

Signal 2b exists because a title says what a chapter is ABOUT, not what happens
in it: the chapter titled "Concerning The Government Of Claudius" covered his
accession, while his DEATH fell in the next chapter. A title hit alone can be
one section early, so read 2b before saving a position.

The gate on signal 3 is the whole trick: an ungated search for "the governor
steals the temple treasury" returns sections 55 past the reader (measured), while
a reader's next report is almost always within a chapter or two of where they
were. Bounding the search to that band turns a wild guess into a shortlist.

A semantic hit is NEVER an answer on its own — measured, it can land 50 sections
away. Read the evidence, show Ben the labels, let him pick. If the candidates
disagree or the best one is far ahead, ask.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.bookbuddy.onboarding import StructureArtifact, artifact_path  # noqa: E402

def _local_label(section) -> str:
    """Same composition notes_range uses: 'BOOK II. / CHAPTER 9.'"""
    label = section.label.strip()
    parent = (section.parent or "").strip()
    if parent and parent != label and not label.startswith(parent):
        return f"{parent} / {label}"
    return label


try:
    from notes_range import composed_label, find_sections
except ImportError:  # pragma: no cover - scripts dir layout changed
    composed_label = _local_label
    find_sections = None

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "into", "that", "this", "these", "those", "his", "her",
    "their", "they", "them", "he", "she", "it", "its", "was", "were", "is",
    "are", "be", "been", "had", "has", "have", "did", "does", "do", "then",
    "when", "where", "which", "who", "whom", "how", "why", "as", "but", "not",
    "up", "out", "over", "after", "before", "part", "where", "reads", "read",
    "about", "again", "against", "all", "any", "because", "been", "being",
    "between", "both", "each", "few", "more", "most", "no", "nor", "only",
    "other", "own", "same", "so", "some", "such", "than", "too", "very",
    "can", "will", "just", "should", "now", "one", "two", "also", "there",
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 2}


def title_matches(sections, query: str, minimum: float = 0.34):
    """Deterministic: fraction of the query's content words present in a title."""
    wanted = tokens(query)
    if not wanted:
        return []
    scored = []
    for section in sections:
        present = tokens(f"{section.title} {section.label}")
        if not present:
            continue
        overlap = len(wanted & present) / len(wanted)
        if overlap >= minimum:
            scored.append((overlap, section))
    scored.sort(key=lambda pair: (-pair[0], pair[1].ordinal))
    return scored


DEATH_WORDS = {
    "die", "dies", "died", "dying", "death", "deaths", "dead",
    "slain", "slay", "slays", "slew", "kill", "kills", "killed", "killing",
    "perished", "perish",
}


def stem(word: str) -> str:
    """Normalise event vocabulary, then strip a common suffix.

    Collapsing die/dies/died/dying/death (and kill/slain/slew) to one token is
    what lets a typed query match the book's own wording. Measured on "the part
    where emperor Claudius dies", without this the signal missed the right
    section entirely (it matched 'dies' against nothing) and returned five
    candidates instead of the two adjacent ones.
    """
    if word in DEATH_WORDS:
        return "death"
    for suffix, cut in (("ies", 3), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if word.endswith(suffix) and len(word) - cut >= 3:
            return word[: -cut]
    return word


def text_matches(book_id: str, sections, query: str, gate: int, top: int, position: int, minimum: int = 2):
    """Deterministic: how many of the query's content words appear in each
    section's TEXT (gated at ``gate``). Prints words, never passage text."""
    path = os.path.join(REPO, "data", f"{book_id}.txt")
    if not os.path.exists(path):
        return None
    from src.bookbuddy.book import load_book

    raw = load_book(path).text
    ordered = sorted(sections, key=lambda s: s.offset)
    offs = [s.offset for s in ordered] + [len(raw)]
    wanted = {stem(word) for word in tokens(query)}
    if not wanted:
        return []
    scored = []
    for index, section in enumerate(ordered):
        if section.ordinal > gate:
            break
        present = {stem(word) for word in tokens(raw[offs[index] : offs[index + 1]])}
        hit = wanted & present
        if len(hit) >= minimum:
            scored.append((len(hit), section, sorted(hit)))
    # Most words first; ties broken by proximity to the reader's position, the
    # same assumption the semantic gate makes — his next report is near where
    # he was, so a hit far down the band is the weaker candidate.
    scored.sort(key=lambda row: (-row[0], abs(row[1].ordinal - position)))
    return scored[:top]


def semantic_candidates(book_id: str, query: str, gate: int, top_k: int):
    from src.bookbuddy.rag import BookStore

    store = BookStore(book_id)
    if not store.exists():
        return None
    hits = store.search(query, gate, top_k=top_k * 3)
    best: dict[int, float] = {}
    for hit in hits:
        if hit.ordinal > gate:  # belt and braces; search() already gates
            continue
        best[hit.ordinal] = max(best.get(hit.ordinal, 0.0), float(getattr(hit, "score", 0.0)))
    ranked = sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("query")
    parser.add_argument("--position", type=int, default=None, help="current saved position")
    parser.add_argument("--window", type=int, default=25, help="sections past the position to search")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    art = StructureArtifact.load(artifact_path(args.book_id))
    sections = art.sections
    by_ord = {s.ordinal: s for s in sections}
    total = len(sections)

    position = args.position
    if position is None:
        path = os.path.join(REPO, "cache", f"{args.book_id}_position.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                position = json.load(handle).get("ordinal")
    position = int(position or 0)
    gate = min(total, position + max(1, args.window))

    print(f"BOOK     : {args.book_id} ({total} sections)")
    print(f"POSITION : {position}")
    print(f"SEARCHED : sections 1..{gate} (position + {args.window}); never past the gate")
    print(f"QUERY    : {args.query}")
    print("=" * 78)

    if find_sections is not None:
        labels = find_sections(sections, args.query)
        if labels:
            print("\n1. LABEL MATCH (the query is a section label):")
            for section in labels[: args.top]:
                print(f"   ordinal {section.ordinal:>3}  {composed_label(section)}")

    titles = title_matches(sections, args.query)
    print("\n2. TITLE MATCH (deterministic, no model):")
    if not titles:
        print("   none — his words are not in any section title")
    for overlap, section in titles[: args.top]:
        flag = "  <= at or before his position" if section.ordinal <= position else ""
        print(
            f"   {overlap:0.2f}  ordinal {section.ordinal:>3}  {composed_label(section)}"
            f"  {section.title[:60]!r}{flag}"
        )

    text_rows = text_matches(args.book_id, sections, args.query, gate, args.top, position)
    print(f"\n2b. TEXT MATCH (deterministic, section text, gated at {gate}):")
    if text_rows is None:
        print(f"   data/{args.book_id}.txt not found")
    elif not text_rows:
        print("   none — no section in the band carries two of his words")
    else:
        total_words = len({stem(word) for word in tokens(args.query)})
        print("   a shortlist, not a verdict: adjacent chapters can both mention")
        print("   the person and an event, so read the candidates before saving.")
        for hits, section, words in text_rows:
            where = (
                f"+{section.ordinal - position} past his position"
                if section.ordinal > position
                else f"{position - section.ordinal} before his position"
            )
            print(
                f"   {hits}/{total_words} words  ordinal {section.ordinal:>3}  "
                f"{composed_label(section)}  matched: {', '.join(words)}  ({where})"
            )

    print(f"\n3. SEMANTIC CANDIDATES (embeddings, gated at {gate}):")
    if args.no_semantic:
        print("   skipped (--no-semantic)")
    else:
        ranked = semantic_candidates(args.book_id, args.query, gate, args.top)
        if ranked is None:
            print(f"   {args.book_id} is not indexed (run scripts/onboard_book.py)")
        elif not ranked:
            print("   no hits")
        else:
            for ordinal, score in ranked:
                section = by_ord[ordinal]
                if ordinal > position:
                    where = f"+{ordinal - position} past his position"
                else:
                    where = f"{position - ordinal} before his position"
                print(
                    f"   {score:0.3f}  ordinal {ordinal:>3}  {composed_label(section)}"
                    f"  {section.title[:60]!r}  ({where})"
                )
            print(
                "   A semantic hit is not an answer: measured, this signal can land "
                "50+ sections away.\n   Show Ben the labels and let him pick."
            )

    window = [s for s in sections if position < s.ordinal <= gate]
    if window:
        print(f"\nWINDOW — sections {position + 1}..{gate}, for eyeballing:")
        for section in window:
            print(f"   {section.ordinal:>3}  {section.title[:70] or composed_label(section)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())