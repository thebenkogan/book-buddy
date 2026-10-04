"""Save a reading position for a book.

    uv run python scripts/save_position.py jewish_war 44 \
        --note "Caligula frees Agrippa, makes him king; Antipas banished to Spain"

The saved position is the reader's high-water mark: ``text_up_to`` and the
Qdrant gate both treat it as "everything up to and including this section".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy.book import load_book, text_up_to, verify_toc  # noqa: E402
from src.bookbuddy.onboarding import StructureArtifact, artifact_path  # noqa: E402
from src.bookbuddy.rag import BookStore  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def positions_path(book_id: str) -> str:
    return os.path.join(REPO, "cache", f"{book_id}_position.json")


def load_position(book_id: str):
    path = positions_path(book_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("ordinal", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--data", default=None, help="path to the .txt")
    args = parser.parse_args()

    art = StructureArtifact.load(artifact_path(args.book_id))
    sections = art.sections
    if not 1 <= args.ordinal <= len(sections):
        print(f"ordinal {args.ordinal} out of range 1..{len(sections)}")
        return 1

    data = args.data or os.path.join(REPO, "data", f"{args.book_id}.txt")
    book = load_book(data, book_id=args.book_id)
    problems = verify_toc(sections, book.text)
    if problems:
        print("refusing to save against an unverified TOC:")
        for problem in problems[:10]:
            print("  ", problem)
        return 1

    section = next(s for s in sections if s.ordinal == args.ordinal)
    portion = text_up_to(book, sections, args.ordinal)
    whole = text_up_to(book, sections, len(sections))
    pct = 100 * len(portion) / len(whole)

    payload = {
        "book_id": args.book_id,
        "ordinal": args.ordinal,
        "label": section.label,
        "note": args.note,
        "chars_read": len(portion),
        "pct_of_book": round(pct, 1),
    }
    path = positions_path(args.book_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"saved position for {args.book_id}")
    print(f"  ordinal   : {args.ordinal} of {len(sections)}")
    print(f"  section   : {section.label}")
    print(f"  read      : {len(portion):,} chars ({pct:.1f}% of the book)")
    if args.note:
        print(f"  note      : {args.note}")
    print(f"  written   : {path}")

    store = BookStore(args.book_id)
    if store.exists():
        probe = store.search("Antipas", args.ordinal, top_k=1)
        blocked = store.search("Antipas", args.ordinal + 1, top_k=1)
        print(
            f"  gate check: 'Antipas' at ordinal {args.ordinal} -> "
            f"{[h.ordinal for h in probe]}; at {args.ordinal + 1} -> "
            f"{[h.ordinal for h in blocked]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())