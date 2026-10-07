#!/usr/bin/env python
"""Print the text the reader has newly finished — the range since the notes
file was last synced. Makes NO model call and writes nothing.

    uv run python scripts/notes_range.py jewish_war 52
    uv run python scripts/notes_range.py jewish_war 52 --out /tmp/range.txt
    uv run python scripts/notes_range.py jewish_war --find "BOOK III. / CHAPTER 5."

The range is EXACTLY the delta between two spoiler gates:
``text_up_to(last)`` and ``text_up_to(new)``. It never reaches past ``new``,
so a summary written from it cannot spoil the reader's position.

State lives in ``cache/<book_id>_notes.json``:
    {"last_ordinal": 44, "doc_id": "<google doc id>", "doc_url": "..."}
``last_ordinal`` is the section the notes file currently ends at. It is the
lower gate. Without that file the lower gate is the saved reading position.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy.book import load_book, section_index, text_up_to  # noqa: E402
from src.bookbuddy.onboarding import StructureArtifact, artifact_path  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAX = 20000


def notes_path(book_id: str) -> str:
    return os.path.join(REPO, "cache", f"{book_id}_notes.json")


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def composed_label(section) -> str:
    """The label the position file uses: 'BOOK II. / CHAPTER 9.'"""
    label = section.label.strip()
    parent = (section.parent or "").strip()
    if parent and parent != label and not label.startswith(parent):
        return f"{parent} / {label}"
    return label


def find_sections(sections, needle: str):
    norm = " ".join(needle.upper().replace("/", " ").split())
    hits = []
    for section in sections:
        hay = " ".join(composed_label(section).upper().replace("/", " ").split())
        hay_bare = " ".join(section.label.upper().replace(".", " ").split())
        title = " ".join((section.title or "").upper().split())
        if norm in hay or norm == hay_bare or (norm and norm in title):
            hits.append(section)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("new_ordinal", type=int, nargs="?")
    parser.add_argument("--find", default=None, help="print ordinals matching a label")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    art = StructureArtifact.load(artifact_path(args.book_id))
    sections = art.sections
    total = len(sections)

    if args.find:
        hits = find_sections(sections, args.find)
        if not hits:
            print(f"no section matches {args.find!r} (1..{total})")
            return 1
        for section in hits:
            print(f"  ordinal {section.ordinal:>3}  {composed_label(section)}")
        return 0

    if args.new_ordinal is None:
        print("need a new ordinal, or --find")
        return 2

    new = args.new_ordinal
    if not 1 <= new <= total:
        print(f"ordinal {new} out of range 1..{total}")
        return 1

    state = load_json(notes_path(args.book_id), {}) or {}
    state_ord = state.get("last_ordinal")

    position = load_json(os.path.join(REPO, "cache", f"{args.book_id}_position.json"), {}) or {}
    pos_ord = position.get("ordinal")

    lower = state_ord if state_ord is not None else (pos_ord if pos_ord is not None else 0)
    if new <= lower:
        print(
            f"ordinal {new} is not past the notes tail (last_ordinal={lower}); "
            "nothing new to summarize."
        )
        return 1

    data = os.path.join(REPO, "data", f"{args.book_id}.txt")
    book = load_book(data, book_id=args.book_id)

    # Deliberately built from two authorised gate slices rather than slicing
    # book.text here: the range is then provably the delta between "read up to
    # lower" and "read up to new", and cannot disagree with what ask.py gates.
    first = section_index(sections, lower + 1) if lower >= 1 else 0
    last = section_index(sections, new)
    before = text_up_to(book, sections, lower) if lower >= 1 else ""
    upto = text_up_to(book, sections, new)
    if not upto.startswith(before):
        print("refusing: gate slices disagree (stale structure artifact?)")
        return 1
    text = upto[len(before) :]

    out = args.out or os.path.join(REPO, "cache", f"{args.book_id}_range.txt")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)

    if not args.quiet:
        print(f"RANGE   : section {lower + 1} .. {new} of {total}")
        print(f"FROM    : {state_ord if state_ord is not None else '(reading position)'} = {lower}")
        print(f"TO      : {composed_label(sections[last])}")
        print(f"CHARS   : {len(text):,}")
        print(f"FILE    : {out}")
        print("SECTIONS IN RANGE:")
        for section in sections[first : last + 1]:
            print(f"  {section.ordinal:>3}  {composed_label(section)}")
        print("=" * 78)
        shown = text[: args.max_chars]
        print(shown)
        if len(text) > len(shown):
            print(f"\n... [{len(text) - len(shown):,} more chars in {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())