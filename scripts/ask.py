#!/usr/bin/env python
"""Retrieve the passages that may answer a question. Makes NO model call.

    uv run python scripts/ask.py "Who is Herod Agrippa?"            # the book being read
    uv run python scripts/ask.py "Who is Jean Valjean?" les_mis     # or name it

Prints which book it answered from, the reader's saved position, the retrieved
chunks in full, and the answering rules. Hermes -- the reader -- then answers
from those chunks and cites them by number.

There is deliberately no OpenRouter call here. Retrieval decides what may be
consulted; the reader is the model and does the answering. Calling out to a
model to summarise passages a model already retrieved costs a round trip and
loses the conversation.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy.answer import ANSWER_SYSTEM, retrieve_for_answer  # noqa: E402
from src.bookbuddy.rag import BookStore  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_book(explicit: "str | None") -> "tuple[str | None, str]":
    """Choose the book to answer from, and say why.

    An explicit book_id always wins. Otherwise the book whose position file was
    written most recently — the one currently being read — and the choice is
    printed, never silent: a bare ``"jewish_war"`` fallback once meant a
    question about any book was answered out of the Jewish War.
    """
    cache = os.path.join(REPO, "cache")
    saved = []
    if os.path.isdir(cache):
        for name in os.listdir(cache):
            if name.endswith("_position.json"):
                path = os.path.join(cache, name)
                saved.append((os.path.getmtime(path), name[: -len("_position.json")]))
    if explicit:
        return explicit, "asked for on the command line"
    if not saved:
        return None, "no saved position in cache/"
    saved.sort(reverse=True)
    chosen = saved[0][1]
    why = f"most recently read ({chosen})"
    others = [book for _, book in saved[1:]]
    if others:
        why += f"; other books with a position: {', '.join(sorted(others))}"
    return chosen, why


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: uv run python scripts/ask.py <question> [book_id]")
        return 2
    question = sys.argv[1]
    book_id, why = resolve_book(sys.argv[2] if len(sys.argv) > 2 else None)
    if book_id is None:
        books = sorted(
            name[: -len(".txt")]
            for name in os.listdir(os.path.join(REPO, "data"))
            if name.endswith(".txt")
        )
        print(f"which book? no position saved yet ({why}).")
        print("available: " + ", ".join(books))
        print("usage: uv run python scripts/ask.py <question> <book_id>")
        return 1

    pos_path = os.path.join(REPO, "cache", f"{book_id}_position.json")
    if not os.path.exists(pos_path):
        print(f"no saved position for {book_id} ({pos_path})")
        print(f"save one: uv run python scripts/save_position.py {book_id} <ordinal>")
        return 1
    with open(pos_path, "r", encoding="utf-8") as handle:
        pos = json.load(handle)

    store = BookStore(book_id)
    if not store.exists():
        print(f"{book_id} is not indexed; run scripts/onboard_book.py first")
        return 1

    hits, prompt = retrieve_for_answer(question, store, int(pos["ordinal"]))

    print(f"BOOK    : {book_id}   [{why}]")
    print(f"POSITION: ordinal {pos['ordinal']} — {pos['label']}")
    if pos.get("note"):
        print(f"NOTE    : {pos['note']}")
    print(f"QUESTION: {question}")
    print("=" * 78)

    if not hits:
        print("\nNothing readable at this position yet.")
        return 0

    print(f"\n{len(hits)} passages, all at or before the reader's position:\n")
    print(prompt)

    print()
    print("=" * 78)
    print("ANSWERING RULES")
    print("=" * 78)
    print(ANSWER_SYSTEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())