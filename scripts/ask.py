#!/usr/bin/env python
"""Retrieve the passages that may answer a question. Makes NO model call.

    uv run python scripts/ask.py "Who is Herod Agrippa?"

Prints the reader's saved position, the retrieved chunks in full, and the
answering rules. Hermes -- the reader -- then answers from those chunks and
cites them by number.

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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: uv run python scripts/ask.py <question> [book_id]")
        return 2
    question = sys.argv[1]
    book_id = sys.argv[2] if len(sys.argv) > 2 else "jewish_war"

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

    print(f"BOOK    : {book_id}")
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