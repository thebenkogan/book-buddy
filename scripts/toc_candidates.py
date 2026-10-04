#!/usr/bin/env python
"""Emit section candidates for Hermes to adjudicate. Makes NO model call.

    uv run python scripts/toc_candidates.py data/jewish_war.txt --slice 1 --of 4

Candidates are split into slices so several subagents can adjudicate them
independently. Nobody has to review one giant list: each subagent returns
verdicts for its own slice, and ``toc_apply.py`` merges them.

Writes cache/{book_id}_candidates.json and prints the requested slice.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy import cache_dir
from src.bookbuddy.book import load_book
from src.bookbuddy.structure import propose_sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--book-id", default=None)
    ap.add_argument("--slice", type=int, default=1, help="1-based slice number")
    ap.add_argument("--of", type=int, default=1, help="total slices")
    ap.add_argument("--preview", type=int, default=240)
    args = ap.parse_args()

    book_id = args.book_id or os.path.splitext(os.path.basename(args.path))[0]
    book = load_book(args.path, book_id=book_id)
    candidates = propose_sections(book)

    out = os.path.join(cache_dir(), f"{book_id}_candidates.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "book_id": book_id,
                "title": book.title,
                "chars": len(book.text),
                "candidates": [
                    {
                        "index": c["index"],
                        "offset": c["offset"],
                        "line": c["line"],
                        "prior": c["prior"],
                        "preview": c["preview"][: args.preview],
                    }
                    for c in candidates
                ],
            },
            fh,
            indent=2,
        )

    n = len(candidates)
    per = max(1, -(-n // max(args.of, 1)))
    lo = (args.slice - 1) * per
    hi = min(lo + per, n)
    chosen = candidates[lo:hi]

    print(f"# {book_id}: candidates {lo + 1}-{hi} of {n} (slice {args.slice}/{args.of})")
    print(f"# full list cached at {out}")
    print()
    for c in chosen:
        print(f"[{c['index']}] ({c['prior']}) {c['line']}")
        print(f"    {c['preview'][: args.preview]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())