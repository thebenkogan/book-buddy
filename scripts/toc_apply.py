#!/usr/bin/env python
"""Apply adjudicated verdicts to build a TOC. Makes NO model call.

    uv run python scripts/toc_apply.py data/jewish_war.txt \
        --verdicts verdicts.json

``verdicts.json`` is a list of ``{"index": <candidate index>, "title": "..."}``
— one entry per candidate Hermes (or a subagent judging one slice) accepted as
a real section heading. Rejected candidates are simply absent.

What this does, all deterministic:
  merge verdicts with the pure-deterministic heading scan
  repair the hierarchy
  verify offsets
  audit for quality problems
  save cache/{book_id}_structure.json and print what a human still needs to fix
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy import cache_dir
from src.bookbuddy.book import Section, build_toc, load_book, verify_toc
from src.bookbuddy.onboarding import (
    StructureArtifact,
    artifact_path,
    audit_structure,
    merge_with_deterministic,
    repair_hierarchy,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--book-id", default=None)
    ap.add_argument("--verdicts", required=True,
                    help='JSON list of {"index": int, "title": str}')
    ap.add_argument("--reviewed-by", default="hermes:adjudicated")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    book_id = args.book_id or os.path.splitext(os.path.basename(args.path))[0]
    book = load_book(args.path, book_id=book_id)

    with open(args.verdicts, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("verdicts", [])
    # cache_dir(), not a "../cache" guess relative to the book file: books do
    # not always live in data/, and this silently broke the moment one did not.
    pool_path = os.path.join(cache_dir(), f"{book_id}_candidates.json")
    if not os.path.exists(pool_path):
        print(f"ERROR: no candidate pool at {pool_path}")
        print("       run scripts/toc_candidates.py for this book first")
        return 1
    with open(pool_path, "r", encoding="utf-8") as fh:
        pool = json.load(fh)
    by_index = {c["index"]: c for c in pool["candidates"]}

    unknown = [v for v in raw if v["index"] not in by_index]
    if unknown:
        print(f"ERROR: {len(unknown)} verdict(s) name an index that is not a "
              f"candidate: {[v['index'] for v in unknown][:10]}")
        return 1

    # Verdict -> Section, with the label built deterministically from the
    # nearest preceding division. The model supplies ONLY the title.
    model_sections: list = []
    for verdict in sorted(raw, key=lambda v: v["index"]):
        candidate = by_index[verdict["index"]]
        model_sections.append(
            Section(
                ordinal=candidate["index"],
                label=candidate["line"],
                title=(verdict.get("title") or "").strip(),
                offset=int(candidate["offset"]),
            )
        )

    merged = merge_with_deterministic(model_sections, book, trust_model=True)
    merged = repair_hierarchy(merged)

    det = build_toc(book, client=lambda p, s: "", use_cache=False)
    problems = verify_toc(merged, book.text)
    issues = audit_structure(merged, book=book)

    print(f"candidates        : {len(pool['candidates'])}")
    print(f"accepted verdicts : {len(model_sections)}")
    print(f"after merge       : {len(merged)} sections")
    print(f"pure deterministic: {len(det)} sections")
    print(f"verify_toc        : {len(problems)} problems")
    print(f"audit             : {len(issues)} issue(s)")

    if problems:
        print("\nOFFSET PROBLEMS (blocking):")
        for problem in problems[:20]:
            print("  ", problem)
        return 1

    if issues:
        print("\nQUALITY ISSUES:")
        for issue in issues[:30]:
            print(f"   {issue}" + (f"   fix={issue.fix}" if issue.fix else ""))

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    artifact = StructureArtifact(
        book_id=book_id,
        sections=merged,
        proposed_by="scripts/toc_apply.py",
        reviewed_by=args.reviewed_by,
        notes=[f"{len(model_sections)} verdicts accepted from candidate pool "
               f"of {len(pool['candidates'])}"],
    )
    path = artifact.save(artifact_path(book_id))
    print(f"\nwrote {path}")
    print("next: uv run python scripts/onboard_book.py data/"
          f"{os.path.basename(args.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())