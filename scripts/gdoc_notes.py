#!/usr/bin/env python
"""Append summarized key points to a book's Google Doc notes file, in the
document's own formatting, and advance the notes tail.

    uv run python scripts/gdoc_notes.py jewish_war 52 --bullets /tmp/bullets.md
    uv run python scripts/gdoc_notes.py jewish_war 52 --bullets /tmp/bullets.md --dry-run
    uv run python scripts/gdoc_notes.py jewish_war --restore cache/gdoc_backups/<f>.json

Bullets file markup (one paragraph per line):

    # Thematic Heading          -> HEADING_1
    - a key point               -> bullet, level 0
      - a sub point             -> bullet, level 1

``**bold**`` inside a row bolds that span in the doc, matching the existing
bullets (a short key phrase per bullet, not the whole line).

Append-only: existing text is never touched. Before writing, the current body
is snapshotted to ``cache/gdoc_backups/`` together with the insertion index, so
``--restore`` can cut exactly what this script added.

State lives in ``cache/<book_id>_notes.json``:
    {"last_ordinal": 44, "doc_id": "...", "doc_url": "..."}
``--bullets`` advances ``last_ordinal`` to the given ordinal on success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy.book import load_book  # noqa: E402
from src.bookbuddy.onboarding import StructureArtifact, artifact_path  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = os.path.expanduser("~/.hermes/google_token.json")
BACKUP_DIR = os.path.join(REPO, "cache", "gdoc_backups")
BULLET_PRESET = "BULLET_DISC_CIRCLE_SQUARE"
BODY_PT = 12
SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]


def notes_path(book_id: str) -> str:
    return os.path.join(REPO, "cache", f"{book_id}_notes.json")


def load_notes(book_id: str) -> dict:
    path = notes_path(book_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_notes(book_id: str, data: dict) -> None:
    path = notes_path(book_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    with open(TOKEN, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    creds = Credentials(
        token=raw.get("token"),
        refresh_token=raw.get("refresh_token"),
        token_uri=raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=raw.get("client_id"),
        client_secret=raw.get("client_secret"),
        scopes=raw.get("scopes") or SCOPES,
    )
    return build("docs", "v1", credentials=creds)


def extract_bold(text: str):
    """Split ``**bold**`` markers out of a row's text.

    -> (clean_text, [(start, end), ...]) with spans relative to clean_text.
    """
    spans, out, i = [], [], 0
    while True:
        start = text.find("**", i)
        if start < 0:
            out.append(text[i:])
            break
        end = text.find("**", start + 2)
        if end < 0:
            out.append(text[i:])
            break
        out.append(text[i:start])
        content = text[start + 2 : end]
        offset = sum(len(part) for part in out)
        out.append(content)
        spans.append((offset, offset + len(content)))
        i = end + 2
    return "".join(out), spans


def parse_bullets(path: str):
    """-> [{'kind': 'heading'|'bullet', 'level': int, 'text': str, 'bold': [(s,e)]}]"""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle.read().split("\n"), 1):
            raw = line.rstrip()
            if not raw.strip():
                continue
            if raw.lstrip().startswith("#"):
                text, bold = extract_bold(raw.lstrip("# ").strip())
                rows.append({"kind": "heading", "level": 0, "text": text, "bold": bold})
                continue
            stripped = raw.lstrip(" \t")
            indent = len(raw) - len(stripped)
            if stripped.startswith("- "):
                text, bold = extract_bold(stripped[2:].strip())
                rows.append(
                    {
                        "kind": "bullet",
                        "level": 1 if indent >= 2 else 0,
                        "text": text,
                        "bold": bold,
                    }
                )
                continue
            raise SystemExit(f"{path}:{number}: line is not '#', '- ' or '  - ': {raw[:60]!r}")
    if not rows:
        raise SystemExit(f"{path}: no bullets found")
    return rows


def body_text(doc) -> str:
    out = []
    for element in doc.get("body", {}).get("content", []):
        para = element.get("paragraph")
        if not para:
            continue
        out.append("".join(e.get("textRun", {}).get("content", "") for e in para.get("elements", [])))
    return "".join(out)


def plan_insert(doc, rows):
    """Absolute index + per-row ranges for an append at the very end."""
    content = doc["body"]["content"]
    last_end = content[-1]["endIndex"]
    index = last_end - 1  # before the doc's final newline

    # If the doc ends with empty paragraphs, the first row joins the last one.
    text = "\n".join(("\t" + r["text"]) if r["kind"] == "bullet" and r["level"] == 1 else r["text"] for r in rows)
    text += "\n"

    ranges, cursor = [], index
    for row, line in zip(
        rows,
        text.split("\n")[:-1],
    ):
        ranges.append({**row, "start": cursor, "end": cursor + len(line)})
        cursor = cursor + len(line) + 1
    return index, text, ranges


def build_requests(index, text, ranges):
    requests = [{"insertText": {"location": {"index": index}, "text": text}}]
    for row in ranges:
        if row["kind"] == "heading":
            requests.append(
                {
                    "updateParagraphStyle": {
                        "range": {"startIndex": row["start"], "endIndex": row["end"] + 1},
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                        "fields": "namedStyleType",
                    }
                }
            )
        else:
            # The doc's own bullets are explicitly 12pt NORMAL_TEXT; the API
            # default for inserted text is 11pt, which reads as a mismatch.
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {"startIndex": row["start"], "endIndex": row["end"]},
                        "textStyle": {
                            "fontSize": {"magnitude": BODY_PT, "unit": "PT"},
                        },
                        "fields": "fontSize",
                    }
                }
            )
    # ``**bold**`` spans from the bullets file, relative to the row text (a
    # level-1 bullet's text is prefixed with a tab in the inserted line).
    for row in ranges:
        prefix = 1 if row["kind"] == "bullet" and row["level"] == 1 else 0
        for start, end in row.get("bold", []):
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": row["start"] + prefix + start,
                            "endIndex": row["start"] + prefix + end,
                        },
                        "textStyle": {"bold": True},
                        "fields": "bold",
                    }
                }
            )
    # Consecutive bullet rows share one createParagraphBullets call; nesting
    # level comes from the leading tab we inserted.
    run = []
    for row in ranges + [None]:
        if row is not None and row["kind"] == "bullet":
            run.append(row)
            continue
        if run:
            requests.append(
                {
                    "createParagraphBullets": {
                        "range": {"startIndex": run[0]["start"], "endIndex": run[-1]["end"] + 1},
                        "bulletPreset": BULLET_PRESET,
                    }
                }
            )
            run = []
    return requests


def snapshot(doc, index, book_id: str, ordinal: int) -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(BACKUP_DIR, f"{book_id}_{ordinal}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "doc_id": doc["documentId"],
                "title": doc.get("title", ""),
                "cut_index": index,
                "book_id": book_id,
                "ordinal": ordinal,
                "when": stamp,
                "prev_last_ordinal": (load_notes(book_id) or {}).get("last_ordinal"),
                "body_before": body_text(doc),
            },
            handle,
            indent=2,
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("ordinal", type=int, nargs="?")
    parser.add_argument("--bullets", default=None)
    parser.add_argument("--doc", default=None, help="override the doc id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-state", action="store_true", help="do not advance cache/<book>_notes.json")
    parser.add_argument("--restore", default=None)
    args = parser.parse_args()

    docs = service()
    notes = load_notes(args.book_id)
    doc_id = args.doc or notes.get("doc_id")
    if not doc_id:
        print(f"no doc_id for {args.book_id} in {notes_path(args.book_id)}; pass --doc")
        return 1

    if args.restore:
        with open(args.restore, "r", encoding="utf-8") as handle:
            backup = json.load(handle)
        if backup["doc_id"] != doc_id:
            print("backup is for a different document; refusing")
            return 1
        doc = docs.documents().get(documentId=doc_id).execute()
        end = doc["body"]["content"][-1]["endIndex"] - 1
        if backup["cut_index"] >= end:
            print("nothing to restore (backup cut index is at or past the body end)")
            return 0
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={
                "requests": [
                    {
                        "deleteContentRange": {
                            "range": {
                                "startIndex": backup["cut_index"],
                                "endIndex": end,
                            }
                        }
                    }
                ]
            },
        ).execute()
        notes["last_ordinal"] = backup.get("prev_last_ordinal")
        if not args.no_state:
            save_notes(args.book_id, notes)
        print(f"restored: cut everything after index {backup['cut_index']}")
        return 0

    if not args.bullets or args.ordinal is None:
        print("need <ordinal> and --bullets")
        return 2

    art = StructureArtifact.load(artifact_path(args.book_id))
    total = len(art.sections)
    if not 1 <= args.ordinal <= total:
        print(f"ordinal {args.ordinal} out of range 1..{total}")
        return 1

    rows = parse_bullets(args.bullets)
    doc = docs.documents().get(documentId=doc_id).execute()
    index, text, ranges = plan_insert(doc, rows)
    requests = build_requests(index, text, ranges)

    notes_now = load_notes(args.book_id) or {}
    lower = notes_now.get("last_ordinal") or 0
    sections = args.ordinal - lower
    bullets = sum(1 for row in rows if row["kind"] != "heading")
    headings = sum(1 for row in rows if row["kind"] == "heading")
    ratio = bullets / sections if sections > 0 else 0
    print(f"concise check: {bullets} bullets + {headings} headings for {sections} sections")
    if ratio > 2:
        print(
            f"  WARNING: {ratio:.1f} bullets per section — the notes are meant as a "
            "digest (~1/section, ceiling 2). Trim before appending."
        )
    if sections <= 0:
        print("  WARNING: new ordinal is not past the notes tail; this would double up text.")

    if args.dry_run:
        print(f"doc      : {doc.get('title')} ({doc_id})")
        print(f"insert at: {index}")
        print(f"requests : {len(requests)}")
        print("-" * 78)
        print(text)
        return 0

    backup = snapshot(doc, index, args.book_id, args.ordinal)
    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    after = docs.documents().get(documentId=doc_id).execute()
    tail = [
        element
        for element in after["body"]["content"]
        if element.get("paragraph")
    ][-len(rows) :]
    print(f"appended {len(rows)} paragraphs to {after.get('title')}")
    print(f"backup   : {backup}")
    print("-" * 78)
    for element in tail:
        para = element["paragraph"]
        line = "".join(e.get("textRun", {}).get("content", "") for e in para["elements"]).rstrip("\n")
        bullet = "*" if para.get("bullet") else " "
        style = para.get("paragraphStyle", {}).get("namedStyleType", "")
        print(f"{bullet} [{style}] {line[:90]}")

    notes["last_ordinal"] = args.ordinal
    notes.setdefault("doc_id", doc_id)
    notes["doc_url"] = f"https://docs.google.com/document/d/{doc_id}/edit"
    if args.no_state:
        print("(--no-state: notes tail NOT advanced)")
    else:
        save_notes(args.book_id, notes)
        print(f"notes tail advanced to ordinal {args.ordinal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())