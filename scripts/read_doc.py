#!/usr/bin/env python
"""Read a book's notes Doc back — verification and visualisation.

    uv run --with google-api-python-client --with google-auth \
      python scripts/read_doc.py jewish_war                  # paragraphs, **bold** marked
    ... python scripts/read_doc.py jewish_war --tail 12      # only the last 12 paragraphs
    ... uv run --with google-api-python-client --with google-auth --with pypdfium2 --with Pillow \
      python scripts/read_doc.py jewish_war --image          # render the Doc to PNG(s)

Google Docs is updated server-side; a browser or phone app can keep showing a
stale copy for a while. When Ben reports that he cannot see an append, run this
instead of appending again: if the text is here, the write landed and he needs
to refresh. ``--image`` writes PNGs under ``cache/doc_images/`` for sending.

Exit code 0 always when the doc is readable; 1 when there is no doc_id for the
book. Nothing here reads the book text, so it cannot spoil.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TOKEN = os.path.expanduser("~/.hermes/google_token.json")
SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]
IMAGE_DIR = os.path.join(REPO, "cache", "doc_images")


def notes_path(book_id: str) -> str:
    return os.path.join(REPO, "cache", f"{book_id}_notes.json")


def load_notes(book_id: str) -> dict:
    path = notes_path(book_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def credentials():
    from google.oauth2.credentials import Credentials

    with open(TOKEN, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Credentials(
        token=raw.get("token"),
        refresh_token=raw.get("refresh_token"),
        token_uri=raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=raw.get("client_id"),
        client_secret=raw.get("client_secret"),
        scopes=raw.get("scopes") or SCOPES,
    )


def paragraphs(doc):
    """-> [(style, [(text, bold), ...])] for every non-empty paragraph."""
    rows = []
    for element in doc.get("body", {}).get("content", []):
        para = element.get("paragraph")
        if not para:
            continue
        runs = []
        for run in para.get("elements", []):
            text_run = run.get("textRun")
            if not text_run:
                continue
            style = text_run.get("textStyle", {})
            runs.append((text_run["content"], bool(style.get("bold"))))
        if any(text.strip() for text, _ in runs):
            rows.append((para.get("paragraphStyle", {}).get("namedStyleType", ""), runs))
    return rows


def show(rows, tail: int | None) -> None:
    if tail:
        rows = rows[-tail:]
    for style, runs in rows:
        line = "".join(("**" + text + "**") if bold else text for text, bold in runs).rstrip("\n")
        if style.startswith("HEADING"):
            print(f"# {line}")
        elif style == "NORMAL_TEXT" and line:
            print(f"- {line}" if not line.startswith(" ") else line)


def render_image(doc_id: str) -> list[str]:
    import pypdfium2 as pdfium
    from googleapiclient.discovery import build

    drive = build("drive", "v3", credentials=credentials())
    pdf_bytes = (
        drive.files()
        .export(fileId=doc_id, mimeType="application/pdf")
        .execute()
    )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_bytes)
    written = []
    for number in range(len(pdf)):
        out = os.path.join(IMAGE_DIR, f"{doc_id[:12]}_p{number + 1:02d}.png")
        pdf[number].render(scale=2.0).to_pil().save(out)
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("--tail", type=int, default=None, help="only the last N paragraphs")
    parser.add_argument("--image", action="store_true", help="render the Doc to PNGs (needs pypdfium2)")
    parser.add_argument("--doc", default=None, help="override the doc id")
    args = parser.parse_args()

    notes = load_notes(args.book_id)
    doc_id = args.doc or notes.get("doc_id")
    if not doc_id:
        print(f"no doc_id for {args.book_id} in {notes_path(args.book_id)}; pass --doc")
        return 1

    if args.image:
        for path in render_image(doc_id):
            print(path)
        return 0

    from googleapiclient.discovery import build

    docs = build("docs", "v1", credentials=credentials())
    doc = docs.documents().get(documentId=doc_id).execute()
    rows = paragraphs(doc)
    print(f"doc      : {doc.get('title')} ({doc_id})")
    print(f"paragraphs: {len(rows)}")
    print("-" * 78)
    show(rows, args.tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
