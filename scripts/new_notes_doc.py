#!/usr/bin/env python
"""Create the Google Doc notes file for a book that was just onboarded.

    uv run --with google-api-python-client --with google-auth \
        python scripts/new_notes_doc.py les_mis --title "Les Miserables (Victor Hugo)"

Creates the Doc in the "Books" Drive folder (the folder the Jewish War notes
live in), then writes ``cache/<book_id>_notes.json``:

    {"last_ordinal": 0, "doc_id": "...", "doc_url": "...", "parent": "..."}

``last_ordinal: 0`` is correct for a fresh book: nothing of the text has been
summarized yet, so the next ``notes_range.py`` call starts at section 1.

The doc starts EMPTY on purpose. Notes accrue one reading-update at a time via
``scripts/gdoc_notes.py``; there is no boilerplate to keep in sync.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = os.path.expanduser("~/.hermes/google_token.json")
FOLDER_NAME = "Books"
SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]


def clients():
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
    return build("docs", "v1", credentials=creds), build("drive", "v3", credentials=creds)


def find_folder(drive, name: str) -> str | None:
    query = (
        f"mimeType='application/vnd.google-apps.folder' and name='{name}' "
        "and trashed=false"
    )
    found = drive.files().list(q=query, fields="files(id,name)").execute().get("files", [])
    return found[0]["id"] if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_id")
    parser.add_argument("--title", required=True, help='Doc name, e.g. "Les Miserables (Victor Hugo)"')
    parser.add_argument("--parent", default=None, help='Drive folder id (default: the "Books" folder)')
    parser.add_argument("--force", action="store_true", help="overwrite an existing notes state")
    args = parser.parse_args()

    state_path = os.path.join(REPO, "cache", f"{args.book_id}_notes.json")
    if os.path.exists(state_path) and not args.force:
        print(f"{state_path} already exists; pass --force to replace it")
        return 1

    docs, drive = clients()
    parent = args.parent or find_folder(drive, FOLDER_NAME)
    created = docs.documents().create(body={"title": args.title}).execute()
    doc_id = created["documentId"]
    if parent:
        drive.files().update(fileId=doc_id, addParents=parent, fields="id,parents").execute()

    payload = {
        "book_id": args.book_id,
        "doc_id": doc_id,
        "doc_url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "parent": parent,
        "last_ordinal": 0,
        "note": "0 = nothing of this book summarized yet; the next update starts at section 1.",
    }
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"created notes doc : {args.title}")
    print(f"  doc id          : {doc_id}")
    print(f"  url             : {payload['doc_url']}")
    print(f"  folder          : {parent or '(Drive root)'}")
    print(f"  state           : {state_path}")
    print("  next: first reading update -> scripts/notes_range.py then gdoc_notes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())