import glob
import logging
import random
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from src.api.config import CACHE_DIR, GUTENDEX_BASE_URL
from src.api.models import GutendexBookResult, SearchResponse

router = APIRouter(prefix="/api/v1/books", tags=["search"])

DEFAULT_PARAMS = {
    "mime_type": "text/plain",
    "languages": "en",
}

TOPICS = [
    "fiction",
    "science",
    "history",
    "children",
    "philosophy",
    "religion",
    "poetry",
    "adventure",
    "mystery",
    "romance",
]
SORT_OPTIONS = ["popular", "ascending", "descending"]


def _get_jittered_params() -> dict:
    page = random.randint(1, 10)
    sort = random.choice(SORT_OPTIONS)

    jitter_type = random.choice(["topic", "page"])

    params = {**DEFAULT_PARAMS, "sort": sort, "page": page}

    if jitter_type == "topic":
        params["topic"] = random.choice(TOPICS)
    elif jitter_type == "page":
        params["page"] = random.randint(1, 50)

    return params


def _is_book_indexed(input_id: str) -> bool:
    for path in glob.glob(str(CACHE_DIR / "*_embeddings.json")):
        book_id = Path(path).name.replace("_embeddings.json", "")
        if book_id == input_id:
            return True
    return False


def _extract_cover_url(formats: dict) -> str | None:
    if not formats:
        return None

    for mime_type, url in formats.items():
        if mime_type.startswith("image/"):
            return url
    return None


@router.get("/search", response_model=SearchResponse)
def search_books(query: str | None = None):
    """Search for books using the Gutendex API. If no query is provided, returns popular books with randomization."""
    if query and len(query.strip()) >= 2:
        params = {**DEFAULT_PARAMS, "search": query.strip()}
    else:
        params = _get_jittered_params()

    try:
        logging.info(f"Fetching from Gutendex: {params}")
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(f"{GUTENDEX_BASE_URL}/books", params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        logging.exception(f"Error fetching from Gutendex: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch from Gutendex API")

    results = []
    for book in data.get("results", []):
        author_name = "Unknown"
        if book.get("authors"):
            author_name = book["authors"][0].get("name", "Unknown")

        book_id = book["title"].replace(" ", "_").lower().strip()
        indexed = _is_book_indexed(book_id)
        cover_url = _extract_cover_url(book.get("formats", {}))

        results.append(
            GutendexBookResult(
                book_id=book_id,
                title=book["title"],
                author=author_name,
                download_count=book.get("download_count", 0),
                languages=book.get("languages", []),
                indexed=indexed,
                cover_url=cover_url,
            )
        )

    return SearchResponse(count=data.get("count", 0), results=results)
