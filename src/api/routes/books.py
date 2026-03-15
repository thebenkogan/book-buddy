import glob

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from src.api.config import CACHE_DIR
from src.api.dependencies import get_mongo_db
from src.api.models import (
    ReadingProgressRequest,
    ReadingProgressResponse,
    RequestBookRequest,
)
from src.index.book import GutenbergBook

router = APIRouter(prefix="/api/v1/books", tags=["books"])


def _is_book_indexed(book_id: str) -> bool:
    for path in glob.glob(str(CACHE_DIR / "*_embeddings.json")):
        cached_book_id = Path(path).name.replace("_embeddings.json", "")
        if cached_book_id == book_id:
            return True
    return False


from pathlib import Path


@router.get("")
def get_books():
    """Return all books on my shelf."""
    books = []
    for path in glob.glob(str(CACHE_DIR / "*_embeddings.json")):
        book = GutenbergBook.model_validate_json(open(path).read())
        books.append(
            {
                "id": book.book_id,
                "title": book.title,
                "progress": 0,
            }
        )
    return books


@router.post("/{book_id}/add")
async def add_book_to_shelf(
    book_id: str,
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    """Add an indexed book to the user's bookshelf."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    if not _is_book_indexed(book_id):
        raise HTTPException(status_code=404, detail="Book not found in index")

    filter_doc = {"user_id": user_id, "book_id": book_id}
    update_doc = {
        "$set": {
            "user_id": user_id,
            "book_id": book_id,
            "current_chapter": 0,
        }
    }

    doc = await db["reading_progress"].find_one_and_update(
        filter_doc, update_doc, return_document=ReturnDocument.AFTER, upsert=True
    )

    return ReadingProgressResponse(
        user_id=doc["user_id"],
        book_id=doc["book_id"],
        current_chapter=doc["current_chapter"],
    )


@router.post("/request")
async def request_book(
    body: RequestBookRequest,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    """Request a new book to be added to the library."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    if _is_book_indexed(body.book_id):
        raise HTTPException(status_code=400, detail="Book is already indexed")

    existing = await db["requested_books"].find_one({"book_id": body.book_id})
    if existing:
        raise HTTPException(status_code=400, detail="Book already requested")

    await db["requested_books"].insert_one(
        {
            "user_id": body.user_id,
            "book_id": body.book_id,
            "title": body.title,
            "author": body.author,
        }
    )

    return {"status": "ok"}


@router.put(
    "/{book_id}/progress",
    response_model=ReadingProgressResponse,
)
async def update_reading_progress(
    book_id: str,
    body: ReadingProgressRequest,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    """Update or create a user's reading progress for a given book."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    filter_doc = {"user_id": body.user_id, "book_id": book_id}
    update_doc = {
        "$set": {
            "user_id": body.user_id,
            "book_id": book_id,
            "current_chapter": body.current_chapter,
        }
    }

    doc = await db["reading_progress"].find_one_and_update(
        filter_doc, update_doc, return_document=ReturnDocument.AFTER
    )
    if not doc:
        raise HTTPException(status_code=500, detail="Failed to update reading progress")

    return ReadingProgressResponse(
        user_id=doc["user_id"],
        book_id=doc["book_id"],
        current_chapter=doc["current_chapter"],
    )
