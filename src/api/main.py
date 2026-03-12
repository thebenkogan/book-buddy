import glob
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel

from openrouter import OpenRouter
from pymongo import ReturnDocument
from src.index.book import GutenbergBook
from src.index.query import query

load_dotenv()


def get_openrouter_client(request: Request) -> OpenRouter:
    return request.app.state.openrouter_client


# Paths relative to project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
router = APIRouter(prefix="/api/v1")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "bookdb")


def get_mongo_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.mongo_db


@router.get("/books")
def get_books():
    """Return all books on my shelf."""
    books = []
    for path in glob.glob(str(CACHE_DIR / "*_embeddings.json")):
        book = GutenbergBook.model_validate_json(Path(path).read_text())
        books.append(
            {
                "id": book.book_id,
                "title": book.title,
                "progress": 0,
            }
        )
    return books


@router.get("/books/{book_id}/content")
def get_book_text(book_id: str):
    """Return the entire book text by reading from the data directory."""
    # Security: ensure path doesn't escape data dir
    filename = f"{book_id}.txt"
    file_path = (DATA_DIR / filename).resolve()
    if not str(file_path).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found")

    try:
        book = GutenbergBook.from_file(file_path)
        return {"text": book.text, "currentPosition": 0}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


@router.get("/books/{book_id}/toc")
def get_table_of_contents(book_id: str):
    """Return the entire Book index object from cache."""
    index_path = CACHE_DIR / f"{book_id}_embeddings.json"
    if not str(index_path).startswith(str(CACHE_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not indexed")

    try:
        book = GutenbergBook.model_validate_json(index_path.read_text())
        chapters = []
        for i, chapter in enumerate(book.chapters):
            chapters.append(
                {
                    "id": i,
                    "name": chapter.name,
                    "context": chapter.context,
                    "startPosition": chapter.start,
                }
            )
        return {"chapters": chapters}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


@router.get("/books/{book_id}/summary")
def get_summary(book_id: str):
    """Return a summary of what was recently read."""
    index_path = CACHE_DIR / f"{book_id}_embeddings.json"
    if not str(index_path).startswith(str(CACHE_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not indexed")

    try:
        book = GutenbergBook.model_validate_json(index_path.read_text())
        return {
            "summary": book.chapters[0].summary,
            "keyPoints": [],  # TODO: remove?
        }
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


class AskQuestionRequest(BaseModel):
    question: str


@router.post("/books/{book_id}/ask")
def ask_question(
    book_id: str,
    request: AskQuestionRequest,
    client: OpenRouter = Depends(get_openrouter_client),
):
    """Ask a question about the book."""
    index_path = CACHE_DIR / f"{book_id}_embeddings.json"
    if not str(index_path).startswith(str(CACHE_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not indexed")

    try:
        book = GutenbergBook.model_validate_json(index_path.read_text())
        answer = query(book, client, request.question)
        return {"answer": answer}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


class ReadingProgressRequest(BaseModel):
    user_id: str
    current_chapter: int
    current_position: int | None = None


class ReadingProgressResponse(BaseModel):
    user_id: str
    book_id: str
    current_chapter: int


@router.put("/books/{book_id}/progress", response_model=ReadingProgressResponse)
async def update_reading_progress(
    book_id: str,
    body: ReadingProgressRequest,
    db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    """Update or create a user's reading progress for a given book."""
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
        current_position=doc.get("current_position"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    app.state.openrouter_client = client
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    app.state.mongo_client = mongo_client
    app.state.mongo_db = mongo_client[MONGO_DB_NAME]

    # Ensure collections and indexes exist
    db = app.state.mongo_db
    await db["books"].create_index("book_id", unique=True)
    await db["reading_progress"].create_index(
        [("user_id", 1), ("book_id", 1)], unique=True
    )

    try:
        yield
    finally:
        mongo_client.close()


app = FastAPI(title="Book API", lifespan=lifespan)
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies/auth headers
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
