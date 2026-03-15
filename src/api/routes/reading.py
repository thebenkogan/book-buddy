import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.config import CACHE_DIR, DATA_DIR
from src.api.dependencies import get_openrouter_client
from src.api.models import AskQuestionRequest
from src.index.book import GutenbergBook
from src.index.query import query

router = APIRouter(prefix="/api/v1/books", tags=["reading"])


def _get_index_path(book_id: str):
    path = CACHE_DIR / f"{book_id}_embeddings.json"
    if not str(path).startswith(str(CACHE_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    return path


@router.get("/{book_id}/content")
def get_book_text(book_id: str):
    """Return the entire book text by reading from the data directory."""
    filename = f"{book_id}.txt"
    file_path = (DATA_DIR / filename).resolve()
    if not str(file_path).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found")

    try:
        book = GutenbergBook.from_file(str(file_path))
        return {"text": book.text, "currentPosition": 0}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


@router.get("/{book_id}/toc")
def get_table_of_contents(book_id: str):
    """Return the table of contents for a book."""
    index_path = _get_index_path(book_id)
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


@router.get("/{book_id}/summary")
def get_summary(book_id: str):
    """Return a summary of the book."""
    index_path = _get_index_path(book_id)
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not indexed")

    try:
        book = GutenbergBook.model_validate_json(index_path.read_text())
        return {
            "summary": book.chapters[0].summary,
            "keyPoints": [],
        }
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


@router.post("/{book_id}/ask")
def ask_question(
    book_id: str,
    request: AskQuestionRequest,
    client=Depends(get_openrouter_client),
):
    """Ask a question about the book."""
    index_path = _get_index_path(book_id)
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not indexed")

    try:
        book = GutenbergBook.model_validate_json(index_path.read_text())
        answer = query(book, client, request.question)
        return {"answer": answer}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)
