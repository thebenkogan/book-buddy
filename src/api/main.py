import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from src.index.book import GutenbergBook

app = FastAPI(title="Book API")

# Paths relative to project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"


@app.get("/books/{book_id}/content")
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
        return {"text": book.text}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)


@app.get("/books/{book_id}/toc")
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
        for chapter in book.chapters:
            chapters.append(
                {
                    "name": chapter.name,
                    "context": chapter.context,
                    "startPosition": chapter.start,
                }
            )
        return {"chapters": chapters}
    except Exception as e:
        logging.exception(f"Error reading book '{book_id}': {e}")
        raise HTTPException(status_code=500)
