from pydantic import BaseModel


class GutendexBookResult(BaseModel):
    book_id: str
    title: str
    author: str
    download_count: int
    languages: list[str]
    indexed: bool
    cover_url: str | None = None


class SearchResponse(BaseModel):
    count: int
    results: list[GutendexBookResult]


class AskQuestionRequest(BaseModel):
    question: str


class ReadingProgressRequest(BaseModel):
    user_id: str
    current_chapter: int
    current_position: int | None = None


class ReadingProgressResponse(BaseModel):
    user_id: str
    book_id: str
    current_chapter: int


class RequestBookRequest(BaseModel):
    user_id: str
    book_id: str
    title: str
    author: str
