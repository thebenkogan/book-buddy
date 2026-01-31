from abc import ABC, abstractmethod
import re
from typing import List, Literal
from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    tokens: int
    embedding: List[float] = []


class Chapter(BaseModel):
    name: str
    summary: str = ""
    text: str = ""
    context: dict = {}
    tokens: int = 0
    start: int = 0
    chunks: List[Chunk] = []


class Book(BaseModel, ABC):
    text: str
    title: str
    embedding_model: str = ""
    chapters: List[Chapter]

    @classmethod
    @abstractmethod
    def from_file(cls, path: str, embedding_model: str) -> "Book":
        pass


guten_start_regex = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK (.+) \*\*\*"
guten_end_regex = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*"


class GutenbergBook(Book):
    type: Literal["gutenberg"] = "gutenberg"

    @classmethod
    def from_file(cls, path: str) -> "GutenbergBook":
        with open(path, "r") as f:
            text = f.read()

        sm = re.search(guten_start_regex, text)
        em = re.search(guten_end_regex, text)
        text = text[sm.end() : em.start()].lower()  # TODO: remove lower?
        title = sm.groups()[0]

        return cls(text=text, title=title, chapters=[])
