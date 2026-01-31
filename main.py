from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize
from embedding import chunk_and_embed
from summarize import summarize

load_dotenv()

book = GutenbergBook.from_file("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

book = chapterize(book, client)
book = summarize(book, client)
book = chunk_and_embed(book, client, model="qwen/qwen3-embedding-8b")
# TODO: support search, do cosine similarity on embeddings


print(book.chapters[22])
