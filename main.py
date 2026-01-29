from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize
from embedding import chunk_and_embed
from summarize import summarize

load_dotenv()

book = GutenbergBook("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

chapters = chapterize(book, client)
chapters = summarize(book, client, chapters)
chapters = chunk_and_embed(book, client, chapters)
# TODO: support search, do cosine similarity on embeddings

print(chapters[0])
