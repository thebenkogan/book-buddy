from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize
from summarize import summarize

load_dotenv()

book = GutenbergBook("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

chapters = chapterize(client, book)
chapters = summarize(client, book, chapters)

del chapters[34]["text"]
print(chapters[34])
