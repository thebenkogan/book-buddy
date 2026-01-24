from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize
from summarize import summarize

load_dotenv()

book = GutenbergBook("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

chapters = chapterize(book, client)
chapters = summarize(book, client, chapters)

del chapters[34]["text"]
print(chapters[34])
