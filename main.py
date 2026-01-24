from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize

load_dotenv()

book = GutenbergBook("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
chapters = chapterize(client, book)

for c in chapters:
    print(c["ctx"], c["chapter"], c["text"][:200], c["start"])
