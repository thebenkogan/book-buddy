from openrouter import OpenRouter
import os
from dotenv import load_dotenv
from book import GutenbergBook
from chapterize import chapterize

load_dotenv()

book = GutenbergBook("data/bk.txt")
client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
chunks = chapterize(client, book)

for chunk in chunks:
    print(chunk["ctx"], chunk["chapter"], chunk["text"][:200], chunk["start"])
