from openrouter import OpenRouter
from book import GutenbergBook
import json
from rapidfuzz import fuzz
from pathlib import Path
from chunking import count_tokens

chapterize_schema = {
    "name": "book_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["rank", "title"],
                    "properties": {
                        "rank": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "The section's rank number.",
                        },
                        "title": {
                            "type": "string",
                            "description": "The full section name (e.g. 'PART', 'CHAPTER', etc), number, and the title if it exists. For example: 'Chapter III. A Little Demon'",
                        },
                    },
                    "additionalProperties": False,
                },
                "description": "List of all section titles and ranks found in the candidate lines. They should have consistent style.",
            },
        },
        "required": ["sections"],
        "additionalProperties": False,
    },
}


def prompt(candidate_list):
    return f"""
f"Below is a list of candidate lines found throughout the book. 
Extract all section headings and discard any lines that are not section headings. 
Each extracted heading should keep as much text as possible, i.e. keep the section name (e.g. 'PART', 'CHAPTER', etc), number, and the title if it exists. 
For each found section heading, also include it's rank. For example, if the book has parts, each with books, and each book with chapters, then:
each PART has rank 1, each BOOK has rank 2, and each CHAPTER has rank 3.
If only chapters are found, then each CHAPTER has rank 1.

\n\nLINES: {candidate_list} \n\n 

Return JSON with keys: 'sections' (list of objects, each containing a section name ('title') and it's rank number ('rank').",
"""


def chapterize(client: OpenRouter, book: GutenbergBook):
    # TODO: make a common cache decorator
    cache_path = Path(f"cache/{book.title.replace(' ', '_').lower()}_chapters.json")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    print(f"Chapterizing {book.title}")

    candidates = []
    offset = 0
    for line in book.text.split("\n\n\n"):
        if 1 < len(line) < 100:
            candidates.append((offset, line.strip()))
        offset += len(line) + 3

    candidate_list = "\n\n".join([c[1] for c in candidates])
    response = client.chat.send(
        model="google/gemma-3-27b-it:free",  # TODO: figure out model fallback
        messages=[
            {
                "role": "system",
                "content": f"You are an expert on the book {book.title}, specifically about how the book is organized. Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": prompt(candidate_list),
            },
        ],
        response_format={"type": "json_schema", "json_schema": chapterize_schema},
        stream=False,
    )

    print(f"Model: {response.model}")
    try:
        data = json.loads(response.choices[0].message.content)
        print("Response data: ", json.dumps(data, indent=4))
    except Exception as e:
        print("Failed to parse response: ", response, e)
        raise e

    sections = data["sections"]
    max_rank = max([s["rank"] for s in sections])
    ctx = {}
    chapters = []
    for s in sections:
        rank = s["rank"]
        title = s["title"].lower()
        if rank == max_rank:
            offset, _ = max(candidates, key=lambda c: fuzz.ratio(title, c[1]))
            chapters.append({"ctx": ctx.copy(), "start": offset, "chapter": title})
        else:
            for r in range(rank + 1, max_rank):
                ctx.pop(r, None)
            ctx[rank] = title

    for i, chapter in enumerate(chapters):
        end = chapters[i + 1]["start"] if i < len(chapters) - 1 else len(book.text)
        chapter["text"] = book.text[chapter["start"] : end]
        chapter["tokens"] = count_tokens(chapter["text"])

    with open(cache_path, "w") as f:
        json.dump(chapters, f)
    return chapters
