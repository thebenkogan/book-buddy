from pathlib import Path
from openrouter import OpenRouter
from book import GutenbergBook
import json

TOKENS_PER_BATCH = 30_000


def summarize(client: OpenRouter, book: GutenbergBook, chapters):
    cache_path = Path(
        f"cache/{book.title.replace(' ', '_').lower()}_chapters_summaries.json"
    )
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    print(f"Summarizing {book.title}")

    # TODO: fix this logic to create batches first, then send them off in a thread pool
    batch = []
    size = 0
    for i, chapter in enumerate(chapters):
        if (
            size == 0
            or size + chapter["tokens"] < TOKENS_PER_BATCH
            or i == len(chapters) - 1
        ):
            batch.append(chapter)
            size += chapter["tokens"]
            if i != len(chapters) - 1:
                continue

        summaries = summarize_batch(client, book, batch)
        for s, c in zip(summaries, batch):
            c["summary"] = s

        size = chapter["tokens"]
        batch = [chapter]

    with open(cache_path, "w") as f:
        json.dump(chapters, f)
    return chapters


def summarize_schema(num_chapters):
    return {
        "name": "chapter_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summaries": {
                    "type": "array",
                    "minItems": num_chapters,
                    "maxItems": num_chapters,
                    "items": {
                        "type": "string",
                    },
                    "description": "List of summaries, one for each chapter.",
                },
            },
            "required": ["summaries"],
            "additionalProperties": False,
        },
    }


def prompt(chapters):
    texts = []
    for c in chapters:
        ctx = "|".join([v for _, v in sorted(c["ctx"].items())])
        texts.append(f"START OF NEW CHAPTER ({ctx}):\n\n{c['text']}\n\n")
    texts = "".join(texts)

    return f"""
Below is a list of the full text of {len(chapters)} chapters in the book. Each chapter is separated by a 'START OF NEW CHAPTER:' line 
contexualizing the chapter. Use your knowledge of the book and the provided chapter texts to generate a concise, spoiler-free 
summary of each chapter. There should be {len(chapters)} summaries in your response.

\n\nCHAPTERS:\n\n {texts} \n\n 

Return JSON with keys: 'summaries' (list of {len(chapters)} strings)",
"""


def summarize_batch(client: OpenRouter, book: GutenbergBook, chapters):
    print(f"Summarizing {len(chapters)} chapters")

    response = client.chat.send(
        model="google/gemma-3-27b-it:free",  # TODO: figure out model fallback
        messages=[
            {
                "role": "system",
                "content": f"You are an expert on the book {book.title} tasked with summarizing some chapters of the book. Return ONLY valid JSON.",
            },
            {
                "role": "user",
                "content": prompt(chapters),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": summarize_schema(len(chapters)),
        },
        stream=False,
    )

    print(f"Model: {response.model}")
    try:
        data = json.loads(response.choices[0].message.content)
        print("Response data: ", json.dumps(data, indent=4))
    except Exception as e:
        print("Failed to parse response: ", response, e)
        raise e

    summaries = data["summaries"]
    if len(summaries) != len(chapters):
        raise ValueError(f"Expected {len(chapters)} summaries, got {len(summaries)}")
    return summaries
