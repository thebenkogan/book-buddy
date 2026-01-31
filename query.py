from typing import List
from openrouter import OpenRouter
from book import Book, Chunk
from embedding import get_best_chunks


def system_prompt(book: Book):
    return f"""
You are an expert on the book {book.title} tasked with answering a question from a reader. 
Use only the provided context chunks to answer the question. You may use your knowledge of
the book to inform your answer, but do not use information found after the chunks to avoid spoilers.
If the answer to the reader's question requires spoilers, then you should guide their thinking and
encourage further reading.
"""


def user_prompt(chunks: List[Chunk], query: str):
    chunk_texts = "\n\n".join([c.text for c in chunks])
    return f"""
Context:

{chunk_texts}

Reader's question: {query}

Answer:
"""


def query(book: Book, client: OpenRouter, query: str):
    chunks = get_best_chunks(book, client, query)
    print(user_prompt(chunks, query))
    response = client.chat.send(
        model="openai/gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": system_prompt(book),
            },
            {
                "role": "user",
                "content": user_prompt(chunks, query),
            },
        ],
        stream=True,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
