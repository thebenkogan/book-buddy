from typing import List
from openrouter import OpenRouter
import semchunk
import tiktoken
from book import Book, Chunk
from checkpoint import checkpoint

TOKENS_PER_EMBED_CALL = 200_000


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


@checkpoint("embeddings")
def chunk_and_embed(book: Book, client: OpenRouter, model: str) -> Book:
    chunker = semchunk.chunkerify("gpt-4", 2500)
    chunks_per_chapter = chunker([c.text for c in book.chapters], overlap=100)
    for chunks, chapter in zip(chunks_per_chapter, book.chapters):
        title_ctx = "|".join([v for _, v in sorted(chapter.context.items())])
        title = title_ctx + "|" + chapter.name
        context = f"This chunk is from the chapter: {title}.\n\nSummary:\n\n{chapter.summary}.\n\nChunk text:\n\n"
        chapter_chunks: List[Chunk] = []
        for chunk in chunks:
            chunk = context + chunk
            chapter_chunks.append(Chunk(text=chunk, tokens=count_tokens(chunk)))
        chapter.chunks = chapter_chunks

    all_chunks = []
    for chapter in book.chapters:
        all_chunks.extend(chapter.chunks)

    # TODO: parallelize batch processing
    chunk_batches = create_chunk_batches(all_chunks)
    embeddings = []
    for batch in chunk_batches:
        print(f"Embedding batch of {len(batch)} chunks")
        texts = [c.text for c in batch]
        resp = client.embeddings.generate(model=model, input=texts)
        for e in resp.data:
            # truncate to 5 decimals to reduce storage size
            embedding = [round(x, 5) for x in e.embedding]
            embeddings.append(embedding)

    i = 0
    for chapter in book.chapters:
        for chunk in chapter.chunks:
            chunk.embedding = embeddings[i]
            i += 1

    book.embedding_model = model
    return book


# TODO: DRY with chapter batching in summarize.py
def create_chunk_batches(chunks: List[Chunk]):
    batches = []
    batch = []
    size = 0
    for chunk in chunks:
        if size == 0 or size + chunk.tokens <= TOKENS_PER_EMBED_CALL:
            batch.append(chunk)
            size += chunk.tokens
        else:
            batches.append(batch)
            batch = [chunk]
            size = chunk.tokens

    if batch:
        batches.append(batch)
    return batches
