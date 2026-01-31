from typing import List
from openrouter import OpenRouter
import semchunk
import tiktoken
from book import Book, Chunk
from checkpoint import checkpoint
import numpy as np

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


NUM_TOP_CHUNKS = 3


def get_best_chunks(book: Book, client: OpenRouter, query: str) -> List[Chunk]:
    resp = client.embeddings.generate(model=book.embedding_model, input=query)
    query_embedding = resp.data[0].embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    all_embeddings = []
    idx_to_chunk = {}
    i = 0
    for chapter in book.chapters:
        for chunk in chapter.chunks:
            all_embeddings.append(chunk.embedding)
            idx_to_chunk[i] = chunk
            i += 1

    chunk_embeddings = np.array(all_embeddings)
    chunk_norm = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True
    )
    similarities = np.dot(chunk_norm, query_norm)
    indices = np.argsort(similarities)[::-1][:NUM_TOP_CHUNKS]
    return [idx_to_chunk[i] for i in indices]
