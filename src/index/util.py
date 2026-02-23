def create_batches(chunks, limit: int):
    batches = []
    batch = []
    size = 0
    for chunk in chunks:
        if size == 0 or size + chunk.tokens <= limit:
            batch.append(chunk)
            size += chunk.tokens
        else:
            batches.append(batch)
            batch = [chunk]
            size = chunk.tokens

    if batch:
        batches.append(batch)
    return batches