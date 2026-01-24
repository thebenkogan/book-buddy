import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# TODO: use semchunk to chunk each chapter for embedding generation
# chunker = semchunk.chunkerify("gpt-4", 2500)
# chunks = chunker([c["text"] for c in chapters], overlap=100)
# for chunk in chunks:
#     print(len(chunk))
# print(len(chunks))
