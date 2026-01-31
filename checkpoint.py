from pathlib import Path
from book import Book, GutenbergBook


def checkpoint(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) == 0 or not isinstance(args[0], Book):
                raise ValueError("First argument must be a Book for checkpointing")
            book: Book = args[0]
            cache_path = Path(
                f"cache/{book.title.replace(' ', '_').lower()}_{name}.json"
            )
            if cache_path.exists():
                # TODO: remove GutenbergBook hardcode, use type selector parser
                return GutenbergBook.model_validate_json(cache_path.read_text())

            result: Book = func(*args, **kwargs)

            cache_path.write_text(result.model_dump_json())

            return result

        return wrapper

    return decorator
