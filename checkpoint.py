import json
from pathlib import Path
from book import GutenbergBook


def checkpoint(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) == 0 or not isinstance(args[0], GutenbergBook):
                raise ValueError(
                    "First argument must be a GutenbergBook for checkpointing"
                )
            book: GutenbergBook = args[0]
            cache_path = Path(
                f"cache/{book.title.replace(' ', '_').lower()}_{name}.json"
            )
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    return json.load(f)

            result = func(*args, **kwargs)

            with open(cache_path, "w") as f:
                json.dump(result, f)

            return result

        return wrapper

    return decorator
