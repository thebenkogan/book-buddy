"""Which book `ask.py` answers from.

The bug this guards: a bare ``"jewish_war"`` default meant that a question
about ANY book was answered out of the Jewish War unless the caller remembered
to pass a book id. Selection must now be explicit or derived from his reading,
and the reason must always be reportable.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("bb_ask", REPO / "scripts" / "ask.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _position(cache: pathlib.Path, book: str, mtime: int) -> None:
    cache.mkdir(exist_ok=True)
    path = cache / f"{book}_position.json"
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_explicit_book_wins(tmp_path):
    ask = _load()
    ask.REPO = str(tmp_path)  # type: ignore[attr-defined]
    assert ask.resolve_book("les_mis")[0] == "les_mis"


def test_default_is_the_book_he_is_reading(tmp_path):
    ask = _load()
    ask.REPO = str(tmp_path)  # type: ignore[attr-defined]
    cache = tmp_path / "cache"
    _position(cache, "jewish_war", 1_000)
    _position(cache, "les_mis", 2_000)

    book, why = ask.resolve_book(None)

    assert book == "les_mis", "the most recently written position is the book in hand"
    assert "jewish_war" in why, "the other books are named, so the choice is checkable"


def test_no_positions_asks_rather_than_guesses(tmp_path):
    ask = _load()
    ask.REPO = str(tmp_path)
    book, why = ask.resolve_book(None)
    assert book is None
    assert "position" in why
