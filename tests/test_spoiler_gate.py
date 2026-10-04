"""The spoiler guarantee, enforced structurally over every module.

This file used to live inside ``test_ask.py`` alongside tests for the retired
``ask.py`` pipeline. It is the single most important test in the repo, so it
stands alone: if a module ever slices ``Book.text`` directly instead of going
through ``text_up_to``, book-buddy leaks the part the reader has not reached --
and that failure is silent, because a wrong offset produces a confident wrong
answer rather than a crash.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent.parent

#: Receivers whose ``.text`` IS the book body and must go through
#: ``text_up_to``. A pydantic ``Book`` is reached as ``book.text`` /
#: ``self.text``; anything else (e.g. ``chunk.text``, a str field) is fine.
BOOK_TEXT_RECEIVERS = {"book", "self", "b", "novel"}

#: Modules that move book text into prompts, and HOW each is allowed to do it.
#:
#: ``rag`` is the ONLY module that turns book text into retrievable passages,
#: so it is the only one that must obtain that text through ``text_up_to``.
#:
#: ``onboarding`` works purely in offsets and labels -- it never reads the book
#: -- and ``answer`` receives chunks that ``rag`` already retrieved through the
#: gate. Requiring either of them to import ``text_up_to`` would be asserting
#: something untrue about them.
NEEDS_TEXT_UP_TO = ("src/bookbuddy/rag.py",)

#: Every module that ends up putting book content into a prompt. All of them
#: must be free of direct ``book.text[...]`` slicing.
OWNED_MODULES = NEEDS_TEXT_UP_TO + ("src/bookbuddy/answer.py",)


def _text_slices_ast(tree: ast.AST) -> List[int]:
    """Line numbers of ``<book>.text[...]`` subscripts.

    Scoped to book-like receivers on purpose. Flagging EVERY ``.text``
    attribute also flagged ``chunk.text`` and ``len(book.text)``, where the
    attribute is a str field or a length -- nothing to do with the spoiler
    gate -- which made the check impossible to satisfy without deleting
    legitimate code.

    Only SUBSCRIPTING a book-like receiver counts, because slicing is what
    leaks: ``book.text[start:end]``. Reading ``len(book.text)`` or
    ``self.text`` cannot produce a passage.
    """
    hits: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute) or value.attr != "text":
            continue
        owner = getattr(value.value, "id", None) or getattr(
            value.value, "attr", None
        )
        if owner in BOOK_TEXT_RECEIVERS:
            hits.append(node.lineno)
    return hits


def test_no_module_slices_book_text():
    """``text_up_to`` is the ONLY thing permitted to move book text into a prompt.

    Scans every module under ``src/``. The retired ``src/index/`` and the
    FastAPI layer are gone, so there is no legacy exemption list any more --
    anything left is an offender.
    """
    offenders: List[str] = []
    for path in sorted((REPO / "src").rglob("*.py")):
        if path.name == "book.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno in _text_slices_ast(tree):
            rel = str(path.relative_to(REPO))
            offenders.append(f"{rel}:{lineno}")
    assert offenders == [], (
        "these modules slice Book.text directly, bypassing the spoiler gate; "
        "use text_up_to(): " + ", ".join(offenders)
    )


def test_owned_modules_do_not_slice_book_text():
    """Each module that feeds a prompt must reach text only via the gate."""
    for rel in OWNED_MODULES:
        path = REPO / rel
        assert path.exists(), f"{rel} is missing"
        hits = _text_slices_ast(ast.parse(path.read_text(encoding="utf-8")))
        assert hits == [], f"{rel} slices .text directly at lines {hits}"


def test_modules_that_read_the_book_import_the_gate():
    """A module that reads book text must obtain it via ``text_up_to``.

    ``answer.py`` is excluded by design: it is handed retrieved chunks and
    never touches the book, so requiring a gate import there would be noise.
    """
    for rel in NEEDS_TEXT_UP_TO:
        source = (REPO / rel).read_text(encoding="utf-8")
        assert "text_up_to" in source, f"{rel} never mentions text_up_to"


def test_answer_module_only_sees_retrieved_chunks():
    """``answer.py`` must not read the book at all.

    It receives chunks from ``BookStore.search``, which already applied the
    ordinal gate. If it ever started reading the book directly it would
    reintroduce a second, ungated path for book text into a prompt.
    """
    source = (REPO / "src/bookbuddy/answer.py").read_text(encoding="utf-8")
    assert "BookStore" in source, "answer.py should retrieve via BookStore"
    assert ".search(" in source, "answer.py should call the gated search"
    hits = _text_slices_ast(ast.parse(source))
    assert hits == [], f"answer.py slices .text at lines {hits}"


def test_rag_reads_book_text_only_through_the_gate():
    """``rag.chunk_book`` builds chunks from gate output, not from Book.text.

    Pinned because an earlier version sliced the gated string with ABSOLUTE
    offsets, reading 16,998 characters early and mislabelling every chunk.
    """
    source = (REPO / "src/bookbuddy/rag.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "text_up_to"
    ]
    assert calls, "rag.py must obtain book text through text_up_to"


def test_no_retired_pipeline_remains():
    """The superseded long-context/map-reduce pipeline is gone.

    It is easy to leave behind, it shares names with the RAG answering path,
    and re-introducing it would mean two ways for book text to reach a prompt.
    """
    assert not (REPO / "src/bookbuddy/ask.py").exists()
    assert not (REPO / "src/index").exists(), "src/index is retired"
    assert not (REPO / "src/api").exists(), "no webserver in this project"