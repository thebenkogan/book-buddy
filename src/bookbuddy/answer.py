"""Answer a reader's question using ONLY chunks from their read portion.

This is the top of the RAG stack. :meth:`BookStore.search` decides *what* the
model is allowed to see; this module decides what it is told to do with it,
and it deliberately narrows the model's freedom rather than trusting it.

The citation rule
-----------------
Every factual claim must cite a numbered block ``[1]``, ``[2]``. This is not
decoration:

* it forces the model to copy from the retrieved passages rather than from
  memory, because a claim without a passage has nowhere to point;
* it makes a hallucination visible to the reader instead of invisible;
* it is machine-checkable. :meth:`RagAnswer.citations_used` reads back which
  markers the answer actually referenced, and an answer with **no** citations
  is the strongest available signal that the model ignored the retrieval and
  answered from pretraining.

That last point matters because the leakage probe measured this exact failure:
given the full read portion, the model volunteered "Titus destroyed
Jerusalem's Temple in 70 CE" and then appended "but not in this excerpt"
*after* the spoiler.

Why generate rather than extractive-only
----------------------------------------
The probe recommended extractive-only when the whole read portion (76k tokens)
was in the prompt. RAG changes the shape: ~3.5k tokens of selected passages,
with no surrounding narrative for the model to infer plot from. Measured with
RAG, "Does Titus destroy the Temple?" at ordinal 40 returns a refusal.

That is better, not proven safe. ``scripts/probe_leakage.py`` measured the old
pipeline; re-run it against this path before trusting the guarantee. This
function is not the spoiler gate -- the gate is the ordinal range filter
inside Qdrant.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.bookbuddy import chat, get_model
from src.bookbuddy.rag import BookStore, Hit, hits_to_prompt

logger = logging.getLogger(__name__)


ANSWER_SYSTEM = """\
You answer questions about a book for a reader who has only read the first \
part of it.

You will be given CONTEXT: a few passages from the part they have read, each \
labelled [1], [2], ... and followed by the section it came from.

Rules, in priority order:

1. Answer ONLY from the CONTEXT. Every factual claim must be supported by a \
passage you were given. If the context does not contain the answer, say so.
2. Quote or closely paraphrase the passage you used, and cite its number, \
like "[2]". Do not make a claim without a citation.
3. You may know how this book ends. You must not say. The reader has not \
reached those events, and telling them destroys their reading. If the answer \
to their question is something that happens later, reply that it has not \
happened yet in their reading, and tell them which section to read next. \
Never name the later event, the person who causes it, or the outcome.
4. If the context is not enough to answer, say which section would answer it \
and stop. Do not fill the gap from memory.

Be concise and concrete. Prefer the book's own words."""


class RagAnswer(BaseModel):
    """A grounded answer plus the provenance needed to check it."""

    question: str
    answer: str
    ordinal: int
    hits: List[dict] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    grounded: bool = True
    def citations_used(self) -> List[str]:
        """Which ``[n]`` markers the answer actually referenced.

        An answer with no citations means the model answered from memory.
        """
        found = re.findall(r"\[(\d+)\]", self.answer)
        return [f"[{n}]" for n in sorted({int(n) for n in found})]


def build_answer_prompt(question: str, hits: List[Hit], max_tokens: int = 4000) -> str:
    context = hits_to_prompt(hits, max_tokens=max_tokens)
    return (
        f"CONTEXT (passages the reader has already read):\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"READER'S QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above, citing the numbers."
    )


def retrieve_for_answer(
    question: str,
    store: BookStore,
    ordinal: int,
    top_k: int = 6,
    lexical=None,
    prefer: str = "hybrid",
    max_context_tokens: int = 4000,
):
    """Retrieve the passages an answer may be built from.

    THIS IS THE PRIMARY ENTRY POINT. It makes NO model call.

    The reader (Hermes) is the model. Retrieval decides *what* may be
    consulted; the reader decides how to phrase it. Sending the retrieved
    chunks back through a second OpenRouter completion -- to have a model
    answer passages a model already retrieved -- costs a round trip and
    produces a worse answer, because it throws away the conversation.

    Earlier this module called OpenRouter itself (``answer_question``). It
    still does, because the leak probe needs a fixed, reproducible second
    opinion, and because it is how the guardrail is measured against
    something other than the reader's own judgement. But the interactive
    path does not use it.

    Returns ``(hits, prompt)``. ``hits`` is empty when the reader has read
    nothing searchable.
    """
    hits = store.search(
        question, ordinal, top_k=top_k, lexical=lexical, prefer=prefer
    )
    if not hits:
        return [], ""
    return hits, build_answer_prompt(question, hits, max_context_tokens)


def answer_question(
    question: str,
    store: BookStore,
    ordinal: int,
    top_k: int = 6,
    lexical=None,
    prefer: str = "hybrid",
    model: Optional[str] = None,
    max_context_tokens: int = 4000,
    max_output_tokens: int = 800,
) -> RagAnswer:
    """Retrieve, then answer in ONE extra OpenRouter call.

    Used by the leak probe and the onboarding spot-check, NOT by the normal
    reading loop -- see :func:`retrieve_for_answer`. Keeping it makes the
    guardrail measurable against a second, independent model call rather than
    against the reader's own judgement, which is the more meaningful test of
    whether the instructions alone are sufficient.
    """
    # The gate lives in Qdrant: `ordinal` becomes a range predicate, so chunks
    # past the reader's position cannot come back regardless of their score.
    hits = store.search(
        question, ordinal, top_k=top_k, lexical=lexical, prefer=prefer
    )

    if not hits:
        return RagAnswer(
            question=question,
            answer=(
                "You haven't read anything I can search yet — "
                "tell me how far you've got and I'll pick it up."
            ),
            ordinal=ordinal,
            hits=[],
            grounded=True,
        )

    prompt = build_answer_prompt(question, hits, max_context_tokens)
    result = chat(
        prompt,
        system=ANSWER_SYSTEM,
        model=model or get_model(),
        temperature=0.0,
        max_tokens=max_output_tokens,
        reasoning_effort="low",
    )
    text = (result.get("content") or "").strip()

    return RagAnswer(
        question=question,
        answer=text,
        ordinal=ordinal,
        hits=[
            {
                "n": i,
                "label": h.label,
                "start": h.start,
                "end": h.end,
                "score": round(h.score, 4),
                "source": h.source,
            }
            for i, h in enumerate(hits, start=1)
        ],
        tokens_in=result.get("tokens_in", 0),
        tokens_out=result.get("tokens_out", 0),
        model=result.get("model", get_model()),
        grounded=bool(text),
    )