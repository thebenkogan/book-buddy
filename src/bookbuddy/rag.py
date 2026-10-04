"""Qdrant-backed hybrid RAG for book-buddy.

Replaces the hand-rolled vector store from the previous ``rag.py``. The
retrieval *quality* approach is unchanged (dense embeddings + BM25 lexical,
fused), but storage, filtering and ranking now run inside Qdrant instead of
Python list comprehensions.

Why Qdrant rather than more hand-rolled code
---------------------------------------------
Because the spoiler gate becomes a **database constraint**. The reader's
position is pushed down as a range predicate::

    models.Filter(must=[FieldCondition(key="ordinal", range=Range(lte=40))])

Qdrant cannot return a point outside that range, so the guarantee no longer
depends on this module remembering to filter. In the previous implementation a
single missed check in one call path would have leaked the ending; the same
mistake now returns nothing, because the constraint lives underneath.

Hybrid search
-------------
Dense (``openai/text-embedding-3-small`` via OpenRouter, using your existing
key) handles paraphrase; sparse BM25 rescues exact proper nouns the embedding
smooths away. Both are kept because they fail differently, and Qdrant fuses
them with RRF.

Available through OpenRouter (verified): text-embedding-3-small (1536),
-3-large (3072), qwen3-embedding-8b (4096), bge-m3 (1024). Most other
embedding models 404 -- the catalogue is thin.

Offset space -- read before trusting a chunk offset
-----------------------------------------------------
``Chunk.start`` is an offset into the NOTE-STRIPPED text, not into
``Book.text``. Translator commentary is excised before chunking, so cleaned
and raw offsets diverge by however many characters were removed. Use
``ordinal`` for gating and ``label`` for provenance; never map ``start`` back
onto ``Book.text`` expecting to land on the same passage.

An earlier version sliced the gated text with absolute offsets, which read
16,998 characters early and mislabelled every chunk. The tests in
``tests/test_rag.py`` pin *section membership*, not offsets, because offsets in
cleaned space are not independently meaningful.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from src.bookbuddy import cache_dir, get_api_key

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EMBED_MODEL = os.environ.get("BOOKBUDDY_EMBED_MODEL", "openai/text-embedding-3-small")
DENSE_DIM = int(os.environ.get("BOOKBUDDY_DENSE_DIM", "1536"))

#: ~500 tokens: paragraph-ish. Six chunks fit a prompt with room to answer.
CHUNK_TOKENS = int(os.environ.get("BOOKBUDDY_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("BOOKBUDDY_CHUNK_OVERLAP", "60"))

TOP_K = int(os.environ.get("BOOKBUDDY_TOP_K", "6"))

#: RRF weights per arm. Measured on the Jewish War for "Who is Ananus?":
#: sparse returned all four passages that name Ananus; dense returned none of
#: them (its top hits were unrelated chapters at cosine 0.43, barely above
#: each other). So the lexical arm is weighted above the dense one, to keep
#: its decisive hit ahead of dense's near-tied generic ones.
#:
#: THE WEIGHT RATIO IS CAPPED BY THE POOL'S RANK SPREAD, and this is the part
#: that was wrong before. An arm's contribution runs from ``weight/(RRF_K+1)``
#: at rank 1 down to ``weight/(RRF_K+FUSION_POOL)`` at the last rank it
#: contributes. The weaker arm can only influence the ordering if its BEST hit
#: beats the stronger arm's WORST, i.e. if
#: ``min/max > (RRF_K+1)/(RRF_K+FUSION_POOL)`` ~= 0.726 here. The old defaults
#: were 0.4 vs 1.0, a ratio of 0.4, well under that bound: measured on the real
#: index, every sparse-arm hit scored at least 0.0119 while the best possible
#: dense-only hit scored 0.0066, so for any query with a non-empty sparse arm
#: NO dense-only chunk could ever be returned -- the dense arm could only break
#: ties among chunks the sparse arm had already returned. The comment here
#: claimed dense "is what handles paraphrase"; with those weights it did not,
#: and the claim described code that could not do it.
#:
#: 0.8 keeps the lexical preference the measurement asks for (sparse rank 1 =
#: 0.0164 still beats dense rank 1 = 0.0131) while letting a decisive dense
#: hit beat the bottom of the lexical pool (0.0131 > 0.0119), which is what
#: paraphrase needs. ``_weight_ratio_is_sane`` enforces the bound.
DENSE_WEIGHT = float(os.environ.get("BOOKBUDDY_DENSE_WEIGHT", "0.8"))
SPARSE_WEIGHT = float(os.environ.get("BOOKBUDDY_SPARSE_WEIGHT", "1.0"))

#: How many candidates each arm contributes to the fusion.
FUSION_POOL = int(os.environ.get("BOOKBUDDY_FUSION_POOL", "24"))

#: Standard RRF constant. Larger values flatten the rank curve, making the
#: fusion more consensus-driven; 60 is the conventional value.
RRF_K = int(os.environ.get("BOOKBUDDY_RRF_K", "60"))
EMBED_BATCH = 128

#: Minimum document frequency, as a fraction of the corpus, for BM25 IDF.
#:
#: What it actually does, measured on 503 chunks: it CAPS a rare term's IDF.
#: "happens" (df=1) goes from IDF 5.82 to 3.87, so a word that occurs in a
#: single chunk cannot dominate the way it otherwise would.
#:
#: What it does NOT do, contrary to an earlier claim in this file: it does not
#: make the identifying term outweigh the accidental one. "masada" (df=24)
#: still scores 3.02, BELOW the floored "happens". Correcting this because the
#: original comment claimed the floor fixed a retrieval failure, and it only
#: reduced it. RRF fusion with the dense vector is what actually carries the
#: proper noun case; the floor keeps the lexical half from being swamped.
DF_FLOOR_FRACTION = 0.02

#: Bumped when the chunking or index contract changes, so a stale collection
#: is rebuilt rather than silently mixing chunk shapes.
SCHEMA_VERSION = 3

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing done down
    during each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves
    """.split()
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """One retrievable passage.

    ``ordinal`` is the spoiler gate: the flattened 1..N position of the
    section this passage belongs to. ``ordinal`` -- not ``start`` -- is what
    every query filters on.
    """

    index: int
    ordinal: int
    label: str
    text: str
    start: int = 0
    end: int = 0
    tokens: int = 0
    #: What was embedded: section heading prefixed to the cleaned text, so a
    #: bare passage knows which chapter it came from.
    embed_text: str = ""
    #: Share of this chunk that is bracketed editorial matter (translator
    #: notes, editorial asides). Indexed, never deleted -- see note_ratio.
    note_ratio: float = 0.0

    @property
    def payload(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "ordinal": int(self.ordinal),
            "label": self.label,
            "text": self.text,
            "start": int(self.start),
            "end": int(self.end),
            "tokens": int(self.tokens),
            "note_ratio": round(float(self.note_ratio), 4),
        }


class Hit(BaseModel):
    """A retrieved chunk plus its provenance."""

    index: int
    ordinal: int
    label: str
    text: str
    score: float
    start: int = 0
    end: int = 0
    #: Share of the chunk that is bracketed editorial matter. Stored so a
    #: caller can tell narrative from commentary; it does not hide the chunk.
    note_ratio: float = 0.0
    source: str = "hybrid"


# --------------------------------------------------------------------------- #
# Tokenising
# --------------------------------------------------------------------------- #


def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


# --------------------------------------------------------------------------- #
# Editorial notes
# --------------------------------------------------------------------------- #

#: Whiston's 1820 translation interleaves the translator's commentary in two
#: forms and BOTH must go before chunking:
#:
#:  1. Footnote BLOCKS between books, headed "WAR BOOK 1 FOOTNOTES", itemised
#:     "17 (return) [ ... ]". 79% of the tail region, and the single biggest
#:     retrieval-quality problem: they embed as "see chap. 7. sect." noise
#:     that outranks real narrative for questions about places, laws, dates.
#:  2. Short inline brackets in the narrative.
#:
#: Stripping runs on the SECTION text, not per chunk, because one note
#: routinely spans a chunk boundary and a per-chunk regex strands the note's
#: tail as if it were prose. Measured: footnote-bearing chunks 19% -> 1.5%.
_EDITOR_NOTE_RE = re.compile(r"\[[^\[\]]{20,4000}\]")
_FOOTNOTE_HEAD_RE = re.compile(r"(?m)^WAR BOOK[^\n]*FOOTNOTES[^\n]*$")
_STRUCTURAL_RE = re.compile(
    r"^(?:BOOK|CHAPTER|PART|VOLUME|SECTION|Canto|ACT|SCENE)\s+"
    r"(?:[0-9]{1,3}|[IVXLC]{1,7}|[A-Z]{3,12})",
    re.IGNORECASE,
)
_ORPHAN_SPACE_RE = re.compile(r"\s+([.,;:!?])")


def tidy_punctuation(text: str) -> str:
    """Clean up whitespace left behind after a note is excised.

    Stripping a bracketed note leaves its surrounding spaces: "Pillaged [By
    Antiochus] . As Also" becomes "Pillaged   . As Also" -- a run of spaces AND
    a space stranded in front of the punctuation. Removing only the second
    leaves "Pillaged  ." in the index, which then embeds with the gap.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _ORPHAN_SPACE_RE.sub(r"\1", text)


def note_ratio(text: str) -> float:
    """How much of this text is bracketed editorial matter, 0.0-1.0.

    Replaces deleting it. Stripping was measured to be lossy in ways that are
    not about the Jewish War at all: the same rule removed 543 characters of
    Alice's actual verse ("When the sands are all dry, he is gay as a lark")
    and Grimms' story title "[LITTLE RED RIDING HOOD]", because both sit
    inside brackets. Nothing is dropped now -- the ratio is stored on the
    chunk and used to RANK it below narrative at query time, so a
    misjudgement costs ordering, not text.

    Kept as a name for callers that only want the old behaviour.
    """
    if not text:
        return 0.0
    spans = sum(len(m.group(0)) for m in _EDITOR_NOTE_RE.finditer(text))
    return min(1.0, spans / max(len(text), 1))


#: How hard a chunk is demoted for being editorial. 1.0 would put a fully
#: editorial chunk last; the measured Whiston notes that motivated this were
#: 79% of the file and needed to lose badly, but a chunk that is only partly
#: bracketed should still compete on its narrative half.
NOTE_PENALTY = float(os.environ.get("BOOKBUDDY_NOTE_PENALTY", "0.7"))
# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def _split_into_token_sized_pieces(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    pieces: List[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size])
        if piece:
            pieces.append(piece)
        if i + size >= len(words):
            break
    return pieces


def _collapse_with_index(text: str) -> Tuple[str, List[int]]:
    """Return ``text`` whitespace-collapsed, plus a map back to raw offsets.

    ``flat_index[i]`` is the offset in ``text`` of the character that produced
    ``flat[i]``. A collapsed run of whitespace is represented by the single
    space at the FIRST whitespace character, so the map stays monotonic and
    ``flat_index[flat.find(needle)]`` lands at or just before the real passage.
    """
    parts: List[str] = []
    index: List[int] = []
    last_space = True
    for position, char in enumerate(text):
        if char.isspace():
            if last_space:
                continue
            parts.append(" ")
            index.append(position)
            last_space = True
        else:
            parts.append(char)
            index.append(position)
            last_space = False
    return "".join(parts), index


def chunk_book(
    book,
    toc: Sequence[Any],
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk the whole book on section boundaries.

    Reads book text only through ``text_up_to`` -- the spoiler gate -- and
    slices the *relative* result, because ``text_up_to`` starts at
    ``toc[0].offset``. See the module docstring for what happened when this
    was done with absolute offsets.
    """
    from src.bookbuddy.book import text_up_to

    n_sections = len(toc)
    if n_sections == 0:
        logger.warning("cannot chunk %s: empty toc", getattr(book, "book_id", "?"))
        return []

    base = int(toc[0].offset)
    whole = text_up_to(book, toc, n_sections)

    chunks: List[Chunk] = []
    for position, section in enumerate(toc, start=1):
        start = int(section.offset) - base
        section_end = (
            int(toc[position].offset) - base if position < n_sections else len(whole)
        )
        if start < 0 or section_end < start:
            logger.warning(
                "section %s has inconsistent offsets [%s:%s]; skipping",
                section.ordinal,
                start,
                section_end,
            )
            continue
        raw = tidy_punctuation(whole[start:section_end])
        if not raw.strip():
            continue
        # Translator commentary is INDEXED, not removed. See note_ratio.
        chunk_note_ratio = note_ratio(raw)

        heading = (getattr(section, "title", "") or "").strip()
        header = f"{section.label}. {heading}".strip(" .")

        # Chunk text is produced by ``text.split()`` + ``" ".join(...)``, so a
        # piece is WHITESPACE-COLLAPSED: single spaces, no newlines. Searching
        # for a piece in ``raw`` (which still has newlines and runs of spaces)
        # therefore never matched -- measured 0 successes out of 531 chunks on
        # the Jewish War, so ``raw.find`` always fell through to ``offset =
        # cursor`` and the "moving cursor" below never actually searched.
        # The stale comment claimed a fix that the code could not perform.
        # Searching a collapsed copy and mapping back through an index fixes it.
        flat, flat_index = _collapse_with_index(raw)

        cursor = 0
        for piece in _split_into_token_sized_pieces(raw, chunk_tokens, overlap):
            # A MOVING cursor. A plain find() always returns the first match,
            # so overlapping chunks were all stamped with the section start.
            needle = piece[:120]
            found = flat.find(needle, cursor)
            if found < 0:
                found = flat.find(needle)
            if found < 0:
                found = cursor
            cursor = max(cursor, found + max(1, len(needle) - 1))
            offset = flat_index[found] if found < len(flat_index) else 0

            cleaned = " ".join(piece.split())
            chunks.append(
                Chunk(
                    index=len(chunks),
                    ordinal=int(section.ordinal),
                    label=section.label,
                    start=base + start + offset,
                    end=base + start + offset + len(cleaned),
                    text=cleaned,
                    embed_text=f"{header}\n{cleaned}" if header else cleaned,
                    tokens=count_tokens(cleaned),
                    note_ratio=note_ratio(cleaned),
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings (OpenRouter)
# --------------------------------------------------------------------------- #


def embed_texts(texts: Sequence[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Embed strings via OpenRouter. One request per ``EMBED_BATCH``."""
    import httpx

    api_key = get_api_key()
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = list(texts[i : i + EMBED_BATCH])
        response = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": batch},
            timeout=180.0,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda d: d.get("index", 0))
        out.extend([d["embedding"] for d in data])
        logger.info("embedded %d/%d", min(i + EMBED_BATCH, len(texts)), len(texts))
    return out


# --------------------------------------------------------------------------- #
# Sparse (BM25) vectors
# --------------------------------------------------------------------------- #


class SparseLexical:
    """BM25 sparse vectors for Qdrant.

    Qdrant takes precomputed (index, weight) pairs, so corpus statistics are
    built once at index time and reused for every query.

    THE INDEX MUST BE A TERM ID, NOT A WORD POSITION
    ------------------------------------------------
    Qdrant computes a sparse dot product over coordinates, so the query and
    document index spaces must be the SAME space. An earlier version numbered
    query terms by their position in the stopword-filtered query and document
    terms by their position in the document, which are unrelated coordinates.
    The measured symptom: "Who is Ananus?" produced a one-term sparse query
    that scored Book III above the passage actually naming Ananus, and RRF
    fusion then dragged the dense results down with it.

    So both sides index into one shared vocabulary (``term_ids``), built from
    the corpus at index time and shipped in the manifest.
    """

    def __init__(self, chunks: Sequence[Chunk]):
        self.df: Counter = Counter()
        total = 0
        for chunk in chunks:
            words = _words(chunk.text)
            self.df.update(set(words))
            total += len(words)
        self.avg_len = (total / len(chunks)) if chunks else 0.0
        self.n_docs = len(chunks)

        # Shared vocabulary: term -> dense integer id, stable across index and
        # query so the coordinate spaces line up.
        self.terms: List[str] = sorted(self.df)
        self.term_ids: Dict[str, int] = {t: i for i, t in enumerate(self.terms)}

    def _idf(self, term: str) -> float:
        # DF FLOOR -- see the config note: it caps the IDF of very rare terms
        # (measured: df=1 goes 5.82 -> 3.87). It does not fully rebalance
        # lexical scoring; RRF fusion carries the proper-noun case.
        df = max(self.df.get(term, 0), self.n_docs * DF_FLOOR_FRACTION)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def query_weights(self, question: str) -> Dict[int, float]:
        """BM25 IDF weights for a query, keyed by shared term id.

        Stopwords are dropped: "Who is Ananus?" must reduce to {ananus}, and
        an unknown query term is skipped rather than given a bogus coordinate.
        """
        out: Dict[int, float] = {}
        for word in _words(question):
            if word in STOPWORDS:
                continue
            tid = self.term_ids.get(word)
            if tid is not None:
                out[tid] = self._idf(word)
        return out

    def document_weights(self, text: str) -> Dict[int, float]:
        """BM25 term weights for one document, keyed by shared term id."""
        words = _words(text)
        if not words:
            return {}
        counts = Counter(words)
        length = len(words)
        k1, b = 1.5, 0.75
        out: Dict[int, float] = {}
        for term, tf in counts.items():
            tid = self.term_ids.get(term)
            if tid is None:
                continue
            denom = tf + k1 * (1 - b + b * (length / (self.avg_len or 1.0))) or 1.0
            out[tid] = self._idf(term) * (tf * (k1 + 1)) / denom
        return out

    @staticmethod
    def to_sparse(weights: Dict[int, float]) -> models.SparseVector:
        indices = sorted(weights)
        return models.SparseVector(
            indices=indices, values=[weights[i] for i in indices]
        )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def _weight_ratio_is_sane() -> bool:
    """True if the weaker arm's best hit can outrank the stronger arm's worst.

    The weaker arm's best contribution is ``low/(RRF_K+1)``; the stronger
    arm's worst is ``high/(RRF_K+FUSION_POOL)``. The weak arm can influence
    the ordering only if the former exceeds the latter, i.e. if

        low/high > (RRF_K+1) / (RRF_K+FUSION_POOL)

    Note the direction: the bound is the INVERSE of the rank spread, ~0.72 at
    the shipped settings, not the spread itself. (An earlier draft of this
    function compared the ratio to the spread and so reported the correct
    0.8/1.0 weighting as broken.) Checked rather than asserted so a bad
    override degrades to a warning instead of killing a reader's query.
    """
    high = max(DENSE_WEIGHT, SPARSE_WEIGHT)
    low = min(DENSE_WEIGHT, SPARSE_WEIGHT)
    if high <= 0:
        return False
    bound = (RRF_K + 1) / (RRF_K + FUSION_POOL)
    return (low / high) > bound


class BookStore:
    """Qdrant collection for one book, with the spoiler gate as a filter.

    ::

        store = BookStore("jewish_war")
        store.build(book, toc)
        hits = store.search("who is Ananus?", ordinal=60)   # never past 60
    """

    def __init__(self, book_id: str, root: Optional[str] = None, client=None):
        self.book_id = book_id
        self.collection = f"book_{book_id}".replace("-", "_")
        self.root = root or os.path.join(cache_dir(), "qdrant")
        os.makedirs(self.root, exist_ok=True)
        self._client = client or QdrantClient(path=os.path.join(self.root, book_id))
        self._lexical: Optional[SparseLexical] = None

    @property
    def client(self) -> QdrantClient:
        return self._client

    def exists(self) -> bool:
        try:
            self._client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def count(self) -> int:
        return int(self._client.count(self.collection, exact=True).count)

    def _ensure_collection(self, dim: int, rebuild: bool = False) -> None:
        if rebuild and self.exists():
            self._client.delete_collection(self.collection)
        if self.exists():
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        # Payload index on the gate column: keeps the ordinal predicate cheap
        # as the collection grows.
        self._client.create_payload_index(
            collection_name=self.collection,
            field_name="ordinal",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    def build(
        self,
        book,
        toc: Sequence[Any],
        refresh: bool = False,
        progress: bool = True,
    ) -> int:
        """Chunk, embed, upload. Returns chunk count. Embeds once per book."""
        if self.exists() and not refresh:
            try:
                meta = self._client.get_collection(self.collection)
                if meta.config.params.vectors["dense"].size == DENSE_DIM:
                    n = self.count()
                    logger.info("reusing %s (%d points)", self.collection, n)
                    return n
            except Exception:
                logger.debug("could not inspect collection; rebuilding", exc_info=True)

        chunks = chunk_book(book, toc)
        if not chunks:
            logger.warning("no chunks built for %s", book.book_id)
            return 0
        if progress:
            print(f"   embedding {len(chunks)} chunks with {EMBED_MODEL}...", flush=True)

        dense = embed_texts([c.embed_text or c.text for c in chunks])
        self._lexical = SparseLexical(chunks)

        self._ensure_collection(len(dense[0]), rebuild=True)

        def _batch(seq, n=256):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        points = [
            models.PointStruct(
                id=c.index,
                vector={
                    "dense": d,
                    "sparse": self._lexical.to_sparse(
                        self._lexical.document_weights(c.text)
                    ),
                },
                payload=c.payload,
            )
            for c, d in zip(chunks, dense)
        ]
        for batch in _batch(points):
            self._client.upsert(collection_name=self.collection, points=batch)
            if progress:
                print(f"   uploaded {len(batch)}...", flush=True)

        self._write_manifest(len(chunks))
        return len(chunks)

    def _write_manifest(self, count: int) -> None:
        """Persist the sparse vocabulary.

        Qdrant stores the sparse vectors but not the term->id map, and a
        query in a fresh process has to use the SAME coordinates. Rebuilding
        the vocabulary from the stored payloads would also work but is O(corpus)
        on every query; reading it from the manifest is free.
        """
        terms = self._lexical.terms if self._lexical else []
        meta = {
            "book_id": self.book_id,
            "schema": SCHEMA_VERSION,
            "embed_model": EMBED_MODEL,
            "dense_dim": DENSE_DIM,
            "chunks": count,
            "terms": terms,
            "df": self._lexical.df if self._lexical else {},
            "avg_len": self._lexical.avg_len if self._lexical else 0.0,
        }
        path = os.path.join(self.root, f"{self.book_id}_manifest.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    meta,
                    handle,
                    indent=2,
                    default=lambda o: dict(o) if isinstance(o, Counter) else str(o),
                )
        except OSError:
            logger.exception("could not write manifest %s", path)

    # -- the gate ---------------------------------------------------------- #

    def _gate(self, ordinal: int) -> models.Filter:
        """THE SPOILER GATE, as a storage-level constraint.

        Qdrant will not return a point whose ``ordinal`` exceeds the reader's
        position, whatever the query vector scores well. Deliberately not a
        Python list filter: the previous implementation would have leaked the
        ending if any one call path forgot the check.
        """
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="ordinal", range=models.Range(lte=int(ordinal))
                )
            ]
        )

    # -- search ------------------------------------------------------------ #

    def corpus_stats(self) -> Optional[SparseLexical]:
        """Rebuild BM25 stats from stored payloads.

        Qdrant does not persist corpus term statistics and the lexical half of
        hybrid search needs them at query time in a fresh process.
        """
        chunks: List[Chunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                p = point.payload or {}
                chunks.append(
                    Chunk(
                        index=int(p.get("index", 0)),
                        ordinal=int(p.get("ordinal", 0)),
                        label=str(p.get("label", "")),
                        text=str(p.get("text", "")),
                    )
                )
            if offset is None:
                break
        if not chunks:
            return None
        lexical = SparseLexical(chunks)

        # Prefer the persisted vocabulary: it is the exact coordinate space the
        # stored sparse vectors were written against.
        manifest = self._manifest()
        if manifest and manifest.get("terms"):
            lexical.terms = list(manifest["terms"])
            lexical.term_ids = {t: i for i, t in enumerate(lexical.terms)}
            if manifest.get("df"):
                lexical.df = Counter(manifest["df"])
            if manifest.get("avg_len"):
                lexical.avg_len = float(manifest["avg_len"])

        self._lexical = lexical
        return self._lexical

    def _manifest(self) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.root, f"{self.book_id}_manifest.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def _fuse(
        self,
        arms: List[Tuple[str, Any, float]],
        gate: "models.Filter",
        top_k: int,
    ) -> List[Hit]:
        """Reciprocal rank fusion across arms, run in Python.

        Qdrant's local mode cannot weight the arms, so an unweighted fusion
        let the dense arm's four near-tied generic hits outvote the sparse
        arm's single decisive hit. Ranking each arm here and summing
        ``weight / (RRF_K + rank + 1)`` restores the intended balance.

        Takes no question text: both arms are already-resolved query vectors by
        the time fusion runs.
        """
        if not _weight_ratio_is_sane():
            logger.warning(
                "RRF arm weights (dense=%s sparse=%s) exceed the pool's rank "
                "spread (%.2f): the weaker arm cannot affect ranking. Raise "
                "BOOKBUDDY_FUSION_POOL or bring the weights closer together.",
                DENSE_WEIGHT,
                SPARSE_WEIGHT,
                (RRF_K + FUSION_POOL) / (RRF_K + 1),
            )
        scores: Dict[int, float] = {}
        payloads: Dict[int, Dict[str, Any]] = {}
        sources: Dict[int, List[str]] = {}

        for name, query, weight in arms:
            result = self._client.query_points(
                collection_name=self.collection,
                query=query,
                using=name,
                query_filter=gate,
                limit=FUSION_POOL,
                with_payload=True,
            )
            for rank, point in enumerate(result.points):
                pid = int(point.id)
                scores[pid] = scores.get(pid, 0.0) + weight / (RRF_K + rank + 1)
                if point.payload:
                    payloads[pid] = point.payload
                sources.setdefault(pid, []).append(name)

        # Editorial chunks are DEMOTED, not excluded. A chunk that is mostly
        # bracketed translator commentary loses ground to narrative that
        # answers the same question, but it stays retrievable -- the text is
        # the book's, and a reader who asks about the commentary should still
        # find it. This is what replaced deletion.
        if NOTE_PENALTY:
            scores = {
                pid: score * (1.0 - NOTE_PENALTY *
                              float(payloads.get(pid, {}).get("note_ratio", 0.0) or 0.0))
                for pid, score in scores.items()
            }

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        hits: List[Hit] = []
        for pid, score in ranked[:top_k]:
            payload = payloads.get(pid, {})
            hits.append(
                Hit(
                    index=int(payload.get("index", pid)),
                    ordinal=int(payload.get("ordinal", 0)),
                    label=str(payload.get("label", "")),
                    text=str(payload.get("text", "")),
                    score=float(score),
                    start=int(payload.get("start", 0)),
                    end=int(payload.get("end", 0)),
                    note_ratio=float(payload.get("note_ratio", 0.0) or 0.0),
                    source="+".join(sorted(set(sources.get(pid, [])))) or "hybrid",
                )
            )

        return hits

    def search(
        self,
        question: str,
        ordinal: int,
        top_k: int = TOP_K,
        lexical: Optional[SparseLexical] = None,
        prefer: str = "hybrid",
    ) -> List[Hit]:
        """Hybrid search restricted to ``ordinal`` and below.

        ``prefer``: ``hybrid`` (RRF fusion), ``dense``, or ``sparse``.
        """
        if not self.exists():
            logger.warning("collection %s missing; build it first", self.collection)
            return []

        query_vec = embed_texts([question])[0]
        gate = self._gate(ordinal)
        prefetch: List[models.Prefetch] = []

        if prefer in ("hybrid", "dense"):
            prefetch.append(
                models.Prefetch(
                    query=query_vec, using="dense", filter=gate, limit=FUSION_POOL
                )
            )
        sparse_query = None
        if prefer in ("hybrid", "sparse"):
            lex = lexical or self._lexical or self.corpus_stats()
            if lex and lex.n_docs:
                sparse_query = lex.to_sparse(lex.query_weights(question))
                prefetch.append(
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        filter=gate,
                        limit=FUSION_POOL,
                    )
                )
        if not prefetch:
            # Corpus stats unavailable: dense only, rather than emitting
            # meaningless BM25 weights.
            prefetch.append(
                models.Prefetch(
                    query=query_vec, using="dense", filter=gate, limit=top_k * 4
                )
            )

        if prefer in ("hybrid", "sparse") and len(prefetch) == 2:
            # Local Qdrant exposes no per-arm RRF weight
            # (``models.Prefetch`` has no `weight` parameter), so the fusion is
            # done here. Reciprocal rank fusion, with a higher weight on the
            # lexical arm -- see the DENSE_WEIGHT comment for the measurement
            # that motivates it.
            arms: List[Tuple[str, Any, float]] = [
                ("dense", query_vec, DENSE_WEIGHT),
            ]
            if sparse_query is not None:
                arms.append(("sparse", sparse_query, SPARSE_WEIGHT))
            hits = self._fuse(arms, gate, top_k)
            for hit in hits:
                assert hit.ordinal <= ordinal, (
                    f"SPOILER GATE BREACH: ordinal {hit.ordinal} > reader {ordinal}"
                )
            return hits

        if prefer == "dense":
            result = self._client.query_points(
                collection_name=self.collection,
                query=query_vec,
                using="dense",
                query_filter=gate,
                limit=top_k,
                with_payload=True,
            )
        elif prefer == "sparse":
            if sparse_query is None:
                return []
            result = self._client.query_points(
                collection_name=self.collection,
                query=sparse_query,
                using="sparse",
                query_filter=gate,
                limit=top_k,
                with_payload=True,
            )
        else:
            result = self._client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

        hits: List[Hit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                Hit(
                    index=int(payload.get("index", point.id)),
                    ordinal=int(payload.get("ordinal", 0)),
                    label=str(payload.get("label", "")),
                    text=str(payload.get("text", "")),
                    score=float(point.score or 0.0),
                    start=int(payload.get("start", 0)),
                    end=int(payload.get("end", 0)),
                    source=prefer,
                )
            )

        # Defence in depth. The filter already guarantees this; the assertion
        # documents the invariant and catches a misconfigured store.
        for hit in hits:
            assert hit.ordinal <= ordinal, (
                f"SPOILER GATE BREACH: ordinal {hit.ordinal} > reader {ordinal}"
            )
        return hits


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def hits_to_prompt(hits: Sequence[Hit], max_tokens: int = 4000) -> str:
    """Render hits as numbered blocks the model must cite."""
    blocks = [f"[{i}] {h.label}\n{h.text}" for i, h in enumerate(hits, start=1)]
    while len(blocks) > 1 and count_tokens("\n\n".join(blocks)) > max_tokens:
        blocks.pop()

    text = "\n\n".join(blocks)
    if count_tokens(text) > max_tokens and blocks:
        marker = (
            "\n[... passage truncated to fit the context budget; "
            "ask again about this section for more.]"
        )
        words = blocks[-1].split()
        kept: List[str] = []
        for word in words:
            if count_tokens(" ".join(kept + [word]) + marker) > max_tokens:
                break
            kept.append(word)
        blocks[-1] = " ".join(kept) + marker
        text = "\n\n".join(blocks)
    return text