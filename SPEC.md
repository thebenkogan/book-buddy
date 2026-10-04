> **SUPERSEDED (2026-10-04).** This spec is for the FastAPI service in
> `src/api/`, which was deleted along with the React client and Docker setup.
> The shipped product is a CLI and library. See `AGENTS.md` and
> `skills/book-buddy/onboard-book/SKILL.md` for what exists.

# SPEC: book-buddy v2 — spoiler-safe reading companion (Hermes-only)

## Non-negotiable constraints (from Ben)

1. **Free models only.** Zero spend. Use `stealth/space-bunny-alpha` (1M ctx, $0) as
   default. Allow an env override `BOOKBUDDY_MODEL`.
2. **Must handle Les Miserables (799,947 tokens) and Monte Cristo at maximum.**
   A 1M context window technically fits Les Mis, but leaves almost no headroom, so
   a map-reduce fallback is REQUIRED, not optional.
3. **Hermes-only.** The React client is out of scope. Do not modify `src/client/`.
   A Hermes skill makes the API reachable by text message.
4. **No spoilers, ever.** A structural guarantee is required; a prompt alone is
   not acceptable.
5. **Free/OpenAI-compatible inference via OpenRouter.** Key already in
   `~/.hermes/.env` as `OPENROUTER_API_KEY`. Do NOT hardcode it. Do NOT print it.

## Environment

- Repo: `/home/benkogan/code/book-buddy` (already cloned; branch `main`)
- Python venv already exists at `.venv` (has tiktoken, rapidfuzz). Use `uv` or the
  venv; do not create a second environment.
- Python 3.10+. Existing deps in `pyproject.toml` include fastapi, httpx,
  motor, openrouter, pydantic, rapidfuzz, tiktoken.
- AGENTS.md in repo root defines code style: Black, 88 cols, absolute imports
  from `src`, Pydantic models, snake_case, `logging.exception()` before re-raise.
- Tests live in `tests/`, run with `pytest`.

## Architecture (decided — implement this, don't re-litigate)

The existing `src/index/embedding.py` (semchunk + OpenRouter embeddings + numpy
cosine) is DELETED. Vector RAG is not needed: the read-portion of a book fits in
a free 1M-context model, verified empirically.

Three modules:

- `src/bookbuddy/book.py` — book loading + TOC + the progress gate
- `src/bookbuddy/ask.py` — long-context answering + map-reduce fallback
- `src/bookbuddy/progress.py` — natural-language "I'm at here" → ordinal

### `book.py`

- `load_book(path) -> Book`: strip Gutenberg header/footer via regex on
  `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` / `*** END ... ***`.
  **Critical bug to avoid:** a naive `text.find("BOOK IV.")` matches the TABLE OF
  CONTENTS. Only accept structural headings that stand alone on their own line
  (`(?m)^BOOK\s+([IVXLC]+)\.\s*$`). Add a regression test for this exact bug.
- `build_toc(book, client) -> list[Section]`: one LLM call per book, cached to
  `cache/{book_id}_toc.json`. A `Section` is
  `{ordinal:int, label:str, parent:str, offset:int}`.
  - Ordinals are 1..N over the FLATTENED list. Progress is an ordinal, never a
    chapter number.
  - Why: in the Jewish War, `CHAPTER 1.` occurs 7 times and 22 of33 bare chapter
    numbers appear in more than one book. "chapter 3" is ambiguous.
- `verify_toc(toc, text) -> list[str]`: **REQUIRED.** Return a list of problems.
  Check every offset lands on/just before a plausible heading, that ordinals are
  contiguous and ascending, and that no offset falls inside the TOC region
  (detect the TOC region as text before the first structural heading). A wrong
  offset does not raise — it silently produces confident wrong answers, which is
  the worst failure mode this tool has. Refuse to answer if verification fails.
- `text_up_to(book, ordinal) -> str`: **THE ONLY function permitted to read book
  text into a prompt.** This is the spoiler guarantee. Nothing else may slice
  `book.text`. Add a test asserting no other module slices text directly.

### `ask.py`

Two strategies, selected automatically:

- `long_context(book, ordinal, question)`: `text_up_to(...)` in one call.
  Used when the read-portion fits the model's context window with headroom
  (reserve >= 15% or >=30k tokens for output).
- `map_reduce(book, ordinal, question)`: per-chapter summaries of the read
  portion (cached), then answer from summaries. REQUIRED for Les Mis-class books
  where the read-portion would not fit. Summaries are per-chapter, computed once,
  cached to `cache/{book_id}_summaries_{ordinal}.json`.
- `answer(book, ordinal, question)`: picks the strategy and returns
  `{answer, strategy, tokens_in, tokens_out}`.

**Leakage validator — REQUIRED, fourth tier.** A structural filter cannot stop
the model's parametric knowledge (it knows Jerusalem falls). After producing an
answer, make one cheap extra call asking a model to flag any statement in the
answer that could only be known from beyond `ordinal`. Return
`{answer, strategy, leakage_flagged: bool}`. Prompt-only guardrails are tier 1
and are known insufficient; this plus the structural gate is the real defense.

### `progress.py`

- `resolve_position(toc, user_text) -> {ordinal:int, confidence:float, ambiguous:bool}`
  Resolves "I read up to chapter 3" / "just finished book 2" / "I'm at the part
  where X happens" against the TOC.
  - **Never silently guess when ambiguous.** If top-2 candidates score close,
    return `ambiguous=True` and ask the user.
  - Use constrained selection: show the model the numbered TOC and require it
    return an INTEGER INDEX into that array, so it cannot invent a chapter.
  - Also return an evidence field with a supporting quote.
- Progress must only move FORWARD, and should clamp DOWN if the user asks to
  rewind without confirmation.

### API (`src/api/`)

Keep the existing FastAPI app shape (`src/api/main.py`, `routes/`). Add/adjust:
- `POST /api/v1/books/{book_id}/progress` — set reading position (ordinal)
- `GET  /api/v1/books/{book_id}/toc` — returns the TOC (so the Hermes skill can
  show the user what is available)
- `POST /api/v1/books/{book_id}/ask` — body `{question}` → spoiler-free answer
- `GET  /api/v1/books/{book_id}/summary` — summary of everything read so far
  (this fixes an existing bug at `routes/reading.py:74` where the whole-book
  summary returns `book.chapters[0].summary`)
- Health endpoint already exists; keep it.

Mongo is optional per existing code — keep it that way, but reading progress
must persist even without Mongo (fall back to a local JSON file).

### Hermes skill

A skill so Ben can text Hermes and reach this API. It must document the
conversational forms:
- "I read up to chapter N in <book>" → resolve + set progress (confirm if ambiguous)
- "what did I read in <book>" → summary of read portion
- "<any question> about <book>" → spoiler-free answer

## Definition of done — VERIFY WITH THE JEWISH WAR

The proof artifact is `data/jewish_war.txt` (already downloaded, 316k tokens,
108 chapters / 7 books, Whiston translation, Gutenberg #2850).

A test/demo script must demonstrate ALL of these with real OpenRouter calls
against a free model:
1. TOC builds and passes `verify_toc` with zero problems.
2. In-scope question answers correctly from the read-portion.
3. **Spoiler probes return no spoilers**: asking "what happens at Masada" and
   "does Titus destroy the Temple" while positioned mid-book must NOT reveal
   those outcomes. Assert on the response text.
4. Boundary probe: "what is the very last event in the text I gave you" returns
   an event from the final READ chapter, not beyond.
5. Les Miserables (or another ~800k token book) takes the map_reduce path and
   still answers correctly with no spoilers.

Report actual measured token counts and real API output. Do NOT fabricate
results. If something fails, say so and leave it failing — a truthful failure is
worth more than a fabricated pass.

## Out of scope

- React client changes
- RAGFlow, Qdrant, or any vector DB (explicitly rejected)
- Paid models
- Multiple users