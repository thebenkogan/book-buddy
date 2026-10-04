> **SUPERSEDED -- DO NOT FOLLOW.**
> This document recommends *dropping the vector RAG* and putting the whole book
> in context with prompt caching. That was implemented, measured, and
> **reversed**: full-book context is both expensive and spoiler-prone, because
> the model sees the entire plot and volunteers it from memory. Retrieval is
> Qdrant with a native ordinal range filter. Kept because the cost analysis and
> the leakage measurements are still useful reference; the recommendation is not.

# Architecture recommendation: spoiler-safe "chat with the book I'm reading"

Date: 2026-10-03. All cost figures measured against live OpenRouter pricing
(`https://openrouter.ai/api/v1/models`) and real token counts of
`data/jewish_war.txt` (Whiston translation, 7 books, 115 sections).

## TL;DR

**Drop the vector RAG for answering. Keep the chapterizer, keep the summaries,
add structural progress filtering, and put the read-so-far text in context with
prompt caching.**

Measured: the whole book is **311,761 tokens**. With a cache-read price, a full
book in context costs **$0.0016/query on gpt-5-nano** — the vector index costs
more in engineering time than it saves in dollars.

## 1. Is full-book long context viable? Yes, decisively.

Measured token count (tiktoken `gpt-4o`, Gutenberg header/footer stripped):

| metric | value |
|---|---|
| raw file | 1,320,911 bytes |
| body chars | 1,301,370 |
| body tokens | 311,761 |
| words | 235,905 |
| sections (7 BOOK + 108 CHAPTER) | 115 |
| avg tokens/section | 2,711 |

Full 312K-token context + 1K output, per query (OpenRouter list prices):

| model | ctx | uncached | **cached read** | 500 q/mo uncached | **500 q/mo cached** |
|---|---|---|---|---|---|
| gpt-5-nano | 400K | $0.0156 | **$0.0016** | $7.79 | **$0.78** |
| gpt-5-mini | 400K | $0.0779 | **$0.0078** | $38.97 | **$3.90** |
| gemini-2.5-flash | 1M | $0.0935 | **$0.0094** | $46.76 | **$4.68** |
| claude-haiku-4.5 | 200K ⚠️ | $0.3118 | $0.0312 | $155.88 | $15.59 |
| claude-sonnet-4.5 | 1M | $0.9353 | $0.0935 | $467.64 | $46.76 |

⚠️ claude-haiku-4.5's 200K window **cannot hold this book** at all. Context
size is a hard filter, not a preference — check `context_length` per model.

The blog-post "1250× cheaper" figures circulating online (Elasticsearch Labs,
usewire, bestaiweb) are **not applicable here**. They measured ~1M-token
corpora on Gemini 2.0 Flash and compare against RAG with 8K retrieved. A
novel-sized corpus is 3× smaller than that and the model prices have fallen
~20× since. Reproduce them here and you get single-digit dollars per *thousand*
queries.

## 2. Map-reduce / precomputed summaries: keep them, but for the right reason

You already compute a per-section summary in `summarize.py`. Measured ingest
cost for all 115 sections of this book:

| summary length | model | one-time ingest | per-query (summaries only) |
|---|---|---|---|
| 400 tok | gpt-5-mini | $0.170 | $0.0115 |
| 400 tok | gpt-5-nano | $0.034 | $0.0023 |
| 800 tok | gpt-5-mini | $0.262 | $0.0230 |

**The summary is cheaper than the raw text at query time, but not by much, and
it loses the thing you need it for.** Note the per-query summaries-only cost
*exceeds* cached full-book cost on the same model ($0.0115 vs $0.0078 on
gpt-5-mini) because all 115 summaries cost more than the raw book after
compression ratios are honest.

Use summaries for what they are good at:

- **"Summarize what I read" / "catch me up"** — this is the product's primary
  feature. Summaries 0..progress are the ideal input, and precomputing makes it
  a single cheap call.
- **Routing / context priming** — prepend section summaries 0..progress as a
  table of contents above the raw text. This mitigates lost-in-the-middle: the
  model gets an index of what's ahead of the answer, so mid-window text is
  locatable. Cheap, and it's the single best mitigation available.
- **Foreshadowing questions** ("is X's death hinted at yet?") — summaries
  compress setup beats that raw chunking fragments.

Do **not** use summaries as the sole answer source for:

- **Exact dialogue** — "what did he literally say?" A 400-token summary cannot
  contain a quote.
- **Specific detail/word choice** — summaries hallucinate plausible-sounding
  specifics. Cost of a wrong answer here is high.
- **Numbers/lists** (Whiston's Book VII speeches enumerate the defenders).

So: summaries as *index + recap layer*, raw text as *evidence layer*.

## 3. The no-spoiler constraint: structural, not prompt

**Amazon ships exactly this product and makes the same split.** Kindle's
"Ask this Book" (iOS, US, rolled out late 2025 / expanded 2026):

- "All responses provide immediate, contextual information **up to your current
  reading position**, but you can also choose to ask questions about the
  entire book."
- "Story So Far" generates spoiler-aware recaps *to where you stopped reading*.
- Sources: https://www.aboutamazon.com/news/books-and-authors/kindle-recaps-feature-ebook-series-refreshers
- User-side confirmation of a whole-book vs up-to-position toggle:
  https://killzoneblog.com/2026/01/amazons-latest-rollout.html

The `else` branch exists — after you finish the book, the same feature answers
about the whole thing. That means progress is a **first-class query parameter**,
not a permanent property of the account. Build it that way.

### Prompt-only guardrails are not enough

Your current `query.py` system prompt says *"do not use information found after
the chunks to avoid spoilers"* — pure instruction. This is the weakest tier:

1. **Prompt-only** — instruction not to spoil. Fails on: model's parametric
   knowledge of a public-domain classic (it knows how Jerusalem falls and will
   volunteer it), question phrasings that beg for the ending, and long-context
   dilution. Amazon's own docs concede it's "not mathematically incapable of
   making a mistake."
2. **Post-hoc output filtering** — generate, then screen for leakage. Expensive,
   brittle, and you show the reader the text before you can retract it.
3. **Structural / filtering** — unread chapters never enter the prompt. The
   right tier, and what you already designed with `chapter_index <= progress`.

**Tier 3 is necessary but not sufficient, because of parametric knowledge.**
Public-domain Josephus is in every model's training data; you cannot structurally
filter what the model *knows*. So you need a fourth layer:

4. **Leakage output validator** — one cheap extra call (or a rubric check) on
   the drafted answer asking: "does this assert or presuppose any event from
   sections > progress?" On reject, regenerate with a stricter instruction or
   return the "you've reached the end of what I can tell you" nudge. On gpt-5-nano
   this is a sub-cent call against a $0.0016 base.

Also note the **model's knowledge leak cuts both ways and you cannot fix it
structurally**: when a user asks "why did Titus destroy the Temple?", the model
answers from weights. Your only defense is the generation instruction + validator.

### Enforce the filter at the data layer

The architectural trap: filtering happens in the *retrieval* function, so any
future code path that assembles context can bypass it. Filter at the **boundary
where raw text enters a prompt**:

```python
class Book:
    def text_up_to(self, progress: int) -> str:
        """The ONLY sanctioned way to read book text into a prompt.
        Structural guarantee: sections > progress cannot be returned.
        """
        if progress >= len(self.sections):
            raise ValueError("progress past end; use entire_book_text()")
        return "".join(s.text for s in self.sections[: progress + 1])
```

Then `grep -r "book.text\|\.text\[" src/` should return **only** hits inside
that function. That's a checkable invariant; a `if chapter_index <= progress`
inside a cosine-similarity loop is not.

## 4. The "I read up to HERE" resolver

This is the sharpest technical problem in the app, and this book is a
deliberately nasty case. Measured on `jewish_war.txt`:

- **115 sections but only 40 distinct labels.** 22 of those 40 labels are
  ambiguous.
- `CHAPTER 1.` occurs **7 times** — at global section indices #2, #36, #59, #70,
  #81, #95, #106 (Books I–VII).
- The Gutenberg text also contains a **108-line table of contents** made of
  identical-looking `CHAPTER n.` / `BOOK n.` strings. A naive regex or a
  `fuzz.ratio` matcher (as in your `chapterize.py`) will happily index the TOC
  itself, creating phantom chapters at low byte offsets.

Your existing `chapterize.py` takes `max(candidates, key=lambda c:
fuzz.ratio(...))` — the *global best* fuzzy match — which on this text will
collapse most chapters onto the TOC occurrences. **This is a real bug in the
current code, not a hypothetical.**

### Robust resolver design

**Layer 0 — fix the index (prerequisite).**
- Restrict heading candidates to the *body region* (exclude front matter/TOC).
  Detect the TOC structurally: a contiguous run of heading-like lines with
  monotonically increasing small numbers and no prose between them.
- Assign every section a **global ordinal** `n` (1..115) in addition to its
  label. `CHAPTER 1.` #2 and `CHAPTER 1.` #59 are different objects. Identity is
  `(label, parent_book, global_ordinal)`, never `label`.
- Keep `parent_book` from your existing `context` dict — `BOOK III / CHAPTER 1`
  is unambiguous even though `CHAPTER 1` alone is not.
- Store a monotonic `progress` in Mongo as an **ordinal**, and derive a display
  label from it. Never parse a label back into progress.

**Layer 1 — deterministic normalization, before any LLM.**
Parse user text with regex: roman/arabic numerals, ordinals ("the third
chapter"), fractions ("about two thirds of the way in"), relative position
("just after the siege begins"), quotes of in-book text.
Normalize both sides: lowercase, strip punctuation/whitespace, expand roman
numerals, drop leading "the".

**Layer 2 — candidate retrieval (cheap, exhaustive).**
Score *every* section against the user's phrase, don't just take the argmax:

| signal | weight |
|---|---|
| exact normalized label match | 100 |
| `rapidfuzz.token_set_ratio` on label | 0–90 |
| parent-book agreement ("Book III, chapter 1") | +50 |
| global-ordinal match ("chapter 35 overall") | +40 |
| BM25/embedding over the chapter's own opening lines (catches "the part where Vespasian arrives") | 0–60 |

**Layer 3 — LLM disambiguation over the top-K candidates only.**
Send the user's phrase plus ~5–8 candidate sections (label, parent, ordinal,
first 200 chars) and ask for a structured pick. **Constrained decoding, not free
generation** — force the model to emit an index into the candidate array, so it
*cannot* invent a chapter:

```python
# use OpenRouter json_schema / structured outputs
{
  "type": "object",
  "required": ["choice", "confidence"],
  "properties": {
    "choice": {"type": "integer", "minimum": 0, "maximum": len(candidates)-1},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "quote_matched": {"type": "string",
      "description": "exact text from the user that identified the section"}
  },
  "additionalProperties": False
}
```

`quote_matched` forces the model to show its evidence, which makes the
disambiguation auditable and debuggable. Pattern reference:
https://github.com/google/langextract (source-grounded extraction, parallel
passes for long docs).

**Layer 4 — explicit disambiguation, do not guess.**
This is the step most implementations skip and it's what makes the feature
trustworthy. If top-2 scores are within a small margin, **ask**:

> You said "Chapter 1" — this book has seven Books, each with its own Chapter 1.
> Did you mean Book II Chapter 1, or Book VI Chapter 1?

Never silently pick. A silent wrong pick means either (a) the user is told they
read things they didn't, or (b) they're shown content from a much earlier
section — a spoiler bug manufactured by the resolver.

**Layer 5 — clamp and round *down* for safety.**
`progress = min(resolved, stored_progress)`. Users reread; never let a fuzzy
match move progress *forward* past a confirmed value. When the two disagree,
prefer the lower. Rounding up is the only way this feature can spoil.

**Layer 6 — progress must be monotonic and user-confirmed.**
Reading position is sticky. Persist ordinals, expose "I'm at section 43 of 115"
in the UI, and let the user correct it. The resolver should *suggest* a position,
not own it.

## 5. Recommended build

```
PUT progress        → ordinal int (sticky, monotonic, user-confirmed)
POST ask            → question
  1. if "I read up to <phrase>" in question → resolver (§4), update progress
  2. context = summaries[0..p]  +  raw_text_up_to(p)     ← structural filter
  3. generate with cache breakpoint right after raw_text_up_to(p)
  4. leakage validator (§3 tier 4); on reject, regenerate or return a nudge
```

**Delete:** `embedding.py` chunk+embed path, `qwen/qwen3-embedding-8b` call,
numpy cosine similarity, the embedding JSON cache files. Mongo still needed for
progress.

**Keep:** `book.py` (extend `Chapter` with `ordinal`, `parent`), `chapterize.py`
(fix the TOC bug), `summarize.py` (promote to a first-class recap layer),
`checkpoint.py` (the `@checkpoint` decorator is genuinely good — it made this
analysis cheap).

**Immediate bug to fix regardless of architecture:** `chapterize.py`'s
`max(candidates, key=fuzz.ratio)` will mis-map most chapters of this book. Run it
and print the resulting `chapter.start` offsets before trusting anything.

**Verify before believing any of this:** on 10–20 real questions of your own,
compare answers from (a) full read-so-far text vs (b) current RAG top-3 chunks.
The literature says LC > RAG for single-document QA
(https://arxiv.org/abs/2501.01880 — "LC generally outperforms RAG in
question-answering benchmarks"; RAG retains an edge on dialogue-based queries,
which is the one category that matters here). Build the eval before you delete
anything.

## Caveats I did not resolve

- **No live API key in this environment**, so all costs are list-price
  arithmetic, not measured invoices. Confirm with 3 real calls.
- **No head-to-head quality eval was run.** The recommendation rests on measured
  token counts, live prices, and the cited literature — not on a benchmark of
  your own queries.
- Cost tables above assume the *whole book* is cached; if you only cache up to
  progress, cache hits stop when the user advances past them.
- `google/gemini-2.5-flash` cache pricing on OpenRouter is not the documented
  Gemini explicit-cache price; verify before relying on the cached column.

## Sources

- Chroma, *Context Rot: How Increasing Input Tokens Impacts LLM Performance*
  (18 models; degradation with length even on trivial tasks; coherent text
  performs *worse* than shuffled) — https://research.trychroma.com/context-rot
- Li et al., *Long Context vs. RAG for LLMs* — https://arxiv.org/abs/2501.01880
- Liu et al., *Lost in the Middle* (TACL 2024) — https://arxiv.org/abs/2307.03172
- Modarressi et al., *NoLiMa* — https://arxiv.org/abs/2502.05167
- Hsieh et al., *RULER: What's the Real Context Size* (effective context is a
  fraction of advertised; budget ~25–65%) — https://arxiv.org/abs/2404.06654
- Amazon, Kindle Story So Far / Ask this Book —
  https://www.aboutamazon.com/news/books-and-authors/kindle-recaps-feature-ebook-series-refreshers
- Kindle toggle (whole book vs up-to-position), first-hand:
  https://killzoneblog.com/2026/01/amazons-latest-rollout.html
- Google, LangExtract (source-grounded structured extraction, parallel passes)
  — https://github.com/google/langextract
- OpenRouter live model/pricing data — https://openrouter.ai/api/v1/models