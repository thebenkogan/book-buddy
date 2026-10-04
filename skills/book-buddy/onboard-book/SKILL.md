---
name: onboard-book
description: Onboard a book: build its TOC, index it, prove the gate.
version: 2.0.0
author: Ben Kogan (benkogan), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [book-buddy, rag, spoiler, onboarding, gutenberg]
    related_skills: [read-book, probe-leakage]
---

# Onboard a book into book-buddy

Build a chapter list (TOC), index it into Qdrant, and prove the spoiler gate.

**The scripts make no model calls except to embed.** Measured, not assumed.

`find_headings()` + `derive_title()` build a TOC from the text. On 24 Gutenberg
books never used to design the regex, **17 produce a TOC that passes
verification and audit with no model at all; 7 do not.** So a model is the
fallback, not the default -- and `toc_needs_model()` decides which is which by
rule, so nobody has to notice.

You still own every judgement call — which is what the remaining steps are
for. But measurements beat architecture taste, and the measurement says most
books never need a model at all. When the audit does object, that is your cue
to step in, not a step you always take.

## When to Use

- Adding a book, or replacing a wrong TOC.
- The audit reports problems you want to fix by hand.
- `SCHEMA_VERSION` changed, or retrieval looks wrong.

Don't use for: answering questions (`read-book`), measuring leakage
(`probe-leakage`).

## Prerequisites

```bash
cd ~/code/book-buddy && export PATH="$HOME/.local/bin:$PATH"
set -a && . ~/.hermes/.env && set +a      # embeddings need the key
```

## Quick Reference

```bash
# 1. deterministic TOC (no model call) -- the normal path
uv run python scripts/onboard_book.py data/<f>.txt --restructure --skip-answer

# 2. only if the audit objects: subagent adjudication
uv run python scripts/toc_candidates.py data/<f>.txt --slice 1 --of 8
uv run python scripts/toc_apply.py data/<f>.txt --verdicts verdicts.json

# 3. index and prove the gate
uv run python scripts/onboard_book.py data/<f>.txt --force
uv run python scripts/save_position.py <book> <ordinal> --note "..."
uv run pytest -q
```

## Procedure

### 1. Build the TOC deterministically -- no model call

```bash
uv run python scripts/onboard_book.py data/<f>.txt --restructure --skip-answer
```

Shapes `find_headings()` recognises, each read off a real file:

| shape | example | seen in |
|---|---|---|
| keyword + number, alone on its line | `CHAPTER 12.` | most books |
| mixed case + closing bracket | `Chapter I.]` | Pride and Prejudice |
| no trailing period | `BOOK ONE: 1805`, `CHAPTER I` | War and Peace |
| title on the same line | `Chapter I. Into the Primitive` | Call of the Wild |
| other keywords | `Canto 1`, `PART ONE`, `ACT III` | Divine Comedy |
| bare roman, no keyword | `XII` | Treasure Island |

The bare-roman branch runs **only** when the keyword convention found almost
nothing. Unconditionally it takes Dracula from 27 correct sections to 54 with
26 spurious.

`derive_title()` reads a title from the lines after a heading when they are
title-shaped. On the Jewish War that recovers 116/117 titles for free; where
chapters are untitled it returns nothing rather than inventing one from the
opening sentence of body prose.

*Done: TOC written, verification clean.*

### 2. Read the router

```
ROUTER: deterministic is good -- 117 sections at 7.39 per 100k chars, audit clean
ROUTER: NEEDS A MODEL -- no headings matched at all -- either an unrecognised
        convention or a book with no chapters
```

Rule: nothing found, **or** the audit objected, **or** section density below
0.5 per 100k characters -- a floor every genuinely chaptered book clears. On
the 24-book corpus it catches **7/7** of the books the deterministic path fails
on, with **0 false alarms** on the 17 that pass.

The two signals are both required. Either alone false-alarms: a book with no
chapters can still pass a density check if the regex hallucinates a heading or
two (Metamorphosis returns 2, and the router calls that fine -- read a chapter
if a very long book's TOC looks suspiciously short).

**The 17/24 ratio flatters the regex.** It is 24 books and the pattern was
started on the Jewish War. Re-run the benchmark before trusting it on a new
corpus.

### 3. If the router says MODEL: delegate adjudication

```bash
uv run python scripts/toc_candidates.py data/<f>.txt --slice 1 --of 8
```

That also writes the whole pool to `cache/<book_id>_candidates.json` -- a
JSON list of `{index, offset, line, prior, preview}`. **Point each subagent at
that file and give it its index range.** Do not paste the candidates into the
prompt: on Grimms a 165-candidate pool is ~35k characters of preview that you
pay for twice (once reading, once writing) and that goes stale the moment the
pool changes. A subagent that reads the file itself gets current data and
costs you nothing.

*Done: slice count chosen, candidates file written.*

### 4. Delegate every slice, in one `delegate_task` call

One `tasks` entry per slice, 40-60 candidates each. Each subagent's `context`
MUST contain the verbatim candidate text from step 3. A subagent given only
"decide on candidates 1-46" has nothing to read and will (correctly) return an
empty array rather than invent data -- I made that mistake and it cost a round.

Put this AFTER the candidate block in each task's `context`:

```
Decide which candidate lines are REAL chapter/volume headings in this book.
A heading is a real section start. Reject:
 - wrapped prose that happens to be short
 - the translator's own footnote blocks ("N (return) [ ... ]", "WAR BOOK n FOOTNOTES")
 - table-of-contents entries (heading-shaped lines with no prose between them)
 - a line that is the CONTINUATION of a heading that wrapped onto two lines
Accept a real heading and give it a 3-12 word title. Prefer the book's own
words for the title.

Reply with ONLY a JSON array, nothing else:
[{"index": 4, "title": "Book I: One Hundred and Sixty-Seven Years"}, ...]
Include only candidates you ACCEPT. Never invent an index.
```

Slice each subagent's **own** candidates only. They cannot see each other's
work, which is the point -- no giant list, no coordination, no one agent
holding 363 decisions.

*Done: N subagents dispatched, each context containing its own candidates, each
returning a JSON array.*

### 5. Index and prove the gate

```bash
uv run python scripts/onboard_book.py data/<f>.txt --force
```

`--force` matters. Without it `build()` reuses an existing collection whose
dimension matches, and leaves a **stale ordinal layout** behind a changed TOC
while reporting success. This actually happened.

Confirm: chunk count > 0, gate holds at ordinals 0/1/mid/max, and
"ordinal=0 -> 0 hits".

*Done: index built, gate proven at four positions.*

### 6. Re-anchor any saved position

If a TOC already existed, ordinals shifted. Re-anchor **by offset**, never by
ordinal — a stale ordinal silently points at different text. Read
`cache/<book>_position.json`, resolve the old offset against the new TOC, and
rewrite it. Assert the resolved section is the one you expect before saving.

*Done: position points at the intended passage.*

## Pitfalls

- **Real headings are ALL-CAPS; real prose is mixed case.** In these books
  that single signal catches most false positives. An audit ignoring case
  flags real chapters as prose, and the auto-drop then breaks parents.
- **The model supplies TITLES only.** Labels, parents and offsets come from
  the deterministic scan. Letting a model supply `parent` once mislabelled 72
  sections while every offset stayed valid.
- **Merging must be the identity.** Merging the deterministic list with
  itself must give the same list. A first-match-within-tolerance bug silently
  deleted seven chapters. There is a test for it.
- **Wrapped headings are the classic false positive.** A chapter title that
  wraps onto two lines presents its second line as a short standalone line.
  Reject those — the run above rejected exactly that.
- **Translator footnote blocks are not chapters.** They can be 79% of the raw
  file. Reject `N (return) [` and `WAR BOOK n FOOTNOTES`.
- **Every audit check traces to an observed failure.** If you add one, name
  the book and say how it broke.

## Verification

`uv run pytest -q` → 197 passed. The gate holds at ordinal 0 (nothing), 1
(only chapter 1), mid, and max. `scripts/ask.py "<question>"` returns passages
numbered and all at or before the saved position.