---
name: read-book
description: Answer questions about a book at the reader's position.
version: 1.0.0
author: Ben Kogan (benkogan), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [book-buddy, rag, reading, spoiler, citation]
    related_skills: [onboard-book, probe-leakage]
---

# Read a book with book-buddy

The reader asks questions about a book they are partway through. Retrieve the
passages they have already read, answer from those alone, cite them.

**You are the model.** Retrieval decides what you may consult; you do the
answering. Never call OpenRouter to summarise passages you already retrieved.

## When to Use

- The reader asks a question about a book they are reading.
- They move their reading position ("I've read to where...").
- They report a wrong or unhelpful answer.

Don't use for: adding a new book (`onboard-book`), measuring leakage
(`probe-leakage`).

## Prerequisites

- Repo: `~/code/book-buddy`.
- `OPENROUTER_API_KEY` — **embeddings only**. Export before retrieving:
  `set -a && . ~/.hermes/.env && set +a`

## How to Run

```bash
cd ~/code/book-buddy && set -a && . ~/.hermes/.env && set +a
uv run python scripts/ask.py "<their question>"
```

Prints the saved position, the numbered passages, and the answering rules.
Make NO further model call. Answer from what it printed.

Optional second arg for a different book: `scripts/ask.py "q" les_mis`.

## Quick Reference

```bash
uv run python scripts/ask.py "<question>" [book_id]   # retrieve, no model call
uv run python scripts/save_position.py <book> <ordinal> --note "..."
uv run python scripts/onboard_book.py data/<file>.txt # new book
uv run python scripts/probe_leakage.py                # leak measurement
uv run pytest -q                                      # 167 tests, offline
```

## Procedure

1. **Run `scripts/ask.py` with their question.** One retrieval, zero model calls.
   *Done: numbered passages printed, all at or before their position.*

2. **Answer in 2–4 sentences.** Direct. No preamble, no restating the question.
   *Done: the answer is shorter than the passages you were given.*

3. **Do not quote unless asked.** Paraphrase. Verbatim quotation is for when
   they say "quote" or "show me the passage". Exception: a short phrase
   (under ~15 words) when the book's own wording is the clearest way to say it.
   *Done: no long block quotes in a normal answer.*

4. **Mark every claim inline with a bare `[n]`.** One marker per sentence or
   clause that asserts something. The number is the passage it came from.
   This is the part that makes the answer checkable: with sources only at the
   bottom, two paragraphs citing `[1][2]` tell the reader nothing about which
   claim rests on which passage.
   *Done: no factual sentence lacks a marker; each marker maps to one passage.*

5. **Then list sources at the bottom** under a `Sources:` heading, one line per
   cited passage: the section label, the number, and a short note on what that
   passage supplied. No commentary on retrieval quality unless something went
   wrong.
   *Done: every inline marker appears in the list below it.*

6. **If the passages don't answer it, say so** — plainly, in one line. Name the
   section that would. Do not fill the gap from memory. This is the single most
   important rule below.
   *Done: no claim lacks a citation; gaps are stated, not papered over.*

## Answer Format

```
<The answer, 2-4 sentences. Each factual sentence ends in [n].>

Sources:
- <Section label> [n] — <what this passage supplied>
- <Section label> [n] — <what this passage supplied>
```

Worked example (question: *what did Pilate do*):

> Pilate twice raised a disturbance in Jerusalem. He first brought the
> imperial ensigns into the city by night, which the Jewish law forbade, and
> when the people lay prostrate for five days rather than comply, surrounded
> them with troops in three ranks; they offered their necks bare instead, and
> Josephus says he was greatly surprised at what he called their prodigious
> superstition, so he withdrew the ensigns [1]. He then diverted the sacred
> Corban treasury to build an aqueduct over four hundred furlongs [2]. Fearing
> the uproar, he hid armed soldiers among the crowd disguised as civilians, with
> staves rather than swords, and gave the signal; many died of the beating and
> many were trampled to death in the panic, and the rest were so astonished they
> fell silent [2].
>
> Sources:
> - BOOK II. / CHAPTER 9. [1] — the ensigns, the prostration, the withdrawal
> - BOOK II. / CHAPTER 9. [2] — the Corban aqueduct, the ambush, the deaths

Note what that buys: a reader can check each claim against one passage. The
same answer with sources only at the bottom would be unverifiable.

## Pitfalls

- **Do not use outside knowledge of the book.** The model knows how these
  books end. Never mention it. Measured: given the full read portion it
  volunteered "Titus destroyed Jerusalem's Temple in 70 CE" and then appended
  "but not in this excerpt" — the disclaimer came after the spoiler.
- **A wrong family relationship is the common failure.** Composition error, not
  retrieval: the passages were right and the answer spliced a sibling into a
  parent slot. Keep relations exactly as stated; if the text doesn't state one,
  say it doesn't.
- **Retrieval is weakest deep inside a chapter** and on questions with one
  rare name. Top scores are near-tied, so *which* chunk answers can be luck.
  If an answer looks wrong, re-run the retrieval and read the passages before
  blaming yourself — or say which it was.
- **Never re-run retrieval hoping for a luckier draw** without telling the
  reader retrieval was the problem. Silent retries hide a real defect.
- **Two Agrippas exist** in the Jewish War: Agrippa I (son of Aristobulus and
  Bernice, grandson of Herod the Great) and the city Agrippias named for the
  elder Agrippa. Retrieval returns both.
- **Ordinals shift when structure sections are dropped.** After editing a
  structure artifact, re-anchor the saved position by OFFSET, not ordinal —
  a stale ordinal silently points at different text.
- **No spoilers.** If asked about something past their position, say it hasn't
  happened yet in their reading and point at the next section.

## Verification

The answer cites at least one numbered passage, every passage cited is at or
before the saved ordinal, and the answer is materially shorter than the
retrieved text. If they correct you, work out whether retrieval or reasoning
caused it before changing anything — they differ, and the fix differs.