---
name: book-notes-sync
description: Use when Ben reports a new reading point. Sync notes file.
version: 1.0.0
author: Ben Kogan (benkogan), Hermes Agent
license: MIT
platforms: [linux, macos]
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token with documents + drive scopes
metadata:
  hermes:
    tags: [book-buddy, notes, google-docs, position, mandatory]
    related_skills: [read-book, google-workspace]
    always_load_for: ["read to", new reading position, notes update]
---

# Keep the Google Doc notes file in step with Ben's reading

Ben keeps one Google Doc per book: a running outline of concise key points, so
that when he finishes he can review the whole book from it. When he reports a
new reading point — "I read to BOOK III. CHAPTER 5", "I'm up to the part where
Lucy opens the wardrobe", "finished chapter 12" — do the whole sync yourself
and report it.

Four steps, in this order: resolve the ordinal, pull the newly-read range,
write the bullets, append them and move his position. The bullets must come
from the range text only — never from memory of the book, never from a raw
grep of `data/<book>.txt`.

## When to Use

Triggers: "I read to X", "I'm up to X", "finished chapter N", "add that to the
notes", any report of progress in a book he is reading. Also when he corrects
a note that was just added.

Do NOT use for content questions ("who was X", "what happened at Y") — those
are `read-book`. Do not use for onboarding a new book (`onboard-book`).

## Where things live (repo `~/code/book-buddy`)

- `scripts/find_event.py` — resolves a described event ("up to where X
  happened") to candidate ordinals. Four signals, never fused: label match,
  title overlap, **text overlap** (deterministic, gated), embedding search.
  Evidence only; never writes anything.
- `scripts/read_doc.py` — reads the notes Doc back: paragraphs with their
  `**bold**` marked (`--tail N`), or the whole Doc rendered to PNGs (`--image`).
  Use it to check that an append landed, and to show Ben the doc.
- `scripts/notes_range.py` — prints the newly-read range; the lower gate is the
  notes tail. Writes the full range to `cache/<book>_range.txt`.
- `scripts/gdoc_notes.py` — appends bullets to the doc, backs up first,
  advances the notes tail. `--dry-run`, `--restore <backup.json>`, `--no-state`.
- `cache/<book>_notes.json` — `{last_ordinal, doc_id, doc_url}`. `last_ordinal`
  is the section the DOC currently ends at — read the file, don't trust a
  number quoted in this skill.
- `cache/<book>_position.json` — his reading position, written ONLY by
  `scripts/save_position.py`.
- `cache/gdoc_backups/<book>_<ordinal>_<ts>.json` — pre-append snapshot +
  insert index, input to `--restore`.

## Running the scripts

Every command below carries `env -u PYTHONPATH` in front of `uv run`, and it has
to stay. The shell exports `PYTHONPATH` into Hermes's own venv, so a plain
`uv run` imports `pydantic` (and `PIL`) from the wrong tree and dies with
`ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`. The fix
is the prefix, not another interpreter. (See the `python-project-workflow`
skill.)

## Concision — the file is a digest, not a retelling

Its purpose is that Ben can re-read the whole book from it in one sitting,
months later, and get the key points and gists. Every bullet that survives must
earn its place against that test.

- **Budget ≈ 1 bullet per section, hard ceiling 2.** `gdoc_notes.py` prints the
  ratio (`N bullets for M sections`) — if it is above 2, cut before appending.
  The budget is per-section on purpose: it holds whatever the book's size or
  granularity, so a long history and a short novel both end up as a digest you
  can read in one sitting (a 117-section history → roughly 150 bullets).
- **Bold a short key phrase in every bullet** with `**...**` (a few words to a
  clause — e.g. "**Claudius became emperor**", "**killed 10,000 Jews**").
  `gdoc_notes.py` turns the markers into real bold runs; the existing doc uses
  this on every bullet, so a plain bullet reads as unfinished to Ben.
- One line each, and short: a single sentence of roughly 25–30 words. If a
  bullet needs a semicolon to hold a second event, it is usually two events too
  many — keep the one that changed something. Measured: six long bullets for a
  three-section range came back as "way too many" and too verbose; four short
  ones were right.
- **What counts as a key point depends on the kind of book.** He reads
  anything on Gutenberg, so never assume a war/history register. The test is
  always: *does the state of things change after this?*
  - **plot-driven fiction:** plot turns, revelations, deaths, marriages,
    betrayals, arrivals and departures that matter, decisions, the first
    appearance of a name that recurs. Drop scene-setting, weather, minor
    walk-ons, and anything you would skip when retelling the story.
  - **history / military:** rulers and successions, wars, sieges, treaties,
    the founding of movements, and the numbers that matter.
  - **argument / non-fiction:** the claim, the evidence that carries it, the
    conclusion each section reaches. Drop illustrative anecdotes, repetition
    and rhetorical build-up.
  - **memoir / travel:** the places that matter, the turning points, the
    people who recur. Drop itinerary filler.
- **Delete test:** if a bullet were removed, would the reader lose a fact they
  need in order to follow the rest? If not, it does not belong.
- **No analysis, no adjectives, no "interestingly".** Dense and factual, dates
  first when the text gives one.
- **Headings are arcs, not chapters,** and there should be a handful of them
  across a whole book — not one per reading update. In fiction, the acts or
  phases of the story; in non-fiction, the themes or periods.

When the range is long, resist the pull toward completeness: five bullets
covering the twenty most consequential facts beat twenty bullets covering
everything.

## Procedure

1. **Resolve the ordinal.** Never guess a number, and never take a semantic
   hit as the answer.

   ```bash
   env -u PYTHONPATH uv run python scripts/notes_range.py jewish_war --find "BOOK III. / CHAPTER 5."
   env -u PYTHONPATH uv run python scripts/notes_range.py jewish_war --find "statue should be set up"
   ```

   `--find` matches the composed label (`BOOK III. / CHAPTER 5.`) and the
   section's descriptive title, so event phrasing usually resolves too — it
   works the same for a novel's chapter headings. If two sections match, show
   him both and ask. The `book_id` is the filename in `data/` without `.txt`;
   list them with `ls data/*.txt`.

   **When his words are not in any title** ("I'm up to where they cast lots and
   kill each other"), use the evidence tool instead of guessing:

   ```bash
   env -u PYTHONPATH uv run python scripts/find_event.py jewish_war "the part where they cast lots and kill each other"
   ```

   It prints four signals without fusing them — label match, token overlap
   with the section titles, token overlap with the section TEXT (deterministic,
   gated at position + 25, reported as matched words only), and embedding search
   — plus the section titles in that window for eyeballing. It never prints
   passage text, so nothing in it can spoil.

   How to read it: a title match is usually right, but **not for events**. A
   title says what a chapter is ABOUT, not what happens in it: the chapter
   titled "Concerning The Government Of Claudius" covered his *accession*, while
   his *death* fell in the next chapter — so a title-only `--find "Claudius"`
   hit landed one section early. When his words name an event about a person (a
   death, a birth, a battle), read signal 2b's shortlist and confirm against the
   section text before saving; the two adjacent candidates usually both match.

   A semantic hit is only a shortlist. Measured, an *ungated* embedding search
   for "the governor steals
   the temple treasury and the people riot" put ordinal 100 first with the right
   answer (52) fourth — the +25 gate is what pulls 52 to the top. So confirm with
   him before writing: show the two or three candidate labels and let him pick.
   Ask outright when the candidates disagree, when the best one is more than ten
   sections ahead of his old position, or when two occurrences of the same event
   exist (he means the latest one he had already read).

2. **Pull the range** (deterministic, no model call):

   ```bash
   env -u PYTHONPATH uv run python scripts/notes_range.py jewish_war 63
   ```

   Prints the section list and the range text (up to 20k chars); the full text
   is always in `cache/<book>_range.txt` — read that file in chunks if the
   printed text was truncated. The range is exactly the delta between two
   spoiler gates, so it cannot contain anything past the new ordinal.
   Check that `FROM` equals the doc's current tail; if the script says "nothing
   new to summarize", see Pitfalls.

3. **Write the bullets** to a file, e.g. `$TMPDIR/bullets.md`, in this markup:

   ```
   # Thematic heading
   - a key point with **a bolded key phrase**
     - a sub point
   ```

   Style, matched to the existing doc:
   - `#` headings name arcs of the book, not book/chapter labels. In the Jewish
     War doc that reads "Hasmonean Dynasty and Rise of Herod", "Herodian
     Kingdom", "Roman Province"; a novel wants the phases of its story, a work
     of argument wants its themes. One heading per arc; a new chapter inside the
     same arc needs no heading.
   - One bullet per significant event: one sentence, factual and dense, with
     the names, numbers and dates the text supplies ("In 66 CE, ..." when there
     is a date). Roughly 1–2 bullets per section read.
   - `  - ` sub-bullets only for genuine lists (a set of doctrines, a group of
     siblings, a sequence of named steps).
   - No quotes, no citations, no interpretation, no filler. Skip trivia.
   - Nothing past the new ordinal — no foreshadowing, no "this would later...".

4. **Append and verify:**

   ```bash
   env -u PYTHONPATH uv run --with google-api-python-client --with google-auth \
     python scripts/gdoc_notes.py jewish_war 63 --bullets "$TMPDIR/bullets.md"
   ```

   Run it from the repo root. Bare `uv run` fails — the book-buddy env has no
   Google libraries, hence the `--with` flags. It snapshots the doc, appends,
   re-reads the doc and prints the tail it added, then advances
   `cache/<book>_notes.json`. Read that printed tail — it is the verification
   that the append landed with the right text and formatting. For an
   independent read-back, `env -u PYTHONPATH uv run --with
   google-api-python-client --with google-auth python scripts/read_doc.py
   <book> --tail 12` shows the doc's last paragraphs with bold marked.

5. **Move his reading position** to the same ordinal:

   ```bash
   env -u PYTHONPATH uv run python scripts/save_position.py jewish_war 63 --note "<last bullet, condensed>"
   ```

6. **Report** in a few lines: ordinal + section label, how many bullets were
   added and under what heading, and the doc link. Then stop — no replay of the
   bullets.

## Bullets are additive and reversible

Appending is the whole point of the standing instruction — don't ask permission
for the write, but always report exactly what landed and mention `--restore
cache/gdoc_backups/<file>.json` if he wants it cut. The backup cuts precisely
what that append added, leaving the rest of the doc untouched.

## Pitfalls

- **A title hit is not an event hit.** `notes_range.py --find "Claudius"` and
  the title signal both land on the chapter *titled* for a person, which often
  covers their accession or birth; the event he names can sit one section later.
  Confirm against the section text (signal 2b of `find_event.py`) before saving.
- **Ben may not see an append.** The Doc updates server-side; a browser or
  phone keeps a stale copy for a while. Do not append again — run
  `scripts/read_doc.py <book> --tail 12`: if the text is there, the write landed
  and he needs to refresh. `--image` renders the page to send him, and the
  Drive `modifiedTime` is the timestamp proof.
- **A position is a spoiler gate, so a wrong move is worse than a question.**
  Never advance or rewind it from a retrieval hit alone. `find_event.py`
  proposes; Ben confirms; only then run it.
- **Notes tail vs reading position drift.** `last_ordinal` gates the range and
  is about the DOC; the position file is about HIM. If they disagree, the doc's
  own last lines are the truth: set `last_ordinal` to the section whose content
  the doc ends with, not to the position. A stale `last_ordinal` either re-summarises
  read text or silently skips sections.
- **Long ranges.** He sometimes reads 18 sections at once (200k+ chars). The
  script prints only the first 20k; read `cache/<book>_range.txt` in ~40k
  chunks yourself and keep one heading across chunks when the arc continues.
  Expect roughly 1–2 bullets per section — ~25 bullets for 19 sections.
- **Don't reach past the gate.** If the range looks short or thin, that is the
  answer — never top it up from `data/<book>.txt` or from memory of the book.
- **Content questions are not this skill.** "Who was X", "what happened at Y"
  go through `read-book` / `scripts/ask.py`. This skill only writes notes for
  text he has already finished.
- **`--restore` needs the right backup.** Backups are per-book and stamped;
  a backup from a different document is refused by design.
- **First append for a new book** needs `doc_id`: create the doc and state with
  `scripts/new_notes_doc.py <book_id> --title "<Title> (<Author>)"` (what
  `onboard-book` step 7 does), or pass `--doc <id>` to `gdoc_notes.py`.
