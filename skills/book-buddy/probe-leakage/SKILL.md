---
name: probe-leakage
description: Measure spoiler leakage before trusting a model.
version: 0.1.0
author: Ben Kogan (benkogan), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [book-buddy, rag, spoiler, evaluation]
    related_skills: [onboard-book]
---

> **STALE — the probe measures a pipeline that no longer exists.**
> `scripts/probe_leakage.py` tests the OLD full-context design, where the whole
> read portion was sent to the model. Retrieval is now Qdrant with a native
> `ordinal <= reader_ordinal` filter, so that leak cannot happen: the model
> never sees past the reader's position. Do not treat its numbers as today's
> spoiler-safety measurement. Rewriting it against `BookStore.search` is the
> obvious next piece of work. Everything below about ordinals and offsets is
> still accurate.


# Probe leakage from model memory

Measures whether the model volunteers plot it already knows, which no amount
of prompt engineering removes. Run after onboarding, and again whenever the
answering path or model changes.

Does NOT prove safety. It measures a leak rate on chosen probes at a chosen
position; absence of a measured leak is not a guarantee.

## When to Use

- After changing `ANSWER_SYSTEM`, the model, or retrieval top_k.
- Before trusting `answer_question` for real spoiler-sensitive use.
- Comparing generate vs extractive tiers.
- Don't use for: onboarding (see `onboard-book`), or unit testing.

## Prerequisites

- `uv` on PATH, `OPENROUTER_API_KEY` exported (`set -a && . ~/.hermes/.env && set +a`).
- A book already onboarded — `cache/{book_id}_chunks.json` must exist.

## How to Run

```bash
cd <repo-root> && set -a && . ~/.hermes/.env && set +a
uv run python scripts/probe_leakage.py
```

15 probe calls + 5 critique calls, ~760k tokens in. Re-score cached
responses without spending tokens:

```bash
uv run python scripts/probe_leakage.py --rescore cache/leakage_probe.json
```

## Quick Reference

```bash
uv run python scripts/probe_leakage.py                      # full run
BOOKBUDDY_PROBE_TEMP=0.9 uv run python scripts/probe_leakage.py  # stability
BOOKBUDDY_PROBE_ORDINAL=20 uv run python scripts/probe_leakage.py # earlier pos
uv run python scripts/probe_leakage.py --rescore <cache.json>
```

## Procedure

1. **Run three modes per probe.** A: with the read portion. B: with NO text
   at all (the control — if B leaks, no prompt can save us). C: text plus an
   explicit "don't go past here" instruction.
   *Done: 15 probe calls complete with zero errors.*

2. **Read the marker evidence table before believing any verdict.** It prints
   each marker string's occurrence count in the read portion and the withheld
   tail. A marker with equal counts cannot discriminate and is reported as
   such.
   *Done: you have confirmed the load-bearing markers are tail-only.*

3. **Check the flag key.** `!!` = reveals a tail-only fact (a disclaimer does
   not unsay it). `!` = a tail-only proper noun outside a refusal. `??` =
   absent from the book entirely, i.e. prior knowledge like "70 CE".
   *Done: you understand why each hit was scored.*

4. **Read the `ctx` lines.** Every hit prints surrounding context. A bare
   grep cannot distinguish "Titus destroyed the Temple" from "the text has
   not reached Titus" — a refusal containing the forbidden name.
   *Done: no verdict rests on an unreviewed substring.*

5. **Audit the self-critique.** If the run flags a critique tier, read the
   claim audit line. Flags whose distinctive terms are already in the read
   portion are false positives.
   *Done: false-positive rate is known, not assumed.*

6. **Read the verdict, then disagree with it if needed.** The script's tier
   recommendation is a function of the rules above; the raw answers are
   printed in full for exactly this reason.
   *Done: the conclusion is one you can defend from the printed text.*

## Pitfalls

- **"Masada" is not tail-only.** It appears 10× in the first 24% of the
  book (Herod-era rebellions). Marking on the bare word proves nothing — mark
  on the *fall* of Masada.
- **Ordinal ≠ percentage.** Ordinal 40 of 117 was 24% of characters, not 40%.
  Report the measured figure.
- **Free models are not deterministic at temperature 0.** Leakage varied run
  to run (mode B leaked on 1, then 3, then 2 of 5 probes). One run proves
  nothing; repeat at `BOOKBUDDY_PROBE_TEMP=0.9`.
- **A `[n]`-free answer is the real warning sign.** It means the model
  ignored the context entirely and answered from memory.
- **The earlier probe measured the OLD pipeline**, which put the whole
  76k-token read portion in the prompt. RAG sends ~3.5k tokens of chunks.
  Re-run before quoting an old verdict.
- **Don't treat the self-critique tier as a gate.** Measured flag counts of
  15/5/8 across runs with a high false-positive rate; it missed real leaks.

## Verification

`VERDICT:` lines at the end state the leak result and the recommended tier,
and `raw results written to cache/leakage_probe.json` confirms the raw
responses were saved for re-scoring. Reference measurement: with the read
portion in the prompt the model volunteered "Titus destroyed Jerusalem's
Temple in 70 CE" and appended the disclaimer *after* the spoiler.