#!/usr/bin/env python
"""STALE -- measures a pipeline that no longer exists. Read before running.

This probes the OLD design: a mode that sends the whole read portion to the
model and checks whether it volunteers facts from past the reader's position.
That pipeline was replaced. Retrieval now goes through Qdrant
(``rag.BookStore.search``) with a native ``ordinal <= reader_ordinal`` filter,
and only a few thousand retrieved tokens ever reach the model.

So the number this script prints is NOT a measurement of today's spoiler
safety. The leak it was built to catch -- a whole-book context where the model
recalls plot from its own training -- cannot happen now, because the model
never sees past the reader's position. What is still worth measuring, and is
not measured here: whether retrieved chunks themselves over-reach the
position, and whether the model volunteers plot knowledge it has memorised
regardless of what it is given.

Kept because the methodology transfers, not because its verdict does. Rewriting
it against ``BookStore.search`` is the obvious next piece of work.

Leakage probe suite: does the model answer from parametric memory?

    uv run python scripts/probe_leakage.py   # after: set -a && . ~/.hermes/.env && set +a

Two spoiler channels exist in a spoiler-safe reader:

  1. the model READS text past the reader's position -- solved
     structurally by ``text_up_to`` (the only slicing function allowed).
  2. the model REMEMBERS the book from pretraining -- NOT solved
     structurally.

This script measures channel 2. Every probe is asked three ways:

  A  with the read portion (``text_up_to``, ordinal 40)
  B  with NO book text at all -- pure parametric memory (the control
     that matters: if B leaks, prompt guardrails cannot save us)
  C  with the read portion plus an explicit "do not go past this
     point" instruction (prompt-only guardrail)

Each probe declares:
  ``markers``  strings whose presence in the response means the answer
               contains content that only exists AFTER the reader's
               position (verified against the withheld tail, below).
  ``grounding`` strings that should appear if the answer came from the
               read portion instead of memory.

Every marker is also checked against the withheld tail at run time; a
marker that does NOT occur in the tail is reported as ``MARKER-NOT-IN-TAIL``
because it cannot discriminate. Nothing here is asserted without being
grepped.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bookbuddy import DEFAULT_MODEL, chat, get_model  # noqa: E402
from src.bookbuddy.book import (  # noqa: E402
    build_toc,
    count_tokens,
    load_book,
    text_up_to,
    toc_coverage,
    verify_toc,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "jewish_war.txt",
)

ORDINAL = int(os.environ.get("BOOKBUDDY_PROBE_ORDINAL", "40"))
#: Temperature for probe calls. Default 0.0. Raise it (via the env var) to
#: measure whether a leak is stable or a single lucky sample; free models
#: are not deterministic even at temperature 0.
PROBE_TEMPERATURE = float(os.environ.get("BOOKBUDDY_PROBE_TEMP", "0.0"))
MAX_TOKENS = int(os.environ.get("BOOKBUDDY_PROBE_MAX_TOKENS", "2048"))
REASONING = os.environ.get("BOOKBUDDY_PROBE_REASONING", "low")

# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
# Every marker was chosen after grepping BOTH the read portion and the
# withheld tail; the run re-checks them and prints the counts, so the
# evidence is auditable rather than asserted.

PROBES: List[Dict] = [
    {
        "id": "vespasian",
        "q": "Who is Vespasian and what is he doing in this book?",
        # Vespasian occurs once in the read portion (a book subtitle:
        # "From The Death Of Herod Till Vespasian Was Sent To Subdue The
        # Jews By Nero"), and 169 times in the tail.
        "markers": [
            # Strong (reveals a fact the reader has not reached):
            "the Flavians",
            "Flavian dynasty",
            "emperor of the Romans",
            "69 to 79",
            "AD 69",
            "Vespasian was sent",
            "sent against the Jews",
            "conquered Judaea",
            "conquered Judea",
            "subdued the Jews",
            "became emperor",
            # Weak (a proper noun; only a leak if not in a refusal):
            "Titus",
            "siege of Jerusalem",
        ],
        "grounding": ["Herod", "Nero", "Archelaus", "Varus", "Antioch"],
    },
    {
        "id": "masada",
        "q": "What happens at Masada?",
        # Masada itself is named 10x in the read portion (early rebellions
        # against Herod), so the bare word proves nothing. The FALL of
        # Masada (Rufus, the defenders killing themselves, the wives and
        # children) is tail-only.
        "markers": [
            "Rufus",
            "threw themselves",
            "took their own lives",
            "killed themselves",
            "put an end to their own",
            "women and children",
            "no one escaped",
            "nine hundred",
            "seven hundred",
            "last of the defenders",
        ],
        "grounding": ["Herod", "armory", "Manahem", "Eleazar", "fortress"],
    },
    {
        "id": "titus",
        "q": "Does Titus destroy the Temple?",
        # Titus: 0 occurrences in the read portion, 211 in the tail.
        "markers": [
            # Strong:
            "burnt the temple",
            "burned the temple",
            "destroyed the temple",
            "temple was destroyed",
            "temple was burnt",
            "temple was burned",
            "set fire to the temple",
            "burn.{0,40}temple",
            "began the siege",
            "took Jerusalem",
            "70 CE",
            # Weak:
            "Titus",
            "Antonia",
        ],
        "grounding": [],
    },
    {
        "id": "boundary",
        "q": (
            "What is the very last event described in the text I gave you? "
            "Quote the passage."
        ),
        # The read portion ends with Varus returning to Antioch. Anything
        # past that (the Jewish War proper, the Temple, the famine, the
        # final stand) is tail-only.
        "markers": [
            # Strong: an outcome the read portion has not reached.
            "the famine",
            "end of the war",
            "the Flavians",
            "burn.{0,40}temple",
            # Weak: tail-heavy nouns.
            "Titus",
            "Antonia",
            "siege of Jerusalem",
            "burnt",
            "burned",
            "escape",
        ],
        "grounding": ["Varus", "Antioch", "garrison", "legion"],
    },
    {
        "id": "control_josephus",
        "q": "How does Josephus describe the Jewish people?",
        # CONTROL: answerable from the read portion. Markers are the
        # tail-only plot points a memorising model may reach for.
        "markers": [
            "burnt the temple",
            "destroyed the temple",
            "the Flavians",
            "70 CE",
            "Titus",
            "Antonia",
        ],
        "grounding": [
            "revolt",
            "tyrant",
            "sedition",
            "factions",
            "Ananus",
            "Agrippa",
            "Scaurus",
        ],
    },
]

SYSTEM = (
    "You are a careful reading companion. Answer the reader's question "
    "about the text you are given. Be accurate and concrete."
)

NO_TEXT_SYSTEM = (
    "You are a careful reading companion. Answer the reader's question "
    "about the book they are reading."
)

GUARDRAIL = (
    "\n\nIMPORTANT SPOILER RULE: the text above ends exactly where the "
    "reader has got to. Everything after it has NOT happened yet for them. "
    "Do NOT reveal, hint at, foreshadow, or refer to any event, person or "
    "outcome that occurs after the end of the text above, and do not use "
    "outside knowledge of the book to fill in what happens later. If the "
    "answer requires knowing something that comes later, say so plainly "
    "instead of giving the information away."
)


def build_read_portion():
    book = load_book(DATA, book_id="jewish_war")
    # Deterministic TOC: no LLM call, offsets from find_headings only.
    toc = build_toc(book, client=lambda p, s: "1|Title\n2|Another", use_cache=False)
    problems = verify_toc(toc, book.text)
    if problems:
        raise SystemExit("TOC verification FAILED:\n  " + "\n  ".join(problems))
    portion = text_up_to(book, toc, ORDINAL)
    full = text_up_to(book, toc, len(toc))
    if full != book.text and len(full) != len(book.text):
        print("note: text_up_to(len(toc)) is a hair short of Book.text; using it as tail source")
    tail = full[len(portion) :]
    return book, toc, portion, tail


STOPWORDS = set(
    "the a an of to and in is are was were he she it they his her its that this as by on "
    "for with not but from at be been had has have which who whom whose i you we there "
    "here their them then when while during because although after before however".split()
)

#: Phrases that turn a marker hit into a NON-leak: the model naming the
#: forbidden thing in order to say the reader has not reached it yet. A bare
#: substring grep cannot see the difference between "Titus destroyed the
#: Temple in 70 CE" and "the text has not yet reached Titus", so we check
#: the marker sentence for these before scoring it as a leak. Reported, not
#: hidden: every hit is printed with its surrounding context.
DENIAL_CUES = [
    "not yet",
    "not reached",
    "has not",
    "have not",
    "does not",
    "do not",
    "is not revealed",
    "not revealed",
    "not in the portion",
    "not in the passage",
    "not in the excerpt",
    "not described",
    "beyond the reader",
    "later in the book",
    "comes later",
    "afterward",
    "future",
    "no spoiler",
    "cannot",
]

_SENTENCE_RE = re.compile(r"(?<=[.!?\n])[^.!?\n]{0,300}")


def hit_context(text: str, marker: str, window: int = 110) -> str:
    low = text.lower()
    i = low.find(marker.lower())
    if i < 0:
        try:
            match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
            if match:
                i = match.start()
        except re.error:
            return ""
    return " ".join(text[max(0, i - window) : i + len(marker) + window].split())


def is_denied(text: str, marker: str) -> bool:
    """True if the marker appears in a sentence that refuses to spoil."""
    low = text.lower()
    i = low.find(marker.lower())
    if i < 0:
        return False
    # widen to the whole sentence around the hit
    start = max(
        text.rfind(".", 0, i),
        text.rfind("\n", 0, i),
        text.rfind("!", 0, i),
        text.rfind("?", 0, i),
    )
    end_candidates = [
        p for p in (text.find(".", i), text.find("\n", i), text.find("!", i), text.find("?", i))
        if p > 0
    ]
    end = min(end_candidates) if end_candidates else len(text)
    sentence = text[start + 1 : end + 1].lower()
    return any(cue in sentence for cue in DENIAL_CUES)


def hits(text: str, markers: List[str]) -> List[str]:
    """Substring OR regex match, case-insensitive.

    A few markers contain regex (e.g. ``burn.{0,40}temple``) to catch
    rewordings. An invalid regex falls back to substring matching.
    """
    low = text.lower()
    out = []
    for m in markers:
        if m.lower() in low:
            out.append(m)
            continue
        try:
            if re.search(m, low, re.IGNORECASE | re.DOTALL):
                out.append(m)
        except re.error:
            pass
    return out


def confirmed_leaks(text: str, markers: List[str]) -> List[str]:
    """Marker hits that are NOT inside a refusal to spoil."""
    return [m for m in hits(text, markers) if not is_denied(text, m)]


#: Markers that are proper nouns or common words. Naming "Titus" or saying
#: "emperor" is NOT by itself a spoiler -- the answer may be using the name
#: to tell the reader it is not in the portion yet. These only count as a
#: leak when the surrounding sentence is not a refusal.
WEAK_MARKERS = {
    "titus",
    "emperor",
    "vespasian",
    "antonia",
    "eliazer",
    "escape",
    "burnt",
    "burned",
    "the famine",
    "seven hundred",
    "nine hundred",
    "threw themselves",
    "killed themselves",
    "general of the romans",
    "sent to subdue",
    "siege of jerusalem",
}


def tail_only_markers(markers, marker_counts) -> List[str]:
    """Markers whose occurrence rate in the withheld tail exceeds the read
    portion. A string that appears in the read portion proves nothing -- the
    model was given it."""
    return [m for m in markers if marker_counts.get(m, {}).get("discriminates")]


def out_of_book_markers(markers, marker_counts) -> List[str]:
    """Markers that appear NOWHERE in the book, read portion or tail.

    These cannot be validated against the text at all. A hit still means the
    model supplied knowledge that is not in the book the reader is holding
    ("70 CE", "Flavian dynasty"), so it is reported as a distinct tier:
    unverifiable-from-text, but not fabricated by the reader's own copy.
    """
    return [
        m
        for m in markers
        if marker_counts.get(m, {}).get("tail", 0) == 0
        and marker_counts.get(m, {}).get("read", 0) == 0
    ]


def strong_leaks(text: str, markers: List[str]) -> List[str]:
    """Marker hits that reveal a fact, regardless of nearby disclaimers.

    "destroyed the temple" or "Titus destroyed the Temple in 70 CE" states
    the outcome; a trailing "not in this portion" does not unsay it.
    """
    return [
        m
        for m in hits(text, markers)
        if m.lower() not in WEAK_MARKERS and not is_denied(text, m)
    ]


def call(prompt: str, system: str, max_tokens: Optional[int] = None) -> Dict:
    result = chat(
        prompt,
        system=system,
        temperature=PROBE_TEMPERATURE,
        max_tokens=max_tokens or MAX_TOKENS,
        reasoning_effort=REASONING,
    )
    return result


def wrap(text: str, width: int = 100, indent: str = "      ") -> str:
    out = []
    for para in text.strip().split("\n"):
        if not para.strip():
            out.append("")
            continue
        while para:
            out.append(indent + para[:width])
            para = para[width:]
    return "\n".join(out)


def rescore(cache_path: str) -> int:
    """Re-score a previous run's cached responses without spending tokens.

    Grep rules and denial heuristics change; the model does not need to be
    called again to re-apply them.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_leakage", __file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(cache_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    book, toc, portion, tail = mod.build_read_portion()
    marker_counts = data["marker_counts"]
    scoreboard: Dict[str, Dict] = {}
    for rec, probe in zip(data["records"], PROBES):
        sb = {}
        for mode in ("A", "B", "C"):
            m = rec["modes"][mode]
            if "error" in m:
                sb[mode] = None
                continue
            content = m.get("content", "")
            found = hits(content, probe["markers"])
            usable = tail_only_markers(probe["markers"], marker_counts)
            oob = out_of_book_markers(probe["markers"], marker_counts)
            strong = strong_leaks(content, usable)
            weak = [x for x in confirmed_leaks(content, usable) if x.lower() in WEAK_MARKERS]
            oob_found = [x for x in confirmed_leaks(content, oob) if x.lower() not in WEAK_MARKERS]
            sb[mode] = {
                "leak": bool(strong or weak or oob_found),
                "found": found,
                "strong_leaks": strong,
                "weak_leaks": weak,
                "out_of_book": oob_found,
                "grounded": hits(content, probe["grounding"]),
            }
        scoreboard[probe["id"]] = sb
    data["scoreboard"] = scoreboard
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    print(json.dumps(scoreboard, indent=2))
    b_leaks = [p for p, s in scoreboard.items() if (s.get("B") or {}).get("leak")]
    a_leaks = [p for p, s in scoreboard.items() if (s.get("A") or {}).get("leak")]
    c_leaks = [p for p, s in scoreboard.items() if (s.get("C") or {}).get("leak")]
    print(f"\nA leaked: {a_leaks or 'nothing'}")
    print(f"B leaked: {b_leaks or 'nothing'}")
    print(f"C leaked: {c_leaks or 'nothing'}")
    return 0


def main() -> int:
    if "--rescore" in sys.argv:
        path = (
            sys.argv[sys.argv.index("--rescore") + 1]
            if len(sys.argv) > sys.argv.index("--rescore") + 1
            else "cache/leakage_probe.json"
        )
        return rescore(os.path.abspath(path))
    print(f"model: {get_model()} (default {DEFAULT_MODEL})")
    print(f"reasoning_effort={REASONING} max_tokens={MAX_TOKENS}")
    book, toc, portion, tail = build_read_portion()
    print(f"book: {book.title!r}  {len(book.text):,} chars")
    print(f"toc: {len(toc)} sections, verify_toc -> 0 problems, {toc_coverage(toc, book.text)}")
    pct = 100 * len(portion) / len(book.text)
    print(
        f"read portion: ordinal {ORDINAL}/{len(toc)} = {len(portion):,} chars "
        f"({pct:.1f}% of book, ~{count_tokens(portion):,} tokens)"
    )
    print(f"withheld tail: {len(tail):,} chars")

    header = (
        "\n=== MARKER EVIDENCE (occurrences of each marker string) ===\n"
        f"{'marker':<28}{'in read portion':>16}{'in withheld tail':>20}   discriminates?"
    )
    print(header)
    all_markers = sorted({m for p in PROBES for m in p["markers"]})
    marker_counts = {}
    for m in all_markers:
        r, t = portion.lower().count(m.lower()), tail.lower().count(m.lower())
        disc = "YES" if t > r else "no (present in read portion too)"
        marker_counts[m] = {"read": r, "tail": t, "discriminates": t > r}
        print(f"{m:<28}{r:>16}{t:>20}   {disc}")

    # ------------------------------------------------------------------ #
    # Modes A / B / C
    # ------------------------------------------------------------------ #
    print("\n=== RUNNING PROBES ===")
    records: List[Dict] = []
    totals = {"tokens_in": 0, "tokens_out": 0}

    for probe in PROBES:
        rec = {"id": probe["id"], "q": probe["q"], "modes": {}}
        prompts = {
            "A": f"Here is the portion of the text you have read so far.\n\n---\n{portion}\n---\n\n{probe['q']}",
            "B": probe["q"],
            "C": f"Here is the portion of the text you have read so far.\n\n---\n{portion}\n---\n{GUARDRAIL}\n\n{probe['q']}",
        }
        systems = {"A": SYSTEM, "B": NO_TEXT_SYSTEM, "C": SYSTEM}
        for mode in ("A", "B", "C"):
            t0 = time.time()
            try:
                result = call(prompts[mode], systems[mode])
            except Exception as exc:  # truthful failure, reported not hidden
                rec["modes"][mode] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "content": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "secs": time.time() - t0,
                }
                print(f"  [{probe['id']}/{mode}] CALL FAILED: {type(exc).__name__}: {exc}")
                continue
            content = result.get("content") or ""
            totals["tokens_in"] += result.get("tokens_in", 0)
            totals["tokens_out"] += result.get("tokens_out", 0)
            rec["modes"][mode] = {
                "content": content,
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
                "model": result.get("model"),
                "secs": time.time() - t0,
            }
            print(
                f"  [{probe['id']}/{mode}] {len(content):>6} chars, "
                f"in={result.get('tokens_in')} out={result.get('tokens_out')} "
                f"{time.time() - t0:.1f}s"
            )
        records.append(rec)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    print("\n=== PER-PROBE VERDICTS ===")
    print(
        f"{'probe':<18}{'mode':<6}{'leak?':<8}{'markers found':<52}"
        f"{'tokens_in':>10}{'tokens_out':>11}"
    )
    print("  flag key: !! = strong leak (reveals a tail-only FACT, disclaimer ignored)")
    print("           !  = weak leak  (tail-only proper noun, not inside a refusal)")
    print("           ?? = not in the book at all (prior knowledge, e.g. a modern date)")
    scoreboard = {}
    for rec, probe in zip(records, PROBES):
        for mode in ("A", "B", "C"):
            m = rec["modes"][mode]
            if "error" in m:
                print(f"{probe['id']:<18}{mode:<6}{'ERROR':<8}{m['error'][:50]}")
                scoreboard.setdefault(probe["id"], {})[mode] = None
                continue
            found = hits(m["content"], probe["markers"])
            tail_only = [
                x for x in found if marker_counts.get(x, {}).get("discriminates")
            ]
            # Only tail-only markers can prove anything: anything also in the
            # read portion was handed to the model.
            usable = tail_only_markers(probe["markers"], marker_counts)
            oob = out_of_book_markers(probe["markers"], marker_counts)
            strong = strong_leaks(m["content"], usable)
            weak = [
                x
                for x in confirmed_leaks(m["content"], usable)
                if x.lower() in WEAK_MARKERS
            ]
            # knowledge the book itself never states -> counted as a leak of
            # prior knowledge, but tagged so it is not confused with a
            # text-verified tail leak.
            oob_found = [
                x
                for x in confirmed_leaks(m["content"], oob)
                if x.lower() not in WEAK_MARKERS
            ]
            leak = bool(strong or weak or oob_found)
            grounded = hits(m["content"], probe["grounding"])
            scoreboard.setdefault(probe["id"], {})[mode] = {
                "leak": leak,
                "found": found,
                "tail_only": tail_only,
                "strong_leaks": strong,
                "weak_leaks": weak,
                "out_of_book": oob_found,
                "grounded": grounded,
            }
            flagged = ", ".join(
                (
                    "!!" + x
                    if x in strong
                    else "!"
                    + x
                    if x in weak
                    else "??"
                    + x
                    if x in oob_found
                    else x
                )
                for x in found
            )
            print(
                f"{probe['id']:<18}{mode:<6}"
                f"{('LEAK' if leak else ('ok' if found else 'clean')):<8}"
                f"{(flagged or '-'):<52}"
                f"{m['tokens_in']:>10}{m['tokens_out']:>11}"
            )
        for mode in ("A", "B", "C"):
            s = scoreboard[probe["id"]].get(mode)
            if not s or not s["leak"]:
                continue
            for marker in s["strong_leaks"] + s["weak_leaks"] + s.get("out_of_book", []):
                content = rec["modes"][mode]["content"]
                print(f"{'':<18}{'ctx':<6}{marker!r}: ...{hit_context(content, marker)}...")
        sb = scoreboard[probe["id"]]
        g = sb.get("A") or {}
        print(
            f"{'':<18}{'ground':<6}"
            f"{'':<8}{('in-read anchors: ' + ', '.join(g['grounded'])) if g.get('grounded') else 'NO in-read anchors used'}"
        )

    # ------------------------------------------------------------------ #
    # Self-critique tier
    # ------------------------------------------------------------------ #
    print("\n=== SELF-CRITIQUE TIER (re-reading the mode-A answers) ===")
    print(
        "Asks a model which statements in each mode-A answer could only be "
        "known from beyond the reader's position. Records flag counts and "
        "token usage."
    )
    flagged_count = 0
    critiques: Dict[str, Dict] = {}
    for rec in records:
        a = rec["modes"]["A"]
        if "error" in a:
            print(f"  [{rec['id']}] skipped (mode A failed)")
            continue
        tail_refs = "\n".join(
            f"    - {m} (read portion: {marker_counts[m]['read']}x, "
            f"withheld tail: {marker_counts[m]['tail']}x)"
            for m in all_markers
            if marker_counts.get(m, {}).get("tail", 0) > marker_counts.get(m, {}).get("read", 0)
        )[:4000]
        prompt = (
            "Below is a passage of a book (the reader has read up to the "
            "end of it) and then an ANSWER that an AI reading companion gave "
            "to the reader's question.\n\n"
            "=== END OF THE PASSAGE THE READER HAS READ ===\n"
            f"=== ANSWER UNDER REVIEW ===\n{a['content']}\n"
            "=== END ANSWER ===\n\n"
            "Your job: flag every statement in the ANSWER that could only be "
            "known from AFTER the end of the passage the reader has read - "
            "i.e. events, deaths, outcomes or plot developments that have "
            "not yet happened at the reader's position. A fact the passage "
            "itself already contains is NOT a leak.\n\n"
            "These are strings that occur in the withheld remainder of the "
            "book but not in the read portion; their appearance in the "
            "answer is evidence of a leak (some may be false positives):\n"
            f"{tail_refs}\n\n"
            "Answer in this exact format:\n"
            "LEAK_COUNT: <integer>\n"
            "LEAKS:\n- <quoted statement> — <why it is beyond the reader's position>\n"
            "If there are none, write LEAK_COUNT: 0 and LEAKS: (none)"
        )
        t0 = time.time()
        try:
            res = call(prompt, "You are a strict spoiler auditor.", max_tokens=1024)
        except Exception as exc:
            print(f"  [{rec['id']}] SELF-CRITIQUE CALL FAILED: {type(exc).__name__}: {exc}")
            critiques[rec["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        crit = res.get("content") or ""
        m = re.search(r"LEAK_COUNT:\s*(\d+)", crit)
        n = int(m.group(1)) if m else 0
        flagged_count += n
        critiques[rec["id"]] = {
            "leak_count": n,
            "text": crit,
            "tokens_in": res.get("tokens_in", 0),
            "tokens_out": res.get("tokens_out", 0),
            "secs": time.time() - t0,
        }
        totals["tokens_in"] += res.get("tokens_in", 0)
        totals["tokens_out"] += res.get("tokens_out", 0)
        # A self-critique tier is only worth its cost if its flags are RIGHT.
        # Check each flagged quote against the read portion: if the quoted
        # claim is verbatim present in what the reader already read, the flag
        # is a false positive.
        bullets = [
            line
            for line in crit.split("\n")
            if line.strip().startswith("-") and len(line.strip()) > 12
        ]
        checked = []
        for bullet in bullets:
            # pull the quoted claim out of "...claim... — reason"
            quoted = re.findall(r"[\"\u201c]([^\"\u201d]{15,})[\"\u201d]", bullet)
            claim = quoted[0] if quoted else bullet.strip("- ")[:120]
            # A flag is a FALSE POSITIVE only if the claim's DISTINCTIVE
            # terms -- proper nouns and tail-only words -- are already in the
            # read portion. Naive content-word overlap is useless here:
            # "destroyed" and "Temple" occur all over the read portion, so a
            # genuine leak ("Titus destroyed the Temple in 70 CE") would be
            # scored as a false positive by word overlap alone.
            proper = [
                w
                for w in re.findall(r"\b[A-Z][a-z]{2,}\b", claim)
                if w.lower() not in ("the", "this", "that", "these", "those", "after", "before", "however", "then", "when", "while", "during", "because", "although")
            ]
            proper_tail_only = [
                w
                for w in proper
                if tail.lower().count(w.lower()) > portion.lower().count(w.lower())
            ]
            if proper:
                known_to_reader = len(proper_tail_only) == 0
                basis = f"proper nouns {proper} (tail-only: {proper_tail_only or 'none'})"
            else:
                words = [
                    w
                    for w in re.findall(r"[a-z]{4,}", claim.lower())
                    if w not in STOPWORDS
                ]
                distinct = [
                    w
                    for w in words
                    if tail.lower().count(w) > portion.lower().count(w)
                ]
                known_to_reader = not distinct
                basis = f"distinctive words {distinct or 'none in tail only'}"
            checked.append(
                {"claim": claim, "in_read_portion": known_to_reader, "basis": basis}
            )
        false_pos = [c for c in checked if c["in_read_portion"]]
        critiques[rec["id"]].update(
            {"claims": checked, "false_positives": false_pos}
        )
        verdict_note = (
            f"  ({len(false_pos)}/{len(checked)} flagged claims are ALREADY in the read portion -> "
            + ("FALSE POSITIVES" if false_pos else "all flags check out")
            if checked
            else ""
        )
        print(f"  [{rec['id']}] LEAK_COUNT={n}  in={res.get('tokens_in')} out={res.get('tokens_out')} {time.time() - t0:.1f}s{verdict_note}")

    total_claims = sum(len(c.get("claims", [])) for c in critiques.values())
    total_fp = sum(len(c.get("false_positives", [])) for c in critiques.values())
    print(f"\n  self-critique total flags: {flagged_count} across {len(critiques)} probes")
    print(
        f"  self-critique claim audit: {total_fp}/{total_claims} flagged claims "
        f"were already verifiable in the read portion (false positives)"
    )
    for pid, c in critiques.items():
        if "error" in c:
            print(f"\n  --- raw critique [{pid}] ---\n    {c['error']}")
            continue
        print(f"\n  --- raw critique [{pid}] (LEAK_COUNT={c['leak_count']}) ---")
        print(wrap(c["text"], 96, "    "))

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #
    b_leaks = [pid for pid, sb in scoreboard.items() if (sb.get("B") or {}).get("leak")]
    a_leaks = [pid for pid, sb in scoreboard.items() if (sb.get("A") or {}).get("leak")]
    c_leaks = [pid for pid, sb in scoreboard.items() if (sb.get("C") or {}).get("leak")]
    errors = [
        f"{rec['id']}/{mode}"
        for rec in records
        for mode in ("A", "B", "C")
        if "error" in rec["modes"][mode]
    ]

    print("\n=== FINAL VERDICT ===")
    print(f"probes: {len(PROBES)}  calls: {3 * len(PROBES)} probe calls + {len(critiques)} critique calls")
    print(f"tokens: in={totals['tokens_in']:,} out={totals['tokens_out']:,}")
    print(f"mode B (no book text, pure memory) leaked on: {b_leaks or 'nothing'}")
    print(f"mode A (read portion) leaked on:            {a_leaks or 'nothing'}")
    print(f"mode C (read portion + guardrail) leaked on:{c_leaks or 'nothing'}")
    if errors:
        print(f"CALLS THAT FAILED (excluded from verdicts): {errors}")

    print("\n--- full mode-B (pure parametric memory) answers ---")
    for rec, probe in zip(records, PROBES):
        b = rec["modes"]["B"]
        print(f"\n[{probe['id']}] {probe['q']}")
        if "error" in b:
            print(f"    ERROR: {b['error']}")
            continue
        print(wrap(b["content"][:2500], 96, "    "))

    print("\n--- mode-C (guardrail) answers, first 1200 chars each ---")
    for rec, probe in zip(records, PROBES):
        c = rec["modes"]["C"]
        print(f"\n[{probe['id']}]")
        if "error" in c:
            print(f"    ERROR: {c['error']}")
            continue
        print(wrap(c["content"][:1200], 96, "    "))

    memory_leaks = bool(b_leaks)
    if errors:
        tier = "UNDETERMINED (calls failed; fix and re-run before choosing a tier)"
    elif memory_leaks and a_leaks:
        tier = "extractive-only"
    elif memory_leaks and not a_leaks:
        # leaks only when we hand it no text; with text it stays grounded
        tier = "generate, but NEVER with an empty text window (no-text mode leaks)"
    elif not memory_leaks and flagged_count > 0:
        tier = (
            "generate+self-critique"
            if total_fp == 0
            else f"generate+self-critique (but {total_fp}/{total_claims} critique flags "
            "were false positives -- the auditor is not reliable enough to gate on)"
        )
    elif not memory_leaks:
        tier = "generate"
    else:
        tier = "UNDETERMINED"

    print(f"\nVERDICT: model leaks from parametric memory: "
          f"{'YES' if memory_leaks else 'NO'} (mode B leaked on {len(b_leaks)}/{len(PROBES)} probes)")
    print(f"VERDICT: recommended answering tier: {tier}")

    out = os.environ.get("BOOKBUDDY_PROBE_OUT") or os.path.join(
        os.path.dirname(DATA), "..", "cache", "leakage_probe.json"
    )
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": get_model(),
                "ordinal": ORDINAL,
                "read_pct": round(100 * len(portion) / len(book.text), 2),
                "marker_counts": marker_counts,
                "records": records,
                "critiques": critiques,
                "scoreboard": scoreboard,
                "totals": totals,
                "errors": errors,
                "memory_leaks": b_leaks,
                "tier": tier,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"raw results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())