"""The heading detector, measured against real Gutenberg conventions.

Every expectation here was read off an actual text file, not assumed:

  jewish_war  117  "BOOK I." / "CHAPTER 12."
  pnp          61  "Chapter I."   (mixed case, trailing "]")
  war_peace   380  "BOOK ONE: 1805" / "CHAPTER I"  (no trailing period)
  sherlock     12  "I." / "II. THE RED-HEADED LEAGUE"  (bare roman)

Before the broadening, the old regex found 117 / 0 / 0 / 57 for those four.
"""

import pytest

from src.bookbuddy.book import HEADING_RE, derive_title, find_headings

JW_TITLE_BLOCK = """BOOK I.

CHAPTER 2.
Concerning The Successors Of Judas, Who Were Jonathan And Simon

CHAPTER 3.
How Aristobulus Was The First That Put A Diadem About His Head

1. For after the death of their father the elder of them Aristobulus did.
"""

UNTITLED_BLOCK = """CHAPTER III.

Within a short walk of Longbourn lived a family with a large fortune.
The principal picnics and parties of the season were held in that valley.

CHAPTER IV.

When Jane and Elizabeth were alone the former told her a great deal.
They had been handsome and clever and rich in comparison with the rest.
"""

TOC_CLUSTER = """      CHAPTER I. The Taking Of Jerusalem
      CHAPTER II. Pompey At Jerusalem
      CHAPTER III. Aristobulus Crowned
      CHAPTER IV. Hyrcanus Restored
"""


def _headings(body: str):
    return [h["label"] for h in find_headings(body)]


class TestHeadingShapes:
    @pytest.mark.parametrize("line", [
        "BOOK I.", "BOOK ONE: 1805", "Chapter I.", "CHAPTER 12.",
        "CHAPTER I", "BOOK SIXTEEN",
        # Pride and Prejudice closes the heading with a bracket on its own line
        "Chapter I.]",
    ])
    def test_recognised(self, line):
        assert HEADING_RE.match(line), line

    @pytest.mark.parametrize("line", [
        "I.", "II. THE RED-HEADED LEAGUE", "XII. THE ADVENTURE OF THE COPPER BEECHES",
    ])
    def test_bare_roman_recognised(self, line):
        assert HEADING_RE.match(line), line

    @pytest.mark.parametrize("line", [
        "At three o'clock precisely I was at Baker Street",
        "1. For after the death of their father the elder of them",
        "15",
        "This is a fine book,' said M. Gillenormand.",
    ])
    def test_not_a_heading(self, line):
        assert not HEADING_RE.match(line), line


class TestFindHeadings:
    def test_counts_both_books_and_chapters(self):
        labels = _headings(JW_TITLE_BLOCK)
        assert labels == ["BOOK I.", "CHAPTER 2.", "CHAPTER 3."]

    def test_regression_never_returns_a_toc_entry(self):
        """Indented contents lines must never become sections."""
        assert _headings(TOC_CLUSTER) == []

    def test_untitled_chapters_still_detected(self):
        assert _headings(UNTITLED_BLOCK) == ["CHAPTER III.", "CHAPTER IV."]

    def test_offsets_are_ascending(self):
        offsets = [h["offset"] for h in find_headings(JW_TITLE_BLOCK)]
        assert offsets == sorted(offsets)


class TestDeriveTitle:
    """Titles are read from the text, not asked for.

    A wrong title is worse than an empty one: it shows up in the UI next to a
    chapter and reads as a spoiler. So the rule must refuse to invent.
    """

    def test_reads_a_real_title(self):
        heading = find_headings(JW_TITLE_BLOCK)[1]
        title = derive_title(JW_TITLE_BLOCK, heading["offset"])
        assert title.startswith("Concerning The Successors Of Judas")

    def test_returns_empty_for_an_untitled_chapter(self):
        heading = find_headings(UNTITLED_BLOCK)[0]
        assert derive_title(UNTITLED_BLOCK, heading["offset"]) == ""

    def test_never_fabricates_from_body_prose(self):
        """The regression that motivated the title-shape test.

        Without the word-capital check this returned "Within a short walk of
        Longbourn lived a family with a large fortune" as the chapter title.
        """
        for heading in find_headings(UNTITLED_BLOCK):
            title = derive_title(UNTITLED_BLOCK, heading["offset"])
            assert "Longbourn" not in title
            assert "fortune" not in title
