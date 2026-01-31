import pytest
from book import Chapter
from summarize import create_chapter_batches


@pytest.mark.parametrize(
    "chapters, expected",
    [
        ([], []),
        (
            [
                Chapter(tokens=20_000, name="Chapter 1"),
                Chapter(tokens=10_000, name="Chapter 2"),
            ],
            [["Chapter 1", "Chapter 2"]],
        ),
        (
            [
                Chapter(tokens=40_000, name="Chapter 1"),
                Chapter(tokens=10_000, name="Chapter 2"),
            ],
            [["Chapter 1"], ["Chapter 2"]],
        ),
        (
            [
                Chapter(tokens=20_000, name="Chapter 1"),
                Chapter(tokens=10_000, name="Chapter 2"),
                Chapter(tokens=30_000, name="Chapter 3"),
                Chapter(tokens=40_000, name="Chapter 4"),
                Chapter(tokens=50_000, name="Chapter 5"),
            ],
            [
                ["Chapter 1", "Chapter 2"],
                ["Chapter 3"],
                ["Chapter 4"],
                ["Chapter 5"],
            ],
        ),
        (
            [
                Chapter(tokens=70_000, name="Chapter 1"),
                Chapter(tokens=10_000, name="Chapter 2"),
                Chapter(tokens=60_000, name="Chapter 3"),
            ],
            [["Chapter 1"], ["Chapter 2"], ["Chapter 3"]],
        ),
    ],
)
def test_create_chapter_batches(chapters, expected):
    batches = create_chapter_batches(chapters)
    batches = [[c.name for c in batch] for batch in batches]
    assert batches == expected
