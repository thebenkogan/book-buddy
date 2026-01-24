import pytest
from summarize import create_chapter_batches


@pytest.mark.parametrize(
    "chapters, expected",
    [
        ([], []),
        (
            [
                {"tokens": 20_000, "chapter": "Chapter 1"},
                {"tokens": 10_000, "chapter": "Chapter 2"},
            ],
            [["Chapter 1", "Chapter 2"]],
        ),
        (
            [
                {"tokens": 40_000, "chapter": "Chapter 1"},
                {"tokens": 10_000, "chapter": "Chapter 2"},
            ],
            [["Chapter 1"], ["Chapter 2"]],
        ),
        (
            [
                {"tokens": 20_000, "chapter": "Chapter 1"},
                {"tokens": 10_000, "chapter": "Chapter 2"},
                {"tokens": 30_000, "chapter": "Chapter 3"},
                {"tokens": 40_000, "chapter": "Chapter 4"},
                {"tokens": 50_000, "chapter": "Chapter 5"},
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
                {"tokens": 70_000, "chapter": "Chapter 1"},
                {"tokens": 10_000, "chapter": "Chapter 2"},
                {"tokens": 60_000, "chapter": "Chapter 3"},
            ],
            [["Chapter 1"], ["Chapter 2"], ["Chapter 3"]],
        ),
    ],
)
def test_create_chapter_batches(chapters, expected):
    batches = create_chapter_batches(chapters)
    batches = [[c["chapter"] for c in batch] for batch in batches]
    assert batches == expected
