"""Chunking is pure and decides retrieval quality, so it gets the closest scrutiny."""

import pytest

from gieok.core.chunking import chunk_text

SOURCE = "doc.md"


def chunks(text: str, *, size: int = 100, overlap: int = 20):
    return list(chunk_text(text, source=SOURCE, size=size, overlap=overlap))


def test_empty_text_produces_no_chunks():
    assert chunks("") == []
    assert chunks("   \n\n  \t ") == []


def test_text_shorter_than_size_produces_one_chunk():
    result = chunks("A short paragraph.")
    assert len(result) == 1
    assert result[0].text == "A short paragraph."
    assert result[0].index == 0
    assert result[0].source == SOURCE


def test_paragraphs_are_packed_until_the_size_limit():
    text = "\n\n".join(["word " * 8] * 6)  # ~40 chars per paragraph
    result = chunks(text, size=100, overlap=0)
    assert len(result) > 1
    assert all(len(chunk.text) <= 100 for chunk in result)
    # Packing means a chunk should generally hold more than a single paragraph.
    assert any("\n\n" in chunk.text for chunk in result)


def test_every_chunk_respects_the_size_invariant():
    text = "\n\n".join(f"Paragraph {n}: " + "lorem ipsum dolor sit amet " * 12 for n in range(20))
    for size in (60, 137, 400):
        for chunk in chunks(text, size=size, overlap=size // 5):
            assert 0 < len(chunk.text) <= size


def test_oversized_paragraph_is_hard_split_on_word_boundaries():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 6
    result = chunks(text, size=80, overlap=0)
    assert len(result) > 1
    assert all(len(chunk.text) <= 80 for chunk in result)
    # A word-boundary split never leaves a fragment of a known word.
    words = set(text.split())
    for chunk in result:
        assert set(chunk.text.split()) <= words


def test_consecutive_chunks_share_overlapping_text():
    text = "\n\n".join(f"Section {n} discusses a distinct topic in detail." for n in range(12))
    result = chunks(text, size=120, overlap=40)
    assert len(result) > 1
    overlaps = [
        any(word in result[n + 1].text for word in result[n].text.split()[-3:])
        for n in range(len(result) - 1)
    ]
    assert any(overlaps), "expected at least some boundary context to be carried forward"


def test_indices_are_contiguous_and_ids_are_unique():
    text = "\n\n".join(f"Distinct paragraph number {n} with its own content." for n in range(15))
    result = chunks(text, size=90, overlap=10)
    assert [chunk.index for chunk in result] == list(range(len(result)))
    assert len({chunk.id for chunk in result}) == len(result)


def test_ids_are_deterministic_across_runs():
    text = "\n\n".join(f"Paragraph {n}." for n in range(10))
    first = [chunk.id for chunk in chunks(text)]
    second = [chunk.id for chunk in chunks(text)]
    assert first == second


def test_id_changes_when_content_changes():
    (original,) = chunks("The original sentence.")
    (edited,) = chunks("The edited sentence.")
    assert original.id != edited.id


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-10, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunks("some text", size=size, overlap=overlap)
