"""Chunking is pure and decides retrieval quality, so it gets the closest scrutiny."""

from pathlib import Path

import pytest

from gieok.core.chunking import chunk_document, chunk_text
from gieok.models import Document, Page

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


def test_golden_id_for_a_known_non_paginated_chunk():
    # Fixed against the pre-PDF-support hash formula (`sha256(f"{source}|{index}|{text}")`).
    # This is the only test that can prove the .md/.txt id space -- and therefore every id
    # already sitting in the live .chroma/ collection -- survived adding the `page` field to
    # `Chunk.create`. It must keep passing unmodified for as long as that guarantee holds.
    (chunk,) = chunks("A short paragraph.", size=100, overlap=20)
    assert chunk.id == "662a2c0836f8f59bd26a228cfbed3482"


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-10, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunks("some text", size=size, overlap=overlap)


# --- chunk_document (page-aware packing) -------------------------------------------------


def test_chunk_document_with_no_pages_matches_chunk_text():
    # A non-paginated Document (the .md/.txt case) must produce byte-identical output to
    # chunk_text -- this is the guarantee that makes step 4's refactor a no-op for the
    # formats that already existed before PDF support.
    text = "\n\n".join(f"Paragraph {n} has some words in it." for n in range(8))
    document = Document(source=Path(SOURCE), content=text)
    via_document = list(chunk_document(document, size=90, overlap=15))
    via_text = list(chunk_text(text, source=SOURCE, size=90, overlap=15))
    assert via_document == via_text


def test_indices_stay_contiguous_across_a_page_boundary():
    pages = [Page(number=n, content=f"Page {n} content with a few words in it.") for n in (1, 2, 3)]
    document = Document.paginated(Path("doc.pdf"), pages)
    result = list(chunk_document(document, size=30, overlap=0))
    assert [chunk.index for chunk in result] == list(range(len(result)))


def test_each_chunk_is_stamped_with_the_page_its_content_starts_on():
    # Sized so each page's short paragraph fills its own chunk and none pack together:
    # a direct one-page-per-chunk correspondence, easy to assert against.
    page1 = Page(number=1, content="AAAA AAAA AAAA AAAA")
    page2 = Page(number=2, content="BBBB BBBB BBBB BBBB")
    document = Document.paginated(Path("doc.pdf"), [page1, page2])

    result = list(chunk_document(document, size=25, overlap=0))

    assert [(chunk.page, chunk.text) for chunk in result] == [
        (1, "AAAA AAAA AAAA AAAA"),
        (2, "BBBB BBBB BBBB BBBB"),
    ]


def test_overlap_carried_across_a_page_boundary_keeps_the_earlier_page():
    # The tail carried into the second chunk originates on page 1 (the end of its second
    # paragraph), even though the chunk's bulk is page 2's content. The chunk is stamped
    # with the page it *starts* on -- page 1 -- per the documented "less precise but true"
    # trade-off, not the page most of its text came from.
    page1 = Page(number=1, content="First paragraph one.\n\nSecond paragraph two words here.")
    page2 = Page(number=2, content="Third paragraph three words words.")
    document = Document.paginated(Path("doc.pdf"), [page1, page2])

    result = list(chunk_document(document, size=70, overlap=20))

    assert len(result) == 2
    assert result[0].page == 1
    assert result[1].page == 1
    assert "Third paragraph" in result[1].text
