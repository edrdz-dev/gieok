"""Domain model invariants: page-derived documents and chunk citations."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from gieok.models import Chunk, Document, Page

SOURCE = Path("report.pdf")


def test_paginated_joins_page_content_with_a_blank_line():
    pages = [Page(number=1, content="First page."), Page(number=2, content="Second page.")]
    document = Document.paginated(SOURCE, pages)
    assert document.content == "First page.\n\nSecond page."
    assert document.pages == tuple(pages)


def test_a_gap_in_page_numbers_is_accepted():
    # A scanned (skipped) page legitimately produces a gap; only order matters.
    pages = [Page(number=1, content="One."), Page(number=3, content="Three.")]
    document = Document.paginated(SOURCE, pages)
    assert [p.number for p in document.pages] == [1, 3]


def test_non_increasing_page_numbers_are_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        Document(
            source=SOURCE,
            content="x",
            pages=(Page(number=2, content="a"), Page(number=1, content="b")),
        )


def test_duplicate_page_numbers_are_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        Document(
            source=SOURCE,
            content="x",
            pages=(Page(number=1, content="a"), Page(number=1, content="b")),
        )


def test_non_paginated_document_needs_no_page_validation():
    document = Document(source=SOURCE, content="plain text, no pages")
    assert document.pages == ()


def test_citation_is_bare_source_when_there_is_no_page():
    chunk = Chunk.create("notes.md", 0, "some text")
    assert chunk.citation == "notes.md"


def test_citation_includes_the_page_when_present():
    chunk = Chunk.create("report.pdf", 0, "some text", page=12)
    assert chunk.citation == "report.pdf p. 12"


def test_page_is_folded_into_the_id_so_two_pages_with_identical_text_get_different_ids():
    same_text_page_one = Chunk.create("report.pdf", 0, "Repeated boilerplate.", page=1)
    same_text_page_two = Chunk.create("report.pdf", 0, "Repeated boilerplate.", page=2)
    assert same_text_page_one.id != same_text_page_two.id


def test_a_paginated_chunk_id_differs_from_the_unpaginated_one_for_the_same_text():
    # Guards the conditional hash formula itself: page=None must not accidentally collide
    # with, or systematically differ in a fragile way from, a real page number.
    unpaginated = Chunk.create("notes.md", 0, "Same text.")
    paginated = Chunk.create("notes.md", 0, "Same text.", page=1)
    assert unpaginated.id != paginated.id
