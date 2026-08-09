"""Splitting documents into embedding-sized slices.

Pure functions with no I/O and no dependencies beyond the domain models, which makes this
the most heavily unit-tested module in the project. Retrieval quality is decided here:
chunks that straddle a topic boundary poison the index far more than a slow embedder does.

Strategy, in order of preference:

1. Split on blank lines, so paragraphs stay intact.
2. Pack consecutive paragraphs together until adding another would exceed ``size``.
3. Hard-split any single paragraph that is larger than ``size``, preferring word boundaries.
4. Carry the tail of each emitted chunk into the next one, so a sentence sitting on a
   boundary is still retrievable from at least one chunk.

Invariant: every emitted chunk satisfies ``0 < len(chunk.text) <= size``.

Paginated documents (PDFs) are packed the same way, at document level rather than one chunk
per page: page boundaries fall mid-sentence far more often than Markdown's blank lines, so
chunking page-by-page would both mint undersized chunks at every page break and silence the
overlap exactly where it is needed most. Each atom the packer consumes instead carries its
page number, and an emitted chunk is stamped with the page its content *starts* on.
"""

import re
from collections.abc import Iterator

from gieok.models import Chunk, Document

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_JOIN = "\n\n"

type _Segment = tuple[int | None, str]
"""A unit of text paired with the page it came from (``None`` when the source has no pages)."""


def chunk_text(text: str, *, source: str, size: int, overlap: int) -> Iterator[Chunk]:
    """Split ``text`` into overlapping chunks.

    This is a generator rather than a function returning a list: a large corpus is
    streamed straight into the embedder in batches and never fully materialised in memory.

    Args:
        text: Raw document content.
        source: Path of the originating document, recorded on every chunk.
        size: Maximum chunk length in characters.
        overlap: Characters of trailing context carried into the following chunk.

    Yields:
        Chunks in document order, each with a contiguous ``index`` starting at zero.

    Raises:
        ValueError: If ``size`` is not positive, or ``overlap`` is negative or >= ``size``.
    """
    yield from _pack(((None, text),), source=source, size=size, overlap=overlap)


def chunk_document(document: Document, *, size: int, overlap: int) -> Iterator[Chunk]:
    """Split a document, carrying page provenance when the format has pages.

    Args:
        document: The document to split. ``document.pages`` empty means a non-paginated
            format (``.md``, ``.txt``); populated means a paginated one (PDF).
        size: Maximum chunk length in characters.
        overlap: Characters of trailing context carried into the following chunk.

    Yields:
        Chunks in document order. ``Chunk.page`` is set when ``document.pages`` is non-empty.

    Raises:
        ValueError: If ``size`` is not positive, or ``overlap`` is negative or >= ``size``.
    """
    source = str(document.source)
    if not document.pages:
        yield from chunk_text(document.content, source=source, size=size, overlap=overlap)
        return
    yield from _pack(
        tuple((page.number, page.content) for page in document.pages),
        source=source,
        size=size,
        overlap=overlap,
    )


def _pack(
    segments: tuple[_Segment, ...], *, source: str, size: int, overlap: int
) -> Iterator[Chunk]:
    """Pack labelled segments into size-bounded, overlapping chunks.

    The single implementation behind both ``chunk_text`` (one unpaginated segment) and
    ``chunk_document`` (one segment per page): identical packing logic either way, the only
    difference is how many distinct pages the units passed in carry.

    Args:
        segments: ``(page, text)`` pairs to pack, in document order.
        source: Path of the originating document, recorded on every chunk.
        size: Maximum chunk length in characters.
        overlap: Characters of trailing context carried into the following chunk.

    Yields:
        Chunks in document order, each with a contiguous ``index`` starting at zero.

    Raises:
        ValueError: If ``size`` is not positive, or ``overlap`` is negative or >= ``size``.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap} (size={size})")

    buffer = ""
    buffer_page: int | None = None
    previous_page: int | None = None
    index = 0

    for page, unit in _iter_units(segments, size):
        candidate = f"{buffer}{_JOIN}{unit}" if buffer else unit
        if len(candidate) <= size:
            if not buffer:
                buffer_page = page
            buffer = candidate
            previous_page = page
            continue

        # ``candidate`` overflowed, so the buffer is complete as it stands.
        yield Chunk.create(source, index, buffer, page=buffer_page)
        index += 1
        buffer, carried = _carry_over(buffer, unit, size=size, overlap=overlap)
        buffer_page = previous_page if carried else page
        previous_page = page

    if buffer:
        yield Chunk.create(source, index, buffer, page=buffer_page)


def _iter_units(segments: tuple[_Segment, ...], size: int) -> Iterator[_Segment]:
    """Yield the atoms the packer works with: paragraphs, hard-split if oversized.

    Each yielded unit carries the page of the segment it came from; a hard-split paragraph
    never crosses a page boundary, since segments are split before being hard-split.
    """
    for page, text in segments:
        for paragraph in _PARAGRAPH_BREAK.split(text):
            stripped = paragraph.strip()
            if not stripped:
                continue
            if len(stripped) <= size:
                yield page, stripped
            else:
                yield from ((page, piece) for piece in _hard_split(stripped, size))


def _hard_split(paragraph: str, size: int) -> Iterator[str]:
    """Break an oversized paragraph into ``size``-bounded pieces on word boundaries."""
    start = 0
    length = len(paragraph)
    while start < length:
        end = min(start + size, length)
        if end < length:
            # Prefer the last space inside the window so words are not cut in half.
            pivot = paragraph.rfind(" ", start + 1, end)
            if pivot != -1:
                end = pivot
        piece = paragraph[start:end].strip()
        if piece:
            yield piece
        start = end


def _carry_over(buffer: str, unit: str, *, size: int, overlap: int) -> tuple[str, bool]:
    """Seed the next buffer with ``unit``, prefixed by the tail of the previous chunk.

    The overlap is dropped when it would push the new buffer past ``size``. Preserving the
    "every chunk fits in ``size``" invariant matters more than guaranteeing overlap on
    every single boundary.

    Returns:
        The new buffer, and whether text was actually carried over from the previous chunk.
        Returned explicitly rather than inferred from string identity, since that inference
        breaks silently the moment ``unit`` happens to already start with the carried tail.
    """
    if overlap == 0:
        return unit, False
    tail = _tail(buffer, overlap)
    if tail and len(tail) + len(_JOIN) + len(unit) <= size:
        return f"{tail}{_JOIN}{unit}", True
    return unit, False


def _tail(text: str, overlap: int) -> str:
    """Return roughly the last ``overlap`` characters of ``text``, cut at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    pivot = tail.find(" ")
    return tail[pivot + 1 :] if pivot != -1 else tail
