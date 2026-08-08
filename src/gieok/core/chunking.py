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
"""

import re
from collections.abc import Iterator

from gieok.models import Chunk

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_JOIN = "\n\n"


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
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap} (size={size})")

    buffer = ""
    index = 0

    for unit in _iter_units(text, size):
        candidate = f"{buffer}{_JOIN}{unit}" if buffer else unit
        if len(candidate) <= size:
            buffer = candidate
            continue

        # ``candidate`` overflowed, so the buffer is complete as it stands.
        yield Chunk.create(source, index, buffer)
        index += 1
        buffer = _carry_over(buffer, unit, size=size, overlap=overlap)

    if buffer:
        yield Chunk.create(source, index, buffer)


def _iter_units(text: str, size: int) -> Iterator[str]:
    """Yield the atoms the packer works with: paragraphs, hard-split if oversized."""
    for paragraph in _PARAGRAPH_BREAK.split(text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if len(stripped) <= size:
            yield stripped
        else:
            yield from _hard_split(stripped, size)


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


def _carry_over(buffer: str, unit: str, *, size: int, overlap: int) -> str:
    """Seed the next buffer with ``unit``, prefixed by the tail of the previous chunk.

    The overlap is dropped when it would push the new buffer past ``size``. Preserving the
    "every chunk fits in ``size``" invariant matters more than guaranteeing overlap on
    every single boundary.
    """
    if overlap == 0:
        return unit
    tail = _tail(buffer, overlap)
    if tail and len(tail) + len(_JOIN) + len(unit) <= size:
        return f"{tail}{_JOIN}{unit}"
    return unit


def _tail(text: str, overlap: int) -> str:
    """Return roughly the last ``overlap`` characters of ``text``, cut at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    pivot = tail.find(" ")
    return tail[pivot + 1 :] if pivot != -1 else tail
