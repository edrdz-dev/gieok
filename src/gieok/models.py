"""Immutable domain models shared across every layer.

These are the equivalent of DTOs in a Spring application, with one important
difference: Pydantic *validates and coerces* at construction time, so an object that
exists is by definition well-formed. Every model is frozen (``frozen=True``), which
makes it hashable and safe to pass across layer boundaries without defensive copies.
"""

import hashlib
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Page(BaseModel):
    """One extracted page of a paginated document."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(
        ge=1, description="Physical 1-based page number, as printed by the extractor."
    )
    content: str = Field(min_length=1, description="Extracted and normalised page text.")


class Document(BaseModel):
    r"""A single source file loaded from disk, before any splitting.

    ``content`` is the authoritative flat text -- everything downstream of ``core.chunking``
    reads it. ``pages`` is an optional provenance layer: empty for non-paginated formats
    (``.md``, ``.txt``), populated for PDFs so a chunk can be traced back to a page number.
    Pydantic cannot enforce ``content == "\n\n".join(p.content for p in pages)``, so that
    redundancy is accepted and documented here rather than pretended away.
    """

    model_config = ConfigDict(frozen=True)

    source: Path = Field(description="Path the document was read from.")
    content: str = Field(description="Full decoded text content.")
    pages: tuple[Page, ...] = Field(
        default=(), description="Per-page provenance; empty for non-paginated formats."
    )

    @classmethod
    def paginated(cls, source: Path, pages: Sequence[Page]) -> Document:
        """Build a document from extracted pages, deriving flat content from them.

        A second named constructor alongside the implicit one Pydantic already gives every
        model -- the same idiom as ``Chunk.create`` -- for the one call site (PDF loading)
        that has pages before it has flat content.

        Args:
            source: Path the pages were extracted from.
            pages: Extracted pages, in physical page order.

        Returns:
            A ``Document`` whose ``content`` is the pages joined by a blank line.
        """
        return cls(source=source, content="\n\n".join(p.content for p in pages), pages=tuple(pages))

    @model_validator(mode="after")
    def _page_numbers_strictly_increase(self) -> Self:
        """Reject a page sequence that is not strictly increasing.

        Page numbers must stay the *physical* page number, not a contiguous re-count: a PDF
        with a scanned (skipped) page 2 legitimately produces ``(1, 3, 4)``. What must never
        happen is a duplicate or an out-of-order number, since every citation after the gap
        would then point at the wrong page.

        Returns:
            The validated document.

        Raises:
            ValueError: If ``pages`` is non-empty and not strictly increasing by number.
        """
        numbers = [page.number for page in self.pages]
        if any(later <= earlier for earlier, later in pairwise(numbers)):
            raise ValueError(f"page numbers must be strictly increasing, got {numbers}")
        return self


class Chunk(BaseModel):
    """A slice of a document, sized to fit comfortably in an embedding window."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Content-derived identifier, stable across runs.")
    source: str = Field(description="Path of the document this slice came from.")
    index: int = Field(ge=0, description="Zero-based position within its document.")
    text: str = Field(min_length=1, description="The slice itself.")
    page: int | None = Field(
        default=None, description="Physical page this slice starts on, for paginated sources."
    )

    @classmethod
    def create(cls, source: str, index: int, text: str, *, page: int | None = None) -> Chunk:
        """Build a chunk with a deterministic, content-derived id.

        Using a hash of ``(source, index, text)`` as the primary key turns re-indexing
        into an idempotent upsert: unchanged content produces the same id and overwrites
        itself, while edited content produces a new id. A random UUID would silently
        duplicate the corpus on every run.

        A ``@classmethod`` is Python's idiom for a named alternative constructor -- the
        direct counterpart of a static factory method, and the reason this class needs no
        builder.

        Args:
            source: Path of the originating document.
            index: Zero-based position of this slice within the document.
            text: The slice content.
            page: Physical page this slice starts on, or ``None`` for non-paginated sources.

        Returns:
            A frozen ``Chunk`` whose ``id`` is a truncated SHA-256 digest.
        """
        # The page is folded into the hash only when present. Hashing it unconditionally
        # would change every id ever minted for a .md/.txt chunk (they would all hash
        # "...|None" instead of "..."), turning the next `ingest` into a silent full
        # duplication of an existing Chroma collection instead of an upsert.
        material = f"{source}|{index}|{text}" if page is None else f"{source}|{index}|{text}|{page}"
        digest = hashlib.sha256(material.encode()).hexdigest()
        return cls(id=digest[:32], source=source, index=index, text=text, page=page)

    @property
    def citation(self) -> str:
        """Human-readable provenance label: ``report.pdf p. 12`` or ``notes.md``.

        The single definition of this label, so ``cli/renderers.py`` (the sources table)
        and ``core/rag.py`` (the prompt context) can never drift apart on how a chunk is
        cited.
        """
        return f"{self.source} p. {self.page}" if self.page is not None else self.source


class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search, together with its relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Cosine similarity in [-1, 1]; higher is more relevant.")


class Answer(BaseModel):
    """A fully materialised answer and the evidence it was grounded on."""

    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    sources: tuple[RetrievedChunk, ...] = Field(default=())
