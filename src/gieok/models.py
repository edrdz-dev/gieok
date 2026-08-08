"""Immutable domain models shared across every layer.

These are the equivalent of DTOs in a Spring application, with one important
difference: Pydantic *validates and coerces* at construction time, so an object that
exists is by definition well-formed. Every model is frozen (``frozen=True``), which
makes it hashable and safe to pass across layer boundaries without defensive copies.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A single source file loaded from disk, before any splitting."""

    model_config = ConfigDict(frozen=True)

    source: Path = Field(description="Path the document was read from.")
    content: str = Field(description="Full decoded text content.")


class Chunk(BaseModel):
    """A slice of a document, sized to fit comfortably in an embedding window."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Content-derived identifier, stable across runs.")
    source: str = Field(description="Path of the document this slice came from.")
    index: int = Field(ge=0, description="Zero-based position within its document.")
    text: str = Field(min_length=1, description="The slice itself.")

    @classmethod
    def create(cls, source: str, index: int, text: str) -> Chunk:
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

        Returns:
            A frozen ``Chunk`` whose ``id`` is a truncated SHA-256 digest.
        """
        digest = hashlib.sha256(f"{source}|{index}|{text}".encode()).hexdigest()
        return cls(id=digest[:32], source=source, index=index, text=text)


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
