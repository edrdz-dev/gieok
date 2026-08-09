"""The interfaces the domain layer needs from the outside world.

These are ``typing.Protocol`` classes: *structural* interfaces. An adapter satisfies one
simply by having methods with matching signatures -- there is nothing to inherit and
nothing to register, yet mypy still verifies conformance at check time.

The practical consequence is that ``core`` never imports ``chromadb`` or ``ollama``. The
dependency arrow points inward only, which is what makes the whole domain layer testable
with a handful of in-memory fakes.
"""

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Protocol

from gieok.models import Chunk, Document, RetrievedChunk


class Embedder(Protocol):
    """Turns text into dense vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per input, in the same order.
        """
        ...


class VectorStore(Protocol):
    """Persists chunks alongside their vectors and searches them by similarity."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks keyed by ``Chunk.id``."""
        ...

    def query(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, most relevant first."""
        ...

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        ...

    def sources(self) -> set[str]:
        """Return every distinct ``Chunk.source`` currently stored."""
        ...

    def delete_sources(self, sources: Collection[str]) -> int:
        """Drop every chunk belonging to ``sources``.

        Args:
            sources: Source paths whose chunks should be removed.

        Returns:
            The number of chunks removed.
        """
        ...

    def reset(self) -> None:
        """Drop every chunk in the collection."""
        ...


class ChatModel(Protocol):
    """Generates text, one token at a time."""

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the completion for ``prompt`` as a sequence of text fragments."""
        ...


class DocumentLoader(Protocol):
    """Yields documents from a filesystem location.

    Declaring this as a callable Protocol rather than a one-method class lets a plain
    module-level generator function be injected directly. Python treats functions as
    first-class values, so the single-abstract-method interface Java would require here
    collapses into the function itself.
    """

    def __call__(self, root: Path, patterns: Sequence[str]) -> Iterator[Document]:
        """Yield every matching, readable document under ``root``."""
        ...
