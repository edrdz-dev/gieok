"""Orchestration of the indexing pipeline: load -> chunk -> embed -> persist.

The whole pipeline is a stream. Documents are read lazily, chunked lazily, and pushed to
the embedder in fixed-size batches, so peak memory depends on ``batch_size`` rather than on
corpus size. Indexing a 2 GB folder costs the same RAM as indexing a single file.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gieok.core.chunking import chunk_text
from gieok.core.ports import DocumentLoader, Embedder, VectorStore
from gieok.models import Chunk

ProgressCallback = Callable[[Path, int], None]


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Summary of a completed ingest run.

    A ``dataclass`` rather than a Pydantic model: this value never crosses a trust
    boundary, so validation would be pure overhead. ``slots=True`` drops the per-instance
    ``__dict__``.
    """

    documents: int
    chunks: int


class IngestionService:
    """Indexes documents into a vector store.

    Every collaborator arrives through the constructor. There is no global client and no
    service locator, which is precisely why the tests can drive this class with in-memory
    fakes and no Ollama or Chroma running.
    """

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> None:
        """Wire the pipeline.

        Args:
            loader: Yields documents from a path.
            embedder: Turns chunk text into vectors.
            store: Persists chunks and vectors.
            chunk_size: Maximum chunk length in characters.
            chunk_overlap: Characters carried between consecutive chunks.
            batch_size: Chunks embedded per round trip.
        """
        self._loader = loader
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

    def ingest(
        self,
        root: Path,
        *,
        patterns: Sequence[str],
        on_document: ProgressCallback | None = None,
    ) -> IngestReport:
        """Index every matching document under ``root``.

        Re-running this over unchanged files is a no-op in effect: chunk ids are derived
        from content, so identical chunks overwrite themselves instead of accumulating.

        Args:
            root: File or directory to index.
            patterns: Glob patterns handed to the loader.
            on_document: Optional progress hook, called with each document path and its
                chunk count. Kept as a plain callable so this layer never imports Rich.

        Returns:
            Counts of documents and chunks indexed.

        Raises:
            DocumentNotFoundError: If nothing readable matched.
        """
        documents = 0
        total_chunks = 0
        batch: list[Chunk] = []

        for document in self._loader(root, patterns):
            documents += 1
            document_chunks = 0
            for chunk in chunk_text(
                document.content,
                source=str(document.source),
                size=self._chunk_size,
                overlap=self._chunk_overlap,
            ):
                batch.append(chunk)
                document_chunks += 1
                if len(batch) >= self._batch_size:
                    total_chunks += self._flush(batch)

            if on_document is not None:
                on_document(document.source, document_chunks)

        total_chunks += self._flush(batch)
        return IngestReport(documents=documents, chunks=total_chunks)

    def _flush(self, batch: list[Chunk]) -> int:
        """Embed and persist a batch, then clear it in place.

        Returns:
            The number of chunks written.
        """
        if not batch:
            return 0
        embeddings = self._embedder.embed([chunk.text for chunk in batch])
        self._store.upsert(batch, embeddings)
        written = len(batch)
        batch.clear()
        return written
