"""Orchestration of the indexing pipeline: load -> chunk -> embed -> persist.

The whole pipeline is a stream. Documents are read lazily, chunked lazily, and pushed to
the embedder in fixed-size batches, so peak memory depends on ``batch_size`` rather than on
corpus size. Indexing a 2 GB folder costs the same RAM as indexing a single file.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gieok.core.chunking import chunk_document
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
    pruned: int = 0


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
        prune: bool = True,
    ) -> IngestReport:
        """Index every matching document under ``root``.

        Re-running this over unchanged files is a no-op in effect: chunk ids are derived
        from content, so identical chunks overwrite themselves instead of accumulating.

        Deleting or renaming a file is the case content-derived ids cannot handle on their
        own: the old chunks answer to an id nothing will mint again, so they linger and can
        still be cited. ``prune`` closes that gap by dropping indexed sources that this run
        should have seen and did not.

        Args:
            root: File or directory to index.
            patterns: Glob patterns handed to the loader.
            on_document: Optional progress hook, called with each document path and its
                chunk count. Kept as a plain callable so this layer never imports Rich.
            prune: Drop indexed chunks whose source is in scope but no longer on disk.

        Returns:
            Counts of documents indexed, chunks written, and chunks pruned.

        Raises:
            DocumentNotFoundError: If nothing readable matched.
        """
        documents = 0
        total_chunks = 0
        batch: list[Chunk] = []
        seen: set[str] = set()

        for document in self._loader(root, patterns):
            documents += 1
            seen.add(str(document.source))
            document_chunks = 0
            for chunk in chunk_document(
                document,
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
        pruned = self._prune(root, patterns, seen) if prune else 0
        return IngestReport(documents=documents, chunks=total_chunks, pruned=pruned)

    def _prune(self, root: Path, patterns: Sequence[str], seen: set[str]) -> int:
        """Drop indexed sources this run should have produced but did not.

        The scope check is the entire point. "Delete whatever I did not see" empties the
        index the first time someone indexes a subfolder or narrows ``--pattern``: every
        source outside this run's reach looks orphaned and is not. A source counts as
        stale only if it was reachable -- under ``root`` and matching ``patterns`` -- and
        still failed to turn up.

        Args:
            root: The root this run walked.
            patterns: The globs this run matched against.
            seen: Sources the loader yielded during this run.

        Returns:
            The number of chunks removed.
        """
        stale = {
            source
            for source in self._store.sources() - seen
            if _in_scope(Path(source), root, patterns)
        }
        return self._store.delete_sources(stale) if stale else 0

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


def _in_scope(source: Path, root: Path, patterns: Sequence[str]) -> bool:
    """Return True if ``source`` was reachable by a run over ``root`` with ``patterns``.

    Both sides are resolved before comparison, so indexing ``./docs`` and ``/home/me/docs``
    is recognised as the same scope. Without that, the two spellings mint different chunk
    ids for the same file and quietly accumulate a duplicate copy of the corpus, each
    invisible to the other.

    Args:
        source: Path recorded on an indexed chunk.
        root: The root a run walked.
        patterns: The globs that run matched against.

    Returns:
        True if a run over ``root`` could have produced ``source``.
    """
    resolved = source.resolve()
    target = root.resolve()
    within = resolved == target if target.is_file() else resolved.is_relative_to(target)
    return within and any(resolved.match(pattern, case_sensitive=False) for pattern in patterns)
