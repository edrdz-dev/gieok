"""Orchestration of the indexing pipeline: load -> chunk -> embed -> persist.

The whole pipeline is a stream. Documents are read lazily, chunked lazily, and pushed to
the embedder in fixed-size batches, so peak memory depends on ``batch_size`` rather than on
corpus size. Indexing a 2 GB folder costs the same RAM as indexing a single file.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gieok.core.chunking import chunk_document
from gieok.core.ports import DocumentLoader, Embedder, VectorStore
from gieok.models import Chunk, Document

ProgressCallback = Callable[[Path, int], None]


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Summary of a completed ingest run.

    A ``dataclass`` rather than a Pydantic model: this value never crosses a trust
    boundary, so validation would be pure overhead. ``slots=True`` drops the per-instance
    ``__dict__``.

    Attributes:
        documents: Files embedded this run, because they were new or their fingerprint
            had changed.
        chunks: Chunks written this run.
        pruned: Chunks removed because their file is gone from the indexed scope.
        unchanged: Files skipped because their fingerprint matched what was already
            stored -- the whole point of this feature, since a skip costs no embedding.
    """

    documents: int
    chunks: int
    pruned: int = 0
    unchanged: int = 0


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
        force: bool = False,
    ) -> IngestReport:
        """Index every matching document under ``root``.

        Embedding, not reading or chunking, is the expensive step on CPU, so this method
        skips it for files that have not changed since the last run: each document is
        fingerprinted (content plus ``chunk_size``/``chunk_overlap``, see ``_fingerprint``)
        and compared against the fingerprint recorded on its stored chunks. A match means
        this file would chunk identically to what is already indexed, so it is left alone
        entirely -- not even re-chunked.

        A changed fingerprint means the file is new or edited. An edited file is the case
        content-derived chunk ids cannot handle on their own: paragraphs that were edited
        away mint no chunk this run, but nothing tells their *old* ids to disappear, so
        they would otherwise linger forever and still be cited. This method closes that
        gap by deleting every existing chunk of a changed source before writing its new
        ones, so a file's indexed content never mixes pre- and post-edit chunks.

        Deleting or renaming a file outright is a related but distinct case that
        fingerprinting does not touch: the file never reappears from the loader at all, so
        there is no "changed" comparison to make. ``prune`` covers that gap by dropping
        indexed sources that this run should have seen and did not.

        Args:
            root: File or directory to index.
            patterns: Glob patterns handed to the loader.
            on_document: Optional progress hook, called with each document path and its
                chunk count. Kept as a plain callable so this layer never imports Rich.
                Not called for a file skipped as unchanged, since it embedded zero chunks
                and reporting that would read as a failed or empty file rather than a skip.
            prune: Drop indexed chunks whose source is in scope but no longer on disk.
            force: Re-embed every matched file even if its fingerprint is unchanged. This
                is not a reset: it still touches only the sources this run matches, and
                still deletes-then-rewrites each one's chunks rather than wiping the whole
                collection.

        Returns:
            Counts of documents (re-)embedded, chunks written, chunks pruned, and files
            skipped as unchanged.

        Raises:
            DocumentNotFoundError: If nothing readable matched.
        """
        documents = 0
        total_chunks = 0
        unchanged = 0
        batch: list[Chunk] = []
        seen: set[str] = set()
        existing = self._store.fingerprints()

        for document in self._loader(root, patterns):
            source = str(document.source)
            # Recorded before any skip decision: a file must never be pruned just because
            # this run decided its content did not need re-embedding.
            seen.add(source)

            fingerprint = _fingerprint(document, size=self._chunk_size, overlap=self._chunk_overlap)
            prior = existing.get(source)
            if not force and prior == fingerprint:
                unchanged += 1
                continue

            if prior is not None:
                # The source was indexed before and its fingerprint just changed (or
                # `force` is re-embedding it anyway): drop its old chunks first so an
                # edit's stale paragraphs cannot linger under ids this run will not mint
                # again. Safe to do immediately even though the new chunks below may sit
                # in a not-yet-flushed cross-document `batch`: this only removes the OLD
                # ids, and the replacement chunks are inserted on a later `_flush`.
                self._store.delete_sources([source])

            documents += 1
            document_chunks = 0
            for chunk in chunk_document(
                document,
                size=self._chunk_size,
                overlap=self._chunk_overlap,
            ):
                batch.append(chunk.model_copy(update={"fingerprint": fingerprint}))
                document_chunks += 1
                if len(batch) >= self._batch_size:
                    total_chunks += self._flush(batch)

            if on_document is not None:
                on_document(document.source, document_chunks)

        total_chunks += self._flush(batch)
        pruned = self._prune(root, patterns, seen) if prune else 0
        return IngestReport(
            documents=documents, chunks=total_chunks, pruned=pruned, unchanged=unchanged
        )

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


def _fingerprint(document: Document, *, size: int, overlap: int) -> str:
    """Digest that changes iff this document's chunk output would change.

    Folds in chunk_size and overlap so that changing either invalidates every
    fingerprint and forces a re-embed -- otherwise a size change would leave the index
    holding a mix of old- and new-sized chunks. For paginated documents the per-page
    structure is hashed (not just the joined content), since chunking reads
    ``document.pages``, and a boundary shift can change chunks without changing the
    joined text.

    This is "Approach A": it still pays the read/extract cost for unchanged files but
    saves the dominant cost, embedding. Skipping extraction too -- hashing raw bytes
    before load -- is a possible future optimization, but it would require widening the
    ``DocumentLoader`` port to expose bytes ahead of parsing, which is not worth it now.

    Args:
        document: The loaded document to fingerprint.
        size: The chunk size this run will use.
        overlap: The chunk overlap this run will use.

    Returns:
        A hex digest stable across runs for the same document and chunk parameters.
    """
    h = hashlib.sha256()
    h.update(f"{size}|{overlap}|".encode())
    if document.pages:
        for page in document.pages:
            h.update(f"{page.number}\x00{page.content}\x00".encode())
    else:
        h.update(document.content.encode())
    return h.hexdigest()
