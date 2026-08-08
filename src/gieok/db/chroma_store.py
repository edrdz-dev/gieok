"""Adapter for a persistent ChromaDB collection.

Chroma ships its own ONNX embedding model and will happily embed for you. We disable that
(``embedding_function=None``) and supply vectors from Ollama instead, so exactly one model
is responsible for the vector space -- mixing two would silently wreck retrieval quality.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from gieok.models import Chunk, RetrievedChunk


class ChromaVectorStore:
    """Stores chunks and their vectors in an on-disk Chroma collection."""

    def __init__(self, *, path: Path, collection_name: str) -> None:
        """Open (or create) a persistent collection.

        Args:
            path: Directory Chroma persists to.
            collection_name: Name of the collection holding the chunks.
        """
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._create_collection()

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks, keyed by their content-derived id.

        Args:
            chunks: Chunks to persist.
            embeddings: Matching vectors, one per chunk and in the same order.

        Raises:
            ValueError: If the two sequences have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings; they must match."
            )
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            # Metadata is what lets a retrieved vector be cited back to a file and offset.
            metadatas=[{"source": chunk.source, "index": chunk.index} for chunk in chunks],
        )

    def query(self, embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks, most relevant first.

        Args:
            embedding: Query vector.
            top_k: Maximum number of results.

        Returns:
            Retrieved chunks with cosine similarity scores.
        """
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return list(self._to_retrieved_chunks(result))

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._create_collection()

    def _create_collection(self) -> Any:  # noqa: ANN401 - chromadb's Collection is untyped
        """Get or create the backing collection configured for cosine distance."""
        return self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _to_retrieved_chunks(result: Any) -> list[RetrievedChunk]:  # noqa: ANN401
        """Flatten Chroma's batched result shape into domain objects.

        Chroma answers a *batch* of queries, so every field is a list-of-lists. We only
        ever send one query, hence the ``[0]`` indexing.
        """
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return []

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            meta = metadata or {}
            chunk = Chunk(
                id=chunk_id,
                source=str(meta.get("source", "unknown")),
                index=int(meta.get("index", 0)),
                text=text,
            )
            # Chroma reports cosine *distance* in [0, 2]; similarity is its complement.
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved
