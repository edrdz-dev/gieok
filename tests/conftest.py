"""Shared fixtures and in-memory doubles.

These fakes satisfy the ports in ``core.ports`` structurally -- they inherit nothing and
register nowhere. That is the payoff of Protocol-based interfaces: the entire domain layer
is exercised here with no Ollama daemon and no ChromaDB on disk.
"""

import math

import pytest

from gieok.models import Chunk, RetrievedChunk

DIMENSIONS = 32


class StubEmbedder:
    """Deterministic bag-of-words embedder.

    Not semantically meaningful, but it is stable and gives texts sharing vocabulary a
    higher cosine similarity, which is enough to test retrieval ordering.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return [self._vector(text) for text in batch]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            vector[hash(word) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else [1.0] + [0.0] * (DIMENSIONS - 1)


class InMemoryVectorStore:
    """Dict-backed vector store with brute-force cosine search."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[Chunk, list[float]]] = {}
        self.upsert_calls = 0

    def upsert(self, chunks, embeddings) -> None:
        assert len(chunks) == len(embeddings)
        self.upsert_calls += 1
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.id] = (chunk, list(embedding))

    def query(self, embedding, top_k):
        scored = [
            RetrievedChunk(chunk=chunk, score=_cosine(embedding, vector))
            for chunk, vector in self.records.values()
        ]
        scored.sort(key=lambda retrieved: retrieved.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.records)

    def reset(self) -> None:
        self.records.clear()


class FakeChatModel:
    """Records the prompt it was given and replays a canned answer."""

    def __init__(self, fragments: list[str] | None = None) -> None:
        self.fragments = fragments if fragments is not None else ["Answer ", "from ", "context."]
        self.prompts: list[str] = []

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        yield from self.fragments

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]


def _cosine(left, right) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def chat() -> FakeChatModel:
    return FakeChatModel()
