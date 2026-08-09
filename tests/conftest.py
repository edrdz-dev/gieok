"""Shared fixtures and in-memory doubles.

These fakes satisfy the ports in ``core.ports`` structurally -- they inherit nothing and
register nowhere. That is the payoff of Protocol-based interfaces: the entire domain layer
is exercised here with no Ollama daemon and no ChromaDB on disk.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from gieok.models import Chunk, Document, RetrievedChunk

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

    def sources(self) -> set[str]:
        return {chunk.source for chunk, _ in self.records.values()}

    def delete_sources(self, sources) -> int:
        wanted = set(sources)
        doomed = [
            chunk_id for chunk_id, (chunk, _) in self.records.items() if chunk.source in wanted
        ]
        for chunk_id in doomed:
            del self.records[chunk_id]
        return len(doomed)

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


class FakeDocumentLoader:
    """Minimal stand-in for ``core.ports.DocumentLoader``.

    Every test up to this feature wires the real ``iter_documents`` into the service under
    test, so the ``DocumentLoader`` Protocol itself was never exercised by an actual fake --
    only ever satisfied structurally by the production function. This closes that gap: it
    hands back a canned list of documents regardless of ``root``/``patterns``, and records
    every call so a test can assert on what it was asked for.
    """

    def __init__(self, documents: Sequence[Document] | None = None) -> None:
        self.documents = list(documents) if documents is not None else []
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(self, root: Path, patterns: Sequence[str]) -> Iterator[Document]:
        self.calls.append((root, tuple(patterns)))
        yield from self.documents


def build_pdf_bytes(pages: Sequence[str | None]) -> bytes:
    """Hand-roll a minimal PDF with one text page per entry (``None`` for a blank page).

    pypdf can manipulate existing PDFs (``PdfWriter.add_blank_page``, ``encrypt``, ...) but
    has no API to draw text, so there is no supported way to generate a PDF with
    extractable text through it. This writes the handful of objects a reader actually
    needs: a Catalog, a Pages tree, one Page per entry (carrying the Type1/Helvetica font
    resource pypdf needs to decode a ``Tj`` operator), and a ``BT ... Tj ET`` content
    stream per non-``None`` entry -- omitting the stream entirely simulates a scanned page
    with no text layer. Byte offsets are recorded as each object is appended, so the xref
    table is built from measurements instead of hand-computed and prone to drift.
    """
    buf = bytearray(b"%PDF-1.4\n")
    page_nums = list(range(4, 4 + len(pages)))
    next_num = 4 + len(pages)
    content_nums: list[int | None] = []
    for text in pages:
        content_nums.append(next_num if text is not None else None)
        next_num += 1 if text is not None else 0
    total = next_num
    offsets = [0] * total

    def emit(num: int, body: bytes) -> None:
        offsets[num] = len(buf)
        buf.extend(f"{num} 0 obj\n".encode())
        buf.extend(body)
        buf.extend(b"\nendobj\n")

    kids = " ".join(f"{n} 0 R" for n in page_nums)
    emit(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    emit(2, f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    emit(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for text, page_num, content_num in zip(pages, page_nums, content_nums, strict=True):
        if content_num is None:
            emit(
                page_num,
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>",
            )
            continue
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        emit(
            page_num,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>"
            ).encode(),
        )
        emit(
            content_num, f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    xref_start = len(buf)
    buf += f"xref\n0 {total}\n".encode()
    buf += b"0000000000 65535 f \n"
    for num in range(1, total):
        buf += f"{offsets[num]:010d} 00000 n \n".encode()
    buf += f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode()
    return bytes(buf)


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
