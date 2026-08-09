"""End-to-end pipeline against a real ChromaDB, with only the LLM faked.

Every other test isolates a single unit behind a Protocol. This one wires the genuine
vector store into the genuine services, so it catches the class of bug that in-memory fakes
by construction cannot: a metadata key that does not round-trip, a distance interpreted as
a similarity, a collection configured for the wrong distance function.

Ollama stays stubbed -- the tests must run on a machine with no daemon and no models.
"""

import pytest

from conftest import build_pdf_bytes
from gieok.core.ingestion import IngestionService
from gieok.core.rag import RagService
from gieok.db.chroma_store import ChromaVectorStore
from gieok.filesystem.loader import DEFAULT_PATTERNS, iter_documents

DOCUMENTS = {
    "database.md": "The project stores embeddings in ChromaDB, a local vector database.",
    "cli.md": "The command line interface is built with Typer and rendered using Rich.",
    "models.md": "Generation and embeddings both run on Ollama, entirely on the local host.",
}


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, body in DOCUMENTS.items():
        (docs / name).write_text(body, encoding="utf-8")
    return docs


@pytest.fixture
def chroma(tmp_path):
    return ChromaVectorStore(path=tmp_path / "chroma", collection_name="test")


def index(corpus, chroma, embedder, **kwargs):
    service = IngestionService(
        loader=iter_documents,
        embedder=embedder,
        store=chroma,
        chunk_size=kwargs.get("size", 400),
        chunk_overlap=kwargs.get("overlap", 50),
        batch_size=kwargs.get("batch_size", 2),
    )
    return service.ingest(corpus, patterns=DEFAULT_PATTERNS)


def test_full_pipeline_indexes_and_answers(corpus, chroma, embedder, chat):
    report = index(corpus, chroma, embedder)
    assert report.documents == len(DOCUMENTS)
    assert chroma.count() == report.chunks

    rag = RagService(embedder=embedder, store=chroma, chat=chat, top_k=2)
    answer = rag.ask("Which vector database is used?")

    assert answer.text == "".join(chat.fragments)
    assert len(answer.sources) == 2


def test_metadata_survives_the_round_trip(corpus, chroma, embedder, chat):
    index(corpus, chroma, embedder)
    sources = RagService(embedder=embedder, store=chroma, chat=chat, top_k=3).retrieve("Typer")

    for retrieved in sources:
        assert retrieved.chunk.source.endswith(".md")
        assert retrieved.chunk.index >= 0
        assert retrieved.chunk.text


def test_scores_are_similarities_not_distances(corpus, chroma, embedder, chat):
    index(corpus, chroma, embedder)
    rag = RagService(embedder=embedder, store=chroma, chat=chat, top_k=3)
    sources = rag.retrieve(
        "The command line interface is built with Typer and rendered using Rich."
    )

    assert all(-1.001 <= s.score <= 1.001 for s in sources), "outside cosine similarity range"
    assert [s.score for s in sources] == sorted((s.score for s in sources), reverse=True)
    # Querying with text that is present verbatim must rank that document first.
    assert sources[0].chunk.source.endswith("cli.md")
    assert sources[0].score == pytest.approx(1.0, abs=1e-3)


def test_reingesting_does_not_duplicate_in_chroma(corpus, chroma, embedder):
    first = index(corpus, chroma, embedder)
    index(corpus, chroma, embedder)
    assert chroma.count() == first.chunks


def test_reset_empties_the_persisted_collection(corpus, chroma, embedder):
    index(corpus, chroma, embedder)
    assert chroma.count() > 0

    chroma.reset()
    assert chroma.count() == 0
    # The collection must still be usable after being dropped and recreated.
    assert index(corpus, chroma, embedder).chunks == chroma.count()


def test_page_survives_the_round_trip_for_a_pdf_source(chroma, embedder, chat, tmp_path):
    # A separate, PDF-only corpus rather than adding a PDF to `corpus`: the other tests in
    # this module assert exact document/chunk counts and which file ranks first, and a
    # third source would perturb every one of those.
    docs = tmp_path / "pdf_docs"
    docs.mkdir()
    (docs / "report.pdf").write_bytes(build_pdf_bytes(["Chroma stores the page metadata."]))

    index(docs, chroma, embedder)
    sources = RagService(embedder=embedder, store=chroma, chat=chat, top_k=1).retrieve(
        "page metadata"
    )

    assert sources
    assert sources[0].chunk.source.endswith("report.pdf")
    assert sources[0].chunk.page == 1
    assert sources[0].chunk.citation.endswith("report.pdf p. 1")
