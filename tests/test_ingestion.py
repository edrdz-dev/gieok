"""The indexing pipeline, driven entirely by in-memory doubles."""

import pytest

from gieok.core.ingestion import IngestionService
from gieok.exceptions import DocumentNotFoundError
from gieok.filesystem.loader import DEFAULT_PATTERNS, iter_documents


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "one.md").write_text("\n\n".join(f"Alpha paragraph {n}." for n in range(6)))
    (tmp_path / "two.txt").write_text("\n\n".join(f"Bravo paragraph {n}." for n in range(6)))
    return tmp_path


def make_service(embedder, store, *, size=60, overlap=10, batch_size=4):
    return IngestionService(
        loader=iter_documents,
        embedder=embedder,
        store=store,
        chunk_size=size,
        chunk_overlap=overlap,
        batch_size=batch_size,
    )


def test_reports_documents_and_chunks(corpus, embedder, store):
    report = make_service(embedder, store).ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert report.documents == 2
    assert report.chunks > 2
    assert store.count() == report.chunks


def test_embeddings_are_requested_in_batches(corpus, embedder, store):
    service = make_service(embedder, store, batch_size=3)
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert all(len(call) <= 3 for call in embedder.calls)
    assert sum(len(call) for call in embedder.calls) == report.chunks


def test_trailing_partial_batch_is_flushed(corpus, embedder, store):
    # A batch size that cannot divide the chunk count exactly exercises the final flush.
    report = make_service(embedder, store, batch_size=7).ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert store.count() == report.chunks


def test_reingesting_unchanged_documents_is_idempotent(corpus, embedder, store):
    service = make_service(embedder, store)
    first = service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    after_first = store.count()
    second = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert second.chunks == first.chunks
    assert store.count() == after_first, "content-derived ids must overwrite, not duplicate"


def test_editing_a_document_adds_new_chunks(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    before = store.count()

    (corpus / "one.md").write_text("Completely different content now.")
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert store.count() != before


def test_progress_callback_receives_every_document(corpus, embedder, store):
    seen: list[tuple[str, int]] = []
    make_service(embedder, store).ingest(
        corpus,
        patterns=DEFAULT_PATTERNS,
        on_document=lambda source, count: seen.append((source.name, count)),
    )
    assert sorted(name for name, _ in seen) == ["one.md", "two.txt"]
    assert all(count > 0 for _, count in seen)


def test_pattern_filter_is_passed_through(corpus, embedder, store):
    report = make_service(embedder, store).ingest(corpus, patterns=("*.txt",))
    assert report.documents == 1


def test_empty_directory_raises_domain_error(tmp_path, embedder, store):
    with pytest.raises(DocumentNotFoundError):
        make_service(embedder, store).ingest(tmp_path, patterns=DEFAULT_PATTERNS)
