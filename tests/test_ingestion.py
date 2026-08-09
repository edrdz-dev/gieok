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


def test_deleted_file_is_pruned_from_the_index(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    (corpus / "two.txt").unlink()
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert report.pruned > 0
    assert not any("two.txt" in source for source in store.sources())


def test_renamed_file_does_not_leave_a_duplicate(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    (corpus / "one.md").rename(corpus / "renamed.md")
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    names = {source.rsplit("/", 1)[-1] for source in store.sources()}
    assert names == {"renamed.md", "two.txt"}, "the old path must not survive the rename"


def test_prune_never_reaches_outside_the_indexed_root(tmp_path, embedder, store):
    """Indexing a subfolder must not wipe everything indexed from its siblings."""
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (inside / "kept.md").write_text("Alpha paragraph.")
    (outside / "untouched.md").write_text("Bravo paragraph.")

    service = make_service(embedder, store)
    service.ingest(outside, patterns=DEFAULT_PATTERNS)
    report = service.ingest(inside, patterns=DEFAULT_PATTERNS)

    assert report.pruned == 0
    assert any("untouched.md" in source for source in store.sources())


def test_prune_never_reaches_outside_the_given_patterns(corpus, embedder, store):
    """Narrowing --pattern must not delete what a wider run had indexed."""
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    report = service.ingest(corpus, patterns=["*.md"])

    assert report.pruned == 0
    assert any("two.txt" in source for source in store.sources()), (
        "*.txt was out of scope for this run, so its chunks were not orphaned"
    )


def test_prune_can_be_turned_off(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    (corpus / "two.txt").unlink()
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS, prune=False)

    assert report.pruned == 0
    assert any("two.txt" in source for source in store.sources())


def test_relative_and_absolute_roots_are_the_same_scope(corpus, embedder, store, monkeypatch):
    """The two spellings of one directory must not accumulate duplicate copies."""
    from pathlib import Path

    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    absolute = store.count()

    monkeypatch.chdir(corpus)
    service.ingest(Path(), patterns=DEFAULT_PATTERNS)

    assert store.count() == absolute, "re-indexing via a relative path must replace, not duplicate"
