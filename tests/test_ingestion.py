"""The indexing pipeline, driven entirely by in-memory doubles."""

from pathlib import Path

import pytest

from gieok.core.ingestion import IngestionService, _fingerprint
from gieok.exceptions import DocumentNotFoundError
from gieok.filesystem.loader import DEFAULT_PATTERNS, iter_documents
from gieok.models import Document, Page


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
    # Content-derived ids made re-ingest idempotent from the start; the fingerprint skip
    # added on top of that means the second run does not even attempt to write anything.
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    after_first = store.count()
    second = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert second.chunks == 0
    assert store.count() == after_first, "content-derived ids must overwrite, not duplicate"


def test_editing_a_document_adds_new_chunks(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    before = store.count()

    (corpus / "one.md").write_text("Completely different content now.")
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert store.count() != before


def test_reingesting_an_unchanged_corpus_embeds_nothing(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    embedder.calls.clear()

    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert sum(len(call) for call in embedder.calls) == 0
    assert report.unchanged == 2
    assert report.documents == 0


def test_a_new_file_is_the_only_thing_embedded_on_the_next_run(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    embedder.calls.clear()

    (corpus / "three.md").write_text("\n\n".join(f"Charlie paragraph {n}." for n in range(6)))
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert report.documents == 1
    assert report.unchanged == 2
    embedded_texts = {text for call in embedder.calls for text in call}
    assert embedded_texts and all("Charlie" in text for text in embedded_texts)


def test_editing_a_file_drops_its_stale_chunks(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    stale_ids = {
        chunk_id for chunk_id, (chunk, _) in store.records.items() if "one.md" in chunk.source
    }

    (corpus / "one.md").write_text("Completely different content now.")
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    surviving_ids = set(store.records)
    assert stale_ids.isdisjoint(surviving_ids), "old chunks of the edited file must not remain"
    assert report.documents == 1


def test_chunks_without_a_fingerprint_are_treated_as_changed_once(corpus, embedder, store):
    # Simulates a collection ingested before this feature existed.
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    for chunk_id, (chunk, vector) in list(store.records.items()):
        store.records[chunk_id] = (chunk.model_copy(update={"fingerprint": None}), vector)

    embedder.calls.clear()
    first = service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert first.documents == 2
    assert sum(len(call) for call in embedder.calls) > 0

    embedder.calls.clear()
    second = service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    assert second.documents == 0
    assert second.unchanged == 2
    assert sum(len(call) for call in embedder.calls) == 0


def test_fingerprint_is_sensitive_to_chunk_size_and_overlap():
    document = Document(source=Path("x.md"), content="Some content that is long enough to matter.")
    baseline = _fingerprint(document, size=800, overlap=150)
    assert baseline != _fingerprint(document, size=400, overlap=150)
    assert baseline != _fingerprint(document, size=800, overlap=50)


def test_fingerprint_distinguishes_paginated_structure_from_flat_content():
    pages = (Page(number=1, content="First."), Page(number=2, content="Second."))
    paginated = Document.paginated(Path("report.pdf"), pages)
    flat = Document(source=Path("report.pdf"), content=paginated.content)
    assert _fingerprint(paginated, size=800, overlap=150) != _fingerprint(
        flat, size=800, overlap=150
    )


def test_changing_chunk_params_forces_a_reembed(corpus, embedder, store):
    make_service(embedder, store, size=60, overlap=10).ingest(corpus, patterns=DEFAULT_PATTERNS)
    embedder.calls.clear()

    report = make_service(embedder, store, size=200, overlap=20).ingest(
        corpus, patterns=DEFAULT_PATTERNS
    )
    assert report.documents == 2
    assert sum(len(call) for call in embedder.calls) > 0


def test_unchanged_file_is_not_pruned(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS)

    assert report.pruned == 0
    sources = store.sources()
    assert any("one.md" in source for source in sources)
    assert any("two.txt" in source for source in sources)


def test_force_reembeds_unchanged_files_without_resetting_the_collection(corpus, embedder, store):
    service = make_service(embedder, store)
    service.ingest(corpus, patterns=DEFAULT_PATTERNS)
    before_sources = store.sources()

    embedder.calls.clear()
    report = service.ingest(corpus, patterns=DEFAULT_PATTERNS, force=True)

    assert report.documents == 2
    assert report.unchanged == 0
    assert sum(len(call) for call in embedder.calls) > 0
    assert store.sources() == before_sources


def test_force_does_not_touch_sources_outside_the_run(tmp_path, embedder, store):
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (inside / "kept.md").write_text("Alpha paragraph, long enough to be its own chunk here.")
    (outside / "untouched.md").write_text("Bravo paragraph, long enough to be its own chunk here.")

    service = make_service(embedder, store)
    service.ingest(outside, patterns=DEFAULT_PATTERNS)
    service.ingest(inside, patterns=DEFAULT_PATTERNS)

    report = service.ingest(inside, patterns=DEFAULT_PATTERNS, force=True)

    assert report.pruned == 0
    assert any("untouched.md" in source for source in store.sources())


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
