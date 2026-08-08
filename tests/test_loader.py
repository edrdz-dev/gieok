"""Filesystem discovery: patterns, hidden directories and undecodable files."""

import pytest

from gieok.exceptions import DocumentNotFoundError
from gieok.filesystem.loader import iter_documents


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "top.md").write_text("top level markdown", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain text notes", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF-1.4 binary")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.md").write_text("nested markdown", encoding="utf-8")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "config.txt").write_text("should be skipped", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    return tmp_path


def sources(documents, root):
    return sorted(str(doc.source.relative_to(root)) for doc in documents)


def test_walks_recursively_and_filters_by_pattern(corpus):
    found = sources(iter_documents(corpus), corpus)
    assert found == ["nested/deep.md", "notes.txt", "top.md"]


def test_hidden_directories_are_skipped(corpus):
    assert not any(".git" in path for path in sources(iter_documents(corpus), corpus))


def test_blank_documents_are_skipped(corpus):
    assert "empty.md" not in sources(iter_documents(corpus), corpus)


def test_custom_patterns_are_honoured(corpus):
    found = sources(iter_documents(corpus, ["*.txt"]), corpus)
    assert found == ["notes.txt"]


def test_a_single_file_can_be_ingested(corpus):
    documents = list(iter_documents(corpus / "top.md"))
    assert len(documents) == 1
    assert documents[0].content == "top level markdown"


def test_undecodable_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "good.md").write_text("readable", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00binary\x80\x81")
    documents = list(iter_documents(tmp_path))
    assert [doc.content for doc in documents] == ["readable"]


def test_missing_path_raises_domain_error(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        list(iter_documents(tmp_path / "nope"))


def test_no_matches_raises_domain_error(tmp_path):
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    with pytest.raises(DocumentNotFoundError):
        list(iter_documents(tmp_path))
