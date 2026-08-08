"""Settings resolution and cross-field validation."""

import pytest
from pydantic import ValidationError

from gieok.config import Settings


def test_defaults_are_sane():
    settings = Settings()
    assert settings.ollama_host.startswith("http")
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k > 0


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("LAE_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LAE_TOP_K", "9")
    monkeypatch.setenv("LAE_CHROMA_PATH", "/tmp/custom-chroma")

    settings = Settings()
    assert settings.chat_model == "qwen2.5:7b"
    assert settings.top_k == 9
    assert str(settings.chroma_path) == "/tmp/custom-chroma"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    ("field", "value"), [("chunk_size", 0), ("top_k", 0), ("chunk_overlap", -1)]
)
def test_non_positive_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})
