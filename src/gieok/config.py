"""Application configuration, resolved from the environment.

``BaseSettings`` is the typed counterpart of an externalised ``application.yml``: values
come from environment variables (prefixed ``LAE_``) or a local ``.env`` file, and are
validated once at startup. Misconfiguration therefore fails immediately and loudly rather
than halfway through an ingest run.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the engine.

    Every field can be overridden with an environment variable, e.g.
    ``LAE_CHAT_MODEL=qwen2.5:7b gieok ask "..."``.
    """

    model_config = SettingsConfigDict(env_prefix="LAE_", env_file=".env", extra="ignore")

    ollama_host: str = Field(default="http://localhost:11434")
    # Chosen by measurement on a CPU-only laptop; see "Choosing models" in the README.
    # Several multilingual embedders scored equivalently, so this one wins on speed and
    # size rather than on accuracy. The generator is small and does not emit reasoning.
    embedding_model: str = Field(default="granite-embedding:278m")
    chat_model: str = Field(default="granite4.1:3b")

    chroma_path: Path = Field(default=Path(".chroma"))
    collection_name: str = Field(default="documents")

    chunk_size: int = Field(default=800, gt=0, description="Maximum chunk length in characters.")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters carried between chunks.")
    embed_batch_size: int = Field(default=32, gt=0)
    # 6 rather than 4: on a 106-chunk corpus that lifted recall from 31/44 to 37/44
    # (McNemar p=0.031) for roughly ten extra seconds per question on CPU. Going to 8
    # was not a significant further gain.
    top_k: int = Field(default=6, gt=0, description="Chunks retrieved per question.")

    @model_validator(mode="after")
    def _overlap_must_fit_inside_chunk(self) -> Self:
        """Reject an overlap that would make chunking non-progressing.

        Cross-field invariants belong in an ``after`` validator, which runs once all
        individual fields are populated and typed.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If ``chunk_overlap`` is not strictly smaller than ``chunk_size``.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self
