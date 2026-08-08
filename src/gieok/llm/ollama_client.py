"""Adapter for a locally running Ollama daemon.

One class implements two ports (``Embedder`` and ``ChatModel``) because both are backed by
the same HTTP connection and the same failure modes. Its other job is translation: every
``ollama``/``httpx`` exception that escapes this module would leak infrastructure detail
into the domain, so none does.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import httpx
import ollama

from gieok.exceptions import (
    EmbeddingMismatchError,
    ModelNotFoundError,
    OllamaUnavailableError,
)

_NOT_FOUND = 404


class OllamaClient:
    """Embeds text and generates completions using local Ollama models."""

    def __init__(
        self,
        *,
        host: str,
        embedding_model: str,
        chat_model: str,
        client: ollama.Client | None = None,
    ) -> None:
        """Create an adapter bound to a host and a pair of models.

        Args:
            host: Base URL of the Ollama daemon.
            embedding_model: Model used for embeddings, e.g. ``nomic-embed-text``.
            chat_model: Model used for generation, e.g. ``llama3.1:8b``.
            client: Pre-built client, injected by tests. Built from ``host`` when omitted.
        """
        self._host = host
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self._client = client if client is not None else ollama.Client(host=host)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in a single round trip.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per input, in the same order.

        Raises:
            OllamaUnavailableError: If the daemon is unreachable.
            ModelNotFoundError: If the embedding model is not installed.
            EmbeddingMismatchError: If the daemon returns the wrong number of vectors.
        """
        batch = list(texts)
        if not batch:
            return []

        with _translated_errors(host=self._host, model=self._embedding_model):
            response = self._client.embed(model=self._embedding_model, input=batch)

        vectors = [list(vector) for vector in response.embeddings]
        if len(vectors) != len(batch):
            raise EmbeddingMismatchError(expected=len(batch), received=len(vectors))
        return vectors

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream a completion for ``prompt``.

        Args:
            prompt: The fully-built prompt to send.

        Yields:
            Text fragments in generation order.

        Raises:
            OllamaUnavailableError: If the daemon is unreachable.
            ModelNotFoundError: If the chat model is not installed.
        """
        with _translated_errors(host=self._host, model=self._chat_model):
            stream = self._client.chat(
                model=self._chat_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                # Reasoning models put their scratchpad in `thinking` and leave `content`
                # empty until it ends, so without this a model like qwen3.5 renders as a
                # blank panel and can burn its whole budget before answering. Extractive
                # RAG does not need deliberation, and on CPU those tokens are the scarcest
                # resource there is.
                think=False,
            )
            for part in stream:
                fragment = part.message.content
                if fragment:
                    yield fragment

    def installed_models(self) -> list[str]:
        """List the models available on the daemon.

        Returns:
            Model names, e.g. ``["llama3.1:8b", "nomic-embed-text:latest"]``.

        Raises:
            OllamaUnavailableError: If the daemon is unreachable.
        """
        with _translated_errors(host=self._host, model=None):
            response = self._client.list()
        return [model.model for model in response.models if model.model]


@contextmanager
def _translated_errors(*, host: str, model: str | None) -> Iterator[None]:
    """Rewrite Ollama/httpx failures as domain exceptions.

    A context manager rather than a decorator, because it must stay in effect for the whole
    life of a generator: ``stream`` yields from inside the ``with`` block, so a connection
    dropping mid-stream is translated too, not just a failure on the first call.

    ``@contextmanager`` builds one from a generator -- the try/yield/except body below is
    the whole implementation, with no ``__enter__``/``__exit__`` pair to write.

    Args:
        host: Daemon address, used to build a helpful message.
        model: Model being addressed, or None when the call is not model-specific.

    Yields:
        Nothing; the block runs with translation active.

    Raises:
        ModelNotFoundError: If the daemon reports the model is unknown.
        OllamaUnavailableError: If the daemon is unreachable or otherwise failing.
    """
    try:
        yield
    except ollama.ResponseError as exc:
        if model is not None and exc.status_code == _NOT_FOUND:
            raise ModelNotFoundError(model) from exc
        raise OllamaUnavailableError(host) from exc
    except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as exc:
        raise OllamaUnavailableError(host) from exc
