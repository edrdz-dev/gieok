"""Domain exceptions for the local AI engine.

Every failure the user can plausibly act on is represented here. Infrastructure
adapters translate third-party errors (``ollama.ResponseError``, ``httpx.ConnectError``,
``OSError``) into these types, so the presentation layer only ever catches
``LocalAiEngineError`` and never imports a vendor exception.
"""


class LocalAiEngineError(Exception):
    """Base class for every error this application raises deliberately.

    The CLI installs a single handler for this type, so anything inheriting from it
    is rendered as a friendly message instead of a traceback.
    """


class OllamaUnavailableError(LocalAiEngineError):
    """Raised when the Ollama daemon cannot be reached."""

    def __init__(self, host: str) -> None:
        super().__init__(
            f"Cannot reach the Ollama daemon at {host}.\n"
            "Start it with 'ollama serve', or point GIEOK_OLLAMA_HOST at the right address."
        )
        self.host = host


class ModelNotFoundError(LocalAiEngineError):
    """Raised when Ollama does not have the requested model available locally."""

    def __init__(self, model: str) -> None:
        super().__init__(f"The model '{model}' is not available locally. Run: ollama pull {model}")
        self.model = model


class DocumentNotFoundError(LocalAiEngineError):
    """Raised when an ingestion path yields no readable documents."""

    def __init__(self, path: str, patterns: tuple[str, ...]) -> None:
        super().__init__(f"No readable documents found at '{path}' matching {', '.join(patterns)}.")
        self.path = path
        self.patterns = patterns


class UnreadableDocumentsError(DocumentNotFoundError):
    """Raised when matching files were found but none of them could be read.

    Inheriting from ``DocumentNotFoundError`` rather than ``LocalAiEngineError`` directly
    keeps any existing ``except DocumentNotFoundError`` (in this codebase's own tests, or a
    caller's) still catching this too -- it is a more specific instance of "nothing usable
    was found", not an unrelated failure.

    Without this type, a directory of a dozen unreadable PDFs would surface only the
    generic "No readable documents found" message: ``iter_documents`` raises
    ``DocumentNotFoundError`` when its generator is exhausted with nothing yielded, that
    exception is caught and rendered by ``handle_errors`` in ``cli/app.py``, and control
    never reaches the code that would otherwise print the per-file skip reasons. Carrying
    the reasons on the exception itself is the only way they survive to the terminal.
    """

    def __init__(self, path: str, patterns: tuple[str, ...], reasons: tuple[str, ...]) -> None:
        detail = "; ".join(reasons)
        # Bypass DocumentNotFoundError.__init__: its generic message is exactly what this
        # type exists to replace with the per-file reasons.
        LocalAiEngineError.__init__(
            self, f"Found {len(reasons)} file(s) at '{path}' but none could be read: {detail}"
        )
        self.path = path
        self.patterns = patterns
        self.reasons = reasons


class EmptyCorpusError(LocalAiEngineError):
    """Raised when a question is asked before anything has been indexed."""

    def __init__(self) -> None:
        super().__init__(
            "The collection is empty, so there is nothing to ground an answer on.\n"
            "Index some documents first: gieok ingest <path>"
        )


class EmbeddingMismatchError(LocalAiEngineError):
    """Raised when the embedder returns a different number of vectors than texts sent."""

    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"The embedding model returned {received} vectors for {expected} inputs.")
        self.expected = expected
        self.received = received
