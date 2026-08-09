"""Typer entry point and composition root.

This module is the only place that knows which concrete adapters exist. Commands parse
arguments, build the object graph once, delegate, and render -- there is no business logic
below this line.
"""

import functools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from gieok.cli.renderers import (
    console,
    render_error,
    render_skipped,
    render_sources,
    stream_markdown,
)
from gieok.config import Settings
from gieok.core.ingestion import IngestionService
from gieok.core.rag import RagService
from gieok.db.chroma_store import ChromaVectorStore
from gieok.exceptions import LocalAiEngineError, OllamaUnavailableError
from gieok.filesystem.loader import DEFAULT_PATTERNS, SkipCallback, SkippedDocument, iter_documents
from gieok.llm.ollama_client import OllamaClient
from gieok.models import Document

app = typer.Typer(
    name="gieok",
    help="gieok - ask questions about your own documents, entirely offline.",
    no_args_is_help=True,
    add_completion=False,
)


@dataclass(frozen=True, slots=True)
class Services:
    """The assembled object graph for a single command invocation."""

    settings: Settings
    llm: OllamaClient
    store: ChromaVectorStore
    ingestion: IngestionService
    rag: RagService


def build_services(settings: Settings, *, on_skip: SkipCallback | None = None) -> Services:
    """Construct the object graph.

    The composition root: adapters are instantiated here and injected downwards, so no
    module holds a global client and every service can be built with fakes in a test.

    Args:
        settings: Validated runtime configuration.
        on_skip: Called once per file that matched a pattern but could not be indexed.

    Returns:
        The wired services.
    """
    llm = OllamaClient(
        host=settings.ollama_host,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
    store = ChromaVectorStore(
        path=settings.chroma_path,
        collection_name=settings.collection_name,
    )

    def loader(root: Path, patterns: Sequence[str]) -> Iterator[Document]:
        """Close over ``on_skip`` to satisfy ``DocumentLoader`` without widening it.

        A `def`, not `functools.partial`: `partial`'s `__call__` is typed
        `(*args: Any, **kwargs: Any)`, which satisfies any `Protocol` vacuously under
        `mypy --strict` -- a real type-safety loss, not just a style preference. Not a
        `lambda` either (ruff `E731`). The skip channel is wired entirely here, in the
        composition root, so `core/ports.py`, `core/ingestion.py` and `IngestReport`
        never need to know it exists.
        """
        return iter_documents(root, patterns, on_skip=on_skip)

    ingestion = IngestionService(
        loader=loader,
        embedder=llm,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        batch_size=settings.embed_batch_size,
    )
    rag = RagService(embedder=llm, store=store, chat=llm, top_k=settings.top_k)
    return Services(settings=settings, llm=llm, store=store, ingestion=ingestion, rag=rag)


def handle_errors[F: Callable[..., Any]](func: F) -> F:
    """Turn any domain exception into a rendered message and a non-zero exit code.

    A decorator is Python's answer to a cross-cutting concern like Spring's
    ``@ControllerAdvice``: the error boundary is declared once and applied per command.
    ``functools.wraps`` sets ``__wrapped__``, which is what lets Typer still introspect the
    original signature and derive the CLI options from it.

    The ``[F: Callable[...]]`` syntax is PEP 695: a type parameter scoped to this function,
    which replaces the older module-level ``TypeVar`` declaration.

    Args:
        func: The command function to wrap.

    Returns:
        The wrapped command.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return func(*args, **kwargs)
        except LocalAiEngineError as exc:
            render_error(str(exc))
            raise typer.Exit(code=1) from exc

    return wrapper  # type: ignore[return-value]


@app.command()
@handle_errors
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to index.")],
    pattern: Annotated[
        list[str] | None,
        typer.Option(
            "--pattern", "-p", help="Glob to match, repeatable. Default: *.md, *.txt, *.pdf"
        ),
    ] = None,
    reset: Annotated[
        bool, typer.Option("--reset", help="Drop the existing collection before indexing.")
    ] = False,
    prune: Annotated[
        bool,
        typer.Option(
            "--prune/--no-prune",
            help="Drop indexed chunks whose file is gone from the indexed scope.",
        ),
    ] = True,
) -> None:
    """Index documents into the local vector store."""
    skipped: list[SkippedDocument] = []
    services = build_services(Settings(), on_skip=skipped.append)
    patterns = tuple(pattern) if pattern else DEFAULT_PATTERNS

    if reset:
        services.store.reset()
        console.print("[yellow]Collection reset.[/yellow]")

    console.print(f"Indexing [bold]{path}[/bold] (patterns: {', '.join(patterns)})")
    with console.status("[cyan]Embedding...", spinner="dots"):
        report = services.ingestion.ingest(
            path,
            patterns=patterns,
            on_document=lambda source, count: console.print(
                f"  [green]OK[/green] {source} [dim]({count} chunks)[/dim]"
            ),
            prune=prune,
        )
    # Rendered after the `status` block exits, not inside it: painting a panel while the
    # spinner still owns the terminal line is what produces garbled, overlapping output.
    if skipped:
        render_skipped(skipped)

    if report.pruned:
        console.print(
            f"[yellow]Pruned[/yellow] {report.pruned} chunks whose file is no longer there."
        )

    console.print(
        f"\n[bold green]Indexed[/bold green] {report.documents} documents "
        f"-> {report.chunks} chunks. Collection now holds {services.store.count()}."
    )


@app.command()
@handle_errors
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    top_k: Annotated[
        int | None, typer.Option("--top-k", "-k", min=1, help="Chunks to retrieve.")
    ] = None,
    sources: Annotated[
        bool, typer.Option("--sources/--no-sources", help="Show the citation table.")
    ] = True,
) -> None:
    """Answer a question using only the indexed documents."""
    services = build_services(Settings())
    retrieved, stream = services.rag.ask_stream(question, top_k=top_k)
    stream_markdown(stream, title=services.settings.chat_model)
    if sources:
        render_sources(retrieved)


@app.command()
@handle_errors
def status() -> None:
    """Report daemon reachability, resolved models and collection size."""
    settings = Settings()
    services = build_services(settings)

    console.print(f"[bold]Ollama[/bold]      {settings.ollama_host}")
    try:
        installed = services.llm.installed_models()
    except OllamaUnavailableError as exc:
        console.print("  [red]unreachable[/red]")
        # Domain messages are multi-line; keep continuation lines aligned under the label.
        console.print(f"  [dim]{str(exc).replace(chr(10), chr(10) + '  ')}[/dim]")
    else:
        console.print(f"  [green]reachable[/green] - {len(installed)} models installed")
        roles = (
            (settings.embedding_model, "embeddings"),
            (settings.chat_model, "chat"),
        )
        for model, role in roles:
            mark = "[green]OK[/green]" if _is_installed(model, installed) else "[red]missing[/red]"
            console.print(f"  {mark} {model} [dim]({role})[/dim]")

    console.print(f"\n[bold]Collection[/bold]  {settings.collection_name}")
    console.print(f"  path       {settings.chroma_path.resolve()}")
    console.print(f"  chunks     {services.store.count()}")
    console.print(
        f"  chunking   size={settings.chunk_size} overlap={settings.chunk_overlap} "
        f"top_k={settings.top_k}"
    )


def _is_installed(model: str, installed: list[str]) -> bool:
    """Match a configured model against installed names, tolerating an implicit :latest tag."""
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in installed or model in installed


if __name__ == "__main__":
    app()
