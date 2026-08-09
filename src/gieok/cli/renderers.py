"""Rich rendering helpers.

Every ``rich`` import in the project lives in ``cli/``. Keeping presentation concerns here
is what allows the services to report progress through a plain callable and stay unaware
that a terminal exists at all.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gieok.filesystem.loader import SkippedDocument
from gieok.models import RetrievedChunk

console = Console()
error_console = Console(stderr=True)

_SNIPPET_LENGTH = 90


def render_error(message: str) -> None:
    """Print an error panel to stderr."""
    error_console.print(Panel(message, title="Error", border_style="red", expand=False))


def render_sources(sources: Sequence[RetrievedChunk]) -> None:
    """Print a citation table for the chunks an answer was grounded on."""
    if not sources:
        return

    table = Table(title="Sources", title_justify="left", header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Document", overflow="fold")
    table.add_column("Excerpt", overflow="fold")

    for n, retrieved in enumerate(sources, start=1):
        table.add_row(
            str(n),
            f"{retrieved.score:.3f}",
            # A path or excerpt containing a literal `[` would otherwise be interpreted as
            # Rich markup. `Text` renders as plain content, sidestepping that entirely.
            Text(retrieved.chunk.citation),
            Text(_snippet(retrieved.chunk.text)),
        )
    console.print(table)


def render_skipped(entries: Sequence[SkippedDocument]) -> None:
    """Print a warning panel for files that matched a pattern but could not be indexed.

    Grouped by reason rather than listed flat: a directory of a dozen scanned PDFs should
    read as one explanation with a dozen paths under it, not a dozen repeated sentences.

    Args:
        entries: Files skipped during the most recent ingest run.
    """
    if not entries:
        return

    by_reason: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_reason[entry.reason].append(str(entry.path))

    body = Text()
    for reason, paths in by_reason.items():
        if body.plain:
            body.append("\n")
        body.append(f"{reason}\n", style="bold")
        for path in paths:
            body.append(f"  {path}\n")

    console.print(Panel(body, title="Skipped", border_style="yellow", expand=False))


def stream_markdown(fragments: Iterable[str], *, title: str) -> str:
    """Render streaming text as it arrives, re-parsing it as Markdown on each update.

    Args:
        fragments: Text fragments in generation order.
        title: Panel title shown around the answer.

    Returns:
        The complete accumulated text.
    """
    buffer: list[str] = []
    with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        for fragment in fragments:
            buffer.append(fragment)
            live.update(Panel(Markdown("".join(buffer)), title=title, border_style="green"))
    # Live leaves the cursor flush against the panel; separate it from whatever follows.
    console.print()
    return "".join(buffer)


def _snippet(text: str) -> str:
    """Collapse a chunk to a single-line preview."""
    flattened = " ".join(text.split())
    if len(flattened) <= _SNIPPET_LENGTH:
        return flattened
    return f"{flattened[:_SNIPPET_LENGTH]}..."
