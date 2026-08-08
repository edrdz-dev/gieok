"""Reading documents off the local filesystem.

Infrastructure: this is the only module in the project that touches the OS filesystem for
document discovery, and it translates ``OSError``/``UnicodeDecodeError`` into domain terms.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

from gieok.exceptions import DocumentNotFoundError
from gieok.models import Document

DEFAULT_PATTERNS: tuple[str, ...] = ("*.md", "*.txt")


def iter_documents(root: Path, patterns: Sequence[str] = DEFAULT_PATTERNS) -> Iterator[Document]:
    """Yield every readable text document under ``root``.

    Accepts either a single file or a directory tree. Hidden directories (``.git``,
    ``.venv``, ...) are skipped, as are files that fail to decode as UTF-8 -- a binary blob
    with a ``.txt`` extension should not abort an otherwise valid ingest run.

    Args:
        root: File or directory to scan.
        patterns: Glob patterns matched against file names.

    Yields:
        One ``Document`` per readable match, in sorted path order for reproducibility.

    Raises:
        DocumentNotFoundError: If ``root`` does not exist, or nothing readable matched.
    """
    patterns = tuple(patterns)
    if not root.exists():
        raise DocumentNotFoundError(str(root), patterns)

    found = False
    for path in _iter_paths(root, patterns):
        content = _read_text(path)
        if content is None or not content.strip():
            continue
        found = True
        yield Document(source=path, content=content)

    if not found:
        raise DocumentNotFoundError(str(root), patterns)


def _iter_paths(root: Path, patterns: tuple[str, ...]) -> Iterator[Path]:
    """Yield candidate file paths, de-duplicated and sorted."""
    if root.is_file():
        if any(root.match(pattern) for pattern in patterns):
            yield root
        return

    # A set collapses the duplicates produced when a file matches several patterns.
    seen = {match for pattern in patterns for match in root.rglob(pattern)}
    yield from sorted(path for path in seen if path.is_file() and not _is_hidden(path, root))


def _is_hidden(path: Path, root: Path) -> bool:
    """Return True if any path segment below ``root`` starts with a dot."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _read_text(path: Path) -> str | None:
    """Read ``path`` as UTF-8, returning None when it is unreadable or not text."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        # Specific exceptions only: a permissions problem and a binary file are both
        # "skip this one", but a KeyboardInterrupt must still propagate.
        return None
