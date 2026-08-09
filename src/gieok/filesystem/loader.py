"""Reading documents off the local filesystem.

Infrastructure: this is the only module in the project that touches the OS filesystem for
document discovery. It translates ``OSError``/``UnicodeDecodeError`` (text files) and
``UnreadablePdfError`` (PDFs) into domain terms.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from gieok.exceptions import DocumentNotFoundError, UnreadableDocumentsError
from gieok.filesystem.pdf import UnreadablePdfError, extract_pages
from gieok.models import Document

DEFAULT_PATTERNS: tuple[str, ...] = ("*.md", "*.txt", "*.pdf")


@dataclass(frozen=True, slots=True)
class SkippedDocument:
    """A file that matched a pattern but could not be turned into a ``Document``."""

    path: Path
    reason: str


type SkipCallback = Callable[[SkippedDocument], None]
"""Reports one skipped file at a time.

A one-argument value object rather than ``Callable[[Path, str], None]``: it lets
``on_skip=skips.append`` be handed straight to ``iter_documents`` with no adapter, and lets
the caller group results by reason without reassembling a pair from two loose arguments.
"""


def iter_documents(
    root: Path,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    *,
    on_skip: SkipCallback | None = None,
) -> Iterator[Document]:
    """Yield every matching, readable document under ``root``.

    Accepts either a single file or a directory tree. Hidden directories (``.git``,
    ``.venv``, ...) are skipped. A file that matches a pattern but cannot be turned into
    text -- undecodable bytes for ``.md``/``.txt``, or any of the ways a PDF can fail to
    yield text -- does not abort an otherwise valid ingest run; it is reported through
    ``on_skip`` instead. Blank text files are the one exception and stay silent, as before:
    "this file is empty" is not actionable the way "this file has content nothing can
    reach" is.

    Args:
        root: File or directory to scan.
        patterns: Glob patterns matched against file names.
        on_skip: Called once per file that matched a pattern but failed to load.

    Yields:
        One ``Document`` per readable match, in sorted path order for reproducibility.

    Raises:
        DocumentNotFoundError: If ``root`` does not exist, or nothing matched any pattern
            (or everything that matched was blank).
        UnreadableDocumentsError: If files matched and at least one failed to load, but
            none produced a usable document.
    """
    patterns = tuple(patterns)
    if not root.exists():
        raise DocumentNotFoundError(str(root), patterns)

    found = False
    skipped: list[SkippedDocument] = []
    for path in _iter_paths(root, patterns):
        loaded = _load(path)
        if isinstance(loaded, SkippedDocument):
            skipped.append(loaded)
            if on_skip is not None:
                on_skip(loaded)
            continue
        if not loaded.content.strip():
            continue
        found = True
        yield loaded

    if not found:
        if skipped:
            reasons = tuple(f"{entry.path.name} — {entry.reason}" for entry in skipped)
            raise UnreadableDocumentsError(str(root), patterns, reasons)
        raise DocumentNotFoundError(str(root), patterns)


def _load(path: Path) -> Document | SkippedDocument:
    """Dispatch on suffix and load one file, before any content is decoded.

    Deciding the format from the suffix first -- rather than attempting a UTF-8 decode and
    falling back -- is what keeps a PDF from ever being misread as garbage text: a ``.pdf``
    whose bytes happen to decode as valid UTF-8 must still be extracted as a PDF, not
    indexed as binary noise.
    """
    if path.suffix.lower() == ".pdf":
        try:
            return Document.paginated(path, extract_pages(path))
        except UnreadablePdfError as exc:
            return SkippedDocument(path=path, reason=str(exc))

    content = _read_text(path)
    if content is None:
        return SkippedDocument(path=path, reason="not valid UTF-8 text")
    return Document(source=path, content=content)


def _iter_paths(root: Path, patterns: tuple[str, ...]) -> Iterator[Path]:
    """Yield candidate file paths, de-duplicated and sorted.

    Matching is case-insensitive (``report.PDF``, ``NOTES.TXT``): a ``.suffix.lower()``
    dispatch in ``_load`` without this would insinuate a case-insensitivity that discovery
    itself did not actually provide.
    """
    if root.is_file():
        if any(root.match(pattern, case_sensitive=False) for pattern in patterns):
            yield root
        return

    # A set collapses the duplicates produced when a file matches several patterns.
    seen = {match for pattern in patterns for match in root.rglob(pattern, case_sensitive=False)}
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
