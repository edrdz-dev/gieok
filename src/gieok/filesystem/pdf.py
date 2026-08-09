"""Extracting text from PDF documents.

Infrastructure: the only module that understands the PDF container format. Text
normalisation (de-hyphenation, paragraph reconstruction) lives here too, not in
``core/chunking.py``: it is repair of artifacts specific to this extractor, exactly the
same category of concern as ``_read_text``'s UTF-8 decoding in ``loader.py``, not chunking
logic. It stays a pure, independently testable function regardless of where it is called
from.
"""

import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

from gieok.models import Page

# pypdf logs a warning for every malformed structure it tolerates while parsing a damaged
# PDF. Left at its default level, that spam corrupts the CLI's `console.status` spinner
# mid-render. Set once at import time, module-wide -- there is no per-call handle for it.
logging.getLogger("pypdf").setLevel(logging.ERROR)

_SOFT_HYPHEN = "\xad"
_HYPHENATED_LINE_BREAK = re.compile(r"([a-z])-\n([a-z])")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Justified PDF text fills its line box; a line noticeably short of the page's widest line
# is therefore a genuine paragraph end, not just where a word happened not to fit. Invented
# threshold, not (yet) benchmarked in isolation -- see benchmarks/README.md for the retrieval
# numbers this feeds into.
_PARAGRAPH_FILL_RATIO = 0.75


class UnreadablePdfError(Exception):
    """Raised when a PDF cannot be turned into text. Never escapes ``filesystem/``."""


def extract_pages(path: Path) -> list[Page]:
    """Extract normalised text, one ``Page`` per physical page that yielded any.

    ``list[Page] | None`` was rejected as the return type: it collapses four distinct,
    actionable outcomes (not a PDF at all / corrupt / password-protected / opens fine but
    has no text layer) into a single falsy value, and only an exception can carry the
    reason back to the caller.

    Args:
        path: Path to the PDF file.

    Returns:
        One ``Page`` per physical page containing extractable text, in page order. Pages
        with no text (e.g. an embedded image with no OCR layer) are silently dropped;
        the whole document is only rejected if *none* of its pages have any text.

    Raises:
        UnreadablePdfError: The file is not a valid PDF, is password-protected, or has no
            extractable text anywhere (scanned with no text layer -- gieok does not do OCR).
    """
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            # `decrypt("")` covers the common "owner password, empty user password" case
            # publishers use. A falsy `PasswordType.NOT_DECRYPTED` means a real password is
            # required and we do not have one.
            raise UnreadablePdfError("password-protected")
        pages = [
            Page(number=number, content=normalised)
            for number, raw_page in enumerate(reader.pages, start=1)
            if (normalised := _normalise(raw_page.extract_text()))
        ]
    except (PdfReadError, DependencyError, OSError, NotImplementedError) as exc:
        # pypdf parses lazily: `PdfReader(path)` itself can succeed and the failure only
        # surfaces later, while walking `reader.pages` or inside `extract_text()`. Wrapping
        # the whole body rather than just the constructor call is what catches that.
        raise UnreadablePdfError("not a valid PDF") from exc

    if not pages:
        raise UnreadablePdfError("no extractable text (scanned? gieok does not do OCR)")
    return pages


def _normalise(raw: str) -> str:
    """Repair the artifacts pypdf's text extraction leaves behind.

    Applied per page, because paragraph reconstruction needs that page's own line-width
    statistics -- a page mixes badly with its neighbours' typical line length.

    Args:
        raw: Text extracted from a single PDF page.

    Returns:
        Normalised text: consistent line endings, de-hyphenated line-end word breaks,
        wrapped lines rejoined into paragraphs, and excess blank lines collapsed.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text).replace(_SOFT_HYPHEN, "")
    # De-hyphenation must run before paragraph rejoining: it targets the exact pattern
    # ("well-\nknown") that the generic line-joining step below would otherwise turn into
    # "well- known" instead of "wellknown". Lowercase-only on both sides limits the damage
    # to legitimate line-broken compounds rather than an em-dash or a capitalised acronym.
    text = _HYPHENATED_LINE_BREAK.sub(r"\1\2", text)
    text = _rejoin_wrapped_lines(text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _rejoin_wrapped_lines(text: str) -> str:
    """Turn hard-wrapped lines back into paragraphs.

    Without this, a page arrives at ``chunk_text`` as one giant paragraph and falls
    straight through to ``_hard_split`` -- a character-window cut, the worst-quality split
    the chunker offers. A line break is treated as a paragraph break when the following
    line is blank, or when the current line is short relative to the page's widest line
    (justified text fills its box, so a short line signals the paragraph ended); otherwise
    the break is just word-wrap and the lines are joined with a space.
    """
    lines = text.split("\n")
    lengths = [len(line) for line in lines]
    if not any(lengths):
        return text
    threshold = _PARAGRAPH_FILL_RATIO * max(lengths)

    paragraphs: list[str] = []
    current: list[str] = []
    last_index = len(lines) - 1
    for index, line in enumerate(lines):
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line.strip())
        next_is_blank = index == last_index or not lines[index + 1].strip()
        if next_is_blank or len(line) < threshold:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)
