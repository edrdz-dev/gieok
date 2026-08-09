"""PDF extraction: page numbering, the four failure modes, and text normalisation."""

import pytest
from pypdf import PdfWriter

from conftest import build_pdf_bytes
from gieok.filesystem.pdf import UnreadablePdfError, _normalise, extract_pages


def test_extracts_one_page_per_physical_page(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf_bytes(["First page text.", "Second page text."]))

    pages = extract_pages(path)

    assert [p.number for p in pages] == [1, 2]
    assert pages[0].content == "First page text."
    assert pages[1].content == "Second page text."


def test_a_page_with_no_text_is_dropped_but_others_survive(tmp_path):
    path = tmp_path / "mixed.pdf"
    path.write_bytes(build_pdf_bytes(["Real text.", None, "More text."]))

    pages = extract_pages(path)

    # The blank middle page (physical page 2) is dropped, but its neighbours keep their
    # true physical page numbers -- a gap, not a re-count, is exactly what a citation
    # needs to stay accurate.
    assert [p.number for p in pages] == [1, 3]


def test_corrupt_bytes_raise_with_a_not_a_valid_pdf_reason(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4 binary")

    with pytest.raises(UnreadablePdfError, match="not a valid PDF"):
        extract_pages(path)


def test_a_page_with_no_content_stream_at_all_raises_no_extractable_text(tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(build_pdf_bytes([None]))

    with pytest.raises(UnreadablePdfError, match="no extractable text"):
        extract_pages(path)


def test_encrypted_with_an_empty_user_password_is_transparently_readable(tmp_path):
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(build_pdf_bytes(["Confidential but not locked."]))
    writer = PdfWriter(clone_from=plain)
    writer.encrypt(user_password="", owner_password="owner")
    encrypted = tmp_path / "encrypted.pdf"
    with encrypted.open("wb") as handle:
        writer.write(handle)

    pages = extract_pages(encrypted)

    assert pages[0].content == "Confidential but not locked."


def test_encrypted_with_a_real_password_is_reported_not_silently_dropped(tmp_path):
    plain = tmp_path / "plain.pdf"
    plain.write_bytes(build_pdf_bytes(["Locked content."]))
    writer = PdfWriter(clone_from=plain)
    writer.encrypt(user_password="secret")
    encrypted = tmp_path / "locked.pdf"
    with encrypted.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(UnreadablePdfError, match="password-protected"):
        extract_pages(encrypted)


# --- _normalise -------------------------------------------------------------------------


def test_wrapped_justified_lines_are_rejoined_into_a_paragraph():
    text = (
        "This line fills nearly the entire available page width as justified text does\n"
        "and continues right onto a second physical line of the same wrapped paragraph\n"
        "short end."
    )
    result = _normalise(text)
    assert "\n" not in result.strip("\n")
    assert result == (
        "This line fills nearly the entire available page width as justified text does "
        "and continues right onto a second physical line of the same wrapped paragraph "
        "short end."
    )


def test_a_short_line_still_ends_its_paragraph_even_without_a_blank_line_after_it():
    text = (
        "First long full-width paragraph line that fills the page nicely end to end here\n"
        "Short.\n"
        "Second long full-width paragraph line that also fills the page from end to end\n"
        "and its second wrapped line here too."
    )
    result = _normalise(text)
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].endswith("Short.")


def test_end_of_line_hyphenation_is_repaired_for_lowercase_compounds():
    assert _normalise("a well-\nknown result") == "a wellknown result"


def test_hyphenation_repair_only_applies_between_two_lowercase_letters():
    # The character after the line break is uppercase, so this reads as a genuine dash
    # before a new sentence, not a compound word split across lines. Lowercase-only on
    # both sides limits the repair to legitimate cases like "well-\nknown" and leaves this
    # one alone, merged with a space by the paragraph-rejoin step instead of silently
    # fused into one word.
    result = _normalise("A long full width introductory line ending in a dash-\nWord.")
    assert "dashWord" not in result


def test_soft_hyphen_is_removed():
    assert _normalise("soft\xadhyphen") == "softhyphen"


def test_three_or_more_blank_lines_collapse_to_one_paragraph_break():
    assert _normalise("Para one.\n\n\n\n\nPara two.") == "Para one.\n\nPara two."


def test_normalise_of_empty_text_stays_empty():
    assert _normalise("") == ""
    assert _normalise("   \n  \n ") == ""
