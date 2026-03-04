from __future__ import annotations
import pathlib
import pytest
from textpraline import praline


def _extract_pdf_text(pdf_path: pathlib.Path) -> str:
    pdfminer = pytest.importorskip(
        "pdfminer.high_level",
        reason="pdfminer.six is required for PDF extraction tests. Install with: pip install pdfminer.six",
    )
    return pdfminer.extract_text(str(pdf_path)) or ""


def _extract_pdf_pages(pdf_path: pathlib.Path, max_pages: int = 40) -> list[str]:
    pdfminer = pytest.importorskip(
        "pdfminer.high_level",
        reason="pdfminer.six is required for PDF extraction tests. Install with: pip install pdfminer.six",
    )
    pages: list[str] = []
    for idx in range(max_pages):
        page_text = pdfminer.extract_text(str(pdf_path), page_numbers=[idx]) or ""
        if page_text.strip():
            pages.append(page_text)
    return pages


def test_praline_on_complex_scientific_pdf():
    pdf_path = pathlib.Path("tests/corpus/docu_astro.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Missing fixture: {pdf_path}. Put your PDF there.")

    raw = _extract_pdf_text(pdf_path)
    assert raw.strip(), "PDF extraction returned empty text."

    cleaned = praline(raw)

    # --- Invariants (should never remain) ---
    assert "\ufeff" not in cleaned  # BOM
    assert "\u00ad" not in cleaned  # soft hyphen
    assert "\u200b" not in cleaned  # zero-width space
    assert "\u200c" not in cleaned  # ZWNJ
    assert "\u200d" not in cleaned  # ZWJ
    assert "\u2060" not in cleaned  # word joiner

    # Control chars except \t \n \r
    assert not any((ord(ch) < 32 and ch not in "\t\n\r") for ch in cleaned)

    # PUA range (BMP private use)
    assert not any(0xE000 <= ord(ch) <= 0xF8FF for ch in cleaned)

    # Common broken “mystery” chars from PDF extraction
    # (￾ often renders as U+FFFE; replacement char is U+FFFD)
    assert "\ufffe" not in cleaned
    assert "\ufffd" not in cleaned

    # --- Should preserve scientific signal ---
    # These strings can vary slightly depending on extractor; keep checks flexible.
    assert "ΛCDM" in cleaned or "LCDM" in cleaned
    assert "σ8" in cleaned or "sigma8" in cleaned.lower()

    # Omega may appear as Ω (U+2126) or Ω (U+03A9)
    assert ("Ω" in cleaned) or ("Ω" in cleaned)

    # --- Idempotence (very important for a “standard”) ---
    assert cleaned == praline(cleaned)


def test_footer_removed_on_real_pdf_blocks():
    pdf_path = pathlib.Path("rag_testing/corpus/Morgan_Stanley_2023_ESG_Report.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Missing fixture: {pdf_path}.")

    pages = _extract_pdf_pages(pdf_path, max_pages=40)
    if len(pages) < 6:
        pytest.skip("Not enough extracted pages to validate footer repetition.")

    raw = "\n\n\n".join(pages)
    needle = "MORGAN STANLEY | 2023 ESG REPORT"
    raw_count = raw.count(needle)
    assert (
        raw_count >= 10
    ), "Expected repeated footer not found often enough in raw text."

    cleaned = praline(raw, normalize_extracted=True, drop_repeated_lines="on")
    cleaned_count = cleaned.count(needle)

    assert cleaned_count < raw_count
    assert cleaned_count <= 2
