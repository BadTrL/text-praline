from __future__ import annotations
import pathlib
import logging
import pytest
from textpraline import praline
from pdfminer.high_level import extract_text


def _extract_pdf_text(pdf_path: pathlib.Path) -> str:
    pdfminer = pytest.importorskip(
        "pdfminer.high_level",
        reason="pdfminer.six is required for PDF extraction tests. Install with: pip install pdfminer.six",
    )
    return pdfminer.extract_text(str(pdf_path)) or ""


def ttest_praline_on_complex_scientific_pdf():
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
    assert not any(
        (ord(ch) < 32 and ch not in "\t\n\r")
        for ch in cleaned
    )

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

def test_output_clean_text():
    pdf_path = pathlib.Path("tests/corpus/docu_astro.pdf")
    raw = extract_text(str(pdf_path))
    output_path_raw = pathlib.Path("tests/corpus/docu_astro_extracted.txt")
    output_path_raw.write_text(raw, encoding="utf-8")

    cleaned = praline(raw)

    output_path = pathlib.Path("tests/corpus/docu_astro_cleaned.txt")
    output_path.write_text(cleaned, encoding="utf-8")

    print("Cleaned file written to:", output_path)