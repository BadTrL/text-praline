import logging
from textpraline import praline

logging.basicConfig(level=logging.INFO)


def test_praline_removes_basic_artifacts():
    raw = "Hello\u00a0World\u200b\n\ufeffglyph<abc>"
    cleaned = praline(raw)

    logging.info(f"result: {raw}")

    assert "\u00a0" not in cleaned  # no NBSP
    assert "\u200b" not in cleaned  # no zero-width
    assert "\ufeff" not in cleaned  # no BOM
    assert "glyph<" not in cleaned  # no glyph artefact
    assert "Hello World" in cleaned


def test_praline_unicode_normalization():
    raw = "ﬁ"  # ligature
    cleaned = praline(raw)
    assert cleaned == "fi"


def test_praline_idempotent():
    raw = "Some   text\n\nwith   spacing"
    once = praline(raw)
    twice = praline(once)
    assert once == twice


def test_praline_markdown_profile_bullets():
    raw = "• Item one\n• Item two"
    cleaned = praline(raw, profile="markdown_safe")
    assert cleaned.startswith("- ")
    assert "- Item one" in cleaned


def test_praline_debug_report():
    raw = "Hello\n....... 23"
    cleaned, report = praline(raw, report=True)

    assert isinstance(report.input_len, int)
    assert report.output_len == len(cleaned)


def test_praline_not_over_aggressive():
    text = "A normal scientific paragraph " * 200
    cleaned = praline(text)
    ratio = len(cleaned) / len(text)
    assert ratio > 0.95


def test_math_not_removed():
    text = "The equation is E = mc^2."
    cleaned = praline(text)
    assert "E = mc^2" in cleaned
