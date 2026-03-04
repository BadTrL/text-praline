from textpraline import praline


def _page_block(i: int) -> str:
    return "\n".join(
        [
            "Dataset Build - Internal Header",
            "Confidential - Do Not Share",
            f"Section {i}",
            "Body line one with enough length to look like normal content.",
            "Body line two with enough length to look like normal content.",
            "Body line three with enough length to look like normal content.",
            "Body line four with enough length to look like normal content.",
            "Body line five with enough length to look like normal content.",
            "Body line six with enough length to look like normal content.",
            "Body line seven with enough length to look like normal content.",
            "Page footer - Build 2026",
            "https://example.internal",
        ]
    )


def test_report_false_returns_string():
    out = praline("hello", report=False)
    assert isinstance(out, str)


def test_report_true_returns_tuple_and_core_fields():
    raw = "Hello\u00a0World\u200b\n\ufeffglyph<abc>\n........ 23"
    cleaned, report = praline(raw, report=True)

    assert isinstance(cleaned, str)
    assert report.input_len == len(raw)
    assert report.output_len == len(cleaned)
    assert report.normalized_extracted is True
    assert report.removed_toc_lines == 1
    assert report.text_profile in {"clean_web", "pdf_like", "ocr_like", "unknown"}
    assert report.detail_enabled is False


def test_report_detail_mode_sets_flag():
    cleaned, report = praline("simple input", report="detail")
    assert cleaned == "simple input"
    assert report.detail_enabled is True


def test_report_header_footer_counter_and_compat_counter():
    raw = "\n\n\n".join(_page_block(i) for i in range(1, 7))
    _, report = praline(
        raw, normalize_extracted=True, drop_repeated_lines="on", report=True
    )

    assert report.removed_header_footer_lines > 0
    assert report.removed_repeated_lines == report.removed_header_footer_lines


def test_report_empty_input():
    cleaned, report = praline("", report=True)
    assert cleaned == ""
    assert report.input_len == 0
    assert report.output_len == 0
    assert report.removed_header_footer_lines == 0
