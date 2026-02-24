import json

from textpraline import praline


def test_citation_line_is_kept_in_debug_decisions():
    citation = "[9] J. Chen, Robust Retrieval under Noise, 2023."
    raw = "\n".join(
        [
            "INTRODUCTION",
            citation,
            "This paper evaluates retrieval quality with controlled corruption.",
        ]
    )

    cleaned, report = praline(
        raw,
        normalize_extracted=True,
        debug_decisions=True,
        report="detail",
        doc_id="citations_doc",
    )

    assert citation in cleaned
    hit = [d for d in report.decisions if d.raw_text == citation]
    assert hit, "Citation line should be present in decisions."
    assert hit[0].action == "keep"
    assert hit[0].category == "other"


def test_toc_navigation_is_dropped_and_labeled():
    toc_nav = "INTRODUCTION  METHODS  RESULTS  DISCUSSION  APPENDICES"
    raw = "\n".join(
        [
            toc_nav,
            "This section explains methods and assumptions in detail.",
            "Another body sentence keeps useful content.",
        ]
    )

    cleaned, report = praline(
        raw,
        normalize_extracted=True,
        debug_decisions=True,
        report="detail",
        doc_id="toc_nav_doc",
    )

    assert toc_nav not in cleaned
    dropped = [
        d
        for d in report.decisions
        if d.raw_text == toc_nav
        and d.action == "drop"
        and d.category == "toc_navigation"
    ]
    assert dropped, "ToC navigation line should be dropped and labeled toc_navigation."


def test_debug_jsonl_contains_exact_dropped_lines_and_indices(tmp_path):
    toc_nav = "INTRODUCTION  METHODS  RESULTS  DISCUSSION"
    dotted = "........ 23"
    raw = "\n".join(
        [
            "Useful title",
            toc_nav,
            "[9] J. Chen, Robust Retrieval under Noise, 2023.",
            dotted,
            "Useful body paragraph.",
        ]
    )

    _, report = praline(
        raw,
        normalize_extracted=True,
        profile="safe",
        debug_decisions=True,
        report="detail",
        doc_id="debug_doc",
    )

    out = tmp_path / "decisions.jsonl"
    report.to_jsonl(out, dropped_only=True)

    rows = []
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    dropped_map = {(r["raw_text"], r["line_idx"]): r for r in rows}
    assert (toc_nav, 1) in dropped_map
    assert (dotted, 3) in dropped_map
    assert dropped_map[(toc_nav, 1)]["category"] == "toc_navigation"
    assert dropped_map[(dotted, 3)]["category"] == "toc"
