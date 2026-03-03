from textpraline import praline


def test_structure_light_hyphen_fix():
    raw = "This infor-\nmation is important.\nAnother line."
    cleaned = praline(
        raw,
        normalize_extracted=False,
        drop_layout_noise="off",
        structure_mode="light",
    )
    assert "information is important." in cleaned


def test_structure_light_paragraph_merge():
    raw = "This sentence is broken\nacross two lines for extraction\nNew paragraph starts."
    cleaned = praline(
        raw,
        normalize_extracted=False,
        drop_layout_noise="off",
        structure_mode="light",
    )
    assert "This sentence is broken across two lines for extraction" in cleaned


def test_structure_aggressive_section_markers_and_references_block():
    raw = "\n".join(
        [
            "Introduction",
            "This is body content",
            "References",
            "[1] J. Chen, Paper title, 2023.",
            "[2] A. Smith, Another title, 2022.",
        ]
    )
    cleaned = praline(
        raw,
        normalize_extracted=False,
        drop_references_section="off",
        structure_mode="aggressive",
    )
    assert "## SECTION: Introduction" in cleaned
    assert "## REFERENCES" in cleaned
    assert "[1] J. Chen, Paper title, 2023." in cleaned


def test_structure_mode_decision_categories_present():
    raw = "\n".join(
        [
            "Methods",
            "This infor-",
            "mation helps",
            "References",
            "[1] M. Doe, Title, 2024.",
        ]
    )
    _, rep = praline(
        raw,
        normalize_extracted=False,
        drop_references_section="off",
        structure_mode="aggressive",
        debug_decisions=True,
        report="detail",
    )
    cats = {d.category for d in rep.decisions}
    assert "hyphen_fix" in cats
    assert "section_marker_inserted" in cats
