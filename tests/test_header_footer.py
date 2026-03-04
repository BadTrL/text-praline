from textpraline import praline


def _make_block(i: int) -> str:
    return "\n".join(
        [
            "Acme Research Bulletin - Internal Use Only",
            "Confidential - TextPraline Evaluation Build",
            f"Section {i}",
            (
                "This block contains a normal paragraph with enough words to look "
                "like body text and not a page header."
            ),
            (
                "Another sentence keeps the block long enough so header and footer "
                "detection can use top and bottom positions safely."
            ),
            (
                "A third line makes the synthetic block closer to real extracted "
                "pages where several body lines appear between header and footer."
            ),
            (
                "A fourth body line keeps the block size above conservative "
                "thresholds used for page-like segmentation."
            ),
            (
                "A fifth body line adds stable text so header/footer inference can "
                "measure repeated edge positions across blocks."
            ),
            (
                "A sixth body line ensures we have enough non-empty lines per block "
                "for conservative filtering."
            ),
            (
                "A seventh body line is intentionally plain and should remain after "
                "cleaning."
            ),
            (
                "An eighth body line completes the synthetic page payload for this "
                "test."
            ),
            "Page footer marker - ACME 2026",
            "https://acme.example/internal",
        ]
    )


def test_header_footer_removal_block_aware():
    blocks = [_make_block(i) for i in range(1, 7)]
    text = "\n\n\n".join(blocks)

    cleaned = praline(
        text,
        normalize_extracted=True,
        drop_repeated_lines="on",
        report=False,
    )

    assert "Acme Research Bulletin - Internal Use Only" not in cleaned
    assert "Confidential - TextPraline Evaluation Build" not in cleaned
    assert "Page footer marker - ACME 2026" not in cleaned
    assert "https://acme.example/internal" not in cleaned
    assert "Section 3" in cleaned


def test_header_footer_does_not_remove_toc_like_heading_duplicates():
    body_tail = [
        (
            "Main content paragraph with enough detail to form stable page-like "
            "blocks."
        ),
        (
            "A second body paragraph keeps realistic density and avoids short-line "
            "false positives."
        ),
        "Additional body line one for conservative block sizing.",
        "Additional body line two for conservative block sizing.",
        "Additional body line three for conservative block sizing.",
        "Additional body line four for conservative block sizing.",
        "Additional body line five for conservative block sizing.",
        "Additional body line six for conservative block sizing.",
    ]
    blocks = []
    for i in range(1, 7):
        blocks.append(
            "\n".join(
                [
                    "Project Atlas - Build Snapshot 2026",
                    "Release candidate channel - internal circulation",
                    f"Chapter {i}",
                    "Introduction 4" if i == 2 else f"Body heading {i}",
                    *body_tail,
                    "Printed from ACME publishing backend",
                    "Company confidential - do not distribute",
                ]
            )
        )
    blocks.append(
        "\n".join(
            [
                "Standalone document heading",
                "Introduction 4",
                "This is a real section heading occurrence in body context.",
            ]
        )
    )

    text = "\n\n\n".join(blocks)
    cleaned = praline(text, normalize_extracted=True, drop_repeated_lines="on")

    assert "Introduction 4" in cleaned


def test_dotted_toc_removed_non_dotted_kept():
    raw = "\n".join(
        [
            "Table of Contents",
            "........ 23",
            "Introduction 4",
            "Body starts here.",
        ]
    )

    cleaned_safe = praline(raw, profile="safe")
    cleaned_strict = praline(raw, profile="strict")

    assert "........ 23" not in cleaned_safe
    assert "........ 23" not in cleaned_strict
    assert "Introduction 4" in cleaned_safe
    assert "Introduction 4" in cleaned_strict


def test_clean_web_auto_header_footer_is_safe():
    web_text = (
        "TextPraline focuses on deterministic cleanup for extracted content while "
        "preserving semantics. This paragraph is intentionally long and uses normal "
        "punctuation so it resembles clean web prose rather than PDF extraction.\n\n"
        "A second long paragraph describes how Unicode normalization, layout-noise "
        "detection, and report counters support ingestion quality without adding any "
        "rewriting logic or model-driven transformations in the cleaner itself.\n\n"
        "A third paragraph confirms that automatic repeated-line settings should not "
        "remove content in clean_web scenarios when no extraction-pollution signals "
        "are present."
    )

    cleaned = praline(web_text, drop_repeated_lines="auto")
    assert cleaned == praline(web_text, drop_repeated_lines="off")
