# textpraline/cleaner/clean.py
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from html import unescape
from typing import Iterable, List, Literal, Tuple, Union, Any

from .mappings import PUA_BULLETS, PUA_TRANSLATE_MAP, TRANSLATE_MAP

__all__ = ["praline", "clean_text", "clean_lines", "PralineReport"]

Profile = Literal["safe", "strict", "markdown_safe"]
NormalizeExtracted = Literal[False, True, "auto"]
Toggle = Literal["off", "on", "auto"]

# ---------------------------------------------------------------------------
# Pre-compiled regexes
# ---------------------------------------------------------------------------

# Unicode Private Use Area (BMP)
RE_PUA = re.compile(r"[\uE000-\uF8FF]")

# Control chars except TAB(0x09), LF(0x0A), CR(0x0D)
RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Table-of-contents lines like "....... 23"
RE_TOC_LINE = re.compile(r"\.{3,}\s*\d+\s*$")

# Normalize list bullets at line start
RE_LIST_HEAD = re.compile(r"^\s*(?:•|\*|·|∙|‧|■|▪|●|◦|—|-)\s*")

# Extractor artefacts like: glyph<...>
GLYPH_RUN_RE = re.compile(r"glyph<[^>]*>|glyph<\S*", re.IGNORECASE)
RE_GLYPH_ESCAPED = re.compile(r"glyph&lt;[^&]*&gt;", re.IGNORECASE)

# Heuristic: heavy presence of HTML entities
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z#0-9]+;")

# Extra artefacts frequently seen in extracted text
RE_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060]")  # ZWSP/ZWNJ/ZWJ/WORD JOINER
RE_SOFT_HYPHEN = re.compile(r"\u00AD")
RE_BOM = re.compile(r"\ufeff")
RE_VARIATION_SELECTORS = re.compile(r"[\uFE00-\uFE0F]")

RE_CID = re.compile(r"\(cid:\d+\)")

# Collapse huge blank gaps (e.g. OCR/PDF)
RE_BLANK_GAPS = re.compile(r"\n{3,}")

# Common publisher / arXiv / proofs boilerplate (line-level)
BOILERPLATE_PATTERNS = [
    re.compile(r"^A&A proofs:\s*manuscript no\..*", re.IGNORECASE),
    re.compile(r"^Article number,\s*page\s*\d+\s*of\s*\d+.*", re.IGNORECASE),
    re.compile(r"^arXiv:\s*\d{4}\.\d+(v\d+)?(\s*\[.*\])?.*", re.IGNORECASE),
    re.compile(r"^submitted to.*", re.IGNORECASE),
]

# Keep captions / table mentions even if they look numeric-ish
CAPTION_HINT = re.compile(r"\b(fig\.?|figure|table)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Report (optional)
# ---------------------------------------------------------------------------


@dataclass
class PralineReport:
    """
    Execution report for a TextPraline cleaning run.

    :param input_len: Length of the input string.
    :param output_len: Length of the cleaned output string.
    :param removed_toc_lines: Number of ToC-like dotted lines removed.
    :param normalized_extracted: Whether extraction normalization was applied.
    """

    input_len: int
    output_len: int
    removed_toc_lines: int = 0
    normalized_extracted: bool = False
    removed_layout_noise_lines: int = 0
    removed_repeated_lines: int = 0
    removed_boilerplate_lines: int = 0


# ---------------------------------------------------------------------------
# Extraction pollution detection & normalization
# ---------------------------------------------------------------------------

# Detect common "PDF extraction went wrong" non-characters / replacement chars
RE_REPLACEMENT = re.compile(
    r"[\uFFFD\uFFFE]"
)  # � and ￾-like noncharacters (often seen in bad PDF text)


def _looks_extraction_polluted(s: str) -> bool:
    """
    Heuristically detect extraction pollution (PDF/OCR/HTML artefacts).

    We keep this conservative but practical for real PDFs:
    - raw or HTML-escaped ``glyph<...>`` runs
    - PUA characters
    - many HTML entities
    - replacement/noncharacter markers (U+FFFD, U+FFFE)
    - ``(cid:123)`` markers (common in some PDF→text outputs)

    :param s: Input text.
    :returns: True if text looks polluted by an extractor, else False.
    """
    if not s:
        return False

    if GLYPH_RUN_RE.search(s) or RE_GLYPH_ESCAPED.search(s):
        return True
    if RE_PUA.search(s):
        return True
    if RE_CID.search(s):
        return True
    if RE_REPLACEMENT.search(s):
        return True
    if len(RE_HTML_ENTITY.findall(s)) >= 3:
        return True

    return False


def _strip_extraction_artifacts(s: str) -> str:
    """
    Normalize common artefacts introduced by extractors.

    This step is language-agnostic and aims to preserve meaning while removing
    invisible corruption and extractor noise.

    Operations:
    - HTML entity unescape (``&nbsp;`` etc.)
    - remove ad-hoc glyph runs (``glyph<...>``)
    - remove ``(cid:123)`` markers
    - normalize punctuation via ``TRANSLATE_MAP``
    - Unicode normalization (NFKC)
    - remove PUA leftovers, zero-width chars, soft hyphens, BOM, variation selectors
    - remove disallowed control characters
    - remove replacement/noncharacters (U+FFFD, U+FFFE)

    :param s: Input text.
    :returns: Cleaned text.
    """
    if not s:
        return s

    # Decode HTML entities (e.g., &nbsp;)
    s = unescape(s)

    # Remove extractor-specific glyph runs
    s = GLYPH_RUN_RE.sub("", s)

    # Remove "(cid:NNN)" markers early
    s = RE_CID.sub("", s)

    # Fast PUA bullet mapping to canonical bullet
    if any(ch in s for ch in PUA_BULLETS):
        s = s.translate(str.maketrans(PUA_TRANSLATE_MAP))

    # Basic mapping (quotes/dashes/NBSP etc.)
    s = s.translate(str.maketrans(TRANSLATE_MAP))

    # Unicode normalization: compatibility form handles ligatures/fullwidth
    s = unicodedata.normalize("NFKC", s)

    # Strip residual artefacts
    s = RE_PUA.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    s = RE_SOFT_HYPHEN.sub("", s)
    s = RE_BOM.sub("", s)
    s = RE_VARIATION_SELECTORS.sub("", s)

    # Drop disallowed control chars
    s = RE_CTRL.sub("", s)

    # Drop replacement / noncharacters seen in bad PDF extraction
    s = RE_REPLACEMENT.sub("", s)

    return s


# ---------------------------------------------------------------------------
# Generic PDF layout noise removal (block-based)
# ---------------------------------------------------------------------------


def _is_boilerplate_line(ln: str) -> bool:
    """
    Detect common publisher / arXiv boilerplate lines.

    :param ln: Single line.
    :returns: True if the line looks like boilerplate.
    """
    s = ln.strip()
    if not s:
        return False
    return any(p.match(s) for p in BOILERPLATE_PATTERNS)


def _looks_like_single_char_line(ln: str) -> bool:
    s = ln.strip()
    return len(s) == 1 and s.isalnum()


def _looks_like_axis_noise_line(ln: str) -> bool:
    """
    Catch dense numeric/axis garbage (plots/axes) while keeping captions.
    """
    s = ln.strip()
    if len(s) < 12:
        return False
    if CAPTION_HINT.search(s):
        return False

    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    total = max(len(s), 1)

    ratio_letters = letters / total
    ratio_digits_punct = (digits + punct) / total

    # Dense, compact axis strings (often no spaces)
    if spaces <= 1 and len(s) >= 30 and ratio_digits_punct >= 0.65:
        return True

    return ratio_digits_punct >= 0.75 and ratio_letters <= 0.10


def _drop_layout_noise_blocks(
    lines: List[str],
    *,
    min_single_char_run: int = 8,
    min_axis_run: int = 4,
) -> Tuple[List[str], int]:
    """
    Remove blocks of layout noise common in PDF/OCR extraction:
    - long runs of single-character lines (vertical/rotated headers)
    - consecutive axis/plot-garbage lines

    :param lines: Input lines.
    :param min_single_char_run: Minimum run length to drop (single-char lines).
    :param min_axis_run: Minimum run length to drop (axis-noise lines).
    :returns: (filtered_lines, removed_count)
    """
    out: List[str] = []
    removed = 0

    single_run: List[str] = []
    axis_run: List[str] = []

    def flush_single():
        nonlocal single_run, removed, out
        if not single_run:
            return
        if len(single_run) >= min_single_char_run:
            removed += len(single_run)
        else:
            out.extend(single_run)
        single_run = []

    def flush_axis():
        nonlocal axis_run, removed, out
        if not axis_run:
            return
        if len(axis_run) >= min_axis_run:
            removed += len(axis_run)
        else:
            out.extend(axis_run)
        axis_run = []

    for ln in lines:
        if _looks_like_single_char_line(ln):
            flush_axis()
            single_run.append(ln)
            continue
        flush_single()

        if _looks_like_axis_noise_line(ln):
            axis_run.append(ln)
            continue
        flush_axis()

        out.append(ln)

    flush_single()
    flush_axis()

    return out, removed


def _drop_repeated_lines(
    lines: List[str], *, min_count: int = 5
) -> Tuple[List[str], int]:
    """
    Drop exact repeated lines (header/footer-like), after light normalization.
    Conservative: only removes lines that repeat at least `min_count` times.

    :param lines: Input lines.
    :param min_count: Minimum repetition count to drop.
    :returns: (filtered_lines, removed_count)
    """
    norm = []
    for ln in lines:
        key = re.sub(r"[ \t]+", " ", ln.strip())
        norm.append(key)

    freq: dict[str, int] = {}
    for k in norm:
        if not k:
            continue
        freq[k] = freq.get(k, 0) + 1

    to_drop = {k for k, c in freq.items() if c >= min_count}

    out: List[str] = []
    removed = 0
    for ln, k in zip(lines, norm):
        if k and k in to_drop:
            removed += 1
            continue
        out.append(ln)

    return out, removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(
    s: str,
    *,
    profile: Profile = "safe",
    normalize_extracted: NormalizeExtracted = "auto",
    normalize_form: str = "NFKC",
    preserve_markdown_tables: bool | None = None,
    collapse_blank_lines: bool = True,
    drop_layout_noise: Toggle = "auto",
    drop_repeated_lines: Toggle = "auto",
) -> Tuple[str, PralineReport]:
    """
    Clean a text string for reliable ingestion (chunking, indexing, RAG).

    Profiles:
    - ``safe``: preserve indentation; keep canonical bullet ``•``.
    - ``markdown_safe``: prefer Markdown-friendly list bullets ``-``.
    - ``strict``: aggressive whitespace collapsing.

    :param s: Input text.
    :param profile: Cleaning profile.
    :param normalize_extracted: ``False``/``True``/``"auto"`` for extractor cleanup.
    :param normalize_form: Unicode normalization form, default ``"NFKC"``.
    :param preserve_markdown_tables: Preserve Markdown table spacing (defaults by profile).
    :param collapse_blank_lines: Collapse ``\\n{3,}`` into ``\\n\\n``.
    :param drop_layout_noise: ``off|on|auto`` remove layout-noise blocks (PDF/OCR).
    :param drop_repeated_lines: ``off|on|auto`` remove repeated header/footer-like lines.
    :returns: ``(cleaned_text, report)``.
    """
    if not s:
        return s, PralineReport(input_len=0, output_len=0)

    report = PralineReport(input_len=len(s), output_len=0)

    if preserve_markdown_tables is None:
        preserve_markdown_tables = profile != "strict"

    # 0) extraction normalization (only once)
    do_norm = normalize_extracted is True or (
        normalize_extracted == "auto" and _looks_extraction_polluted(s)
    )
    if do_norm:
        s = _strip_extraction_artifacts(s)
        report.normalized_extracted = True
    else:
        s = s.translate(str.maketrans(TRANSLATE_MAP))
        s = unicodedata.normalize(normalize_form, s)

    # 1) ensure requested normalization
    s = unicodedata.normalize(normalize_form, s)

    # 2) invariant guardrails
    s = RE_PUA.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)

    # 3) line-level processing
    lines = s.splitlines()

    # boilerplate removal is only safe-ish when we are in “extracted/PDF-ish” mode
    if do_norm:
        filtered = []
        for ln in lines:
            if _is_boilerplate_line(ln):
                report.removed_boilerplate_lines += 1
                continue
            filtered.append(ln)
        lines = filtered

    # layout-noise blocks (generic)
    do_layout = (drop_layout_noise == "on") or (drop_layout_noise == "auto" and do_norm)
    if do_layout:
        lines, removed = _drop_layout_noise_blocks(lines)
        report.removed_layout_noise_lines += removed

    # repeated lines (generic headers/footers)
    do_rep = (drop_repeated_lines == "on") or (
        drop_repeated_lines == "auto" and do_norm
    )
    if do_rep:
        lines, removed = _drop_repeated_lines(lines, min_count=5)
        report.removed_repeated_lines += removed

    # ToC dotted lines + list heads
    bullet = "-" if profile == "markdown_safe" else "•"
    out_lines: List[str] = []
    removed_toc = 0

    for ln in lines:
        if profile in ("safe", "strict") and RE_TOC_LINE.search(ln.strip()):
            removed_toc += 1
            continue
        out_lines.append(RE_LIST_HEAD.sub(f"{bullet} ", ln))

    report.removed_toc_lines = removed_toc
    s = "\n".join(out_lines)

    # 4) whitespace normalization
    if profile == "strict":
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" ?\n ?", "\n", s)
        s = s.strip()
    else:
        new_lines: List[str] = []
        for ln in s.splitlines():
            is_table = ln.lstrip().startswith("|") and ("|" in ln.lstrip()[1:])
            if preserve_markdown_tables and is_table:
                new_lines.append(ln.rstrip())
            else:
                new_lines.append(re.sub(r"[ \t]+", " ", ln).rstrip())
        s = "\n".join(new_lines).strip("\n")

    # 5) collapse giant blank gaps (keeps paragraphs)
    if collapse_blank_lines:
        s = RE_BLANK_GAPS.sub("\n\n", s)

    # final guardrails
    s = RE_PUA.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    s = RE_REPLACEMENT.sub("", s)

    report.output_len = len(s)
    return s, report


def praline(
    text: str,
    *,
    profile: Profile = "safe",
    debug: bool = False,
    normalize_extracted: NormalizeExtracted = "auto",
    collapse_blank_lines: bool = True,
    drop_layout_noise: Toggle = "auto",
    drop_repeated_lines: Toggle = "auto",
) -> Union[str, Tuple[str, PralineReport]]:
    """
    One-shot entrypoint: refine any text to be ingestion-ready.

    :param text: Input text.
    :param profile: ``safe|markdown_safe|strict``.
    :param debug: If True, return ``(text, report)``.
    :param normalize_extracted: ``False|True|auto``.
    :param collapse_blank_lines: Collapse ``\\n{3,}`` into ``\\n\\n``.
    :param drop_layout_noise: ``off|on|auto``.
    :param drop_repeated_lines: ``off|on|auto``.
    :returns: Cleaned text (or ``(text, report)`` if debug=True).
    """
    out, rep = clean_text(
        text,
        profile=profile,
        normalize_extracted=normalize_extracted,
        collapse_blank_lines=collapse_blank_lines,
        drop_layout_noise=drop_layout_noise,
        drop_repeated_lines=drop_repeated_lines,
    )
    return (out, rep) if debug else out


def clean_lines(lines: Iterable[str], **kwargs: Any) -> List[str]:
    """
    Clean an iterable of strings with :func:`~textpraline.cleaner.clean.praline`.

    :param lines: Iterable of strings.
    :param kwargs: Forwarded to :func:`~textpraline.cleaner.clean.praline`.
    :returns: List of cleaned strings.
    """
    out: List[str] = []
    for x in lines:
        res = praline(x, **kwargs)
        out.append(res if isinstance(res, str) else res[0])
    return out


def main(argv: List[str] | None = None) -> None:
    """
    CLI entrypoint.

    Usage::

        textpraline < infile.txt > outfile.txt

    :param argv: Optional argv (unused).
    :returns: None.
    """
    data = sys.stdin.read()
    sys.stdout.write(praline(data))
