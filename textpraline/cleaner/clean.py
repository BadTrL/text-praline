# textpraline/cleaner/clean.py
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from html import unescape
from typing import Iterable, List, Literal, Tuple, Union, Any

from .mappings import PUA_BULLETS, PUA_TRANSLATE_MAP, TRANSLATE_MAP

__all__ = [
    "praline",
    "clean_text",
    "clean_lines",
    "PralineReport",
    "detect_text_profile",
]

Profile = Literal["safe", "strict", "markdown_safe"]
NormalizeExtracted = Literal[False, True, "auto"]
Toggle = Literal["off", "on", "auto"]
ReportMode = Literal[False, True, "detail"]

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
    text_profile: Literal["clean_web", "pdf_like", "ocr_like", "unknown"] = "unknown"
    detail_enabled: bool = False


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


def detect_text_profile(
    text: str,
) -> Literal["clean_web", "pdf_like", "ocr_like", "unknown"]:
    """
    Heuristic detection of text extraction profile.

    Returns:
        - "clean_web"  : well-formed HTML/RSS extraction
        - "pdf_like"   : typical PDF text extraction artefacts
        - "ocr_like"   : OCR noise patterns
        - "unknown"    : not enough signal
    """
    if not text or len(text) < 200:
        return "unknown"

    lines = text.splitlines()
    total_lines = len(lines)

    short_lines = [line for line in lines if 0 < len(line.strip()) < 40]
    short_line_ratio = len(short_lines) / max(total_lines, 1)
    avg_line_length = sum(len(line) for line in lines) / max(total_lines, 1)

    cid_count = len(re.findall(r"\(cid:\d+\)", text))
    glyph_count = len(re.findall(r"glyph<[^>]+>", text))
    pua_count = sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF)
    page_markers = len(re.findall(r"Page\s+\d+(\s+of\s+\d+)?", text, re.IGNORECASE))
    hyphen_line_breaks = len(re.findall(r"-\n[a-z]", text))

    pdf_score = (
        cid_count * 2
        + glyph_count * 2
        + pua_count * 0.5
        + page_markers * 2
        + hyphen_line_breaks * 1
        + (short_line_ratio > 0.4) * 2
        + (avg_line_length < 80) * 1
    )

    weird_char_ratio = sum(
        1 for c in text if (not c.isprintable()) and (not c.isspace())
    ) / len(text)
    long_char_runs = len(re.findall(r"(.)\1{4,}", text))
    broken_words = len(re.findall(r"\b[a-zA-Z]{1}\s[a-zA-Z]{1}\s[a-zA-Z]{1}\b", text))

    ocr_score = weird_char_ratio * 100 + long_char_runs * 2 + broken_words * 1.5

    long_paragraphs = len([line for line in lines if len(line.strip()) > 120])
    punctuation_ratio = sum(1 for c in text if c in ".?!:,;") / len(text)

    web_score = (
        long_paragraphs * 1.5
        + (punctuation_ratio > 0.01) * 2
        - pdf_score * 0.5
        - ocr_score * 0.5
    )

    if pdf_score > 5 and pdf_score > ocr_score:
        return "pdf_like"
    if ocr_score > 5 and ocr_score > pdf_score:
        return "ocr_like"
    if web_score > 5 and pdf_score < 3 and ocr_score < 3:
        return "clean_web"
    return "unknown"


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


def _pass_detect_profile(s: str, rep: PralineReport) -> Tuple[str, bool]:
    """
    Detect profile and return (text_profile, web_safe).
    """
    text_profile = detect_text_profile(s)
    rep.text_profile = text_profile
    return text_profile, (text_profile == "clean_web")


def _pass_extraction_normalize(
    s: str,
    rep: PralineReport,
    *,
    normalize_extracted: NormalizeExtracted,
    normalize_form: str,
    text_profile: Literal["clean_web", "pdf_like", "ocr_like", "unknown"],
) -> Tuple[str, bool]:
    """
    Normalize extractor artefacts if needed. Returns (text, do_norm).
    """
    do_norm = normalize_extracted is True or (
        normalize_extracted == "auto"
        and (text_profile in ("pdf_like", "ocr_like") or _looks_extraction_polluted(s))
    )
    if do_norm:
        s = _strip_extraction_artifacts(s)
        rep.normalized_extracted = True
    else:
        s = s.translate(str.maketrans(TRANSLATE_MAP))
        s = unicodedata.normalize(normalize_form, s)

    # ensure requested normalization (always)
    s = unicodedata.normalize(normalize_form, s)
    return s, do_norm


def _pass_invariant_guardrails(s: str) -> str:
    """
    Always-safe removals (no semantics changes).
    """
    s = RE_PUA.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    return s


def _pass_drop_boilerplate(
    lines: List[str], rep: PralineReport, *, enabled: bool
) -> List[str]:
    """
    Drop publisher/arXiv boilerplate lines (only when enabled).
    """
    if not enabled:
        return lines

    out: List[str] = []
    for ln in lines:
        if _is_boilerplate_line(ln):
            rep.removed_boilerplate_lines += 1
            continue
        out.append(ln)
    return out


def _pass_drop_layout_noise(
    lines: List[str],
    rep: PralineReport,
    *,
    enabled: bool,
) -> List[str]:
    """
    Drop PDF/OCR layout noise blocks (only when enabled).
    """
    if not enabled:
        return lines
    lines, removed = _drop_layout_noise_blocks(lines)
    rep.removed_layout_noise_lines += removed
    return lines


def _pass_toc_and_bullets(
    lines: List[str],
    rep: PralineReport,
    *,
    profile: Profile,
) -> List[str]:
    """
    Remove ToC dotted lines (safe/strict only) and normalize list heads.
    """
    bullet = "-" if profile == "markdown_safe" else "•"
    out_lines: List[str] = []
    removed_toc = 0

    for ln in lines:
        if profile in ("safe", "strict") and RE_TOC_LINE.search(ln.strip()):
            removed_toc += 1
            continue
        out_lines.append(RE_LIST_HEAD.sub(f"{bullet} ", ln))

    rep.removed_toc_lines = removed_toc
    return out_lines


def _pass_whitespace(
    s: str,
    *,
    profile: Profile,
    preserve_markdown_tables: bool,
) -> str:
    """
    Normalize whitespace according to profile.
    """
    if profile == "strict":
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" ?\n ?", "\n", s)
        return s.strip()

    new_lines: List[str] = []
    for ln in s.splitlines():
        is_table = ln.lstrip().startswith("|") and ("|" in ln.lstrip()[1:])
        if preserve_markdown_tables and is_table:
            new_lines.append(ln.rstrip())
        else:
            new_lines.append(re.sub(r"[ \t]+", " ", ln).rstrip())
    return "\n".join(new_lines).strip("\n")


def _pass_collapse_blank_gaps(s: str, *, enabled: bool) -> str:
    if not enabled:
        return s
    return RE_BLANK_GAPS.sub("\n\n", s)


def _pass_final_guardrails(s: str) -> str:
    s = RE_PUA.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    s = RE_REPLACEMENT.sub("", s)
    return s


# --- clean_text ------------------------------------------------


def clean_text(
    s: str,
    *,
    profile: Profile = "safe",
    normalize_extracted: NormalizeExtracted = "auto",
    normalize_form: str = "NFKC",
    preserve_markdown_tables: bool | None = None,
    collapse_blank_lines: bool = True,
    drop_layout_noise: Toggle = "auto",
    drop_repeated_lines: Toggle = "off",  # IMPORTANT: keep OFF by default
    report: ReportMode = False,
) -> Tuple[str, PralineReport]:
    """
    Clean a text string for reliable ingestion (chunking, indexing, RAG).

    :param report: ``False|True|"detail"`` controls report enrichment.
                  clean_text ALWAYS returns (text, report).
                  (praline() can hide the report for convenience.)
    """
    if not s:
        rep = PralineReport(input_len=0, output_len=0)
        rep.detail_enabled = report == "detail"
        return s, rep

    rep = PralineReport(input_len=len(s), output_len=0)
    rep.detail_enabled = report == "detail"

    if preserve_markdown_tables is None:
        preserve_markdown_tables = profile != "strict"

    # 0) profile detection
    text_profile, web_safe = _pass_detect_profile(s, rep)

    # 1) extraction normalization (+ unicode normalize)
    s, do_norm = _pass_extraction_normalize(
        s,
        rep,
        normalize_extracted=normalize_extracted,
        normalize_form=normalize_form,
        text_profile=text_profile,
    )

    # 2) invariant guardrails
    s = _pass_invariant_guardrails(s)

    # 3) line-level passes
    lines = s.splitlines()

    # Boilerplate removal only when extracted/PDF-ish
    lines = _pass_drop_boilerplate(lines, rep, enabled=do_norm)

    # Layout-noise: OFF for clean web unless forced
    layout_enabled = (drop_layout_noise == "on") or (
        drop_layout_noise == "auto" and do_norm and not web_safe
    )
    lines = _pass_drop_layout_noise(lines, rep, enabled=layout_enabled)

    # Repeated lines: dangerous without page-awareness -> only if forced ON
    if drop_repeated_lines == "on":
        lines, removed = _drop_repeated_lines(lines, min_count=5)
        rep.removed_repeated_lines += removed
    # if "auto" or "off": do nothing

    # ToC dotted lines + bullets normalization
    lines = _pass_toc_and_bullets(lines, rep, profile=profile)
    s = "\n".join(lines)

    # 4) whitespace normalization
    s = _pass_whitespace(
        s,
        profile=profile,
        preserve_markdown_tables=preserve_markdown_tables,
    )

    # 5) collapse blank gaps
    s = _pass_collapse_blank_gaps(s, enabled=collapse_blank_lines)

    # 6) final guardrails
    s = _pass_final_guardrails(s)

    rep.output_len = len(s)
    return s, rep


# --- praline ----------------------------------------------------


def praline(
    text: str,
    *,
    profile: Profile = "safe",
    normalize_extracted: NormalizeExtracted = "auto",
    collapse_blank_lines: bool = True,
    drop_layout_noise: Toggle = "auto",
    drop_repeated_lines: Toggle = "off",
    report: ReportMode = False,
) -> Union[str, Tuple[str, PralineReport]]:
    """
    One-shot entrypoint: refine any text to be ingestion-ready.

    - report=False  -> returns str
    - report=True/"detail" -> returns (str, PralineReport)
    """
    out, rep = clean_text(
        text,
        profile=profile,
        normalize_extracted=normalize_extracted,
        collapse_blank_lines=collapse_blank_lines,
        drop_layout_noise=drop_layout_noise,
        drop_repeated_lines=drop_repeated_lines,
        report=report,
    )
    return (out, rep) if report in (True, "detail") else out


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
