# textpraline/cleaner/clean.py
from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Iterable, List, Literal, Tuple, Union, Any

from .mappings import PUA_BULLETS, PUA_TRANSLATE_MAP, TRANSLATE_MAP

__all__ = [
    "praline",
    "clean_text",
    "clean_lines",
    "LineDecision",
    "PralineReport",
    "detect_text_profile",
]

Profile = Literal["safe", "strict", "markdown_safe"]
NormalizeExtracted = Literal[False, True, "auto"]
Toggle = Literal["off", "on", "auto"]
ReportMode = Literal[False, True, "detail"]
DecisionAction = Literal["keep", "drop"]
DecisionCategory = Literal[
    "toc",
    "toc_navigation",
    "header_footer",
    "boilerplate",
    "layout_noise",
    "other",
]

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
RE_PAGE_MARKER = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
RE_STRONG_SEPARATOR = re.compile(r"^\s*(?:[-_=*]{6,}|#{6,})\s*$")

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
class LineDecision:
    """
    Per-line decision trace emitted by debug mode.
    """

    doc_id: str | None
    page_idx: int
    line_idx: int
    raw_text: str
    normalized_text: str
    action: DecisionAction
    category: DecisionCategory
    confidence: float | None = None
    repeat_count: int | None = None
    edge_hit_ratio: float | None = None
    caps_ratio: float = 0.0
    digit_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    codepoints: List[str] = field(default_factory=list)


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
    removed_header_footer_lines: int = 0
    # Backward-compatible counter name.
    removed_repeated_lines: int = 0
    removed_boilerplate_lines: int = 0
    text_profile: Literal["clean_web", "pdf_like", "ocr_like", "unknown"] = "unknown"
    detail_enabled: bool = False
    decisions: List[LineDecision] = field(default_factory=list)

    def to_jsonl(self, path: str | Path, *, dropped_only: bool = False) -> None:
        """
        Export line-level decisions as JSONL.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for d in self.decisions:
                if dropped_only and d.action != "drop":
                    continue
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


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
    if not text or len(text) < 240:
        return "unknown"

    lines = text.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    total_lines = len(non_empty)
    if total_lines < 8:
        return "unknown"

    short_lines = [line for line in non_empty if len(line) < 45]
    short_line_ratio = len(short_lines) / max(total_lines, 1)
    avg_line_length = sum(len(line) for line in non_empty) / max(total_lines, 1)
    long_paragraphs = [line for line in non_empty if len(line) >= 120]

    cid_count = len(RE_CID.findall(text))
    glyph_count = len(GLYPH_RUN_RE.findall(text))
    glyph_escaped_count = len(RE_GLYPH_ESCAPED.findall(text))
    pua_count = sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF)
    replacement_count = len(RE_REPLACEMENT.findall(text))
    page_markers = len(RE_PAGE_MARKER.findall(text))
    hyphen_line_breaks = len(re.findall(r"-\n[a-z]", text))
    entity_count = len(RE_HTML_ENTITY.findall(text))

    artifact_signals = sum(
        (
            cid_count > 0,
            glyph_count > 0,
            glyph_escaped_count > 0,
            pua_count > 0,
            replacement_count > 0,
            entity_count >= 4,
        )
    )

    weird_char_ratio = sum(
        1 for c in text if (not c.isprintable()) and (not c.isspace())
    ) / len(text)
    long_char_runs = len(re.findall(r"(.)\1{4,}", text))
    broken_words = len(re.findall(r"\b[a-zA-Z]{1}\s[a-zA-Z]{1}\s[a-zA-Z]{1}\b", text))
    single_char_line_ratio = sum(1 for line in non_empty if len(line) == 1) / max(
        total_lines, 1
    )

    web_candidate = (
        len(long_paragraphs) >= 3
        and short_line_ratio <= 0.30
        and avg_line_length >= 85
        and artifact_signals == 0
        and page_markers == 0
        and hyphen_line_breaks == 0
        and weird_char_ratio < 0.0005
        and long_char_runs == 0
        and broken_words <= 1
        and single_char_line_ratio <= 0.02
    )
    if web_candidate:
        return "clean_web"

    pdf_like = (
        artifact_signals >= 2
        or (page_markers >= 2 and short_line_ratio >= 0.35)
        or (artifact_signals >= 1 and hyphen_line_breaks >= 2)
    )
    if pdf_like:
        return "pdf_like"

    ocr_like = (
        (weird_char_ratio >= 0.0025 and short_line_ratio >= 0.45)
        or (long_char_runs >= 3 and short_line_ratio >= 0.40)
        or (broken_words >= 6 and short_line_ratio >= 0.45)
        or (single_char_line_ratio >= 0.10 and short_line_ratio >= 0.55)
    )
    if ocr_like and not web_candidate:
        return "ocr_like"

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


def _normalize_line_key(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())


def _line_ratios(s: str) -> Tuple[float, float, float]:
    if not s:
        return 0.0, 0.0, 0.0
    total = max(len(s), 1)
    caps = sum(1 for ch in s if ch.isalpha() and ch.isupper()) / total
    digit = sum(ch.isdigit() for ch in s) / total
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s) / total
    return caps, digit, punct


def _looks_like_toc_navigation_line(s: str) -> bool:
    """
    Detect ToC navigation banners (e.g. many ALL-CAPS sections separated by large gaps).
    """
    if not s:
        return False
    parts = [p.strip() for p in re.split(r"\s{2,}", s.strip()) if p.strip()]
    if len(parts) < 3:
        return False

    caps_parts = 0
    for p in parts:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9&/\-]*", p)
        if not words:
            continue
        upp = sum(1 for w in words if w.isupper() and len(w) >= 3)
        if upp >= 1:
            caps_parts += 1
    return caps_parts >= 3


def _looks_like_section_title(s: str) -> bool:
    """
    Conservative title-like detector used to avoid destructive removals.
    """
    if CAPTION_HINT.search(s):
        return True
    if RE_TOC_LINE.search(s):
        return True
    if s.endswith((".", "?", "!", ";")):
        return False

    words = s.split()
    if not words or len(words) > 6:
        return False
    if len(s) < 8:
        return True
    if "://" in s:
        return False
    if not words[0][0].isalpha():
        return False
    if any(sep in s for sep in (" | ", " - ", " -- ", " / ")):
        return False

    alpha = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    if alpha == 0:
        return False

    # "Introduction 4", "Related Work", "Methods and Data", etc.
    return (alpha / len(s) >= 0.65) and (digits <= 3) and (len(words) <= 4)


def _split_candidate_blocks(
    lines: List[str], *, block_min_lines: int
) -> List[List[Tuple[int, str]]]:
    """
    Build page-like blocks from text-only signals.
    """
    blocks: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    blank_run = 0

    def flush_current() -> None:
        nonlocal current
        if current:
            non_empty = sum(1 for _, ln in current if ln.strip())
            if non_empty >= block_min_lines:
                blocks.append(current)
        current = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if RE_PAGE_MARKER.match(stripped):
            flush_current()
            current = [(idx, line)]
            blank_run = 0
            continue

        if RE_STRONG_SEPARATOR.match(stripped):
            flush_current()
            blank_run = 0
            continue

        if not stripped:
            blank_run += 1
            # "\n\n\n" between blocks becomes two empty lines in splitlines().
            if blank_run >= 2:
                flush_current()
            elif current:
                current.append((idx, line))
            continue

        blank_run = 0
        current.append((idx, line))

    flush_current()
    return blocks


def _drop_header_footer_lines(
    lines: List[str],
    *,
    min_count: int = 5,
    min_len: int = 12,
    max_len: int = 160,
    top_k: int = 2,
    bottom_k: int = 2,
    block_min_lines: int = 12,
    presence_ratio: float = 0.6,
) -> Tuple[List[str], int]:
    """
    Drop only lines that behave like page headers/footers.

    A line is removable only if it appears across many inferred blocks and mostly
    at top/bottom positions; this avoids global duplicate suppression.
    """
    blocks = _split_candidate_blocks(lines, block_min_lines=block_min_lines)
    if len(blocks) < 2:
        return lines, 0

    n_blocks = len(blocks)
    key_all_blocks: dict[str, set[int]] = {}
    key_edge_blocks: dict[str, set[int]] = {}
    key_edge_indices: dict[str, List[int]] = {}
    key_non_edge_seen: dict[str, bool] = {}

    for block_id, block in enumerate(blocks):
        non_empty = [(idx, ln) for idx, ln in block if ln.strip()]
        if len(non_empty) < block_min_lines:
            continue

        edge_rows = non_empty[:top_k] + non_empty[-bottom_k:]
        edge_keys: set[str] = set()
        all_keys: set[str] = set()

        for _, ln in non_empty:
            key = _normalize_line_key(ln)
            if not key:
                continue
            all_keys.add(key)
            key_all_blocks.setdefault(key, set()).add(block_id)

        for idx, ln in edge_rows:
            key = _normalize_line_key(ln)
            if not key:
                continue

            if len(key) < min_len or len(key) > max_len:
                continue
            if CAPTION_HINT.search(key):
                continue
            if _looks_like_section_title(key):
                continue

            edge_keys.add(key)
            key_edge_blocks.setdefault(key, set()).add(block_id)
            key_edge_indices.setdefault(key, []).append(idx)

        for key in all_keys:
            if key in edge_keys:
                continue
            key_non_edge_seen[key] = True

    removable_indices: set[int] = set()
    for key, edge_blocks in key_edge_blocks.items():
        all_blocks = key_all_blocks.get(key, set())
        if len(all_blocks) < min_count:
            continue
        if len(edge_blocks) / n_blocks < presence_ratio:
            continue
        if len(edge_blocks) / len(all_blocks) < 0.80:
            continue
        if key_non_edge_seen.get(key, False):
            continue

        removable_indices.update(key_edge_indices.get(key, []))

    if not removable_indices:
        return lines, 0

    out: List[str] = []
    removed = 0
    for idx, ln in enumerate(lines):
        if idx in removable_indices:
            removed += 1
            continue
        out.append(ln)
    return out, removed


def _detect_header_footer_candidates(
    lines: List[str],
    *,
    min_count: int = 5,
    min_len: int = 12,
    max_len: int = 160,
    top_k: int = 2,
    bottom_k: int = 2,
    block_min_lines: int = 12,
    presence_ratio: float = 0.6,
) -> Tuple[set[int], dict[int, int], dict[int, float], dict[int, int]]:
    """
    Return candidate local line indices + diagnostic metadata.
    """
    blocks = _split_candidate_blocks(lines, block_min_lines=block_min_lines)
    if len(blocks) < 2:
        return set(), {}, {}, {}

    n_blocks = len(blocks)
    key_all_blocks: dict[str, set[int]] = {}
    key_edge_blocks: dict[str, set[int]] = {}
    key_edge_indices: dict[str, List[int]] = {}
    key_non_edge_seen: dict[str, bool] = {}
    page_idx_by_local: dict[int, int] = {}

    for block_id, block in enumerate(blocks):
        non_empty = [(idx, ln) for idx, ln in block if ln.strip()]
        if len(non_empty) < block_min_lines:
            continue
        for idx, _ in non_empty:
            page_idx_by_local[idx] = block_id

        edge_rows = non_empty[:top_k] + non_empty[-bottom_k:]
        edge_keys: set[str] = set()
        all_keys: set[str] = set()

        for _, ln in non_empty:
            key = _normalize_line_key(ln)
            if not key:
                continue
            all_keys.add(key)
            key_all_blocks.setdefault(key, set()).add(block_id)

        for idx, ln in edge_rows:
            key = _normalize_line_key(ln)
            if not key:
                continue
            if len(key) < min_len or len(key) > max_len:
                continue
            if CAPTION_HINT.search(key):
                continue
            if _looks_like_section_title(key):
                continue
            if _looks_like_toc_navigation_line(key):
                continue

            edge_keys.add(key)
            key_edge_blocks.setdefault(key, set()).add(block_id)
            key_edge_indices.setdefault(key, []).append(idx)

        for key in all_keys:
            if key in edge_keys:
                continue
            key_non_edge_seen[key] = True

    removable_indices: set[int] = set()
    repeat_count_by_local: dict[int, int] = {}
    edge_ratio_by_local: dict[int, float] = {}
    for key, edge_blocks in key_edge_blocks.items():
        all_blocks = key_all_blocks.get(key, set())
        if len(all_blocks) < min_count:
            continue
        if len(edge_blocks) / n_blocks < presence_ratio:
            continue
        if len(edge_blocks) / len(all_blocks) < 0.80:
            continue
        if key_non_edge_seen.get(key, False):
            continue

        edge_ratio = len(edge_blocks) / n_blocks
        for idx in key_edge_indices.get(key, []):
            removable_indices.add(idx)
            repeat_count_by_local[idx] = len(all_blocks)
            edge_ratio_by_local[idx] = edge_ratio

    return (
        removable_indices,
        repeat_count_by_local,
        edge_ratio_by_local,
        page_idx_by_local,
    )


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
    Normalize extractor artefacts if needed using cleanup-only transforms.
    Returns (text, do_norm).
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
    Remove dotted ToC lines (Table of Contents, safe/strict only) and
    normalize list heads. Non-dotted ToC entries are intentionally kept.
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


def _pass_drop_header_footer(
    lines: List[str],
    rep: PralineReport,
    *,
    enabled: bool,
) -> List[str]:
    """
    Drop only high-confidence header/footer lines.
    """
    if not enabled:
        return lines
    lines, removed = _drop_header_footer_lines(lines)
    rep.removed_header_footer_lines += removed
    rep.removed_repeated_lines += removed
    return lines


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


@dataclass
class _WorkingLine:
    line_idx: int
    raw_text: str
    normalized_text: str
    page_idx: int = -1
    action: DecisionAction = "keep"
    category: DecisionCategory = "other"
    confidence: float | None = None
    repeat_count: int | None = None
    edge_hit_ratio: float | None = None


def _drop_layout_noise_records(
    records: List[_WorkingLine],
) -> Tuple[List[_WorkingLine], int]:
    out: List[_WorkingLine] = []
    removed = 0
    single_run: List[_WorkingLine] = []
    axis_run: List[_WorkingLine] = []

    def flush_single() -> None:
        nonlocal single_run, removed
        if not single_run:
            return
        if len(single_run) >= 8:
            for rec in single_run:
                rec.action = "drop"
                rec.category = "layout_noise"
                rec.confidence = 0.82
                removed += 1
        else:
            out.extend(single_run)
        single_run = []

    def flush_axis() -> None:
        nonlocal axis_run, removed
        if not axis_run:
            return
        if len(axis_run) >= 4:
            for rec in axis_run:
                rec.action = "drop"
                rec.category = "layout_noise"
                rec.confidence = 0.82
                removed += 1
        else:
            out.extend(axis_run)
        axis_run = []

    for rec in records:
        ln = rec.normalized_text
        if _looks_like_single_char_line(ln):
            flush_axis()
            single_run.append(rec)
            continue
        flush_single()

        if _looks_like_axis_noise_line(ln):
            axis_run.append(rec)
            continue
        flush_axis()
        out.append(rec)

    flush_single()
    flush_axis()
    return out, removed


def _build_line_decisions(
    raw_line_texts: List[str],
    lines: List[str],
    *,
    rep: PralineReport,
    profile: Profile,
    do_norm: bool,
    web_safe: bool,
    layout_enabled: bool,
    header_footer_enabled: bool,
    toc_navigation_enabled: bool,
    debug_decisions: bool,
    doc_id: str | None,
) -> Tuple[List[str], List[LineDecision]]:
    records: List[_WorkingLine] = []
    for idx, ln in enumerate(lines):
        raw = raw_line_texts[idx] if idx < len(raw_line_texts) else ln
        records.append(_WorkingLine(line_idx=idx, raw_text=raw, normalized_text=ln))

    page_idx_by_local: dict[int, int] = {}
    for block_id, block in enumerate(
        _split_candidate_blocks(lines, block_min_lines=12)
    ):
        for local_idx, _ in block:
            page_idx_by_local[local_idx] = block_id
    for rec in records:
        rec.page_idx = page_idx_by_local.get(rec.line_idx, -1)

    active = records

    if do_norm:
        kept: List[_WorkingLine] = []
        for rec in active:
            if _is_boilerplate_line(rec.normalized_text):
                rec.action = "drop"
                rec.category = "boilerplate"
                rec.confidence = 0.90
                rep.removed_boilerplate_lines += 1
            else:
                kept.append(rec)
        active = kept

    if layout_enabled:
        active, removed = _drop_layout_noise_records(active)
        rep.removed_layout_noise_lines += removed

    if toc_navigation_enabled:
        kept = []
        for rec in active:
            if _looks_like_toc_navigation_line(rec.normalized_text):
                rec.action = "drop"
                rec.category = "toc_navigation"
                rec.confidence = 0.88
            else:
                kept.append(rec)
        active = kept

    if header_footer_enabled and active:
        active_lines = [r.normalized_text for r in active]
        removable, rep_count, edge_ratio, page_map = _detect_header_footer_candidates(
            active_lines
        )
        kept = []
        for local_idx, rec in enumerate(active):
            if local_idx in page_map:
                rec.page_idx = page_map[local_idx]
            if local_idx in removable:
                rec.action = "drop"
                rec.category = "header_footer"
                rec.confidence = 0.85
                rec.repeat_count = rep_count.get(local_idx)
                rec.edge_hit_ratio = edge_ratio.get(local_idx)
                rep.removed_header_footer_lines += 1
                rep.removed_repeated_lines += 1
            else:
                kept.append(rec)
        active = kept

    bullet = "-" if profile == "markdown_safe" else "•"
    kept_final: List[_WorkingLine] = []
    for rec in active:
        s = rec.normalized_text
        if profile in ("safe", "strict") and RE_TOC_LINE.search(s.strip()):
            rec.action = "drop"
            rec.category = "toc"
            rec.confidence = 0.98
            rep.removed_toc_lines += 1
            continue
        rec.normalized_text = RE_LIST_HEAD.sub(f"{bullet} ", s)
        kept_final.append(rec)

    out_lines = [r.normalized_text for r in kept_final]
    if not debug_decisions:
        return out_lines, []

    decisions: List[LineDecision] = []
    for rec in records:
        caps, digit, punct = _line_ratios(rec.normalized_text)
        decisions.append(
            LineDecision(
                doc_id=doc_id,
                page_idx=rec.page_idx,
                line_idx=rec.line_idx,
                raw_text=rec.raw_text,
                normalized_text=rec.normalized_text,
                action=rec.action,
                category=rec.category if rec.action == "drop" else "other",
                confidence=rec.confidence,
                repeat_count=rec.repeat_count,
                edge_hit_ratio=rec.edge_hit_ratio,
                caps_ratio=caps,
                digit_ratio=digit,
                punctuation_ratio=punct,
                codepoints=[f"U+{ord(ch):04X}" for ch in rec.raw_text],
            )
        )
    return out_lines, decisions


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
    debug_decisions: bool = False,
    doc_id: str | None = None,
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

    original_text = s
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

    # 3) line-level passes + optional debug decisions
    lines = s.splitlines()
    raw_line_texts = original_text.splitlines()

    # Layout-noise: OFF for clean web unless forced
    layout_enabled = (drop_layout_noise == "on") or (
        drop_layout_noise == "auto" and do_norm and not web_safe
    )
    header_footer_enabled = (drop_repeated_lines == "on") or (
        drop_repeated_lines == "auto"
        and do_norm
        and text_profile in ("pdf_like", "ocr_like")
        and not web_safe
    )
    toc_navigation_enabled = do_norm and not web_safe
    lines, decisions = _build_line_decisions(
        raw_line_texts,
        lines,
        rep=rep,
        profile=profile,
        do_norm=do_norm,
        web_safe=web_safe,
        layout_enabled=layout_enabled,
        header_footer_enabled=header_footer_enabled,
        toc_navigation_enabled=toc_navigation_enabled,
        debug_decisions=debug_decisions,
        doc_id=doc_id,
    )
    rep.decisions = decisions
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
    debug_decisions: bool = False,
    doc_id: str | None = None,
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
        debug_decisions=debug_decisions,
        doc_id=doc_id,
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
