# textpraline/cleaner/clean.py
from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple, Union

from .mappings import PUA_BULLETS, PUA_TRANSLATE_MAP, TRANSLATE_MAP

__all__ = [
    "praline",
    "clean_text",
    "clean_lines",
    "LineDecision",
    "PralineReport",
    "detect_text_profile",
    "PralineConfig",
    "PRESETS",
]

Profile = Literal["safe", "strict", "markdown_safe"]
NormalizeExtracted = Literal[False, True, "auto"]
Toggle = Literal["off", "on", "auto"]
StructureMode = Literal["off", "light", "aggressive"]
ReportMode = Literal[False, True, "detail"]
DecisionAction = Literal["keep", "drop"]
DecisionCategory = Literal[
    "toc",
    "toc_navigation",
    "header_footer",
    "boilerplate",
    "layout_noise",
    "references_section",
    "paragraph_merge",
    "hyphen_fix",
    "section_marker_inserted",
    "other",
]
Preset = Literal["raw", "safe", "bench", "strict"]

# ---------------------------------------------------------------------------
# Presets / config (bench-friendly separation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PralineConfig:
    profile: Profile = "safe"
    normalize_extracted: NormalizeExtracted = "auto"
    normalize_form: str = "NFKC"
    preserve_markdown_tables: bool | None = None
    collapse_blank_lines: bool = True
    drop_layout_noise: Toggle = "auto"
    drop_repeated_lines: Toggle = "off"
    drop_references_section: Toggle = "off"
    structure_mode: StructureMode = "off"


PRESETS: Dict[str, PralineConfig] = {
    "safe": PralineConfig(),
    # Bench preset: conservative, but cuts common PDF tails (References) in auto mode.
    "bench": PralineConfig(
        profile="safe",
        drop_layout_noise="auto",
        drop_repeated_lines="off",
        drop_references_section="off",
    ),
    # Strict preset: more aggressive, but still keeps repeated-lines OFF unless forced.
    "strict": PralineConfig(
        profile="strict",
        drop_layout_noise="auto",
        drop_repeated_lines="off",
        drop_references_section="on",
        structure_mode="light",
    ),
}


def _resolve_cfg(
    preset: Preset,
    *,
    profile: Profile | None = None,
    normalize_extracted: NormalizeExtracted | None = None,
    normalize_form: str | None = None,
    preserve_markdown_tables: bool | None = None,
    collapse_blank_lines: bool | None = None,
    drop_layout_noise: Toggle | None = None,
    drop_repeated_lines: Toggle | None = None,
    drop_references_section: Toggle | None = None,
    structure_mode: StructureMode | None = None,
) -> PralineConfig:
    if preset == "raw":
        # Not used, but keep a valid object.
        base = PRESETS["safe"]
    else:
        base = PRESETS[preset]

    return PralineConfig(
        profile=profile if profile is not None else base.profile,
        normalize_extracted=(
            normalize_extracted
            if normalize_extracted is not None
            else base.normalize_extracted
        ),
        normalize_form=normalize_form
        if normalize_form is not None
        else base.normalize_form,
        preserve_markdown_tables=(
            preserve_markdown_tables
            if preserve_markdown_tables is not None
            else base.preserve_markdown_tables
        ),
        collapse_blank_lines=(
            collapse_blank_lines
            if collapse_blank_lines is not None
            else base.collapse_blank_lines
        ),
        drop_layout_noise=drop_layout_noise
        if drop_layout_noise is not None
        else base.drop_layout_noise,
        drop_repeated_lines=(
            drop_repeated_lines
            if drop_repeated_lines is not None
            else base.drop_repeated_lines
        ),
        drop_references_section=(
            drop_references_section
            if drop_references_section is not None
            else base.drop_references_section
        ),
        structure_mode=structure_mode
        if structure_mode is not None
        else base.structure_mode,
    )


# ---------------------------------------------------------------------------
# Pre-compiled regexes
# ---------------------------------------------------------------------------

RE_PUA = re.compile(r"[\uE000-\uF8FF]")
RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
RE_TOC_LINE = re.compile(r"\.{3,}\s*\d+\s*$")
RE_LIST_HEAD = re.compile(r"^\s*(?:•|\*|·|∙|‧|■|▪|●|◦|—|-)\s*")
GLYPH_RUN_RE = re.compile(r"glyph<[^>]*>|glyph<\S*", re.IGNORECASE)
RE_GLYPH_ESCAPED = re.compile(r"glyph&lt;[^&]*&gt;", re.IGNORECASE)
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z#0-9]+;")
RE_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060]")
RE_SOFT_HYPHEN = re.compile(r"\u00AD")
RE_BOM = re.compile(r"\ufeff")
RE_VARIATION_SELECTORS = re.compile(r"[\uFE00-\uFE0F]")
RE_CID = re.compile(r"\(cid:\d+\)")
RE_BLANK_GAPS = re.compile(r"\n{3,}")
RE_PAGE_MARKER = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
RE_STRONG_SEPARATOR = re.compile(r"^\s*(?:[-_=*]{6,}|#{6,})\s*$")
RE_REPLACEMENT = re.compile(r"[\uFFFD\uFFFE]")

# References tail section (bench-friendly)
RE_REF_HEADER = re.compile(
    r"^\s*(references|bibliography|références)\s*$", re.IGNORECASE
)
RE_REF_HEADER_INLINE = re.compile(
    r"^\s*(references|bibliography|références)\b[:\s]*", re.IGNORECASE
)

BOILERPLATE_PATTERNS = [
    re.compile(r"^A&A proofs:\s*manuscript no\..*", re.IGNORECASE),
    re.compile(r"^Article number,\s*page\s*\d+\s*of\s*\d+.*", re.IGNORECASE),
    re.compile(r"^arXiv:\s*\d{4}\.\d+(v\d+)?(\s*\[.*\])?.*", re.IGNORECASE),
    re.compile(r"^submitted to.*", re.IGNORECASE),
]

CAPTION_HINT = re.compile(r"\b(fig\.?|figure|table)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Report (optional)
# ---------------------------------------------------------------------------


@dataclass
class LineDecision:
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
    input_len: int
    output_len: int
    removed_toc_lines: int = 0
    normalized_extracted: bool = False
    removed_layout_noise_lines: int = 0
    removed_header_footer_lines: int = 0
    removed_repeated_lines: int = 0
    removed_boilerplate_lines: int = 0
    removed_references_lines: int = 0
    text_profile: Literal["clean_web", "pdf_like", "ocr_like", "unknown"] = "unknown"
    detail_enabled: bool = False
    decisions: List[LineDecision] = field(default_factory=list)

    def to_jsonl(self, path: str | Path, *, dropped_only: bool = False) -> None:
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


def _looks_extraction_polluted(s: str) -> bool:
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
    if not s:
        return s

    s = unescape(s)
    s = GLYPH_RUN_RE.sub("", s)
    s = RE_CID.sub("", s)

    if any(ch in s for ch in PUA_BULLETS):
        s = s.translate(str.maketrans(PUA_TRANSLATE_MAP))

    s = s.translate(str.maketrans(TRANSLATE_MAP))
    s = unicodedata.normalize("NFKC", s)

    s = RE_PUA.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    s = RE_SOFT_HYPHEN.sub("", s)
    s = RE_BOM.sub("", s)
    s = RE_VARIATION_SELECTORS.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_REPLACEMENT.sub("", s)
    return s


# ---------------------------------------------------------------------------
# Profile detection
# ---------------------------------------------------------------------------


def detect_text_profile(
    text: str,
) -> Literal["clean_web", "pdf_like", "ocr_like", "unknown"]:
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
    s = ln.strip()
    if not s:
        return False
    return any(p.match(s) for p in BOILERPLATE_PATTERNS)


def _looks_like_single_char_line(ln: str) -> bool:
    s = ln.strip()
    return len(s) == 1 and s.isalnum()


def _looks_like_axis_noise_line(ln: str) -> bool:
    s = ln.strip()
    if len(s) < 12:
        return False
    if CAPTION_HINT.search(s):
        return False
    # Preserve likely scientific signal even when numeric-heavy.
    if re.search(
        r"\b(km|m|cm|mm|ms|s|kg|g|hz|khz|mhz|ghz|ev|kev|mev|gev|sigma|lambda)\b",
        s,
        re.IGNORECASE,
    ):
        return False
    if len(re.findall(r"[A-Za-z]{3,}", s)) >= 3:
        return False

    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    total = max(len(s), 1)

    ratio_letters = letters / total
    ratio_digits_punct = (digits + punct) / total

    if (
        spaces <= 1
        and len(s) >= 36
        and ratio_digits_punct >= 0.75
        and ratio_letters <= 0.10
    ):
        return True
    return ratio_digits_punct >= 0.85 and ratio_letters <= 0.08


def _drop_layout_noise_blocks(
    lines: List[str],
    *,
    min_single_char_run: int = 8,
    min_axis_run: int = 4,
) -> Tuple[List[str], int]:
    out: List[str] = []
    removed = 0

    single_run: List[str] = []
    axis_run: List[str] = []

    def flush_single() -> None:
        nonlocal single_run, removed, out
        if not single_run:
            return
        if len(single_run) >= min_single_char_run:
            removed += len(single_run)
        else:
            out.extend(single_run)
        single_run = []

    def flush_axis() -> None:
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


def _is_markdown_table_line(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("|") and ("|" in s[1:])


def _looks_like_sentence_terminal(line: str) -> bool:
    s = line.rstrip()
    return bool(s) and s[-1] in ".?!:;)]}\"'"


def _starts_with_lowercase(line: str) -> bool:
    s = line.lstrip()
    for ch in s:
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit():
            return False
    return False


def _is_reference_heading(line: str) -> bool:
    s = line.strip().lower()
    return s in {"references", "reference", "bibliography", "works cited"}


def _is_reference_entry_line(line: str) -> bool:
    s = line.strip()
    if re.match(r"^\[\d+\]\s+", s):
        return True
    if re.match(r"^\d+\.\s+", s):
        return True
    if re.match(r"^[A-Z][a-zA-Z\-]+,\s+[A-Z]\.", s):
        return True
    return False


def _reconstruct_paragraphs(
    lines: List[str], *, preserve_markdown_tables: bool
) -> Tuple[List[str], List[Tuple[str, int, str, str]]]:
    """
    Rebuild paragraph continuity from PDF-broken line wraps.
    """
    if not lines:
        return lines, []

    out: List[str] = []
    events: List[Tuple[str, int, str, str]] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i == len(lines) - 1:
            out.append(cur)
            break

        nxt = lines[i + 1]
        cur_s = cur.rstrip()
        nxt_s = nxt.lstrip()

        if cur_s.startswith("## ") or nxt_s.startswith("## "):
            out.append(cur)
            i += 1
            continue
        if preserve_markdown_tables and (
            _is_markdown_table_line(cur) or _is_markdown_table_line(nxt)
        ):
            out.append(cur)
            i += 1
            continue
        if _is_likely_section_title_line(cur) or _is_likely_section_title_line(nxt):
            out.append(cur)
            i += 1
            continue

        # Hyphenated line-break fix: "infor-\\nmation" -> "information".
        if cur_s.endswith("-") and nxt_s and nxt_s[:1].islower():
            merged = cur_s[:-1] + nxt_s
            events.append(("hyphen_fix", i, f"{cur}\n{nxt}", merged))
            lines[i + 1] = merged
            i += 1
            continue

        # Paragraph reconstruction for wrapped lines.
        if (not _looks_like_sentence_terminal(cur_s)) and _starts_with_lowercase(nxt):
            merged = f"{cur_s} {nxt_s}"
            events.append(("paragraph_merge", i, f"{cur}\n{nxt}", merged))
            lines[i + 1] = merged
            i += 1
            continue

        out.append(cur)
        i += 1

    return out, events


def _looks_like_toc_navigation_line(s: str) -> bool:
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


def _is_likely_section_title_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) < 3 or len(s) > 90:
        return False
    if s.endswith("-"):
        return False
    if _looks_like_sentence_terminal(s):
        return False
    if _is_markdown_table_line(s):
        return False
    if CAPTION_HINT.search(s):
        return False
    if _is_reference_heading(s):
        return True

    words = s.split()
    if len(words) > 8:
        return False
    # Heading-like lines usually start uppercase.
    first_alpha = next((ch for ch in s if ch.isalpha()), "")
    if not first_alpha or first_alpha.islower():
        return False

    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    alpha_ratio = letters / max(len(s), 1)
    title_like_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    long_words = sum(1 for w in words if len(w) >= 5)
    return (
        alpha_ratio >= 0.55
        and digits <= max(3, len(s) // 12)
        and (len(words) <= 3 or title_like_ratio >= 0.85)
        and long_words <= 3
    )


def _looks_like_section_title(s: str) -> bool:
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
    return (alpha / len(s) >= 0.65) and (digits <= 3) and (len(words) <= 4)


def _insert_section_markers(
    lines: List[str],
) -> Tuple[List[str], List[Tuple[str, int, str, str]]]:
    out: List[str] = []
    events: List[Tuple[str, int, str, str]] = []
    for idx, ln in enumerate(lines):
        if _is_likely_section_title_line(ln):
            marker = f"## SECTION: {ln.strip()}"
            if not out or out[-1].strip() != marker:
                out.append(marker)
                events.append(("section_marker_inserted", idx, ln, marker))
        out.append(ln)
    return out, events


def _isolate_references_block(
    lines: List[str],
) -> Tuple[List[str], List[Tuple[str, int, str, str]]]:
    if not lines:
        return lines, []
    ref_idx = -1
    for i, ln in enumerate(lines):
        if _is_reference_heading(ln):
            ref_idx = i
            break
    if ref_idx == -1:
        tail_start = max(0, len(lines) - 50)
        tail = lines[tail_start:]
        ref_like = sum(1 for ln in tail if _is_reference_entry_line(ln))
        if ref_like >= 5:
            for j in range(tail_start, len(lines)):
                if _is_reference_entry_line(lines[j]):
                    ref_idx = j
                    break
    if ref_idx == -1:
        return lines, []

    marker = "## REFERENCES"
    if ref_idx > 0 and lines[ref_idx - 1].strip() == marker:
        return lines, []

    out = lines[:ref_idx] + [marker] + lines[ref_idx:]
    return out, [("section_marker_inserted", ref_idx, lines[ref_idx], marker)]


def _split_candidate_blocks(
    lines: List[str], *, block_min_lines: int
) -> List[List[Tuple[int, str]]]:
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
            if blank_run >= 2:
                flush_current()
            elif current:
                current.append((idx, line))
            continue

        blank_run = 0
        current.append((idx, line))

    flush_current()
    return blocks


def _detect_header_footer_candidates(
    lines: List[str],
    *,
    min_count: int = 6,
    min_len: int = 12,
    max_len: int = 160,
    top_k: int = 2,
    bottom_k: int = 2,
    block_min_lines: int = 12,
    presence_ratio: float = 0.8,
) -> Tuple[set[int], dict[int, int], dict[int, float], dict[int, int]]:
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
        if len(edge_blocks) / len(all_blocks) < 0.9:
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

    s = unicodedata.normalize(normalize_form, s)
    return s, do_norm


def _pass_invariant_guardrails(s: str) -> str:
    s = RE_PUA.sub("", s)
    s = RE_CTRL.sub("", s)
    s = RE_ZERO_WIDTH.sub("", s)
    return s


def _pass_whitespace(
    s: str,
    *,
    profile: Profile,
    preserve_markdown_tables: bool,
) -> str:
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


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


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
    references_enabled: bool,
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

    # Boilerplate (only when extractor-normalization happened)
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

    # Layout noise blocks
    if layout_enabled:
        active, removed = _drop_layout_noise_records(active)
        rep.removed_layout_noise_lines += removed

    # TOC navigation banners (ONLY meaningful for pdf/ocr, gating done outside)
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

    # Header/footer candidates
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

    in_refs = False
    for rec in active:
        s = rec.normalized_text

        # References tail cut (bench-friendly; traceable)
        if references_enabled:
            if in_refs:
                rec.action = "drop"
                rec.category = "references_section"
                rec.confidence = 0.90
                rep.removed_references_lines += 1
                continue
            st = s.strip()
            if RE_REF_HEADER.match(st) or RE_REF_HEADER_INLINE.match(st):
                in_refs = True
                rec.action = "drop"
                rec.category = "references_section"
                rec.confidence = 0.92
                rep.removed_references_lines += 1
                continue

        # Dotted ToC lines
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
                category=rec.category,
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


def _line_decision_from_event(
    event: Tuple[str, int, str, str],
    *,
    doc_id: str | None,
    page_map: dict[int, int],
) -> LineDecision:
    category, idx, raw_text, normalized_text = event
    caps, digit, punct = _line_ratios(normalized_text)
    return LineDecision(
        doc_id=doc_id,
        page_idx=page_map.get(idx, -1),
        line_idx=idx,
        raw_text=raw_text,
        normalized_text=normalized_text,
        action="keep",
        category=category,  # paragraph_merge / hyphen_fix / section_marker_inserted
        confidence=0.85,
        caps_ratio=caps,
        digit_ratio=digit,
        punctuation_ratio=punct,
        codepoints=[f"U+{ord(ch):04X}" for ch in raw_text],
    )


def _apply_structure_mode(
    lines: List[str],
    *,
    structure_mode: StructureMode,
    preserve_markdown_tables: bool,
    debug_decisions: bool,
    doc_id: str | None,
    existing_decisions: List[LineDecision],
) -> Tuple[List[str], List[LineDecision]]:
    if structure_mode == "off" or not lines:
        return lines, existing_decisions

    page_map = {d.line_idx: d.page_idx for d in existing_decisions}
    events: List[Tuple[str, int, str, str]] = []

    # Light mode: continuity restoration only.
    lines, ev = _reconstruct_paragraphs(
        lines, preserve_markdown_tables=preserve_markdown_tables
    )
    events.extend(ev)

    # Aggressive mode: add structural markers for sections and references.
    if structure_mode == "aggressive":
        lines, ev = _insert_section_markers(lines)
        events.extend(ev)
        lines, ev = _isolate_references_block(lines)
        events.extend(ev)

    if not debug_decisions:
        return lines, existing_decisions

    extra = [
        _line_decision_from_event(e, doc_id=doc_id, page_map=page_map) for e in events
    ]
    return lines, existing_decisions + extra


# ---------------------------------------------------------------------------
# clean_text / praline API
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
    drop_repeated_lines: Toggle = "off",
    drop_references_section: Toggle = "off",
    structure_mode: StructureMode = "off",
    debug_decisions: bool = False,
    doc_id: str | None = None,
    report: ReportMode = False,
) -> Tuple[str, PralineReport]:
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
    is_pdfish = text_profile in ("pdf_like", "ocr_like")

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

    layout_enabled = (drop_layout_noise == "on") or (
        drop_layout_noise == "auto" and do_norm and is_pdfish and not web_safe
    )
    header_footer_enabled = (drop_repeated_lines == "on") or (
        drop_repeated_lines == "auto" and do_norm and is_pdfish and not web_safe
    )
    toc_navigation_enabled = bool(do_norm and not web_safe)

    references_enabled = (drop_references_section == "on") or (
        drop_references_section == "auto" and do_norm and is_pdfish and not web_safe
    )

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
        references_enabled=references_enabled,
        debug_decisions=debug_decisions,
        doc_id=doc_id,
    )
    lines, decisions = _apply_structure_mode(
        lines,
        structure_mode=structure_mode,
        preserve_markdown_tables=preserve_markdown_tables,
        debug_decisions=debug_decisions,
        doc_id=doc_id,
        existing_decisions=decisions,
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


def praline(
    text: str,
    *,
    preset: Preset = "safe",
    # overrides (optional)
    profile: Profile | None = None,
    normalize_extracted: NormalizeExtracted | None = None,
    normalize_form: str | None = None,
    preserve_markdown_tables: bool | None = None,
    collapse_blank_lines: bool | None = None,
    drop_layout_noise: Toggle | None = None,
    drop_repeated_lines: Toggle | None = None,
    drop_references_section: Toggle | None = None,
    structure_mode: StructureMode | None = None,
    debug_decisions: bool = False,
    doc_id: str | None = None,
    report: ReportMode = False,
) -> Union[str, Tuple[str, PralineReport]]:
    """
    One-shot entrypoint: refine any text to be ingestion-ready.

    - preset="raw" -> returns original text (+ report if requested)
    - report=False -> returns str
    - report=True/"detail" -> returns (str, PralineReport)
    """
    if preset == "raw":
        rep = PralineReport(input_len=len(text or ""), output_len=len(text or ""))
        rep.detail_enabled = report == "detail"
        return (text, rep) if report in (True, "detail") else text

    cfg = _resolve_cfg(
        preset,
        profile=profile,
        normalize_extracted=normalize_extracted,
        normalize_form=normalize_form,
        preserve_markdown_tables=preserve_markdown_tables,
        collapse_blank_lines=collapse_blank_lines,
        drop_layout_noise=drop_layout_noise,
        drop_repeated_lines=drop_repeated_lines,
        drop_references_section=drop_references_section,
        structure_mode=structure_mode,
    )

    out, rep = clean_text(
        text,
        profile=cfg.profile,
        normalize_extracted=cfg.normalize_extracted,
        normalize_form=cfg.normalize_form,
        preserve_markdown_tables=(
            cfg.preserve_markdown_tables
            if cfg.preserve_markdown_tables is not None
            else None
        ),
        collapse_blank_lines=cfg.collapse_blank_lines,
        drop_layout_noise=cfg.drop_layout_noise,
        drop_repeated_lines=cfg.drop_repeated_lines,
        drop_references_section=cfg.drop_references_section,
        structure_mode=cfg.structure_mode,
        debug_decisions=debug_decisions,
        doc_id=doc_id,
        report=report,
    )
    return (out, rep) if report in (True, "detail") else out


def clean_lines(
    lines: Iterable[str],
    *,
    return_reports: bool = False,
    **kwargs: Any,
) -> List[str] | Tuple[List[str], List[PralineReport]]:
    texts: List[str] = []
    reports: List[PralineReport] = []

    kwargs["report"] = True if return_reports else kwargs.get("report", False)

    for x in lines:
        res = praline(x, **kwargs)
        if isinstance(res, str):
            texts.append(res)
        else:
            txt, rep = res
            texts.append(txt)
            if return_reports:
                reports.append(rep)

    return (texts, reports) if return_reports else texts


def main(argv: List[str] | None = None) -> None:
    data = sys.stdin.read()
    sys.stdout.write(praline(data))


if __name__ == "__main__":
    main()
