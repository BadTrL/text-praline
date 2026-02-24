from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import zipfile
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

import chromadb
import fitz
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from textpraline.cleaner.clean import PralineReport, praline


# =============================================================================
# Config
# =============================================================================

GENERATE_DATASET = True
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
DATASETS_DIR = BASE_DIR / "datasets"
QA_JSONL_PATH = DATASETS_DIR / "generated_qa.jsonl"
XLSX_PATH = DATASETS_DIR / "rag_eval_dataset.xlsx"
CSV_PATH = DATASETS_DIR / "rag_eval_dataset.csv"
REMOVED_DEBUG_JSONL_PATH = DATASETS_DIR / "removed_debug.jsonl"

EMBED_MODEL = "all-MiniLM-L6-v2"  # same model as rag_testing.ipynb
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 10
TARGET_QUESTIONS_PER_DOC = 10
MIN_QUESTIONS_PER_DOC = 8
MAX_QUESTIONS_PER_DOC = 15

RE_WS = re.compile(r"[ \t]+")
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
RE_TOC_DOTTED = re.compile(r"\.{3,}\s*\d+\s*$")
RE_BOILERPLATE = [
    re.compile(r"^A&A proofs:\s*manuscript no\..*", re.IGNORECASE),
    re.compile(r"^Article number,\s*page\s*\d+\s*of\s*\d+.*", re.IGNORECASE),
    re.compile(r"^arXiv:\s*\d{4}\.\d+(v\d+)?(\s*\[.*\])?.*", re.IGNORECASE),
    re.compile(r"^submitted to.*", re.IGNORECASE),
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "have",
    "has",
    "are",
    "was",
    "were",
    "will",
    "about",
    "their",
    "there",
    "they",
    "them",
    "what",
    "when",
    "where",
    "which",
    "while",
    "than",
    "then",
    "also",
    "such",
    "using",
    "used",
    "each",
    "over",
    "under",
    "between",
    "after",
    "before",
    "more",
    "most",
    "only",
    "into",
    "onto",
    "your",
    "you",
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DocData:
    doc_id: str
    pdf_path: str
    pages: List[str]
    raw_doc_text: str
    praline_doc_text: str
    report: PralineReport
    page_count: int
    raw_len: int
    praline_len: int


@dataclass
class ChunkData:
    doc_id: str
    chunk_id: str
    chunk_index: int
    variant: str
    text: str
    start_char: int
    end_char: int
    page_span: str


@dataclass
class QAItem:
    qid: str
    question: str
    expected_doc_id: str
    expected_answer_snippet: str


# =============================================================================
# Utility
# =============================================================================


def norm_ws(text: str) -> str:
    return RE_WS.sub(" ", text.strip())


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9/_%-]*", text)


def col_to_excel(col_idx: int) -> str:
    out = []
    n = col_idx
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out.append(chr(65 + rem))
    return "".join(reversed(out))


def chunk_text_with_offsets(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> List[Tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >=0 and < chunk_size")

    chunks: List[Tuple[int, int, str]] = []
    step = chunk_size - chunk_overlap
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j]
        if chunk.strip():
            chunks.append((i, j, chunk))
        i += step
    return chunks


def compute_page_offsets(pages: Sequence[str]) -> List[int]:
    offsets = [0]
    total = 0
    sep = "\n\n\n"
    for i, page in enumerate(pages):
        total += len(page)
        if i < len(pages) - 1:
            total += len(sep)
        offsets.append(total)
    return offsets


def char_span_to_page_span(start: int, end: int, page_offsets: Sequence[int]) -> str:
    if len(page_offsets) < 2:
        return ""
    s = max(0, bisect_right(page_offsets, start) - 1)
    e = max(0, bisect_right(page_offsets, max(start, end - 1)) - 1)
    s = min(s, len(page_offsets) - 2)
    e = min(e, len(page_offsets) - 2)
    return str(s + 1) if s == e else f"{s + 1}-{e + 1}"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def _xlsx_sheet_xml(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    body_rows: List[str] = []
    header = {c: c for c in columns}
    table = [header] + rows

    for r_idx, row in enumerate(table, start=1):
        cell_xml: List[str] = []
        for c_idx, col in enumerate(columns, start=1):
            cell_ref = f"{col_to_excel(c_idx)}{r_idx}"
            val = row.get(col, "")
            if val is None:
                continue
            if isinstance(val, bool):
                cell_xml.append(f'<c r="{cell_ref}" t="b"><v>{1 if val else 0}</v></c>')
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                if isinstance(val, float):
                    if math.isnan(val) or math.isinf(val):
                        txt = ""
                        cell_xml.append(
                            f'<c r="{cell_ref}" t="inlineStr"><is><t>{txt}</t></is></c>'
                        )
                    else:
                        cell_xml.append(f'<c r="{cell_ref}"><v>{val}</v></c>')
                else:
                    cell_xml.append(f'<c r="{cell_ref}"><v>{val}</v></c>')
            else:
                txt = xml_escape(str(val))
                cell_xml.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{txt}</t></is></c>'
                )
        body_rows.append(f'<row r="{r_idx}">{"".join(cell_xml)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(body_rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: Dict[str, Tuple[List[Dict[str, Any]], List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheet_names) + 1)
            )
            + "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(sheet_names, start=1)
            )
            + "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheet_names) + 1)
            )
            + f'<Relationship Id="rId{len(sheet_names) + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            "</styleSheet>",
        )
        for i, name in enumerate(sheet_names, start=1):
            rows, cols = sheets[name]
            z.writestr(f"xl/worksheets/sheet{i}.xml", _xlsx_sheet_xml(rows, cols))


class HashEmbeddingFunction:
    """
    Deterministic offline fallback embedding function.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def name(self) -> str:
        return "hash-embedding-function"

    def embed_query(self, input: Any) -> List[List[float]]:
        if isinstance(input, str):
            return self([input])
        return self(input)

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:
        return self(input)

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            for tok in tokenize_words((text or "").lower()):
                h = hashlib.sha1(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                weight = 1.0 + min(len(tok), 20) / 40.0
                vec[idx] += sign * weight
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


# =============================================================================
# Pipeline
# =============================================================================


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text")
            pages.append(txt if txt is not None else "")
    return pages


def build_documents(corpus_dir: Path) -> List[DocData]:
    docs: List[DocData] = []
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {corpus_dir}")

    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        raw_doc_text = "\n\n\n".join(pages)
        praline_doc_text, report = praline(
            raw_doc_text,
            normalize_extracted="auto",
            drop_layout_noise="auto",
            drop_repeated_lines="auto",
            debug_decisions=True,
            doc_id=pdf_path.stem,
            report="detail",
        )
        docs.append(
            DocData(
                doc_id=pdf_path.stem,
                pdf_path=str(pdf_path),
                pages=pages,
                raw_doc_text=raw_doc_text,
                praline_doc_text=praline_doc_text,
                report=report,
                page_count=len(pages),
                raw_len=len(raw_doc_text),
                praline_len=len(praline_doc_text),
            )
        )
    return docs


def build_chunks(docs: Sequence[DocData], variant: str) -> List[ChunkData]:
    out: List[ChunkData] = []
    for doc in docs:
        text = doc.raw_doc_text if variant == "raw" else doc.praline_doc_text
        page_offsets = compute_page_offsets(doc.pages) if variant == "raw" else [0, len(text)]
        chunks = chunk_text_with_offsets(
            text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        for idx, (start, end, ch) in enumerate(chunks):
            out.append(
                ChunkData(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}__{variant}__{idx:05d}",
                    chunk_index=idx,
                    variant=variant,
                    text=ch,
                    start_char=start,
                    end_char=end,
                    page_span=char_span_to_page_span(start, end, page_offsets)
                    if variant == "raw"
                    else "",
                )
            )
    return out


def build_collection(
    client: chromadb.ClientAPI,
    name: str,
    embed_fn: SentenceTransformerEmbeddingFunction,
    chunks: Sequence[ChunkData],
):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(name=name, embedding_function=embed_fn)
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "variant": c.variant,
            "page_span": c.page_span,
        }
        for c in chunks
    ]
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def split_sentences(text: str) -> List[str]:
    normalized = norm_ws(text.replace("\n", " "))
    if not normalized:
        return []
    sents = [norm_ws(s) for s in RE_SENT_SPLIT.split(normalized) if norm_ws(s)]
    return sents


def is_boilerplate_sentence(sent: str) -> bool:
    s = sent.strip()
    if not s:
        return True
    if RE_TOC_DOTTED.search(s):
        return True
    if any(p.match(s) for p in RE_BOILERPLATE):
        return True
    boilerplate_terms = (
        "all rights reserved",
        "copyright",
        "table of contents",
        "references",
        "appendix",
    )
    low = s.lower()
    return any(t in low for t in boilerplate_terms)


def sentence_informativeness(sent: str) -> float:
    tokens = tokenize_words(sent)
    if not tokens:
        return -1.0
    n = len(tokens)
    if n < 12 or n > 30:
        return -1.0
    if is_boilerplate_sentence(sent):
        return -1.0

    cap = sum(1 for t in tokens if t[:1].isupper())
    acr = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    dig = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    long_terms = sum(1 for t in tokens if len(t) >= 7)
    uniq_ratio = len(set(t.lower() for t in tokens)) / n
    return cap * 0.4 + acr * 0.9 + dig * 0.8 + long_terms * 0.2 + uniq_ratio


def build_focus_phrase(sent: str) -> str:
    tokens = tokenize_words(sent)
    ranked: List[Tuple[float, str]] = []
    for tok in tokens:
        low = tok.lower()
        if low in STOPWORDS:
            continue
        score = 0.0
        if tok[:1].isupper():
            score += 1.0
        if tok.isupper() and len(tok) > 1:
            score += 1.2
        if any(ch.isdigit() for ch in tok):
            score += 1.1
        score += min(len(tok), 12) / 20
        ranked.append((score, tok))
    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    picked = [tok for _, tok in ranked[:8]]
    if not picked:
        picked = tokens[:8]
    return " ".join(picked)


def build_question(anchor: str, qid_seed: int) -> str:
    focus = build_focus_phrase(anchor)
    templates = [
        "What does the document state about {focus}?",
        "According to the document, what is said about {focus}?",
        "What key point is made regarding {focus}?",
    ]
    t = templates[qid_seed % len(templates)]
    return t.format(focus=focus)


def build_answer_snippet(text: str, anchor: str) -> str:
    idx = text.find(anchor)
    if idx >= 0:
        tail = text[idx : idx + 1200]
    else:
        tail = anchor
    words = tokenize_words(tail)
    if len(words) >= 20:
        return " ".join(words[: min(40, max(20, 28))])
    anchor_words = tokenize_words(anchor)
    if len(anchor_words) >= 20:
        return " ".join(anchor_words[:40])
    # deterministic fallback padding
    merged = (anchor_words + words)[:40]
    return " ".join(merged[:20] if len(merged) >= 20 else merged)


def generate_questions(docs: Sequence[DocData]) -> List[QAItem]:
    qa_items: List[QAItem] = []
    qid_counter = 1
    for doc in docs:
        sents = split_sentences(doc.praline_doc_text)
        scored: List[Tuple[float, int, str]] = []
        for i, sent in enumerate(sents):
            score = sentence_informativeness(sent)
            if score < 0:
                continue
            scored.append((score, i, sent))
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected: List[str] = []
        seen = set()
        for _, _, sent in scored:
            key = sent.lower()
            if key in seen:
                continue
            selected.append(sent)
            seen.add(key)
            if len(selected) >= TARGET_QUESTIONS_PER_DOC:
                break

        if len(selected) < MIN_QUESTIONS_PER_DOC:
            for sent in sents:
                if is_boilerplate_sentence(sent):
                    continue
                n = len(tokenize_words(sent))
                if n < 10 or n > 35:
                    continue
                key = sent.lower()
                if key in seen:
                    continue
                selected.append(sent)
                seen.add(key)
                if len(selected) >= MIN_QUESTIONS_PER_DOC:
                    break

        selected = selected[:MAX_QUESTIONS_PER_DOC]
        for sent in selected:
            qid = f"q{qid_counter:04d}"
            question = build_question(sent, qid_counter)
            answer = build_answer_snippet(doc.praline_doc_text, sent)
            qa_items.append(
                QAItem(
                    qid=qid,
                    question=question,
                    expected_doc_id=doc.doc_id,
                    expected_answer_snippet=answer,
                )
            )
            qid_counter += 1
    return qa_items


def query_topk(col, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    res = col.query(
        query_texts=[question],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )
    out: List[Dict[str, Any]] = []
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    docs = res.get("documents", [[]])[0]
    for md, dist, doc in zip(metas, dists, docs):
        out.append(
            {
                "doc_id": md.get("doc_id", ""),
                "chunk_id": md.get("chunk_id", ""),
                "chunk_index": md.get("chunk_index", -1),
                "distance": float(dist) if dist is not None else None,
                "snippet": (doc or "")[:180].replace("\n", " "),
            }
        )
    return out


def top_docs_unique(topk: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    out = []
    for item in topk:
        doc_id = item["doc_id"]
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(ranked_docs: Sequence[str], expected: str, k: int) -> int:
    return int(expected in ranked_docs[:k])


def mrr(ranked_docs: Sequence[str], expected: str) -> float:
    if expected not in ranked_docs:
        return 0.0
    return 1.0 / (ranked_docs.index(expected) + 1)


def best_chunk_for_answer(
    chunks: Sequence[ChunkData], answer_snippet: str, expected_doc_id: str
) -> str:
    cand = [c for c in chunks if c.doc_id == expected_doc_id]
    if not cand:
        return ""
    low_snip = answer_snippet.lower()
    for c in cand:
        if low_snip and low_snip in c.text.lower():
            return c.chunk_id

    snip_tokens = set(t.lower() for t in tokenize_words(answer_snippet))
    best = ("", -1.0)
    for c in cand:
        toks = set(t.lower() for t in tokenize_words(c.text))
        if not toks:
            continue
        inter = len(snip_tokens & toks)
        if inter == 0:
            continue
        score = inter / max(len(snip_tokens), 1)
        if score > best[1]:
            best = (c.chunk_id, score)
    return best[0]


def compact_topk_json(items: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "doc_id": x["doc_id"],
            "chunk_id": x["chunk_id"],
            "chunk_index": x["chunk_index"],
            "d": round(x["distance"], 4) if x["distance"] is not None else None,
        }
        for x in items
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def line_category_guess(
    normalized_line: str,
    raw_count: int,
    edge_hits: int,
    edge_total: int,
) -> str:
    if RE_TOC_DOTTED.search(normalized_line):
        return "toc_dotted"
    if any(p.match(normalized_line) for p in RE_BOILERPLATE):
        return "boilerplate"
    if edge_total > 0 and edge_hits / edge_total >= 0.6 and raw_count >= 3:
        return "header_footer"
    return "other"


def looks_like_layout_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) == 1 and s.isalnum():
        return True
    if len(s) < 12:
        return False

    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    total = max(len(s), 1)
    ratio_letters = letters / total
    ratio_digits_punct = (digits + punct) / total
    if spaces <= 1 and len(s) >= 30 and ratio_digits_punct >= 0.65:
        return True
    return ratio_digits_punct >= 0.75 and ratio_letters <= 0.10


def classify_removed_line(
    line: str,
    *,
    raw_count: int,
    edge_hits: int,
    edge_total: int,
) -> str:
    s = line.strip()
    if RE_TOC_DOTTED.search(s):
        return "toc"
    if any(p.match(s) for p in RE_BOILERPLATE):
        return "boilerplate"
    if looks_like_layout_noise_line(s):
        return "layout_noise"
    if edge_total > 0 and edge_hits / edge_total >= 0.6 and raw_count >= 3:
        return "header_footer"
    return "other"


def codepoints(text: str) -> List[str]:
    return [f"U+{ord(ch):04X}" for ch in text]


def build_removed_lines_overview(docs: Sequence[DocData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        raw_lines = [ln for ln in doc.raw_doc_text.splitlines() if ln.strip()]
        pra_lines = [ln for ln in doc.praline_doc_text.splitlines() if ln.strip()]

        raw_norm_map: Dict[str, List[str]] = {}
        raw_counts: Dict[str, int] = {}
        pra_counts: Dict[str, int] = {}

        for ln in raw_lines:
            n = norm_ws(ln)
            raw_counts[n] = raw_counts.get(n, 0) + 1
            raw_norm_map.setdefault(n, []).append(ln)
        for ln in pra_lines:
            n = norm_ws(ln)
            pra_counts[n] = pra_counts.get(n, 0) + 1

        edge_hits: Dict[str, int] = {}
        edge_total: Dict[str, int] = {}
        for page in doc.pages:
            lines = [norm_ws(ln) for ln in page.splitlines() if norm_ws(ln)]
            if not lines:
                continue
            edges = set(lines[:2] + lines[-2:])
            for n in set(lines):
                edge_total[n] = edge_total.get(n, 0) + 1
            for n in edges:
                edge_hits[n] = edge_hits.get(n, 0) + 1

        all_lines = sorted(set(raw_counts.keys()) | set(pra_counts.keys()))
        changed = [
            n
            for n in all_lines
            if raw_counts.get(n, 0) != pra_counts.get(n, 0)
            and n
            and len(n) >= 2
        ]

        def score_key(n: str) -> Tuple[int, int, int]:
            rc = raw_counts.get(n, 0)
            pc = pra_counts.get(n, 0)
            removed = int(rc > 0 and pc == 0)
            delta = abs(rc - pc)
            return (removed, delta, rc)

        changed_sorted = sorted(changed, key=score_key, reverse=True)
        # Keep overview tractable while still representative.
        limited = changed_sorted[:400]
        for n in limited:
            rc = raw_counts.get(n, 0)
            pc = pra_counts.get(n, 0)
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    # repr keeps XLSX safe while preserving visual fidelity of hidden chars.
                    "normalized_line": repr(n),
                    "raw_count": rc,
                    "praline_count": pc,
                    "removed": bool(rc > 0 and pc == 0),
                    "category_guess": line_category_guess(
                        n,
                        rc,
                        edge_hits.get(n, 0),
                        edge_total.get(n, 0),
                    ),
                    "example_raw_line": repr(raw_norm_map.get(n, [""])[0]),
                }
            )
    return rows


def build_removed_debug_rows(docs: Sequence[DocData]) -> List[Dict[str, Any]]:
    """
    Raw removed-line debug rows from PralineReport decisions.
    """
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        for d in doc.report.decisions:
            if d.action != "drop":
                continue
            rows.append(
                {
                    "doc_id": d.doc_id or doc.doc_id,
                    "page_idx": d.page_idx,
                    "line_index": d.line_idx,
                    "removal_type": d.category,
                    "raw_text": d.raw_text,
                    "repr_text": repr(d.raw_text),
                    "codepoints": d.codepoints,
                }
            )
    return rows


def summarize_metrics(question_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not question_rows:
        return []

    def avg(col: str) -> float:
        return sum(float(r[col]) for r in question_rows) / len(question_rows)

    summary_rows: List[Dict[str, Any]] = []
    metrics = [
        ("recall@1", "recall@1_raw", "recall@1_praline"),
        ("recall@5", "recall@5_raw", "recall@5_praline"),
        ("recall@10", "recall@10_raw", "recall@10_praline"),
        ("mrr", "mrr_raw", "mrr_praline"),
        ("top1_accuracy", "accuracy_top1_raw", "accuracy_top1_praline"),
    ]
    for label, raw_col, pra_col in metrics:
        raw_v = avg(raw_col)
        pra_v = avg(pra_col)
        summary_rows.append(
            {
                "section": "global_metrics",
                "item": label,
                "raw_value": raw_v,
                "praline_value": pra_v,
                "delta": pra_v - raw_v,
                "notes": "",
            }
        )

    raw_mrr = avg("mrr_raw")
    pra_mrr = avg("mrr_praline")
    raw_top1 = avg("accuracy_top1_raw")
    pra_top1 = avg("accuracy_top1_praline")
    improved = sum(1 for r in question_rows if float(r["delta_mrr"]) > 0)
    regressed = sum(1 for r in question_rows if float(r["delta_mrr"]) < 0)
    tied = len(question_rows) - improved - regressed

    winner_by_mrr = "praline" if pra_mrr > raw_mrr else ("raw" if raw_mrr > pra_mrr else "tie")
    winner_by_top1 = (
        "praline" if pra_top1 > raw_top1 else ("raw" if raw_top1 > pra_top1 else "tie")
    )

    summary_rows.append(
        {
            "section": "overall_result",
            "item": "winner_by_mrr",
            "raw_value": raw_mrr,
            "praline_value": pra_mrr,
            "delta": pra_mrr - raw_mrr,
            "notes": winner_by_mrr,
        }
    )
    summary_rows.append(
        {
            "section": "overall_result",
            "item": "winner_by_top1",
            "raw_value": raw_top1,
            "praline_value": pra_top1,
            "delta": pra_top1 - raw_top1,
            "notes": winner_by_top1,
        }
    )
    summary_rows.append(
        {
            "section": "overall_result",
            "item": "question_delta_counts",
            "raw_value": improved,
            "praline_value": regressed,
            "delta": improved - regressed,
            "notes": f"improved={improved}, regressed={regressed}, tied={tied}",
        }
    )

    sorted_q = sorted(question_rows, key=lambda r: float(r["delta_mrr"]), reverse=True)
    for row in sorted_q[:10]:
        summary_rows.append(
            {
                "section": "top_improvements",
                "item": row["qid"],
                "raw_value": row["mrr_raw"],
                "praline_value": row["mrr_praline"],
                "delta": row["delta_mrr"],
                "notes": row["question"],
            }
        )
    for row in sorted(question_rows, key=lambda r: float(r["delta_mrr"]))[:10]:
        summary_rows.append(
            {
                "section": "top_regressions",
                "item": row["qid"],
                "raw_value": row["mrr_raw"],
                "praline_value": row["mrr_praline"],
                "delta": row["delta_mrr"],
                "notes": row["question"],
            }
        )
    return summary_rows


def make_embedding_function() -> tuple[Any, str]:
    """
    Prefer notebook model; fallback to deterministic offline embeddings.
    """
    try:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        # force one call so download/caching failure happens here
        _ = embed_fn(["healthcheck"])
        return embed_fn, f"sentence-transformers:{EMBED_MODEL}"
    except Exception:
        return HashEmbeddingFunction(dim=768), "hash-fallback-768"


def main() -> None:
    if not GENERATE_DATASET:
        print("GENERATE_DATASET is False. Exiting.")
        return

    random.seed(SEED)

    docs = build_documents(CORPUS_DIR)
    raw_chunks = build_chunks(docs, "raw")
    praline_chunks = build_chunks(docs, "praline")

    embed_fn, embedding_backend = make_embedding_function()
    client = chromadb.Client()
    raw_col = build_collection(client, "rag_eval_raw", embed_fn, raw_chunks)
    praline_col = build_collection(client, "rag_eval_praline", embed_fn, praline_chunks)

    qa_items = generate_questions(docs)
    write_jsonl(QA_JSONL_PATH, [asdict(x) for x in qa_items])

    question_rows: List[Dict[str, Any]] = []
    for qa in qa_items:
        raw_topk = query_topk(raw_col, qa.question, top_k=TOP_K)
        pra_topk = query_topk(praline_col, qa.question, top_k=TOP_K)

        raw_docs = top_docs_unique(raw_topk)
        pra_docs = top_docs_unique(pra_topk)

        mrr_raw = mrr(raw_docs, qa.expected_doc_id)
        mrr_pra = mrr(pra_docs, qa.expected_doc_id)

        question_rows.append(
            {
                "qid": qa.qid,
                "question": qa.question,
                "expected_doc_id": qa.expected_doc_id,
                "expected_answer_snippet": qa.expected_answer_snippet,
                "expected_chunk_raw": best_chunk_for_answer(
                    raw_chunks, qa.expected_answer_snippet, qa.expected_doc_id
                ),
                "expected_chunk_praline": best_chunk_for_answer(
                    praline_chunks, qa.expected_answer_snippet, qa.expected_doc_id
                ),
                "top1_doc_raw": raw_docs[0] if raw_docs else "",
                "top1_doc_praline": pra_docs[0] if pra_docs else "",
                "recall@1_raw": recall_at_k(raw_docs, qa.expected_doc_id, 1),
                "recall@5_raw": recall_at_k(raw_docs, qa.expected_doc_id, 5),
                "recall@10_raw": recall_at_k(raw_docs, qa.expected_doc_id, 10),
                "mrr_raw": mrr_raw,
                "recall@1_praline": recall_at_k(pra_docs, qa.expected_doc_id, 1),
                "recall@5_praline": recall_at_k(pra_docs, qa.expected_doc_id, 5),
                "recall@10_praline": recall_at_k(pra_docs, qa.expected_doc_id, 10),
                "mrr_praline": mrr_pra,
                "accuracy_top1_raw": bool(raw_docs and raw_docs[0] == qa.expected_doc_id),
                "accuracy_top1_praline": bool(
                    pra_docs and pra_docs[0] == qa.expected_doc_id
                ),
                "delta_mrr": mrr_pra - mrr_raw,
                "raw_topk_chunks_json": compact_topk_json(raw_topk),
                "praline_topk_chunks_json": compact_topk_json(pra_topk),
            }
        )

    doc_report_rows: List[Dict[str, Any]] = []
    for d in docs:
        removed_chars = d.raw_len - d.praline_len
        doc_report_rows.append(
            {
                "doc_id": d.doc_id,
                "page_count": d.page_count,
                "raw_len": d.raw_len,
                "praline_len": d.praline_len,
                "removed_chars": removed_chars,
                "removed_ratio": (removed_chars / d.raw_len) if d.raw_len else 0.0,
                "text_profile": d.report.text_profile,
                "normalized_extracted": d.report.normalized_extracted,
                "removed_layout_noise_lines": d.report.removed_layout_noise_lines,
                "removed_header_footer_lines": d.report.removed_header_footer_lines,
                "removed_toc_lines": d.report.removed_toc_lines,
                "removed_boilerplate_lines": d.report.removed_boilerplate_lines,
            }
        )

    removed_lines_rows = build_removed_lines_overview(docs)
    removed_debug_rows = build_removed_debug_rows(docs)
    summary_rows = summarize_metrics(question_rows)
    summary_rows.insert(
        0,
        {
            "section": "run_info",
            "item": "embedding_backend",
            "raw_value": "",
            "praline_value": "",
            "delta": "",
            "notes": embedding_backend,
        },
    )

    questions_columns = [
        "qid",
        "question",
        "expected_doc_id",
        "expected_answer_snippet",
        "expected_chunk_raw",
        "expected_chunk_praline",
        "top1_doc_raw",
        "top1_doc_praline",
        "recall@1_raw",
        "recall@5_raw",
        "recall@10_raw",
        "mrr_raw",
        "recall@1_praline",
        "recall@5_praline",
        "recall@10_praline",
        "mrr_praline",
        "accuracy_top1_raw",
        "accuracy_top1_praline",
        "delta_mrr",
        "raw_topk_chunks_json",
        "praline_topk_chunks_json",
    ]
    doc_report_columns = [
        "doc_id",
        "page_count",
        "raw_len",
        "praline_len",
        "removed_chars",
        "removed_ratio",
        "text_profile",
        "normalized_extracted",
        "removed_layout_noise_lines",
        "removed_header_footer_lines",
        "removed_toc_lines",
        "removed_boilerplate_lines",
    ]
    removed_columns = [
        "doc_id",
        "normalized_line",
        "raw_count",
        "praline_count",
        "removed",
        "category_guess",
        "example_raw_line",
    ]
    summary_columns = ["section", "item", "raw_value", "praline_value", "delta", "notes"]

    write_csv(CSV_PATH, question_rows, questions_columns)
    write_jsonl(REMOVED_DEBUG_JSONL_PATH, removed_debug_rows)
    write_xlsx(
        XLSX_PATH,
        {
            "questions": (question_rows, questions_columns),
            "doc_reports": (doc_report_rows, doc_report_columns),
            "removed_lines_overview": (removed_lines_rows, removed_columns),
            "summary": (summary_rows, summary_columns),
        },
    )

    print("Done.")
    print(f"Docs processed: {len(docs)}")
    print(f"Questions generated: {len(question_rows)}")
    print(f"RAW chunks: {len(raw_chunks)}")
    print(f"PRALINE chunks: {len(praline_chunks)}")
    print(f"Embedding backend: {embedding_backend}")
    if question_rows:
        raw_mrr = sum(float(r["mrr_raw"]) for r in question_rows) / len(question_rows)
        pra_mrr = sum(float(r["mrr_praline"]) for r in question_rows) / len(question_rows)
        raw_top1 = sum(float(r["accuracy_top1_raw"]) for r in question_rows) / len(question_rows)
        pra_top1 = sum(float(r["accuracy_top1_praline"]) for r in question_rows) / len(question_rows)
        winner = "PRALINE" if pra_mrr > raw_mrr else ("RAW" if raw_mrr > pra_mrr else "TIE")
        print(
            "Overall winner by MRR: "
            f"{winner} (raw={raw_mrr:.4f}, praline={pra_mrr:.4f}, "
            f"top1_raw={raw_top1:.4f}, top1_praline={pra_top1:.4f})"
        )
    print(f"QA JSONL: {QA_JSONL_PATH}")
    print(f"XLSX: {XLSX_PATH}")
    print(f"CSV: {CSV_PATH}")
    print(f"Removed debug JSONL: {REMOVED_DEBUG_JSONL_PATH}")


if __name__ == "__main__":
    main()
