#!/usr/bin/env python3

"""
Simplified evaluation script using hash embeddings to avoid large downloads.
"""

import csv
import hashlib
import json
import math
import random
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import chromadb
import fitz

from textpraline.cleaner.clean import PralineReport, praline

# =============================================================================
# Config - Keep identical across all variants as required
# =============================================================================

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "rag_testing" / "corpus"
RESULTS_DIR = BASE_DIR / "rag_testing" / "datasets" / "light_experiment_simple"

# Identical parameters across all variants
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 10

RE_WS = re.compile(r"[ \t]+")

# =============================================================================
# Hash Embedding Function (deterministic, no downloads needed)
# =============================================================================

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
            for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9/_%-]*", (text or "").lower()):
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
# Data Models
# =============================================================================

@dataclass
class DocData:
    doc_id: str
    pdf_path: str
    pages: List[str]
    raw_doc_text: str
    variants: Dict[str, Tuple[str, PralineReport]]
    page_count: int
    raw_len: int

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
# Focused Variants - Only the three we need for this experiment
# =============================================================================

def build_documents_focused(corpus_dir: Path) -> List[DocData]:
    """Build documents with only the three variants needed for the experiment."""
    docs: List[DocData] = []
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {corpus_dir}")

    # Focused variants for this experiment
    VARIANTS_FOCUSED = {
        "raw": dict(kind="raw"),
        "praline_full": dict(preset="safe"),  # This is the "full" praline cleaning
        "praline_light": dict(preset="light"),  # Our new light variant
    }

    for pdf_path in pdf_paths:
        # Extract pages
        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                txt = page.get_text("text")
                pages.append(txt if txt is not None else "")
        
        raw_doc_text = "\n\n\n".join(pages)

        variants: Dict[str, Tuple[str, PralineReport]] = {}

        # raw variant
        variants["raw"] = (
            raw_doc_text,
            PralineReport(
                input_len=len(raw_doc_text),
                output_len=len(raw_doc_text),
                detail_enabled=False,
            ),
        )

        # praline variants
        for name, cfg in VARIANTS_FOCUSED.items():
            if cfg.get("kind") == "raw":
                continue
                
            cleaned, rep = praline(
                raw_doc_text,
                **cfg,
                debug_decisions=True,
                doc_id=pdf_path.stem,
                report="detail",
            )
            variants[name] = (cleaned, rep)

        docs.append(
            DocData(
                doc_id=pdf_path.stem,
                pdf_path=str(pdf_path),
                pages=pages,
                raw_doc_text=raw_doc_text,
                variants=variants,
                page_count=len(pages),
                raw_len=len(raw_doc_text),
            )
        )

    return docs

# =============================================================================
# Utility Functions
# =============================================================================

def norm_ws(text: str) -> str:
    return RE_WS.sub(" ", text.strip())

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9/_%-]*", text)

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

def build_chunks(docs: Sequence[DocData], variant: str) -> List[ChunkData]:
    out: List[ChunkData] = []
    for doc in docs:
        if variant not in doc.variants:
            raise KeyError(f"Unknown variant '{variant}' for doc '{doc.doc_id}'")

        text = doc.variants[variant][0]

        raw_page_offsets = compute_page_offsets(doc.pages)
        chunks = chunk_text_with_offsets(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

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
                    page_span=char_span_to_page_span(start, end, raw_page_offsets) if variant == "raw" else "",
                )
            )
    return out

def build_collection(
    client: chromadb.ClientAPI,
    name: str,
    embed_fn: HashEmbeddingFunction,
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

# =============================================================================
# QA Generation (simplified version)
# =============================================================================

def split_sentences(text: str) -> List[str]:
    normalized = norm_ws(text.replace("\n", " "))
    if not normalized:
        return []
    sents = [norm_ws(s) for s in re.split(r"(?<=[.!?])\s+", normalized) if norm_ws(s)]
    return sents

def is_boilerplate_sentence(sent: str) -> bool:
    s = sent.strip()
    if not s:
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
        if low in {"the", "and", "for", "with", "that", "this", "from", "into"}:
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
    merged = (anchor_words + words)[:40]
    return " ".join(merged[:20] if len(merged) >= 20 else merged)

def generate_questions(docs: Sequence[DocData], *, qa_variant: str) -> List[QAItem]:
    qa_items: List[QAItem] = []
    qid_counter = 1

    for doc in docs:
        if qa_variant not in doc.variants:
            raise KeyError(f"QA variant '{qa_variant}' missing for doc '{doc.doc_id}'")

        qa_text = doc.variants[qa_variant][0]

        sents = split_sentences(qa_text)
        scored: List[Tuple[float, int, str]] = []
        for i, sent in enumerate(sents):
            score = sentence_informativeness(sent)
            if score >= 0:
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
            if len(selected) >= 10:  # Target 10 questions per doc
                break

        if len(selected) < 8:  # Minimum 8 questions per doc
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
                if len(selected) >= 8:
                    break

        selected = selected[:15]  # Maximum 15 questions per doc
        for sent in selected:
            qid = f"q{qid_counter:04d}"
            qa_items.append(
                QAItem(
                    qid=qid,
                    question=build_question(sent, qid_counter),
                    expected_doc_id=doc.doc_id,
                    expected_answer_snippet=build_answer_snippet(qa_text, sent),
                )
            )
            qid_counter += 1

    return qa_items

# =============================================================================
# Evaluation Metrics
# =============================================================================

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

# =============================================================================
# Main Evaluation Function
# =============================================================================

def run_focused_evaluation():
    """Run the focused evaluation comparing RAW vs Praline Full vs Praline Light."""
    
    print("Building documents with focused variants...")
    docs = build_documents_focused(CORPUS_DIR)
    print(f"Processed {len(docs)} documents")
    
    # Use raw variant for QA generation
    qa_items = generate_questions(docs, qa_variant="raw")
    print(f"Generated {len(qa_items)} questions")
    
    # Build chunks for each variant
    variant_names = ["raw", "praline_full", "praline_light"]
    chunks_by_variant = {v: build_chunks(docs, v) for v in variant_names}
    
    print("Chunk statistics:")
    for v, chunks in chunks_by_variant.items():
        print(f"  {v}: {len(chunks)} chunks")
    
    # Build Chroma collections with hash embeddings
    print("Building Chroma collections with hash embeddings...")
    embed_fn = HashEmbeddingFunction(dim=768)
    client = chromadb.Client()
    
    cols = {}
    for v in variant_names:
        cols[v] = build_collection(client, f"light_exp__{v}", embed_fn, chunks_by_variant[v])
        print(f"  Indexed {v}: {len(chunks_by_variant[v])} chunks")
    
    # Run evaluation
    print("\nRunning evaluation...")
    question_rows = []
    
    # First pass: get RAW baseline
    raw_results = {}
    for qa in qa_items:
        raw_topk = query_topk(cols["raw"], qa.question, top_k=TOP_K)
        raw_docs = top_docs_unique(raw_topk)
        raw_results[qa.qid] = {
            "mrr": mrr(raw_docs, qa.expected_doc_id),
            "recall@1": recall_at_k(raw_docs, qa.expected_doc_id, 1),
            "recall@5": recall_at_k(raw_docs, qa.expected_doc_id, 5),
            "recall@10": recall_at_k(raw_docs, qa.expected_doc_id, 10),
            "top1_doc": raw_docs[0] if raw_docs else "",
        }
    
    # Second pass: evaluate all variants against RAW baseline
    for qa in qa_items:
        for v in variant_names:
            topk = query_topk(cols[v], qa.question, top_k=TOP_K)
            ranked = top_docs_unique(topk)
            
            mrr_v = mrr(ranked, qa.expected_doc_id)
            recall1 = recall_at_k(ranked, qa.expected_doc_id, 1)
            recall5 = recall_at_k(ranked, qa.expected_doc_id, 5)
            recall10 = recall_at_k(ranked, qa.expected_doc_id, 10)
            
            # Compare against RAW baseline
            raw_baseline = raw_results[qa.qid]
            delta_mrr = mrr_v - raw_baseline["mrr"]
            
            question_rows.append({
                "qid": qa.qid,
                "variant": v,
                "question": qa.question,
                "expected_doc_id": qa.expected_doc_id,
                "mrr": mrr_v,
                "recall@1": recall1,
                "recall@5": recall5,
                "recall@10": recall10,
                "top1_doc": ranked[0] if ranked else "",
                "delta_mrr_vs_raw": delta_mrr,
                "raw_mrr": raw_baseline["mrr"],
                "raw_recall@1": raw_baseline["recall@1"],
            })
    
    # Compute metrics
    def avg(rows, col):
        return sum(float(r[col]) for r in rows) / max(len(rows), 1)
    
    # Summary metrics
    summary = {}
    for v in variant_names:
        vr = [r for r in question_rows if r["variant"] == v]
        summary[v] = {
            "mrr": avg(vr, "mrr"),
            "recall@1": avg(vr, "recall@1"),
            "recall@5": avg(vr, "recall@5"),
            "recall@10": avg(vr, "recall@10"),
            "delta_mrr_vs_raw": avg(vr, "delta_mrr_vs_raw"),
            "count": len(vr),
        }
    
    # Analysis: queries where RAW wins R@1 over others
    raw_wins_r1 = []
    for qa in qa_items:
        raw_recall1 = raw_results[qa.qid]["recall@1"]
        for v in ["praline_full", "praline_light"]:
            vr = [r for r in question_rows if r["qid"] == qa.qid and r["variant"] == v]
            if vr:
                variant_recall1 = vr[0]["recall@1"]
                if raw_recall1 > variant_recall1:
                    raw_wins_r1.append({
                        "qid": qa.qid,
                        "question": qa.question,
                        "expected_doc_id": qa.expected_doc_id,
                        "variant": v,
                        "raw_recall@1": raw_recall1,
                        "variant_recall@1": variant_recall1,
                    })
    
    # Analysis: queries where praline_light improves over RAW
    light_improvements = []
    for qa in qa_items:
        raw_mrr = raw_results[qa.qid]["mrr"]
        vr = [r for r in question_rows if r["qid"] == qa.qid and r["variant"] == "praline_light"]
        if vr:
            light_mrr = vr[0]["mrr"]
            if light_mrr > raw_mrr:
                light_improvements.append({
                    "qid": qa.qid,
                    "question": qa.question,
                    "expected_doc_id": qa.expected_doc_id,
                    "raw_mrr": raw_mrr,
                    "light_mrr": light_mrr,
                    "delta_mrr": light_mrr - raw_mrr,
                })
    
    # Export results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write detailed results
    with (RESULTS_DIR / "detailed_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "qid", "variant", "question", "expected_doc_id",
            "mrr", "recall@1", "recall@5", "recall@10", "top1_doc",
            "delta_mrr_vs_raw", "raw_mrr", "raw_recall@1"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(question_rows)
    
    # Write summary
    with (RESULTS_DIR / "summary_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant", "mrr", "recall@1", "recall@5", "recall@10", "delta_mrr_vs_raw", "count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v, metrics in summary.items():
            writer.writerow({**metrics, "variant": v})
    
    # Write analysis results
    with (RESULTS_DIR / "raw_wins_r1.json").open("w", encoding="utf-8") as f:
        json.dump(raw_wins_r1, f, indent=2, ensure_ascii=False)
    
    with (RESULTS_DIR / "light_improvements.json").open("w", encoding="utf-8") as f:
        json.dump(light_improvements, f, indent=2, ensure_ascii=False)
    
    # Export top-1 chunks for manual inspection
    top1_chunks = []
    for qa in qa_items:
        for v in variant_names:
            topk = query_topk(cols[v], qa.question, top_k=1)
            if topk:
                top1_chunks.append({
                    "qid": qa.qid,
                    "variant": v,
                    "question": qa.question,
                    "expected_doc_id": qa.expected_doc_id,
                    "top1_chunk": topk[0],
                })
    
    with (RESULTS_DIR / "top1_chunks.json").open("w", encoding="utf-8") as f:
        json.dump(top1_chunks, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Embedding backend: hash-embedding-function (deterministic)")
    print(f"Documents: {len(docs)}")
    print(f"Questions: {len(qa_items)}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print()
    
    print("Metrics by variant:")
    print(f"{'Variant':<15} {'MRR':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10} {'ΔMRR vs RAW':<12}")
    print("-" * 70)
    
    for v in variant_names:
        m = summary[v]
        print(f"{v:<15} {m['mrr']:.4f} {m['recall@1']:.4f} {m['recall@5']:.4f} {m['recall@10']:.4f} {m['delta_mrr_vs_raw']:+.4f}")
    
    print(f"\nRAW wins R@1 over others: {len(raw_wins_r1)} queries")
    print(f"Praline Light improvements over RAW: {len(light_improvements)} queries")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nFiles generated:")
    print("  - detailed_results.csv: All question-level results")
    print("  - summary_results.csv: Aggregated metrics")
    print("  - raw_wins_r1.json: Queries where RAW wins R@1")
    print("  - light_improvements.json: Queries where Light improves over RAW")
    print("  - top1_chunks.json: Top-1 chunks for manual inspection")

if __name__ == "__main__":
    run_focused_evaluation()