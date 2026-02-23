TextPraline — Research Log

1. Initial Observation

While inspecting RAG chunks extracted from real-world PDFs, several issues were observed:
	•	Repeated headers and footers
	•	Academic affiliations polluting chunks
	•	(cid:XX) artifacts
	•	glyph<...> runs
	•	Private Use Area (PUA) characters
	•	Soft hyphens and zero-width characters
	•	Broken line layouts
	•	Boilerplate repeated across pages

These artifacts were negatively affecting:
	•	Chunk quality
	•	Embedding density
	•	Retrieval precision
	•	Ranking stability

This led to the hypothesis:

Removing extraction artifacts (without rewriting content) should improve retrieval quality.

⸻

2. Philosophy of TextPraline

TextPraline is designed as a:

Neutral, deterministic text cleaning layer for RAG ingestion.

It is not:
	•	A paraphraser
	•	A rewriter
	•	A summarizer
	•	A prompt optimizer

It strictly performs:
	•	Unicode normalization
	•	Artifact removal
	•	Layout stabilization
	•	Noise reduction

Design Constraint

Every transformation must pass this rule:

If it removes noise → allowed
If it alters semantic content → forbidden

3. Architecture Separation

Pipeline structure:

Extraction (PDF / OCR / Parser)
        ↓
TextPraline (neutral cleaner)
        ↓
Chunking layer
        ↓
Embedding
        ↓
Vector Store
        ↓
Retrieval
        ↓
LLM

Each layer has a clearly defined responsibility.

TextPraline does not:
	•	Chunk
	•	Rerank
	•	Reformulate
	•	Interpret

⸻

4. Experimental Setup

Corpus

Multi-domain benchmark including:
	•	Astrophysics research paper
	•	LLM academic paper
	•	IMF institutional report
	•	ESG corporate report
	•	PEP8 technical document

This ensures evaluation across:
	•	Scientific layout
	•	Corporate layout
	•	Technical text
	•	Institutional text
	•	Clean structured documents

⸻

5. Chunking Strategy for Evaluation

To isolate the cleaning effect:
	•	Character-based chunking
	•	Fixed chunk_size
	•	Fixed chunk_overlap
	•	No paragraph-sensitive splitting

Rationale:

Avoid chunk boundary variations caused by newline normalization.

This ensures evaluation measures cleaning impact only.

⸻

6. Metrics Used

Doc-level evaluation:
	•	Recall@1
	•	Recall@5
	•	Recall@10
	•	MRR (Mean Reciprocal Rank)

Additionally:
	•	Number of chunks
	•	Distance comparison (RAW vs PRALINE)
	•	Debug top-k inspection

⸻

7. Observations

Indexed RAW:     docs=5, chunks=1238
Indexed PRALINE: docs=5, chunks=1221

Praline reduced:
	•	Total characters
	•	Slight number of chunks
	•	Extraction noise

Retrieval quality remained:
	•	Stable
	•	Non-degrading
	•	Slightly altered distances but same ranking

Conclusion:

TextPraline does not harm retrieval signal under neutral chunking.

⸻

8. Important Insight

Optimizing text like a prompt is NOT equivalent to optimizing for embeddings.

LLM prompting:
	•	Improves reasoning clarity.

Embedding ingestion:
	•	Requires signal density preservation.

Therefore:

TextPraline avoids rewriting or enriching text.

⸻

9. Future Work

Short-term
	•	Add per-document PralineReport analysis
	•	Quantify noise reduction ratio
	•	Detect repetitive header/footer patterns automatically
	•	Improve layout-noise heuristics

Mid-term
	•	Benchmark on larger corpus (10–20 PDFs)
	•	Add chunk-level evaluation
	•	Measure embedding norm shifts

Long-term
	•	Introduce ChocoChunk (structure-aware chunking)
	•	Scientific layout detection
	•	Section-aware splitting
	•	Multi-column PDF heuristics

⸻

10. Guiding Principle

TextPraline should remain:
	•	Multi-domain
	•	Deterministic
	•	Neutral
	•	Extraction-agnostic
	•	Embedding-safe

It is a layout sanitizer, not a language transformer.

⸻

11. Open Questions
	•	Does noise removal consistently improve retrieval in large corpora?
	•	Can layout noise detection be made fully domain-agnostic?
	•	Should Praline expose cleaning profiles (academic / corporate / legal)?
	•	How to quantify “semantic density gain”?

⸻

12. Current Status

✔ Multi-domain tested
✔ Stable chunking evaluation
✔ No retrieval degradation observed
✔ Architecture separation validated

TextPraline is currently functioning as:

A reliable universal pre-ingestion cleaning layer for RAG pipelines.

19/02/2026
----------
13. Automatic Text Profile Detection

Motivation

During testing, an important issue emerged:

Aggressive layout heuristics (e.g. repeated line removal) can damage clean HTML/RSS documents while being necessary for PDF/OCR extraction noise.

This revealed a structural problem:

TextPraline must adapt to text extraction profiles without relying on file type.

Implementation

A new function was introduced:

.. code:

	detect_text_profile(text) -> Literal[
    "clean_web",
    "pdf_like",
    "ocr_like",
    "unknown"
	]

Detection is based on structural signals rather than file extensions:

PDF-like signals:
	•	(cid:XX) patterns
	•	glyph<...> runs
	•	Private Use Area (PUA) characters
	•	Page markers (“Page X of Y”)
	•	High short-line ratio
	•	Hyphenated line breaks (-\nword)

OCR-like signals:
	•	Broken word spacing
	•	Character repetition runs
	•	Non-printable character ratio
	•	Fragmented line layout

Clean web signals:
	•	Long natural paragraphs
	•	Normal punctuation density
	•	Low artifact presence

Key Insight

A PDF is not necessarily “pdf_like”.
A PDF may be classified as “ocr_like” if its extraction resembles OCR.

The classification reflects text structure, not file origin.

Conclusion:

Behavioral detection is more robust than extension-based branching.

⸻

14. Safe Mode Cleaning Strategy

An important correction was introduced:

If profile == “clean_web”:
	•	No aggressive repeated-line removal
	•	No destructive layout pruning

If profile in {“pdf_like”, “ocr_like”}:
	•	Layout normalization enabled
	•	Repeated-line heuristics allowed (with safeguards)

This prevents HTML pages (e.g. news live feeds) from being partially deleted.

This was a critical stability improvement.

⸻

15. Major Conceptual Shift

Initial assumption:
Noise reduction is the primary RAG ingestion challenge.

New realization:
The core problem of RAG is ambiguity, not noise.

Noise vs Ambiguity

Noise:
	•	Glyph artifacts
	•	OCR corruption
	•	Headers/footers
	•	Boilerplate

Effect:
	•	Dilutes embedding signal.

Ambiguity:
	•	Multi-topic chunks
	•	Overlapping semantic domains
	•	Vague queries
	•	Concept collision

Effect:
	•	Retrieves the wrong relevant chunk.

Ambiguity is structurally more dangerous than noise.

Noise reduces signal density.
Ambiguity corrupts retrieval direction.

⸻

16. Architectural Clarification

TextPraline handles:

Syntactic noise reduction.

It does not handle:

Semantic disambiguation.

This responsibility will belong to:

ChocoChunk (future module)

Proposed direction for ChocoChunk:
	•	Information-aware chunking
	•	Chunk-level scoring
	•	Pre-embedding ranking
	•	Context vector enrichment
	•	Optional judge-based answerability scoring

This separates:

Signal cleaning (Praline)
from
Signal structuring (ChocoChunk)

⸻

17. Theoretical Insight

Cleaning improves text stability.

But optimizing RAG requires:

Maximizing conceptual separability.

The problem shifts from:

“Remove extraction artifacts”

to

“Reduce semantic collision”

This reframes the roadmap from a layout problem to an information theory problem.

⸻

18. Next Steps

Short-term:
	•	Expose detected text profile in PralineReport
	•	Add debug observability for profile reasoning
	•	Strengthen safeguards for repeated-line removal
	•	Validate profile detection on 10+ mixed documents

Mid-term:
	•	Evaluate ambiguity impact separately from noise impact
	•	Measure retrieval stability under semantically overlapping chunks
	•	Define a chunk-level ambiguity metric

Long-term:
	•	Develop ChocoChunk as a semantic structuring layer
	•	Introduce context-aware chunk scoring
	•	Explore dual-embedding strategy (content + context)
	•	Experiment with answerability-aware ranking

⸻

19. Updated Understanding

TextPraline is not solving the hardest part of RAG.

It is stabilizing the foundation.

Ambiguity control will be the next frontier.

⸻

20. Current Status (Updated)

✔ Profile-based adaptive cleaning implemented
✔ Web-safe mode established
✔ PDF/OCR detection functional
✔ Retrieval stability preserved
✔ Conceptual architecture clarified

TextPraline remains:

A deterministic, extraction-agnostic, embedding-safe ingestion sanitizer.

The next phase will move from cleaning to structuring.
