<p align="center">
  <img src="assets/TextPraline_logo.png" alt="TextPraline Logo" width="300"/>
</p>
<h1 align="center">TextPraline 🍬</h1>

<p align="center">
TextPraline is the layer used after text extraction to "sweeten" the extracted lines.  
Perfect before a RAG ingestion 😉
</p>

---

## What is TextPraline?

**TextPraline** is a deterministic, extraction-agnostic text refinement layer designed to run **after**:

- PDF text extraction
- OCR pipelines
- Vision encoders
- HTML scrapers
- Any raw text source

Its goal is simple:

> Turn unstable, noisy, extraction-level text into clean, ingestion-ready text.

TextPraline does **not** rewrite content.  
It stabilizes and refines it.

---

## Why does this matter?

Raw extracted text is often:

- polluted with invisible Unicode artifacts
- broken by layout issues (PDF column noise, vertical text)
- full of repeated headers/footers
- corrupted by glyph artefacts or `(cid:123)` markers
- unstable in whitespace formatting
- inconsistent in punctuation normalization

These artifacts degrade:

- embedding quality
- chunking accuracy
- retrieval performance
- downstream reasoning

TextPraline ensures that what gets embedded is structurally stable and semantically clean.

---

## What TextPraline Does

### Universal Cleaning
- Unicode normalization (NFKC)
- Removes:
  - control characters
  - zero-width characters
  - BOM
  - soft hyphen
  - variation selectors
  - Private Use Area (PUA)
  - replacement/noncharacters (�)
- Normalizes typographic quotes and dashes
- Stabilizes list markers

### Extraction-Aware Cleaning (auto-detected)
- Removes `glyph<...>` artifacts
- Removes `(cid:NNN)` markers
- Unescapes HTML entities
- Collapses excessive blank lines
- Drops layout-noise blocks (PDF axis garbage / vertical text runs)
- Removes repeated header/footer lines (frequency-based)
- Removes common boilerplate patterns

### Profiles
- `safe` → preserves structure, canonical bullet `•`
- `markdown_safe` → markdown-friendly `-`
- `strict` → aggressive whitespace collapsing

### Monitoring
`debug=True` returns a `PralineReport` with metrics about:
- layout noise removed
- repeated lines removed
- ToC lines removed
- etc.

---

## Future Perspective

TextPraline currently focuses on **post-extraction refinement**.

A natural evolution would be:

- optional native PDF extraction adapters
- OCR integration modules
- multimodal-aware extraction helpers
- structured text reconstruction utilities

However, extraction is intentionally kept out of scope for now to preserve:
- modularity
- composability
- separation of concerns

---

## Contributing

TextPraline is built with real-world ingestion in mind.

If you encounter:
- extraction edge cases
- layout patterns not properly handled
- PDFs that break the heuristics
- corpus-specific artifacts

Please open an issue or propose a pull request.

We actively encourage improvements, new heuristics, and robustness enhancements.

<p align="center">Let’s make ingestion sweeter 🍬</p>
