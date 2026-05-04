---
name: doc2ml-json
description: >
  Ingest any document (PDF, EPUB, DOCX, TXT, HTML, Markdown) and convert it
  into structured, ML-ready JSON. Use when the user asks to: extract text from
  documents, convert documents to structured data, create ML datasets from
  documents, parse PDF/EPUB/DOCX content, chunk documents for RAG, or produce
  tokenizable JSON from unstructured documents. Triggers on file uploads of
  supported formats or requests mentioning document conversion, text extraction,
  dataset creation from documents.
---

# doc2ml-json: Document-to-ML-JSON Converter

## Overview

This skill provides a complete pipeline for converting unstructured documents into
structured JSON optimized for machine learning workflows. It handles format detection,
content extraction, structural understanding, normalization, and JSON generation.

## Supported Formats

- PDF (text-based and scanned/OCR)
- EPUB (ebooks)
- DOCX (Microsoft Word)
- TXT / Markdown / MD
- HTML (single file or archive)

## Workflow Decision Tree

Start here. Based on the user's request, follow the appropriate path:

**User uploaded a document file?** → Follow "Full Pipeline" below
**User provided a directory of documents?** → Follow "Batch Processing" below
**User wants to extract specific sections?** → Follow "Selective Extraction" below
**User wants document converted to training data?** → Follow "ML Dataset Generation" below

## Full Pipeline

### Phase 1: Format Detection

Run `scripts/detect_format.py` on the input file. If unavailable, implement the
three-layer cascade manually:

1. **File extension** → map to MIME type (`.pdf` → `application/pdf`, `.epub` → `application/epub+zip`, `.docx` → `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `.md` → `text/markdown`, `.html` → `text/html`, `.txt` → `text/plain`)
2. **Magic bytes** → inspect first 16 bytes (`%PDF-` = PDF, `PK\x03\x04` = ZIP-based, `<?xml` = XML, `<html` = HTML)
3. **Deep inspection** → for ZIP-based files, inspect internal structure (DOCX has `word/document.xml`, EPUB has `mimetype` containing `epub`)

Confidence scoring: all three agree = 1.0; signature + extension agree = 0.85; only signature = 0.70; only extension = 0.50. If confidence < 0.5, flag for manual review.

For PDFs specifically, check if scanned: open with `fitz` (PyMuPDF), if `page.get_text()` is empty but `page.get_images()` has entries on >50% of pages, set `is_scanned: true`.

### Phase 2: Content Extraction

Route by detected format. Use these libraries:

| Format | Primary Library | Fallback | Key Extraction Target |
|--------|-----------------|----------|----------------------|
| PDF | `pdfplumber` + `fitz` (PyMuPDF) | `PyPDF2` | Text blocks with bbox, tables, metadata |
| EPUB | `ebooklib` + `BeautifulSoup` | Manual ZIP + XML | Spine reading order, chapter HTML, TOC |
| DOCX | `python-docx` + `lxml` | `zipfile` + manual XML | Paragraphs, tables, lists, styles, footnotes |
| TXT/MD | Native Python + `markdown` | `yaml` for frontmatter | Headings, paragraphs, code blocks, lists, tables |
| HTML | `BeautifulSoup` | `lxml` | Semantic tags, heading hierarchy, links |

**PDF extraction details:**
- Use `pdfplumber.open(filepath)` → iterate pages → `page.extract_words()` for layout, `page.extract_tables()` for tables
- Use `fitz.open(filepath)` → `page.get_text("dict")` for font/size/style metadata, `doc.metadata` for document properties
- Detect headings by font size: >1.3x average and <200 chars = heading; level by size tier relative to max
- Remove headers/footers: collect text in top/bottom 10% margin across pages, drop lines appearing on >50% of pages

**EPUB extraction details:**
- Open as ZIP → read `META-INF/container.xml` → find OPF path
- Parse OPF: extract `dc:title`, `dc:creator`, `dc:language`, manifest, spine
- Navigate spine sequentially; parse each chapter with BeautifulSoup; remove `script`, `style`, `nav`
- Extract TOC from NCX (EPUB2) or NavDoc (EPUB3)

**DOCX extraction details:**
- Open with `python-docx` → iterate `doc.paragraphs`, `doc.tables`
- Detect headings from paragraph style names (`Heading1` → level 1)
- Detect lists from `w:numPr` in paragraph XML; resolve list types via `word/numbering.xml`
- Extract footnotes from `word/footnotes.xml`; hyperlinks from `word/_rels/document.xml.rels`
- Handle merged cells: check `w:gridSpan` in table cell XML

**Markdown extraction details:**
- Detect YAML frontmatter between `---` delimiters; parse with `yaml.safe_load`
- Parse line-by-line: ATX headings (`#`), fenced code blocks (```` ``` ````), GFM tables (`|`), lists (`-`, `*`, `1.`), blockquotes (`>`)
- Setext headings: underlined with `=` (h1) or `-` (h2)

### Phase 3: Structure Understanding

1. **Heading hierarchy reconstruction:**
   - Collect all heading blocks; if none exist, infer from short all-caps paragraphs
   - Normalize skipped levels: `[1, 3, 3, 5]` → `[1, 2, 2, 3]` by remapping unique sorted levels
   - If first heading level > 1, insert inferred h1 from first substantial text or filename
   - Promote paragraph-as-heading: short (<100 chars) all-caps or bold-styled paragraphs

2. **Section boundary detection:**
   - Group blocks into sections based on heading hierarchy
   - Maintain a section stack: on heading at level L, pop stack to find parent with level < L
   - Content before first heading → anonymous section at level 0

3. **List nesting reconstruction:**
   - From flat items with `level` fields, build nested tree using stack depth
   - If `level` jumps by >1, create intermediate nesting levels

4. **Table normalization:**
   - Build rectangular grid handling `colspan` and `rowspan`
   - Track span_map for cells propagated from above rows
   - Pad rows to uniform `max_cols`

### Phase 4: Normalization

Apply in order:

1. **Unicode NFKC** → `unicodedata.normalize("NFKC", text)` for compatibility characters
2. **Whitespace** → collapse multiple spaces/tabs to single space; normalize `\r\n` → `\n`; remove zero-width chars (`\u200b`, `\ufeff`); collapse >2 newlines to 2
3. **Encoding** → re-encode as UTF-8 strict, fall back to replacement on error
4. **Dates** → parse common formats (`%Y-%m-%d`, `%d/%m/%Y`, `%B %d, %Y`) to ISO 8601; keep original on failure
5. **Links** → remove tracking params (`utm_*`, `fbclid`, `gclid`); prefer `https`; resolve relative URLs if `base_url` known

### Phase 5: JSON Generation

Build the canonical `Doc2MLDocument` structure (see schema quick reference below).

Key construction rules:
- `chunk_id` format: `blk-{zero-padded-index:03d}`
- Every block gets: `content`, `text_plain`, `char_count`, `token_count_est` (chars/4), `embedding_ready`, `context_window`, `provenance`, `language`, `semantics`, `relations`
- `context_window`: link `prev_chunk_id`, `next_chunk_id`, `parent_heading_chunk_id`, `parent_structure_node_id`, and 100-char preview of surrounding text
- `provenance`: `page_number`, `source_location` (e.g., `h1`, `p tag`, `line 45-67`), `extraction_method`, `confidence`
- Build `structure` tree with `node_id`, `node_type` (document/part/chapter/section/...), `level`, `chunk_ids`, `children`
- Build `ml_index`: `chunk_id_map` (index + structure_path per block), `heading_map`, `embedding_candidates` (all `embedding_ready=True` blocks), `chunk_boundaries` (sections with token_count_est)

**Note on `metadata.source`:** When a Markdown file contains YAML frontmatter with a `source` field (e.g., `source: "Journal of Machine Learning"`), that declared source takes precedence over filesystem metadata. The output JSON stores both `declared_source` (frontmatter value) and `source_type: "frontmatter"` to distinguish it from `source_type: "filesystem"`. If no frontmatter source is present, `declared_source` falls back to the file path.

**Note on token counts:** `token_count_est` values are estimates computed with a whitespace-based heuristic (`words * 0.75`), not formal tokenizer counts. They may vary by ±30% from actual counts produced by subword tokenizers such as BPE or SentencePiece. Use them for relative sizing and chunking decisions, not for exact budget calculations.

### Phase 6: Validation & Output

Run `scripts/validate_output.py` on the generated JSON. Manual checks:

1. **Schema compliance** → validate against JSON Schema in `references/schema.md` section 9 (use `jsonschema` library if available, else manual required-field check)
2. **Roundtrip sanity** → text coverage ratio (JSON text chars / original text chars) should be >0.85; heading count should match; metadata fields preserved
3. **ML readiness** → all chunks have non-empty text; token counts are reasonable; no empty `embedding_ready` blocks; `chunk_boundaries` cover all `embedding_ready` blocks

Output: write `{document_id}.doc2ml.json` with indentation=2.

## Batch Processing

For directories or file lists:

1. Collect all files with supported extensions (recurse if requested)
2. Run Phase 1-6 on each file independently
3. Aggregate outputs:
   - **Per-document JSONs**: write individual `.doc2ml.json` files
   - **Dataset manifest**: generate `manifest.json` with `document_id`, `filename`, `block_count`, `token_count_est` per file
   - **Combined chunks**: optionally merge all `embedding_ready` blocks into a single flat JSONL for training
4. Parallelize with `multiprocessing.Pool` if >10 files; set `max_workers = min(8, cpu_count())`
5. Log failures per file with error code; do not fail the entire batch

## Selective Extraction

When user requests specific content only:

1. Parse the full document through Phase 1-4 (do not skip — structure needed for context)
2. Filter by criteria:
   - **Section/chapter**: collect `chunk_ids` from target `StructureNode`, emit only those blocks
   - **Block type**: filter `blocks` array by `type` (e.g., only `table`, only `code_block`)
   - **Page range**: filter by `provenance.page_number` within range
   - **Keyword search**: scan `text_plain` for regex matches, emit matched blocks + their `context_window` previews
3. Rebuild a minimal `Doc2MLDocument` from filtered subset:
   - Update `structure` to only include nodes with matching `chunk_ids`
   - Update `ml_index.chunk_id_map` and `ml_index.heading_map`
   - Recalculate `metadata.statistics` for the subset

## ML Dataset Generation

Derive task-specific training data from the canonical JSON:

| Task | Strategy | Source Fields |
|------|----------|---------------|
| Instruction-following | Pair each heading with its section paragraphs as instruction → response | `ml_index.heading_map` + section blocks |
| Summarization | Abstract block as target, full body as input; or last paragraph per section as target | `abstract` blocks, `chunk_boundaries` |
| RAG retrieval | Sliding-window chunks over `embedding_candidates` with overlap | `ml_index.embedding_candidates`, `context_window` |
| Embeddings | One payload per `embedding_ready` block with metadata for vector DB filtering | `text_plain`, `structure_path`, `section_role` |
| Table-to-text | Table `records` + `caption` as input, nearest paragraph as target | `table` blocks, spatial proximity in `blocks` array |
| Code LLM | `code_block` content as output, preceding paragraph as instruction | `code_block` blocks, `context_window.prev_chunk_id` |
| QA | Citation cross-references as pseudo-QA pairs | `cross_references` with `ref_type=citation` |
| Classification | Document-level metadata + block statistics as features | `metadata.classification`, `metadata.statistics` |

Implementation: iterate over `blocks` or `ml_index` fields, transform to target format (Alpaca, ChatML, JSONL), write to task-specific output files.

## Error Handling

13 defined error codes with recovery strategies:

| Code | Severity | Recovery |
|------|----------|----------|
| `DETECT_001` Extension/signature mismatch | Warning | Use signature result, flag for review |
| `DETECT_002` Unknown file format | Error | Try generic text extraction; fail if binary |
| `DETECT_003` Corrupted ZIP (DOCX/EPUB) | Error | Try to repair ZIP, extract readable parts with `zipfile` |
| `EXTRACT_001` Password-protected PDF | Error | Report failure, request password |
| `EXTRACT_002` Scanned PDF without OCR text | Warning | Set `is_scanned=true`, note OCR recommendation |
| `EXTRACT_003` Missing fonts in PDF | Warning | Use raw text extraction fallback (`PyPDF2`) |
| `EXTRACT_004` Broken XML in DOCX | Error | Parse with `lxml` recovery mode (`recover=True`) |
| `EXTRACT_005` Encoding detection failure | Warning | Use UTF-8 with replacement characters |
| `STRUCT_001` No headings detected | Warning | Infer from paragraph patterns (all-caps, short lines) |
| `STRUCT_002` Broken heading hierarchy | Warning | Normalize levels, insert inferred headings |
| `STRUCT_003` Table extraction failed | Warning | Emit table as `unknown` block with raw text |
| `NORM_001` Invalid date format | Warning | Keep original, set `normalized: null` |
| `NORM_002` URL parse error | Warning | Keep original URL, log warning |

Recovery principle: **progressive enhancement**. Extract what you can; never fail entirely if partial extraction is possible. Log all errors in `metadata.ingestion` under `warnings` or `errors` arrays.

## Output Schema Quick Reference

Top-level structure:

```json
{
  "doc2ml_version": "1.0.0",
  "document_id": "uuid-v4",
  "metadata": {
    "title": "string",
    "authors": [{"name": "string"}],
    "source": {"uri": "...", "mime_type": "...", "filename": "...", "checksum_sha256": "...", "file_size_bytes": 0},
    "ingestion": {"ingestion_date": "ISO-8601", "processing_version": "doc2ml-json v1.0.0", "extractor": "...", "extractor_version": "...", "ingestion_pipeline": ["..."], "processing_duration_ms": 0},
    "language": {"detected": "en", "confidence": 0.97},
    "statistics": {"page_count": 0, "chapter_count": 0, "section_count": 0, "block_count": 0, "table_count": 0, "figure_count": 0, "footnote_count": 0, "total_char_count": 0, "total_token_count_est": 0, "total_word_count": 0},
    "classification": {"doc_type": "...", "genre": "...", "keywords": ["..."], "topics_ml": [{"label": "...", "score": 0.92}]},
    "dates": {"created": "...", "modified": "...", "published": "..."},
    "rights": {"license": "...", "copyright": "...", "open_access": true}
  },
  "structure": {"node_id": "root", "node_type": "document", "title": "...", "level": 0, "chunk_ids": ["..."], "children": [...]},
  "blocks": [{"chunk_id": "blk-000", "type": "heading|paragraph|table|list|code_block|...", "content": {...}, "text_plain": "...", "char_count": 0, "token_count_est": 0, "embedding_ready": true, "context_window": {...}, "provenance": {...}, "language": {...}, "semantics": {...}, "relations": {...}, "custom": {}}],
  "cross_references": [{"ref_id": "...", "ref_type": "citation|internal_link|figure_ref|...", "source_chunk_id": "...", "target_chunk_id": "...", "resolved": true}],
  "ml_index": {"chunk_id_map": {...}, "heading_map": [...], "embedding_candidates": [...], "chunk_boundaries": [...]},
  "custom": {}
}
```

## Resources

- `references/schema.md` — Full JSON schema specification, block types, examples, ML use-case mappings, and JSON Schema validator (Draft 2020-12)
- `references/workflow.md` — Complete per-format extraction recipes, edge case guide (scanned PDFs, password protection), tool mapping, and decision trees
- `scripts/detect_format.py` — Document format detection with extension, magic bytes, and deep inspection
- `scripts/extract_structure.py` — Main extraction pipeline for all supported formats
- `scripts/validate_output.py` — JSON schema validation with ML readiness metrics
- `scripts/chunk_for_ml.py` — Chunk JSON for ML context windows with section boundary respect
