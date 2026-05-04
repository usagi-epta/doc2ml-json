# doc2ml-json v0.6.2 (Pre-release)

> Turn any document into structured, machine-learning-ready JSON — fully offline, zero API keys.

`doc2ml-json` is an **AI agent skill** that ingests documents in any format (PDF, EPUB, DOCX, HTML, Markdown, code files, and more) and converts them into a canonical JSON schema optimized for machine learning pipelines: fine-tuning, RAG chunking, embedding generation, and sequence-to-sequence training.

Unlike cloud-based alternatives, `doc2ml-json` runs **entirely locally** with no API keys, no GPU, and no network dependencies. It is packaged as a portable `SKILL.md` file that any agent framework supporting skill files (Claude Code, OpenClaw, Cursor, etc.) can load and execute.

---

## Features

| Feature | Details |
|---------|---------|
| **9 supported formats** | PDF, EPUB, DOCX, TXT, HTML, Markdown, Python, JavaScript, and 50+ code languages |
| **OCR for scanned PDFs** | Automatic Tesseract OCR fallback when PDFs have no extractable text |
| **Structure preservation** | Headings, sections, subsections, tables, lists, code blocks, quotes, figures, footnotes |
| **ML-native schema** | Every block carries `chunk_id`, `token_count_est`, `char_count`, `context_window`, `embedding_ready` |
| **Built-in chunking** | `chunk_for_ml.py` splits documents into context-window-compliant chunks (default ≤512 tokens, 50-token overlap) |
| **Validation pipeline** | `validate_output.py` enforces schema compliance and ML readiness checks |
| **Multilingual** | Automatic language detection with script analysis (supports bilingual docs like en/ja) |
| **Inline element preservation** | Links, bold, italic, inline code, abbreviations, highlights, strikethrough preserved as Markdown-style markers |
| **Robust error recovery** | DOCX XML fallback when `python-docx` crashes; scanned PDF detection; corrupted EPUB handling |
| **Zero external dependencies** | No API keys, no cloud services, no GPU required |

---

## Supported Formats & Strategies

| Format | Detection | Extraction Strategy | Special Handling |
|--------|-----------|---------------------|------------------|
| **PDF** | Extension + magic bytes | `PyPDF2` → `pdfplumber` → `PyMuPDF` → **Tesseract OCR** fallback | Scanned PDF detection, layout analysis, font-based heading inference, OCR for image-based pages |
| **EPUB** | Extension + ZIP structure | `ebooklib` OPF parsing → `BeautifulSoup` chapter extraction | Spine navigation, chapter boundary detection, CSS style stripping |
| **DOCX** | Extension + ZIP structure | `python-docx` → **XML fallback** on crash | Footnote/endnote extraction, table parsing, heading style detection |
| **HTML** | Extension + `<html>` tag | `BeautifulSoup` DOM parsing | Nav/header/footer stripping, semantic tag usage (`<main>`, `<article>`), Open Graph meta extraction |
| **Markdown** | Extension | `markdown` lib → `BeautifulSoup` | YAML frontmatter, fenced code blocks, heading hierarchy |
| **TXT** | Extension + content | Line-based structure inference | Encoding detection, heading pattern matching, key-value extraction |
| **Code files** | 50+ extensions + shebang | Full-file `code_block` with metadata | Function/class/import extraction via regex, language detection |

---

## Installation

### As an Agent Skill

Download the `.skill` file and place it in your agent's skills directory:

```bash
# Claude Code
cp doc2ml-json-v0.6.2.skill ~/.claude/skills/

# OpenClaw / generic
cp doc2ml-json-v0.6.2.skill ~/.config/openclaw/skills/
```

### From Source

```bash
git clone https://github.com/yourusername/doc2ml-json.git
cd doc2ml-json
# The skill is self-contained; scripts run with standard Python libraries
```

### Python Dependencies (optional but recommended)

```bash
# Core dependencies
pip install beautifulsoup4 lxml pyyaml python-docx PyPDF2 pdfplumber ebooklib markdown

# Optional: OCR support for scanned PDFs
pip install pytesseract pdf2image
# Also install system binaries: apt install tesseract-ocr poppler-utils  # Ubuntu/Debian
```

> The skill degrades gracefully if libraries are missing — it will use file-format conversion or simpler parsers. OCR is completely optional.

---

## Quick Start

### 1. Process a Document

```bash
python3 scripts/detect_format.py my_document.pdf
python3 scripts/extract_structure.py my_document.pdf --output-dir ./output
```

### 2. Validate the Output

```bash
python3 scripts/validate_output.py ./output/*.doc2ml.json
```

### 3. Chunk for ML

```bash
python3 scripts/chunk_for_ml.py ./output/*.doc2ml.json \
  --output ./chunks.json \
  --max-tokens 512 \
  --overlap 50
```

---

## Output Schema

The canonical output is a single JSON file per document with this top-level structure:

```json
{
  "metadata": {
    "title": "...",
    "authors": [{"name": "..."}],
    "source": { "uri": "...", "mime_type": "...", "checksum_sha256": "..." },
    "language": { "detected": "en", "confidence": 0.99 },
    "statistics": { "block_count": 279, "total_token_count_est": 85708, ... },
    "dates": { "created": "...", "modified": "..." }
  },
  "structure": {
    "node_type": "root", "level": 0,
    "title": "...", "chunk_ids": [...],
    "children": [ /* nested sections */ ]
  },
  "blocks": [
    {
      "chunk_id": "blk-000",
      "type": "heading",
      "text_plain": "Introduction",
      "text_original": "Introduction",
      "token_count_est": 3,
      "char_count": 14,
      "language": "en",
      "embedding_ready": true,
      "content": { "text": "Introduction", "level": 1, ... },
      "provenance": { "page_number": 1, "source_format": "pdf" },
      "context_window": { "prev_chunk_id": "...", "next_chunk_id": "...", "parent_heading_chunk_id": "..." }
    }
  ],
  "ml_index": {
    "chunk_id_map": { "blk-000": 0, ... },
    "heading_map": { "blk-000": "Introduction" },
    "embedding_candidates": ["blk-000", ...],
    "chunk_boundaries": [[0, 5], [5, 10], ...]
  }
}
```

### Block Types

- `heading` — Document headings (h1–h6)
- `paragraph` — Body text with inline elements preserved
- `table` — Structured data with headers, rows, cells, and HTML representation
- `list` — Ordered/unordered lists with nested list support
- `code_block` — Source code with language detection and metadata
- `quote` — Block quotations with attribution
- `figure` — Images with captions and alt text
- `definition_list` — Term/definition pairs
- `footnote` — Footnotes and endnotes
- `divider` — Horizontal rules (`<hr>`)
- `image_placeholder` — Images without captions

### ML Use Case Mappings

| ML Task | How to use the JSON |
|---------|---------------------|
| **LLM Fine-tuning** | Feed `blocks[].text_plain` as training examples; use `heading_path` as instruction context |
| **RAG / Vector DB** | Embed `blocks[].text_plain` where `embedding_ready: true`; use `chunk_id` as key and `context_window` for retrieval augmentation |
| **Document QA** | Map answers to `block_ids` for provenance; use `section_path` for scoped retrieval |
| **Text Classification** | Label `blocks[]` by `structure_path` or `heading_path`; uniform chunk sizes prevent length bias |
| **Seq2Seq / Summarization** | Use adjacent chunks as input-output pairs; `context_window` provides surrounding context |
| **Embedding Generation** | Batch-encode `embedding_candidates` with `token_count_est` for padding decisions |

---

## Example: Full Pipeline

```python
import subprocess
import json

# Step 1: Detect format
subprocess.run(["python3", "scripts/detect_format.py", "paper.pdf"])

# Step 2: Extract
subprocess.run(["python3", "scripts/extract_structure.py", "paper.pdf", "--output-dir", "./out"])

# Step 3: Load and inspect
with open("./out/paper.doc2ml.json") as f:
    doc = json.load(f)

print(f"Title: {doc['metadata']['title']}")
print(f"Blocks: {doc['metadata']['statistics']['block_count']}")
print(f"Tokens: {doc['metadata']['statistics']['total_token_count_est']}")

# Step 4: Chunk for RAG
subprocess.run([
    "python3", "scripts/chunk_for_ml.py",
    "./out/paper.doc2ml.json",
    "--output", "./out/chunks.json",
    "--max-tokens", "512",
    "--overlap", "50"
])

# Step 5: Load chunks
with open("./out/chunks.json") as f:
    chunks = json.load(f)

for chunk in chunks["chunks"][:3]:
    print(f"Chunk {chunk['chunk_id']}: {chunk['token_count_est']} tokens")
    print(f"  Path: {'/'.join(chunk['structure_path'])}")
    print(f"  Text: {chunk['text'][:100]}...")
```

---

## Scripts Reference

| Script | Purpose | CLI |
|--------|---------|-----|
| `detect_format.py` | MIME type, encoding, format detection | `python3 detect_format.py <file> --output <json>` |
| `extract_structure.py` | Main extraction pipeline | `python3 extract_structure.py <file> --output-dir <dir>` |
| `validate_output.py` | Schema + ML readiness validation | `python3 validate_output.py <json> --output <report>` |
| `chunk_for_ml.py` | Context-window chunking | `python3 chunk_for_ml.py <json> --output <json> --max-tokens 512 --overlap 50` |

---

## How It Compares

| | **doc2ml-json** | LlamaParse | MinerU | Unstructured-IO |
|---|---|---|---|---|
| **Format** | Agent skill | Agent skill | Agent skill | Python library |
| **Runs offline** | ✅ Yes | ❌ Cloud API | ❌ API/GPU | ⚠️ Heavy deps |
| **ML schema** | ✅ Purpose-built | Partial | Partial | Partial |
| **Built-in chunking** | ✅ Yes | ❌ Separate | ❌ Separate | ⚠️ Extra step |
| **ML validation** | ✅ Yes | ❌ | ❌ | ❌ |
| **Code files** | ✅ 50+ languages | ❌ | ❌ | ❌ |
| **No API keys** | ✅ Yes | ❌ | ❌ | ✅ Yes |
| **Bilingual docs** | ✅ Script detection | Limited | Limited | Limited |

---

## Contributing

Contributions are welcome. Priority areas:

- Additional format support (RTF, ODT, LaTeX, Jupyter notebooks)
- OCR integration for scanned PDFs (Tesseract, EasyOCR)
- Table structure extraction (header/row/cell semantics)
- Language-specific tokenization for more accurate `token_count_est`
- Additional inline element types (math, chemistry, musical notation)

Please open an issue before submitting a PR.

---

## License

MIT License — free for commercial and non-commercial use.

---

## Acknowledgements

This skill was developed and built with assistance from [Kimi](https://kimi.moonshot.cn), using the **skill-creator-swarm** built-in agent framework for multi-agent orchestration and skill packaging framework. Evaluated against [LlamaParse](https://github.com/run-llama/llama_parse), [MinerU](https://github.com/opendatalab/MinerU), and [Unstructured-IO](https://github.com/Unstructured-IO/unstructured) as baseline comparisons.
