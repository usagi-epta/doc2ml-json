# `doc2ml-json` Canonical Schema v0.6.2

**Status:** Pre-Release  
**Version:** 0.6.2  
**Date:** 2026-05-04  
**Purpose:** Universal document-to-ML JSON ingestion schema for fine-tuning, RAG, embeddings, and sequence modeling pipelines.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Main Schema Definition](#2-main-schema-definition)
3. [Block Types Enum](#3-block-types-enum)
4. [Per-Block Schema Reference](#4-per-block-schema-reference)
5. [Example 1: Simple Article / Blog Post](#5-example-1-simple-article--blog-post)
6. [Example 2: Academic Paper (Complex)](#6-example-2-academic-paper-complex)
7. [Example 3: Book / Chapter Hierarchy (EPUB-like)](#7-example-3-book--chapter-hierarchy-epub-like)
8. [ML Use-Case Mappings](#8-ml-use-case-mappings)
9. [Schema Validation JSON Schema (Draft 2020-12)](#9-schema-validation-json-schema-draft-2020-12)

---

## 1. Design Principles

| Principle            | Rationale                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Self-describing**  | Every object carries enough metadata to be processed without external context.                                 |
| **Hierarchical**     | Preserves original document semantics (parts → chapters → sections → subsections).                             |
| **Chunkable**        | Every content block is independently addressable via `chunk_id` for sliding-window or random-access pipelines. |
| **Normalized**       | Common metadata lives at the document level; block-level metadata is additive only.                            |
| **Extensible**       | The `custom` key-value store allows format-specific extensions without schema changes.                         |
| **Embedding-ready**  | Each block declares its own `embedding_ready` suitability and `context_window` references.                     |
| **Provenance-first** | Every block knows exactly where it came from in the original file.                                             |

---

## 2. Main Schema Definition

The top-level object is a **`Doc2MLDocument`**. It contains metadata, a structural hierarchy, and a flat registry of all content blocks.

```json
{
  "doc2ml_version": "0.6.2",
  "document_id": "uuid-v4",
  "metadata": { ...DocumentMetadata... },
  "structure": { ...DocumentStructure... },
  "blocks": [ ...BlockRegistry... ],
  "cross_references": [ ...CrossReference... ],
  "ml_index": { ...MLIndex... },
  "custom": {}
}
```

### 2.1 DocumentMetadata

```json
{
  "title": "string (required)",
  "subtitle": "string (optional)",
  "authors": [
    {
      "name": "string (required)",
      "orcid": "string (optional)",
      "affiliation": "string (optional)",
      "email": "string (optional)"
    }
  ],
  "source": {
    "uri": "string — original file path or URL",
    "mime_type": "string — e.g. application/pdf",
    "filename": "string — original filename",
    "checksum_sha256": "string — SHA-256 of original bytes",
    "file_size_bytes": 0
  },
  "ingestion": {
    "ingestion_date": "ISO-8601 datetime",
    "processing_version": "doc2ml-json v0.6.2",
    "extractor": "string — e.g. 'pypdfium2', 'pandoc', 'marker'",
    "extractor_version": "string",
    "ingestion_pipeline": ["step1", "step2"],
    "processing_duration_ms": 0
  },
  "language": {
    "detected": "ISO-639-1 code, e.g. 'en'",
    "confidence": 0.97,
    "declared": "string (optional — from document metadata)"
  },
  "statistics": {
    "page_count": 0,
    "chapter_count": 0,
    "section_count": 0,
    "block_count": 0,
    "table_count": 0,
    "figure_count": 0,
    "footnote_count": 0,
    "total_char_count": 0,
    "total_token_count_est": 0,
    "total_word_count": 0
  },
  "classification": {
    "doc_type": "string — e.g. 'academic_paper', 'book', 'blog_post', 'legal_contract'",
    "genre": "string (optional)",
    "keywords": ["keyword1", "keyword2"],
    "topics_ml": [
      {"label": "machine learning", "score": 0.92}
    ]
  },
  "dates": {
    "created": "ISO-8601 (optional)",
    "modified": "ISO-8601 (optional)",
    "published": "ISO-8601 (optional)"
  },
  "rights": {
    "license": "string (optional)",
    "copyright": "string (optional)",
    "open_access": true
  }
}
```

### 2.2 DocumentStructure

The `structure` field is a **tree of `StructureNode` objects** that reflects the logical hierarchy of the document. It does **not** duplicate content — it only holds references (`chunk_id`s) to blocks in the flat `blocks` registry.

```json
{
  "structure": {
    "node_id": "root",
    "node_type": "document",
    "title": "Document Title",
    "level": 0,
    "children": [
      {
        "node_id": "part-001",
        "node_type": "part",
        "title": "Part I: Introduction",
        "level": 1,
        "chunk_ids": ["blk-000", "blk-001"],
        "children": [
          {
            "node_id": "ch-001",
            "node_type": "chapter",
            "title": "Chapter 1",
            "level": 2,
            "chunk_ids": ["blk-002", "blk-003", "blk-004"],
            "children": [ ... ]
          }
        ]
      }
    ]
  }
}
```

**StructureNode fields:**

| Field        | Type            | Required | Description                                                                                                                         |
| ------------ | --------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `node_id`    | string          | Yes      | Unique identifier for this structural node.                                                                                         |
| `node_type`  | enum            | Yes      | One of: `document`, `part`, `chapter`, `section`, `subsection`, `subsubsection`, `appendix`, `front_matter`, `back_matter`, `page`. |
| `title`      | string          | No       | Human-readable title of this structural unit.                                                                                       |
| `level`      | integer         | Yes      | Depth in hierarchy (0 = root, 1 = part, 2 = chapter, 3 = section, etc.).                                                            |
| `chunk_ids`  | [string]        | No       | Ordered list of block IDs that belong directly to this node (not recursively).                                                      |
| `children`   | [StructureNode] | No       | Nested child nodes.                                                                                                                 |
| `page_start` | integer         | No       | Starting page number in original document.                                                                                          |
| `page_end`   | integer         | No       | Ending page number in original document.                                                                                            |
| `custom`     | object          | No       | Format-specific structural metadata.                                                                                                |

### 2.3 BlockRegistry

The `blocks` array is a **flat, ordered list** of all content blocks in the document. Every block has a globally unique `chunk_id` within the document. The order in the array reflects the **logical reading order**.

### 2.4 CrossReference

```json
{
  "cross_references": [
    {
      "ref_id": "ref-001",
      "ref_type": "citation",
      "source_chunk_id": "blk-042",
      "target_chunk_id": "blk-089",
      "target_structure_node_id": "sec-methods",
      "label": "Smith et al., 2023",
      "context_text": "As shown by Smith et al. (2023)...",
      "resolved": true
    }
  ]
}
```

**CrossReference fields:**

| Field                      | Type    | Required | Description                                                                                              |
| -------------------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `ref_id`                   | string  | Yes      | Unique reference identifier.                                                                             |
| `ref_type`                 | enum    | Yes      | `citation`, `internal_link`, `figure_ref`, `table_ref`, `footnote_ref`, `equation_ref`, `external_link`. |
| `source_chunk_id`          | string  | Yes      | Block containing the reference.                                                                          |
| `target_chunk_id`          | string  | No       | Target block (if resolved).                                                                              |
| `target_structure_node_id` | string  | No       | Target structural node (if resolved).                                                                    |
| `label`                    | string  | No       | Display text of the reference.                                                                           |
| `context_text`             | string  | No       | Surrounding sentence for context.                                                                        |
| `resolved`                 | boolean | Yes      | Whether the target was successfully located.                                                             |

### 2.5 MLIndex

The `ml_index` section provides pre-computed indices that accelerate downstream ML pipelines without requiring a full scan of the document.

```json
{
  "ml_index": {
    "chunk_id_map": {
      "blk-001": { "index": 0, "structure_path": ["root", "part-001", "ch-001", "sec-001"] }
    },
    "heading_map": [
      {"chunk_id": "blk-003", "heading_text": "Methods", "level": 3, "node_id": "sec-methods"}
    ],
    "embedding_candidates": ["blk-001", "blk-002", "blk-005"],
    "chunk_boundaries": [
      {"start_chunk_id": "blk-001", "end_chunk_id": "blk-010", "boundary_type": "semantic", "token_count_est": 512}
    ]
  }
}
```

---

## 3. Block Types Enum

Every block in the `blocks` array **must** declare a `type` from this enum.

| Block Type          | Description                                                 | Common ML Use                                                 |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| `heading`           | Section, chapter, or sub-section heading.                   | Structure-aware chunking, outline extraction, TOC generation. |
| `paragraph`         | Standard prose paragraph.                                   | Core text for embeddings, summarization, QA.                  |
| `table`             | Tabular data (with dual representation).                    | Structured QA, table-to-text, code generation.                |
| `list`              | Ordered or unordered list (bulleted, numbered, definition). | Feature extraction, hierarchical reasoning.                   |
| `code_block`        | Monospaced code or pseudo-code.                             | Code LLM fine-tuning, syntax-aware embeddings.                |
| `figure_caption`    | Caption associated with an image, diagram, or chart.        | Vision-language grounding, image captioning pairs.            |
| `footnote`          | Footnote or endnote content.                                | Citation analysis, reference-grounded QA.                     |
| `quote`             | Block quote, epigraph, or verbatim excerpt.                 | Attribution tasks, style transfer.                            |
| `metadata`          | Document-level metadata block (title page, DOI, ISBN).      | Bibliographic extraction, cataloging.                         |
| `equation`          | Mathematical equation or formula (LaTeX/MathML).            | Scientific QA, formula-to-text.                               |
| `abstract`          | Document abstract or summary.                               | Pre-training signal, summarization targets.                   |
| `keyword_block`     | Author-supplied keywords or index terms.                    | Topic modeling, tagging.                                      |
| `reference_entry`   | Single bibliographic reference entry.                       | Citation recommendation, citation text extraction.            |
| `page_header`       | Running header from original pagination.                    | Page-aware chunking, layout analysis.                         |
| `page_footer`       | Running footer, page numbers.                               | Layout analysis.                                              |
| `image_placeholder` | Placeholder for an image that was extracted separately.     | Vision-language pipelines.                                    |
| `annotation`        | Highlight, comment, or marginalia.                          | Review/summarization from annotations.                        |
| `divider`           | Horizontal rule or visual separator.                        | Structural boundary detection.                                |
| `unknown`           | Unclassified block that could not be typed.                 | Fallback for robustness.                                      |

---

## 4. Per-Block Schema Reference

Every block in the `blocks` array conforms to the following **base schema**. Specific block types extend this with additional typed fields under a `content` key.

### 4.1 Base Block Schema (all types)

```json
{
  "chunk_id": "blk-{zero-padded-index}",
  "type": "paragraph",
  "content": { ...type-specific... },
  "text_plain": "string — lossless plain-text rendering for tokenization",
  "char_count": 0,
  "token_count_est": 0,
  "embedding_ready": true,
  "context_window": {
    "prev_chunk_id": "blk-001",
    "next_chunk_id": "blk-003",
    "parent_heading_chunk_id": "blk-000",
    "parent_structure_node_id": "sec-001",
    "surrounding_text_preview": "string (first 100 chars of prev + next)"
  },
  "provenance": {
    "page_number": 1,
    "page_range": [1, 2],
    "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
    "source_location": "string — e.g. 'line 45-67' or 'char offset 1234-5678'",
    "extraction_method": "string — e.g. 'ocr', 'native_text', 'pandoc'",
    "confidence": 1.0
  },
  "language": {
    "detected": "en",
    "confidence": 0.99
  },
  "semantics": {
    "heading_level": null,
    "is_first_paragraph": false,
    "is_last_paragraph": false,
    "section_role": "string — e.g. 'introduction', 'method', 'result', 'discussion'",
    "sentiment_score": null,
    "readability_flesch": null
  },
  "relations": {
    "part_of_table": null,
    "part_of_list": null,
    "part_of_figure": null,
    "footnotes_for": [],
    "references": []
  },
  "custom": {}
}
```

### 4.2 Type-Specific Content Schemas

#### `heading`

```json
{
  "content": {
    "text": "1. Introduction",
    "level": 1,
    "numbered": true,
    "label": "string — e.g. '1.1' or 'Chapter 3'"
  }
}
```

#### `paragraph`

```json
{
  "content": {
    "text": "string — the full paragraph text with inline formatting preserved as markdown",
    "inline_elements": [
      {
        "type": "bold | italic | code | link | math | sub | sup | underline | strikethrough",
        "start": 10,
        "end": 20,
        "text": "highlighted text",
        "href": "string (for links)"
      }
    ],
    "sentences": [
      {"text": "First sentence.", "start_char": 0, "end_char": 15, "sentence_id": "s-001"}
    ]
  }
}
```

#### `table`

Tables have **dual representation** for maximum ML flexibility.

```json
{
  "content": {
    "caption": "Table 1: Experimental results",
    "caption_chunk_id": "blk-cap-001",
    "table_id": "tbl-001",
    "column_count": 4,
    "row_count": 5,
    "header_row_count": 1,
    "html": "<table><thead>...</thead><tbody>...</tbody></table>",
    "markdown": "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |",
    "records": [
      {"Col A": "1", "Col B": "2", "_row_index": 0, "_is_header": false}
    ],
    "cells": [
      {
        "row": 0,
        "col": 0,
        "text": "Header 1",
        "is_header": true,
        "rowspan": 1,
        "colspan": 1,
        "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 0.25, "y2": 0.1}
      }
    ],
    "footnotes": ["blk-ft-001"]
  }
}
```

#### `list`

```json
{
  "content": {
    "list_type": "ordered | unordered | definition",
    "start_number": 1,
    "items": [
      {
        "text": "First item",
        "item_id": "li-001",
        "level": 0,
        "marker": "• or 1. or (a)",
        "children": []
      }
    ]
  }
}
```

#### `code_block`

```json
{
  "content": {
    "code": "string — raw source code",
    "language": "python",
    "filename": "example.py (optional)",
    "line_numbers": true,
    "lines": [
      {"line_number": 1, "text": "def hello():"}
    ]
  }
}
```

#### `figure_caption`

```json
{
  "content": {
    "label": "Figure 3",
    "text": "Accuracy over training epochs.",
    "figure_id": "fig-003",
    "image_uris": ["path/to/extracted_fig_003.png"],
    "bounding_boxes": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5}]
  }
}
```

#### `footnote`

```json
{
  "content": {
    "note_id": "fn-001",
    "note_label": "1",
    "text": "See supplementary material.",
    "referenced_by": ["blk-042"],
    "backlinks_to": ["blk-042"]
  }
}
```

#### `quote`

```json
{
  "content": {
    "text": "quoted text",
    "attribution": "Author Name",
    "source": "Source Title",
    "cite_chunk_id": "blk-ref-001"
  }
}
```

#### `metadata`

```json
{
  "content": {
    "field": "doi",
    "value": "10.1000/example",
    "display_text": "DOI: 10.1000/example"
  }
}
```

#### `equation`

```json
{
  "content": {
    "latex": "E = mc^2",
    "mathml": "<math>...</math>",
    "plain_text": "E equals m c squared",
    "equation_id": "eq-001",
    "display_mode": true
  }
}
```

#### `reference_entry`

```json
{
  "content": {
    "raw_text": "Smith, J. (2023). Title. Journal, 1(1), 1-10.",
    "parsed": {
      "authors": [{"given": "J", "family": "Smith"}],
      "title": "Title",
      "journal": "Journal",
      "year": 2023,
      "volume": "1",
      "issue": "1",
      "pages": "1-10",
      "doi": "10.1000/example"
    },
    "reference_id": "ref-001",
    "cited_by": ["blk-012", "blk-045"]
  }
}
```

#### `image_placeholder`

```json
{
  "content": {
    "image_id": "img-001",
    "image_uri": "extracted/img_001.png",
    "alt_text": "description",
    "width_px": 800,
    "height_px": 600,
    "format": "png"
  }
}
```

---

## 5. Example 1: Simple Article / Blog Post

This example demonstrates a minimal but complete document: a short tech blog post with a heading, paragraphs, a code block, and a list.

```json
{
  "doc2ml_version": "0.6.2",
  "document_id": "doc-7f3a9b2e-4c1d-4e5f-8a6b-2c3d4e5f6a7b",
  "metadata": {
    "title": "Understanding Vector Embeddings",
    "subtitle": "A Gentle Introduction for Developers",
    "authors": [
      {
        "name": "Alex Chen",
        "email": "alex@example.com"
      }
    ],
    "source": {
      "uri": "https://example.com/blog/vector-embeddings",
      "mime_type": "text/html",
      "filename": "vector-embeddings.html",
      "checksum_sha256": "a1b2c3d4e5f6789012345678901234567890abcdef",
      "file_size_bytes": 15234
    },
    "ingestion": {
      "ingestion_date": "2025-01-16T09:30:00Z",
      "processing_version": "doc2ml-json v0.6.2",
      "extractor": "pandoc",
      "extractor_version": "3.1.11",
      "ingestion_pipeline": ["html_parse", "structure_infer", "block_split", "metadata_enrich"],
      "processing_duration_ms": 245
    },
    "language": {
      "detected": "en",
      "confidence": 0.99,
      "declared": "en"
    },
    "statistics": {
      "page_count": 1,
      "chapter_count": 0,
      "section_count": 1,
      "block_count": 6,
      "table_count": 0,
      "figure_count": 0,
      "footnote_count": 0,
      "total_char_count": 1247,
      "total_token_count_est": 312,
      "total_word_count": 198
    },
    "classification": {
      "doc_type": "blog_post",
      "genre": "technical_education",
      "keywords": ["embeddings", "vectors", "machine learning", "nlp"],
      "topics_ml": [
        {"label": "machine learning", "score": 0.95},
        {"label": "natural language processing", "score": 0.88}
      ]
    },
    "dates": {
      "published": "2024-11-20T00:00:00Z"
    },
    "rights": {
      "license": "CC-BY-4.0",
      "open_access": true
    }
  },
  "structure": {
    "node_id": "root",
    "node_type": "document",
    "title": "Understanding Vector Embeddings",
    "level": 0,
    "chunk_ids": ["blk-000"],
    "children": [
      {
        "node_id": "sec-001",
        "node_type": "section",
        "title": "Understanding Vector Embeddings",
        "level": 1,
        "chunk_ids": ["blk-001", "blk-002", "blk-003", "blk-004", "blk-005"],
        "children": []
      }
    ]
  },
  "blocks": [
    {
      "chunk_id": "blk-000",
      "type": "metadata",
      "content": {
        "field": "author_bio",
        "value": "Alex Chen is a senior ML engineer.",
        "display_text": "Alex Chen is a senior ML engineer."
      },
      "text_plain": "Alex Chen is a senior ML engineer.",
      "char_count": 37,
      "token_count_est": 9,
      "embedding_ready": false,
      "context_window": {
        "prev_chunk_id": null,
        "next_chunk_id": "blk-001",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "root",
        "surrounding_text_preview": "Alex Chen is a senior ML engineer. | Understanding Vector..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "header div",
        "extraction_method": "pandoc",
        "confidence": 0.98
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": false, "section_role": "metadata"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-001",
      "type": "heading",
      "content": {
        "text": "Understanding Vector Embeddings",
        "level": 1,
        "numbered": false,
        "label": null
      },
      "text_plain": "Understanding Vector Embeddings",
      "char_count": 31,
      "token_count_est": 5,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-000",
        "next_chunk_id": "blk-002",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "sec-001",
        "surrounding_text_preview": "Alex Chen is a senior ML engineer. | A Gentle Introduction for Developers"
      },
      "provenance": {
        "page_number": 1,
        "source_location": "h1 tag",
        "extraction_method": "pandoc",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "is_first_paragraph": false, "section_role": "title"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-002",
      "type": "heading",
      "content": {
        "text": "A Gentle Introduction for Developers",
        "level": 2,
        "numbered": false,
        "label": null
      },
      "text_plain": "A Gentle Introduction for Developers",
      "char_count": 36,
      "token_count_est": 5,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-001",
        "next_chunk_id": "blk-003",
        "parent_heading_chunk_id": "blk-001",
        "parent_structure_node_id": "sec-001",
        "surrounding_text_preview": "Understanding Vector Embeddings | At their core, vector..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "h2 tag",
        "extraction_method": "pandoc",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 2, "is_first_paragraph": false, "section_role": "subtitle"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-003",
      "type": "paragraph",
      "content": {
        "text": "At their core, vector embeddings are numerical representations of data. Whether it's a sentence, an image, or a user profile, embeddings compress meaning into a fixed-length array of floats.",
        "inline_elements": [],
        "sentences": [
          {"text": "At their core, vector embeddings are numerical representations of data.", "start_char": 0, "end_char": 66, "sentence_id": "s-001"},
          {"text": "Whether it's a sentence, an image, or a user profile, embeddings compress meaning into a fixed-length array of floats.", "start_char": 67, "end_char": 175, "sentence_id": "s-002"}
        ]
      },
      "text_plain": "At their core, vector embeddings are numerical representations of data. Whether it's a sentence, an image, or a user profile, embeddings compress meaning into a fixed-length array of floats.",
      "char_count": 175,
      "token_count_est": 35,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-002",
        "next_chunk_id": "blk-004",
        "parent_heading_chunk_id": "blk-001",
        "parent_structure_node_id": "sec-001",
        "surrounding_text_preview": "A Gentle Introduction for Developers | At their core, vector..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "p tag",
        "extraction_method": "pandoc",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "introduction"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-004",
      "type": "code_block",
      "content": {
        "code": "import numpy as np\n\n# Example: 3-dimensional embedding for the word 'king'\nembedding = np.array([0.21, -0.45, 0.88, ..., 0.12])  # shape: (768,)",
        "language": "python",
        "filename": null,
        "line_numbers": false,
        "lines": [
          {"line_number": 1, "text": "import numpy as np"},
          {"line_number": 2, "text": ""},
          {"line_number": 3, "text": "# Example: 3-dimensional embedding for the word 'king'"},
          {"line_number": 4, "text": "embedding = np.array([0.21, -0.45, 0.88, ..., 0.12])  # shape: (768,)"}
        ]
      },
      "text_plain": "import numpy as np\n\n# Example: 3-dimensional embedding for the word 'king'\nembedding = np.array([0.21, -0.45, 0.88, ..., 0.12])  # shape: (768,)",
      "char_count": 132,
      "token_count_est": 38,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-003",
        "next_chunk_id": "blk-005",
        "parent_heading_chunk_id": "blk-001",
        "parent_structure_node_id": "sec-001",
        "surrounding_text_preview": "...fixed-length array of floats. | import numpy as np..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "pre > code tag",
        "extraction_method": "pandoc",
        "confidence": 0.98
      },
      "language": {"detected": "en", "confidence": 0.95},
      "semantics": {"heading_level": null, "is_first_paragraph": false, "section_role": "example"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"syntax_highlighting": "true"}
    },
    {
      "chunk_id": "blk-005",
      "type": "list",
      "content": {
        "list_type": "unordered",
        "items": [
          {
            "text": "Dense representations capture semantic similarity.",
            "item_id": "li-001",
            "level": 0,
            "marker": "•",
            "children": []
          },
          {
            "text": "Dimensionality is typically 128, 512, 768, or 1024.",
            "item_id": "li-002",
            "level": 0,
            "marker": "•",
            "children": []
          },
          {
            "text": "Popular models include Word2Vec, GloVe, and modern transformer outputs.",
            "item_id": "li-003",
            "level": 0,
            "marker": "•",
            "children": []
          }
        ]
      },
      "text_plain": "• Dense representations capture semantic similarity.\n• Dimensionality is typically 128, 512, 768, or 1024.\n• Popular models include Word2Vec, GloVe, and modern transformer outputs.",
      "char_count": 167,
      "token_count_est": 30,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-004",
        "next_chunk_id": null,
        "parent_heading_chunk_id": "blk-001",
        "parent_structure_node_id": "sec-001",
        "surrounding_text_preview": "...shape: (768,) | Dense representations capture..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "ul > li tags",
        "extraction_method": "pandoc",
        "confidence": 0.97
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": false, "section_role": "summary"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    }
  ],
  "cross_references": [],
  "ml_index": {
    "chunk_id_map": {
      "blk-000": {"index": 0, "structure_path": ["root"]},
      "blk-001": {"index": 1, "structure_path": ["root", "sec-001"]},
      "blk-002": {"index": 2, "structure_path": ["root", "sec-001"]},
      "blk-003": {"index": 3, "structure_path": ["root", "sec-001"]},
      "blk-004": {"index": 4, "structure_path": ["root", "sec-001"]},
      "blk-005": {"index": 5, "structure_path": ["root", "sec-001"]}
    },
    "heading_map": [
      {"chunk_id": "blk-001", "heading_text": "Understanding Vector Embeddings", "level": 1, "node_id": "sec-001"},
      {"chunk_id": "blk-002", "heading_text": "A Gentle Introduction for Developers", "level": 2, "node_id": "sec-001"}
    ],
    "embedding_candidates": ["blk-001", "blk-002", "blk-003", "blk-004", "blk-005"],
    "chunk_boundaries": [
      {"start_chunk_id": "blk-001", "end_chunk_id": "blk-005", "boundary_type": "semantic", "token_count_est": 113}
    ]
  },
  "custom": {}
}
```

---

## 6. Example 2: Academic Paper (Complex)

This example demonstrates a full academic paper with: abstract, multi-level headings, a table with dual representation, figure caption, inline equations, citations, and a references section.

```json
{
  "doc2ml_version": "0.6.2",
  "document_id": "doc-8b4c2d1e-9a3f-4b5c-8d7e-1a2b3c4d5e6f",
  "metadata": {
    "title": "Attention Is All You Need",
    "subtitle": null,
    "authors": [
      {"name": "Ashish Vaswani", "affiliation": "Google Brain"},
      {"name": "Noam Shazeer", "affiliation": "Google Brain"},
      {"name": "Niki Parmar", "affiliation": "Google Research"},
      {"name": "Jakob Uszkoreit", "affiliation": "Google Research"},
      {"name": "Llion Jones", "affiliation": "Google Research"},
      {"name": "Aidan N. Gomez", "affiliation": "University of Toronto"},
      {"name": "Lukasz Kaiser", "affiliation": "Google Brain"},
      {"name": "Illia Polosukhin", "affiliation": "Google Research"}
    ],
    "source": {
      "uri": "https://arxiv.org/abs/1706.03762",
      "mime_type": "application/pdf",
      "filename": "1706.03762.pdf",
      "checksum_sha256": "f3c4d5e6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
      "file_size_bytes": 2180038
    },
    "ingestion": {
      "ingestion_date": "2025-01-16T10:00:00Z",
      "processing_version": "doc2ml-json v0.6.2",
      "extractor": "marker",
      "extractor_version": "0.3.0",
      "ingestion_pipeline": ["pdf_extract", "structure_infer", "table_ocr", "reference_parse", "crossref_resolve", "block_split", "ml_index_build"],
      "processing_duration_ms": 3847
    },
    "language": {"detected": "en", "confidence": 0.99, "declared": "en"},
    "statistics": {
      "page_count": 15,
      "chapter_count": 0,
      "section_count": 8,
      "block_count": 89,
      "table_count": 3,
      "figure_count": 4,
      "footnote_count": 0,
      "total_char_count": 45231,
      "total_token_count_est": 11308,
      "total_word_count": 7124
    },
    "classification": {
      "doc_type": "academic_paper",
      "genre": "computer_science",
      "keywords": ["transformer", "attention mechanism", "neural machine translation", "NLP"],
      "topics_ml": [
        {"label": "deep learning", "score": 0.98},
        {"label": "natural language processing", "score": 0.96}
      ]
    },
    "dates": {
      "created": "2017-06-12T00:00:00Z",
      "published": "2017-06-12T00:00:00Z"
    },
    "rights": {
      "license": "arXiv non-exclusive perpetual irrevocable license",
      "open_access": true
    }
  },
  "structure": {
    "node_id": "root",
    "node_type": "document",
    "title": "Attention Is All You Need",
    "level": 0,
    "children": [
      {
        "node_id": "front",
        "node_type": "front_matter",
        "title": "Front Matter",
        "level": 1,
        "chunk_ids": ["blk-000", "blk-001"],
        "children": []
      },
      {
        "node_id": "sec-intro",
        "node_type": "section",
        "title": "1. Introduction",
        "level": 1,
        "chunk_ids": ["blk-002", "blk-003", "blk-004"],
        "children": []
      },
      {
        "node_id": "sec-background",
        "node_type": "section",
        "title": "2. Background",
        "level": 1,
        "chunk_ids": ["blk-005", "blk-006"],
        "children": []
      },
      {
        "node_id": "sec-model",
        "node_type": "section",
        "title": "3. Model Architecture",
        "level": 1,
        "chunk_ids": ["blk-007", "blk-008"],
        "children": [
          {
            "node_id": "sec-encoder",
            "node_type": "subsection",
            "title": "3.1 Encoder and Decoder Stacks",
            "level": 2,
            "chunk_ids": ["blk-009", "blk-010"],
            "children": []
          },
          {
            "node_id": "sec-attention",
            "node_type": "subsection",
            "title": "3.2 Attention",
            "level": 2,
            "chunk_ids": ["blk-011", "blk-012", "blk-013", "blk-014"],
            "children": [
              {
                "node_id": "sec-scaled-dot",
                "node_type": "subsubsection",
                "title": "3.2.1 Scaled Dot-Product Attention",
                "level": 3,
                "chunk_ids": ["blk-015", "blk-016"],
                "children": []
              }
            ]
          }
        ]
      },
      {
        "node_id": "sec-experiments",
        "node_type": "section",
        "title": "4. Experiments",
        "level": 1,
        "chunk_ids": ["blk-020", "blk-021", "blk-022", "blk-023"],
        "children": []
      },
      {
        "node_id": "sec-references",
        "node_type": "back_matter",
        "title": "References",
        "level": 1,
        "chunk_ids": ["blk-080"],
        "children": []
      }
    ]
  },
  "blocks": [
    {
      "chunk_id": "blk-000",
      "type": "metadata",
      "content": {
        "field": "doi",
        "value": "10.48550/arXiv.1706.03762",
        "display_text": "DOI: 10.48550/arXiv.1706.03762"
      },
      "text_plain": "DOI: 10.48550/arXiv.1706.03762",
      "char_count": 34,
      "token_count_est": 9,
      "embedding_ready": false,
      "context_window": {
        "prev_chunk_id": null,
        "next_chunk_id": "blk-001",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "front",
        "surrounding_text_preview": "| Attention Is All You Need"
      },
      "provenance": {
        "page_number": 1,
        "source_location": "header meta",
        "extraction_method": "marker",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "section_role": "metadata"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-001",
      "type": "abstract",
      "content": {
        "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
      },
      "text_plain": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
      "char_count": 342,
      "token_count_est": 68,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-000",
        "next_chunk_id": "blk-002",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "front",
        "surrounding_text_preview": "DOI: 10.48550/arXiv.1706.03762 | 1. Introduction"
      },
      "provenance": {
        "page_number": 1,
        "source_location": "abstract region",
        "extraction_method": "marker",
        "confidence": 0.97
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "abstract"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-002",
      "type": "heading",
      "content": {
        "text": "1. Introduction",
        "level": 1,
        "numbered": true,
        "label": "1"
      },
      "text_plain": "1. Introduction",
      "char_count": 15,
      "token_count_est": 3,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-001",
        "next_chunk_id": "blk-003",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "sec-intro",
        "surrounding_text_preview": "...convolutions entirely. | Recurrent neural networks..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "heading region",
        "extraction_method": "marker",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "section_role": "introduction"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-003",
      "type": "paragraph",
      "content": {
        "text": "Recurrent neural networks (RNN), long short-term memory (LSTM) [13] and gated recurrent neural networks (GRU) [7] have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5].",
        "inline_elements": [
          {"type": "link", "start": 54, "end": 57, "text": "13", "href": "#ref-13"},
          {"type": "link", "start": 118, "end": 120, "text": "7", "href": "#ref-7"},
          {"type": "link", "start": 218, "end": 220, "text": "35", "href": "#ref-35"},
          {"type": "link", "start": 222, "end": 223, "text": "2", "href": "#ref-2"},
          {"type": "link", "start": 225, "end": 226, "text": "5", "href": "#ref-5"}
        ],
        "sentences": [
          {"text": "Recurrent neural networks (RNN), long short-term memory (LSTM) [13] and gated recurrent neural networks (GRU) [7] have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5].", "start_char": 0, "end_char": 227, "sentence_id": "s-001"}
        ]
      },
      "text_plain": "Recurrent neural networks (RNN), long short-term memory (LSTM) [13] and gated recurrent neural networks (GRU) [7] have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5].",
      "char_count": 227,
      "token_count_est": 43,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-002",
        "next_chunk_id": "blk-004",
        "parent_heading_chunk_id": "blk-002",
        "parent_structure_node_id": "sec-intro",
        "surrounding_text_preview": "1. Introduction | Recurrent neural networks..."
      },
      "provenance": {
        "page_number": 1,
        "source_location": "body text",
        "extraction_method": "marker",
        "confidence": 0.98
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "introduction"},
      "relations": {"footnotes_for": [], "references": ["ref-cite-001"]},
      "custom": {}
    },
    {
      "chunk_id": "blk-011",
      "type": "heading",
      "content": {
        "text": "3.2 Attention",
        "level": 2,
        "numbered": true,
        "label": "3.2"
      },
      "text_plain": "3.2 Attention",
      "char_count": 11,
      "token_count_est": 3,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-010",
        "next_chunk_id": "blk-012",
        "parent_heading_chunk_id": "blk-007",
        "parent_structure_node_id": "sec-attention",
        "surrounding_text_preview": "...encoder and decoder stacks. | 3.2 Attention"
      },
      "provenance": {
        "page_number": 3,
        "source_location": "heading region",
        "extraction_method": "marker",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 2, "section_role": "method"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-015",
      "type": "heading",
      "content": {
        "text": "3.2.1 Scaled Dot-Product Attention",
        "level": 3,
        "numbered": true,
        "label": "3.2.1"
      },
      "text_plain": "3.2.1 Scaled Dot-Product Attention",
      "char_count": 34,
      "token_count_est": 6,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-014",
        "next_chunk_id": "blk-016",
        "parent_heading_chunk_id": "blk-011",
        "parent_structure_node_id": "sec-scaled-dot",
        "surrounding_text_preview": "3.2 Attention | We call our particular attention..."
      },
      "provenance": {
        "page_number": 3,
        "source_location": "heading region",
        "extraction_method": "marker",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 3, "section_role": "method"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-016",
      "type": "equation",
      "content": {
        "latex": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
        "mathml": "<math><mi>Attention</mi><mo>(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>=</mo><mi>softmax</mi><mrow><mo>(</mo><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>)</mo></mrow><mi>V</mi></math>",
        "plain_text": "Attention of Q, K, V equals softmax of Q K transpose divided by square root of d_k, times V",
        "equation_id": "eq-001",
        "display_mode": true
      },
      "text_plain": "Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V",
      "char_count": 65,
      "token_count_est": 22,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-015",
        "next_chunk_id": "blk-017",
        "parent_heading_chunk_id": "blk-015",
        "parent_structure_node_id": "sec-scaled-dot",
        "surrounding_text_preview": "3.2.1 Scaled Dot-Product Attention | In practice, we..."
      },
      "provenance": {
        "page_number": 3,
        "source_location": "equation region",
        "extraction_method": "marker",
        "confidence": 0.95
      },
      "language": {"detected": "en", "confidence": 0.90},
      "semantics": {"heading_level": null, "section_role": "method"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"latex_source_confidence": 0.94}
    },
    {
      "chunk_id": "blk-020",
      "type": "heading",
      "content": {
        "text": "4. Experiments",
        "level": 1,
        "numbered": true,
        "label": "4"
      },
      "text_plain": "4. Experiments",
      "char_count": 14,
      "token_count_est": 3,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-019",
        "next_chunk_id": "blk-021",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "sec-experiments",
        "surrounding_text_preview": "...multi-head attention. | 4. Experiments"
      },
      "provenance": {
        "page_number": 7,
        "source_location": "heading region",
        "extraction_method": "marker",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "section_role": "results"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-021",
      "type": "table",
      "content": {
        "caption": "Table 2: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types.",
        "caption_chunk_id": "blk-cap-021",
        "table_id": "tbl-002",
        "column_count": 4,
        "row_count": 4,
        "header_row_count": 1,
        "html": "<table><thead><tr><th>Layer Type</th><th>Complexity per Layer</th><th>Sequential Operations</th><th>Maximum Path Length</th></tr></thead><tbody><tr><td>Self-Attention</td><td>O(n^2 * d)</td><td>O(1)</td><td>O(1)</td></tr><tr><td>Recurrent</td><td>O(n * d^2)</td><td>O(n)</td><td>O(n)</td></tr><tr><td>Convolutional</td><td>O(k * n * d^2)</td><td>O(1)</td><td>O(log_k(n))</td></tr></tbody></table>",
        "markdown": "| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |\n|---|---|---|---|\n| Self-Attention | O(n^2 * d) | O(1) | O(1) |\n| Recurrent | O(n * d^2) | O(n) | O(n) |\n| Convolutional | O(k * n * d^2) | O(1) | O(log_k(n)) |",
        "records": [
          {"Layer Type": "Self-Attention", "Complexity per Layer": "O(n^2 * d)", "Sequential Operations": "O(1)", "Maximum Path Length": "O(1)", "_row_index": 0, "_is_header": false},
          {"Layer Type": "Recurrent", "Complexity per Layer": "O(n * d^2)", "Sequential Operations": "O(n)", "Maximum Path Length": "O(n)", "_row_index": 1, "_is_header": false},
          {"Layer Type": "Convolutional", "Complexity per Layer": "O(k * n * d^2)", "Sequential Operations": "O(1)", "Maximum Path Length": "O(log_k(n))", "_row_index": 2, "_is_header": false}
        ],
        "cells": [
          {"row": 0, "col": 0, "text": "Layer Type", "is_header": true, "rowspan": 1, "colspan": 1},
          {"row": 0, "col": 1, "text": "Complexity per Layer", "is_header": true, "rowspan": 1, "colspan": 1},
          {"row": 0, "col": 2, "text": "Sequential Operations", "is_header": true, "rowspan": 1, "colspan": 1},
          {"row": 0, "col": 3, "text": "Maximum Path Length", "is_header": true, "rowspan": 1, "colspan": 1},
          {"row": 1, "col": 0, "text": "Self-Attention", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 1, "col": 1, "text": "O(n^2 * d)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 1, "col": 2, "text": "O(1)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 1, "col": 3, "text": "O(1)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 2, "col": 0, "text": "Recurrent", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 2, "col": 1, "text": "O(n * d^2)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 2, "col": 2, "text": "O(n)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 2, "col": 3, "text": "O(n)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 3, "col": 0, "text": "Convolutional", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 3, "col": 1, "text": "O(k * n * d^2)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 3, "col": 2, "text": "O(1)", "is_header": false, "rowspan": 1, "colspan": 1},
          {"row": 3, "col": 3, "text": "O(log_k(n))", "is_header": false, "rowspan": 1, "colspan": 1}
        ],
        "footnotes": []
      },
      "text_plain": "Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length\nSelf-Attention | O(n^2 * d) | O(1) | O(1)\nRecurrent | O(n * d^2) | O(n) | O(n)\nConvolutional | O(k * n * d^2) | O(1) | O(log_k(n))",
      "char_count": 280,
      "token_count_est": 72,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-020",
        "next_chunk_id": "blk-022",
        "parent_heading_chunk_id": "blk-020",
        "parent_structure_node_id": "sec-experiments",
        "surrounding_text_preview": "4. Experiments | ...training results..."
      },
      "provenance": {
        "page_number": 7,
        "source_location": "table region",
        "extraction_method": "table_ocr",
        "confidence": 0.96
      },
      "language": {"detected": "en", "confidence": 0.95},
      "semantics": {"heading_level": null, "section_role": "results"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"table_extraction_quality": "high"}
    },
    {
      "chunk_id": "blk-022",
      "type": "figure_caption",
      "content": {
        "label": "Figure 1",
        "text": "The Transformer - model architecture.",
        "figure_id": "fig-001",
        "image_uris": ["extracted/fig_001.png"],
        "bounding_boxes": [{"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.6}]
      },
      "text_plain": "Figure 1: The Transformer - model architecture.",
      "char_count": 43,
      "token_count_est": 8,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-021",
        "next_chunk_id": "blk-023",
        "parent_heading_chunk_id": "blk-020",
        "parent_structure_node_id": "sec-experiments",
        "surrounding_text_preview": "...O(log_k(n)) | Figure 1: The Transformer..."
      },
      "provenance": {
        "page_number": 8,
        "source_location": "caption region",
        "extraction_method": "marker",
        "confidence": 0.98
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "section_role": "results"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-023",
      "type": "paragraph",
      "content": {
        "text": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs [9].",
        "inline_elements": [
          {"type": "link", "start": 107, "end": 109, "text": "9", "href": "#ref-9"}
        ],
        "sentences": [
          {"text": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs [9].", "start_char": 0, "end_char": 110, "sentence_id": "s-001"}
        ]
      },
      "text_plain": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs [9].",
      "char_count": 110,
      "token_count_est": 21,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-022",
        "next_chunk_id": "blk-024",
        "parent_heading_chunk_id": "blk-020",
        "parent_structure_node_id": "sec-experiments",
        "surrounding_text_preview": "Figure 1: The Transformer... | ...tokenized using byte-pair..."
      },
      "provenance": {
        "page_number": 8,
        "source_location": "body text",
        "extraction_method": "marker",
        "confidence": 0.98
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "results"},
      "relations": {"footnotes_for": [], "references": ["ref-cite-010"]},
      "custom": {}
    },
    {
      "chunk_id": "blk-080",
      "type": "reference_entry",
      "content": {
        "raw_text": "[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. ICLR, 2015.",
        "parsed": {
          "authors": [
            {"given": "Dzmitry", "family": "Bahdanau"},
            {"given": "Kyunghyun", "family": "Cho"},
            {"given": "Yoshua", "family": "Bengio"}
          ],
          "title": "Neural machine translation by jointly learning to align and translate",
          "journal": "ICLR",
          "year": 2015,
          "volume": null,
          "issue": null,
          "pages": null,
          "doi": null
        },
        "reference_id": "ref-2",
        "cited_by": ["blk-003"]
      },
      "text_plain": "[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. ICLR, 2015.",
      "char_count": 145,
      "token_count_est": 22,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-079",
        "next_chunk_id": "blk-081",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "sec-references",
        "surrounding_text_preview": "[1] ... | [2] Dzmitry Bahdanau... | [3] ..."
      },
      "provenance": {
        "page_number": 15,
        "source_location": "reference list",
        "extraction_method": "reference_parse",
        "confidence": 0.92
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "section_role": "reference"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"parser": "anystyle"}
    }
  ],
  "cross_references": [
    {
      "ref_id": "ref-cite-001",
      "ref_type": "citation",
      "source_chunk_id": "blk-003",
      "target_chunk_id": null,
      "target_structure_node_id": "sec-references",
      "label": "[13, 7, 35, 2, 5]",
      "context_text": "Recurrent neural networks (RNN), long short-term memory (LSTM) [13] and gated recurrent neural networks (GRU) [7] have been firmly established...",
      "resolved": false
    },
    {
      "ref_id": "ref-cite-010",
      "ref_type": "citation",
      "source_chunk_id": "blk-023",
      "target_chunk_id": null,
      "target_structure_node_id": "sec-references",
      "label": "[9]",
      "context_text": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs [9].",
      "resolved": false
    },
    {
      "ref_id": "ref-fig-001",
      "ref_type": "figure_ref",
      "source_chunk_id": "blk-010",
      "target_chunk_id": "blk-022",
      "target_structure_node_id": null,
      "label": "Figure 1",
      "context_text": "The encoder is composed of a stack of N = 6 identical layers (see Figure 1).",
      "resolved": true
    }
  ],
  "ml_index": {
    "chunk_id_map": {
      "blk-000": {"index": 0, "structure_path": ["root", "front"]},
      "blk-001": {"index": 1, "structure_path": ["root", "front"]},
      "blk-002": {"index": 2, "structure_path": ["root", "sec-intro"]},
      "blk-003": {"index": 3, "structure_path": ["root", "sec-intro"]},
      "blk-004": {"index": 4, "structure_path": ["root", "sec-intro"]},
      "blk-005": {"index": 5, "structure_path": ["root", "sec-background"]},
      "blk-006": {"index": 6, "structure_path": ["root", "sec-background"]},
      "blk-007": {"index": 7, "structure_path": ["root", "sec-model"]},
      "blk-008": {"index": 8, "structure_path": ["root", "sec-model"]},
      "blk-009": {"index": 9, "structure_path": ["root", "sec-model", "sec-encoder"]},
      "blk-010": {"index": 10, "structure_path": ["root", "sec-model", "sec-encoder"]},
      "blk-011": {"index": 11, "structure_path": ["root", "sec-model", "sec-attention"]},
      "blk-012": {"index": 12, "structure_path": ["root", "sec-model", "sec-attention"]},
      "blk-013": {"index": 13, "structure_path": ["root", "sec-model", "sec-attention"]},
      "blk-014": {"index": 14, "structure_path": ["root", "sec-model", "sec-attention"]},
      "blk-015": {"index": 15, "structure_path": ["root", "sec-model", "sec-attention", "sec-scaled-dot"]},
      "blk-016": {"index": 16, "structure_path": ["root", "sec-model", "sec-attention", "sec-scaled-dot"]},
      "blk-020": {"index": 20, "structure_path": ["root", "sec-experiments"]},
      "blk-021": {"index": 21, "structure_path": ["root", "sec-experiments"]},
      "blk-022": {"index": 22, "structure_path": ["root", "sec-experiments"]},
      "blk-023": {"index": 23, "structure_path": ["root", "sec-experiments"]},
      "blk-080": {"index": 80, "structure_path": ["root", "sec-references"]}
    },
    "heading_map": [
      {"chunk_id": "blk-002", "heading_text": "1. Introduction", "level": 1, "node_id": "sec-intro"},
      {"chunk_id": "blk-005", "heading_text": "2. Background", "level": 1, "node_id": "sec-background"},
      {"chunk_id": "blk-007", "heading_text": "3. Model Architecture", "level": 1, "node_id": "sec-model"},
      {"chunk_id": "blk-009", "heading_text": "3.1 Encoder and Decoder Stacks", "level": 2, "node_id": "sec-encoder"},
      {"chunk_id": "blk-011", "heading_text": "3.2 Attention", "level": 2, "node_id": "sec-attention"},
      {"chunk_id": "blk-015", "heading_text": "3.2.1 Scaled Dot-Product Attention", "level": 3, "node_id": "sec-scaled-dot"},
      {"chunk_id": "blk-020", "heading_text": "4. Experiments", "level": 1, "node_id": "sec-experiments"}
    ],
    "embedding_candidates": [
      "blk-001", "blk-003", "blk-005", "blk-006", "blk-007",
      "blk-009", "blk-010", "blk-011", "blk-012", "blk-013",
      "blk-014", "blk-015", "blk-016", "blk-020", "blk-023",
      "blk-080"
    ],
    "chunk_boundaries": [
      {"start_chunk_id": "blk-002", "end_chunk_id": "blk-004", "boundary_type": "section", "token_count_est": 74},
      {"start_chunk_id": "blk-005", "end_chunk_id": "blk-006", "boundary_type": "section", "token_count_est": 340},
      {"start_chunk_id": "blk-007", "end_chunk_id": "blk-019", "boundary_type": "section", "token_count_est": 2100},
      {"start_chunk_id": "blk-020", "end_chunk_id": "blk-023", "boundary_type": "section", "token_count_est": 450}
    ]
  },
  "custom": {}
}
```

---

## 7. Example 3: Book / Chapter Hierarchy (EPUB-like)

This example demonstrates a book with front matter, multiple parts, chapters, and nested sections — the deepest hierarchy the schema supports.

```json
{
  "doc2ml_version": "0.6.2",
  "document_id": "doc-1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
  "metadata": {
    "title": "The Structure of Scientific Revolutions",
    "subtitle": "Second Edition, Enlarged",
    "authors": [
      {
        "name": "Thomas S. Kuhn",
        "affiliation": "Princeton University"
      }
    ],
    "source": {
      "uri": "/books/kuhn-scientific-revolutions.epub",
      "mime_type": "application/epub+zip",
      "filename": "kuhn-scientific-revolutions.epub",
      "checksum_sha256": "9f8e7d6c5b4a3928170654433221100998877665544332211aabbccdd1122334",
      "file_size_bytes": 2847561
    },
    "ingestion": {
      "ingestion_date": "2025-01-16T11:00:00Z",
      "processing_version": "doc2ml-json v0.6.2",
      "extractor": "epub2text",
      "extractor_version": "2.1.0",
      "ingestion_pipeline": ["epub_unpack", "opf_parse", "ncx_nav", "html_chapter_extract", "structure_infer", "block_split", "ml_index_build"],
      "processing_duration_ms": 5234
    },
    "language": {"detected": "en", "confidence": 0.99, "declared": "en"},
    "statistics": {
      "page_count": 210,
      "chapter_count": 10,
      "section_count": 32,
      "block_count": 1247,
      "table_count": 2,
      "figure_count": 1,
      "footnote_count": 45,
      "total_char_count": 142356,
      "total_token_count_est": 35589,
      "total_word_count": 25432
    },
    "classification": {
      "doc_type": "book",
      "genre": "philosophy_of_science",
      "keywords": ["paradigm", "scientific revolution", "normal science", "epistemology"],
      "topics_ml": [
        {"label": "philosophy of science", "score": 0.97},
        {"label": "history of science", "score": 0.91}
      ]
    },
    "dates": {
      "created": "1962-01-01T00:00:00Z",
      "published": "1970-01-01T00:00:00Z",
      "modified": "1970-06-15T00:00:00Z"
    },
    "rights": {
      "license": "All rights reserved",
      "copyright": "© 1962, 1970 The University of Chicago",
      "open_access": false
    }
  },
  "structure": {
    "node_id": "root",
    "node_type": "document",
    "title": "The Structure of Scientific Revolutions",
    "level": 0,
    "children": [
      {
        "node_id": "front",
        "node_type": "front_matter",
        "title": "Front Matter",
        "level": 1,
        "chunk_ids": ["blk-000", "blk-001", "blk-002"],
        "children": [
          {
            "node_id": "front-title",
            "node_type": "section",
            "title": "Title Page",
            "level": 2,
            "chunk_ids": ["blk-000"],
            "children": []
          },
          {
            "node_id": "front-preface",
            "node_type": "section",
            "title": "Preface to the First Edition",
            "level": 2,
            "chunk_ids": ["blk-001", "blk-002"],
            "children": []
          }
        ]
      },
      {
        "node_id": "part-1",
        "node_type": "part",
        "title": "Part I: The Route to Normal Science",
        "level": 1,
        "children": [
          {
            "node_id": "ch-1",
            "node_type": "chapter",
            "title": "I. The Role of Paradigms",
            "level": 2,
            "page_start": 10,
            "page_end": 22,
            "chunk_ids": ["blk-003", "blk-004"],
            "children": [
              {
                "node_id": "ch-1-sec-1",
                "node_type": "subsection",
                "title": "The Study of History",
                "level": 3,
                "chunk_ids": ["blk-005", "blk-006", "blk-007"],
                "children": []
              },
              {
                "node_id": "ch-1-sec-2",
                "node_type": "subsection",
                "title": "The Emergence of Paradigms",
                "level": 3,
                "chunk_ids": ["blk-008", "blk-009"],
                "children": []
              }
            ]
          },
          {
            "node_id": "ch-2",
            "node_type": "chapter",
            "title": "II. The Nature of Normal Science",
            "level": 2,
            "page_start": 23,
            "page_end": 34,
            "chunk_ids": ["blk-010", "blk-011"],
            "children": []
          }
        ]
      },
      {
        "node_id": "part-2",
        "node_type": "part",
        "title": "Part II: Crisis and the Emergence of Scientific Theories",
        "level": 1,
        "children": [
          {
            "node_id": "ch-6",
            "node_type": "chapter",
            "title": "VI. Anomaly and the Emergence of Scientific Discoveries",
            "level": 2,
            "page_start": 52,
            "page_end": 65,
            "chunk_ids": ["blk-060", "blk-061"],
            "children": [
              {
                "node_id": "ch-6-sec-1",
                "node_type": "subsection",
                "title": "The Detection of Anomaly",
                "level": 3,
                "chunk_ids": ["blk-062", "blk-063"],
                "children": []
              },
              {
                "node_id": "ch-6-sec-2",
                "node_type": "subsection",
                "title": "The Resistance to Change",
                "level": 3,
                "chunk_ids": ["blk-064", "blk-065", "blk-066"],
                "children": [
                  {
                    "node_id": "ch-6-sec-2-1",
                    "node_type": "subsubsection",
                    "title": "Cognitive Factors",
                    "level": 4,
                    "chunk_ids": ["blk-067"],
                    "children": []
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "node_id": "back",
        "node_type": "back_matter",
        "title": "Back Matter",
        "level": 1,
        "chunk_ids": ["blk-200", "blk-201"],
        "children": [
          {
            "node_id": "back-index",
            "node_type": "section",
            "title": "Index",
            "level": 2,
            "chunk_ids": ["blk-200"],
            "children": []
          },
          {
            "node_id": "back-biblio",
            "node_type": "section",
            "title": "Bibliography",
            "level": 2,
            "chunk_ids": ["blk-201"],
            "children": []
          }
        ]
      }
    ]
  },
  "blocks": [
    {
      "chunk_id": "blk-000",
      "type": "metadata",
      "content": {
        "field": "publisher",
        "value": "The University of Chicago Press",
        "display_text": "The University of Chicago Press, Chicago and London"
      },
      "text_plain": "The University of Chicago Press, Chicago and London",
      "char_count": 49,
      "token_count_est": 9,
      "embedding_ready": false,
      "context_window": {
        "prev_chunk_id": null,
        "next_chunk_id": "blk-001",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "front-title",
        "surrounding_text_preview": "| The Structure of Scientific Revolutions"
      },
      "provenance": {
        "page_number": 1,
        "source_location": "title-page",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "section_role": "metadata"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"epub_element": "publisher"}
    },
    {
      "chunk_id": "blk-001",
      "type": "heading",
      "content": {
        "text": "Preface to the First Edition",
        "level": 1,
        "numbered": false,
        "label": null
      },
      "text_plain": "Preface to the First Edition",
      "char_count": 28,
      "token_count_est": 6,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-000",
        "next_chunk_id": "blk-002",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "front-preface",
        "surrounding_text_preview": "The University of Chicago Press... | A preliminary version..."
      },
      "provenance": {
        "page_number": 3,
        "source_location": "h1",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "section_role": "preface"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-002",
      "type": "paragraph",
      "content": {
        "text": "A preliminary version of this essay was presented as a paper at the meeting of the American Psychological Association in Boston in August 1959.",
        "inline_elements": [],
        "sentences": [
          {"text": "A preliminary version of this essay was presented as a paper at the meeting of the American Psychological Association in Boston in August 1959.", "start_char": 0, "end_char": 134, "sentence_id": "s-001"}
        ]
      },
      "text_plain": "A preliminary version of this essay was presented as a paper at the meeting of the American Psychological Association in Boston in August 1959.",
      "char_count": 134,
      "token_count_est": 24,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-001",
        "next_chunk_id": "blk-003",
        "parent_heading_chunk_id": "blk-001",
        "parent_structure_node_id": "front-preface",
        "surrounding_text_preview": "Preface to the First Edition | I. The Role of Paradigms"
      },
      "provenance": {
        "page_number": 3,
        "source_location": "p",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "preface"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-003",
      "type": "heading",
      "content": {
        "text": "I. The Role of Paradigms",
        "level": 1,
        "numbered": true,
        "label": "I"
      },
      "text_plain": "I. The Role of Paradigms",
      "char_count": 22,
      "token_count_est": 5,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-002",
        "next_chunk_id": "blk-004",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "ch-1",
        "surrounding_text_preview": "...August 1959. | What is the nature..."
      },
      "provenance": {
        "page_number": 10,
        "source_location": "h1",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "section_role": "chapter"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"epub_id": "chapter001"}
    },
    {
      "chunk_id": "blk-004",
      "type": "paragraph",
      "content": {
        "text": "What is the nature of the processes by which scientific knowledge develops? In this essay I argue that these processes, or at least the major ones, have a recurrent structure.",
        "inline_elements": [],
        "sentences": [
          {"text": "What is the nature of the processes by which scientific knowledge develops?", "start_char": 0, "end_char": 75, "sentence_id": "s-001"},
          {"text": "In this essay I argue that these processes, or at least the major ones, have a recurrent structure.", "start_char": 76, "end_char": 174, "sentence_id": "s-002"}
        ]
      },
      "text_plain": "What is the nature of the processes by which scientific knowledge develops? In this essay I argue that these processes, or at least the major ones, have a recurrent structure.",
      "char_count": 174,
      "token_count_est": 32,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-003",
        "next_chunk_id": "blk-005",
        "parent_heading_chunk_id": "blk-003",
        "parent_structure_node_id": "ch-1",
        "surrounding_text_preview": "I. The Role of Paradigms | What is the nature..."
      },
      "provenance": {
        "page_number": 10,
        "source_location": "p",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "introduction"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-005",
      "type": "heading",
      "content": {
        "text": "The Study of History",
        "level": 2,
        "numbered": false,
        "label": null
      },
      "text_plain": "The Study of History",
      "char_count": 20,
      "token_count_est": 4,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-004",
        "next_chunk_id": "blk-006",
        "parent_heading_chunk_id": "blk-003",
        "parent_structure_node_id": "ch-1-sec-1",
        "surrounding_text_preview": "...recurrent structure. | Historians of science..."
      },
      "provenance": {
        "page_number": 11,
        "source_location": "h2",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 2, "section_role": "subsection"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-006",
      "type": "paragraph",
      "content": {
        "text": "Historians of science have begun to ask new sorts of questions and to employ new sorts of methods. Their field has been transformed from a largely step-by-step chronicle of accumulating positive achievement to an attempt to understand scientific development as a species of social change.",
        "inline_elements": [],
        "sentences": [
          {"text": "Historians of science have begun to ask new sorts of questions and to employ new sorts of methods.", "start_char": 0, "end_char": 97, "sentence_id": "s-001"},
          {"text": "Their field has been transformed from a largely step-by-step chronicle of accumulating positive achievement to an attempt to understand scientific development as a species of social change.", "start_char": 98, "end_char": 245, "sentence_id": "s-002"}
        ]
      },
      "text_plain": "Historians of science have begun to ask new sorts of questions and to employ new sorts of methods. Their field has been transformed from a largely step-by-step chronicle of accumulating positive achievement to an attempt to understand scientific development as a species of social change.",
      "char_count": 245,
      "token_count_est": 42,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-005",
        "next_chunk_id": "blk-007",
        "parent_heading_chunk_id": "blk-005",
        "parent_structure_node_id": "ch-1-sec-1",
        "surrounding_text_preview": "The Study of History | Historians of science..."
      },
      "provenance": {
        "page_number": 11,
        "source_location": "p",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "is_first_paragraph": true, "section_role": "discussion"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    },
    {
      "chunk_id": "blk-007",
      "type": "footnote",
      "content": {
        "note_id": "fn-001",
        "note_label": "1",
        "text": "For the view that historical study is a source of data for the philosophy of science, see K. R. Popper, The Poverty of Historicism (London, 1957), pp. 130-143.",
        "referenced_by": ["blk-006"],
        "backlinks_to": ["blk-006"]
      },
      "text_plain": "1 For the view that historical study is a source of data for the philosophy of science, see K. R. Popper, The Poverty of Historicism (London, 1957), pp. 130-143.",
      "char_count": 155,
      "token_count_est": 27,
      "embedding_ready": false,
      "context_window": {
        "prev_chunk_id": "blk-006",
        "next_chunk_id": "blk-008",
        "parent_heading_chunk_id": "blk-005",
        "parent_structure_node_id": "ch-1-sec-1",
        "surrounding_text_preview": "...species of social change. | The Emergence of Paradigms"
      },
      "provenance": {
        "page_number": 11,
        "source_location": "footnote",
        "extraction_method": "epub2text",
        "confidence": 0.95
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": null, "section_role": "footnote"},
      "relations": {"footnotes_for": ["blk-006"], "references": []},
      "custom": {"epub_notebacklink": "chapter001.html#noteref-1"}
    },
    {
      "chunk_id": "blk-060",
      "type": "heading",
      "content": {
        "text": "VI. Anomaly and the Emergence of Scientific Discoveries",
        "level": 1,
        "numbered": true,
        "label": "VI"
      },
      "text_plain": "VI. Anomaly and the Emergence of Scientific Discoveries",
      "char_count": 53,
      "token_count_est": 8,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-059",
        "next_chunk_id": "blk-061",
        "parent_heading_chunk_id": null,
        "parent_structure_node_id": "ch-6",
        "surrounding_text_preview": "...V. The Priority of Paradigms | Normal science..."
      },
      "provenance": {
        "page_number": 52,
        "source_location": "h1",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 1, "section_role": "chapter"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {"epub_id": "chapter006"}
    },
    {
      "chunk_id": "blk-067",
      "type": "heading",
      "content": {
        "text": "Cognitive Factors",
        "level": 3,
        "numbered": false,
        "label": null
      },
      "text_plain": "Cognitive Factors",
      "char_count": 17,
      "token_count_est": 3,
      "embedding_ready": true,
      "context_window": {
        "prev_chunk_id": "blk-066",
        "next_chunk_id": "blk-068",
        "parent_heading_chunk_id": "blk-064",
        "parent_structure_node_id": "ch-6-sec-2-1",
        "surrounding_text_preview": "The Resistance to Change | What causes..."
      },
      "provenance": {
        "page_number": 61,
        "source_location": "h3",
        "extraction_method": "epub2text",
        "confidence": 0.99
      },
      "language": {"detected": "en", "confidence": 0.99},
      "semantics": {"heading_level": 3, "section_role": "subsubsection"},
      "relations": {"footnotes_for": [], "references": []},
      "custom": {}
    }
  ],
  "cross_references": [
    {
      "ref_id": "ref-fn-001",
      "ref_type": "footnote_ref",
      "source_chunk_id": "blk-006",
      "target_chunk_id": "blk-007",
      "target_structure_node_id": null,
      "label": "1",
      "context_text": "...to an attempt to understand scientific development as a species of social change.1",
      "resolved": true
    }
  ],
  "ml_index": {
    "chunk_id_map": {
      "blk-000": {"index": 0, "structure_path": ["root", "front", "front-title"]},
      "blk-001": {"index": 1, "structure_path": ["root", "front", "front-preface"]},
      "blk-002": {"index": 2, "structure_path": ["root", "front", "front-preface"]},
      "blk-003": {"index": 3, "structure_path": ["root", "part-1", "ch-1"]},
      "blk-004": {"index": 4, "structure_path": ["root", "part-1", "ch-1"]},
      "blk-005": {"index": 5, "structure_path": ["root", "part-1", "ch-1", "ch-1-sec-1"]},
      "blk-006": {"index": 6, "structure_path": ["root", "part-1", "ch-1", "ch-1-sec-1"]},
      "blk-007": {"index": 7, "structure_path": ["root", "part-1", "ch-1", "ch-1-sec-1"]},
      "blk-008": {"index": 8, "structure_path": ["root", "part-1", "ch-1", "ch-1-sec-2"]},
      "blk-009": {"index": 9, "structure_path": ["root", "part-1", "ch-1", "ch-1-sec-2"]},
      "blk-010": {"index": 10, "structure_path": ["root", "part-1", "ch-2"]},
      "blk-011": {"index": 11, "structure_path": ["root", "part-1", "ch-2"]},
      "blk-060": {"index": 60, "structure_path": ["root", "part-2", "ch-6"]},
      "blk-061": {"index": 61, "structure_path": ["root", "part-2", "ch-6"]},
      "blk-062": {"index": 62, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-1"]},
      "blk-063": {"index": 63, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-1"]},
      "blk-064": {"index": 64, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-2"]},
      "blk-065": {"index": 65, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-2"]},
      "blk-066": {"index": 66, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-2"]},
      "blk-067": {"index": 67, "structure_path": ["root", "part-2", "ch-6", "ch-6-sec-2", "ch-6-sec-2-1"]}
    },
    "heading_map": [
      {"chunk_id": "blk-001", "heading_text": "Preface to the First Edition", "level": 1, "node_id": "front-preface"},
      {"chunk_id": "blk-003", "heading_text": "I. The Role of Paradigms", "level": 1, "node_id": "ch-1"},
      {"chunk_id": "blk-005", "heading_text": "The Study of History", "level": 2, "node_id": "ch-1-sec-1"},
      {"chunk_id": "blk-008", "heading_text": "The Emergence of Paradigms", "level": 2, "node_id": "ch-1-sec-2"},
      {"chunk_id": "blk-060", "heading_text": "VI. Anomaly and the Emergence of Scientific Discoveries", "level": 1, "node_id": "ch-6"},
      {"chunk_id": "blk-062", "heading_text": "The Detection of Anomaly", "level": 2, "node_id": "ch-6-sec-1"},
      {"chunk_id": "blk-064", "heading_text": "The Resistance to Change", "level": 2, "node_id": "ch-6-sec-2"},
      {"chunk_id": "blk-067", "heading_text": "Cognitive Factors", "level": 3, "node_id": "ch-6-sec-2-1"}
    ],
    "embedding_candidates": [
      "blk-001", "blk-002", "blk-003", "blk-004", "blk-005",
      "blk-006", "blk-008", "blk-009", "blk-010", "blk-011",
      "blk-060", "blk-061", "blk-062", "blk-063", "blk-064",
      "blk-065", "blk-066", "blk-067"
    ],
    "chunk_boundaries": [
      {"start_chunk_id": "blk-001", "end_chunk_id": "blk-002", "boundary_type": "section", "token_count_est": 33},
      {"start_chunk_id": "blk-003", "end_chunk_id": "blk-004", "boundary_type": "section", "token_count_est": 37},
      {"start_chunk_id": "blk-005", "end_chunk_id": "blk-007", "boundary_type": "section", "token_count_est": 73},
      {"start_chunk_id": "blk-008", "end_chunk_id": "blk-009", "boundary_type": "section", "token_count_est": 210},
      {"start_chunk_id": "blk-060", "end_chunk_id": "blk-067", "boundary_type": "chapter", "token_count_est": 4850}
    ]
  },
  "custom": {
    "epub_metadata": {
      "identifier": "urn:isbn:9780226458120",
      "rights": "Copyright © 1962, 1970 The University of Chicago",
      "subject": ["Science--Philosophy", "Science--History"]
    }
  }
}
```

---

## 8. ML Use-Case Mappings

This section specifies how downstream ML pipelines should derive training/inference examples from the `doc2ml-json` schema.

### 8.1 Fine-Tuning: Instruction-Following (Alpaca / ChatML format)

**Strategy:** Pair headings and their surrounding paragraphs as instruction-response pairs.

```python
def derive_instruction_pairs(doc):
    pairs = []
    for heading in doc["ml_index"]["heading_map"]:
        h_block = get_block(doc, heading["chunk_id"])
        # Collect all paragraph blocks under this heading's section
        section_blocks = get_blocks_in_node(doc, heading["node_id"])
        context_text = " ".join([b["text_plain"] for b in section_blocks if b["type"] == "paragraph"])
        if context_text:
            pairs.append({
                "instruction": f"Explain: {h_block['content']['text']}",
                "input": "",
                "output": context_text[:2000],  # truncate to model context
                "metadata": {
                    "source_doc_id": doc["document_id"],
                    "heading_chunk_id": heading["chunk_id"],
                    "context_window_token_est": sum(b["token_count_est"] for b in section_blocks)
                }
            })
    return pairs
```

### 8.2 Fine-Tuning: Summarization

**Strategy:** Use `abstract` blocks as targets and the full document body as input. Or, use section-level chunk boundaries.

```python
def derive_summarization_examples(doc):
    examples = []
    # Document-level: abstract → full text
    abstract_block = next((b for b in doc["blocks"] if b["type"] == "abstract"), None)
    if abstract_block:
        body_text = " ".join([b["text_plain"] for b in doc["blocks"]
                              if b["type"] in ("paragraph", "heading") and b["embedding_ready"]])
        examples.append({
            "input": body_text[:4000],
            "target": abstract_block["text_plain"],
            "task": "abstractive_summarization"
        })
    # Section-level: section heading + first paragraph → section summary (if available)
    for boundary in doc["ml_index"]["chunk_boundaries"]:
        section_blocks = get_blocks_range(doc, boundary["start_chunk_id"], boundary["end_chunk_id"])
        section_text = " ".join([b["text_plain"] for b in section_blocks])
        # Heuristic: last paragraph in section is often a mini-summary
        paragraphs = [b for b in section_blocks if b["type"] == "paragraph"]
        if len(paragraphs) >= 3:
            examples.append({
                "input": section_text[:2000],
                "target": paragraphs[-1]["text_plain"],
                "task": "section_summarization"
            })
    return examples
```

### 8.3 RAG: Semantic Chunking for Retrieval

**Strategy:** Create overlapping chunks using the `chunk_boundaries` and `context_window` fields.

```python
def derive_rag_chunks(doc, max_tokens=512, overlap_tokens=64):
    chunks = []
    candidates = [get_block(doc, cid) for cid in doc["ml_index"]["embedding_candidates"]]
    # Sliding window over embedding-ready blocks
    current_chunk = []
    current_tokens = 0
    for block in candidates:
        if current_tokens + block["token_count_est"] > max_tokens:
            # Emit chunk with metadata
            chunks.append({
                "text": " ".join([b["text_plain"] for b in current_chunk]),
                "chunk_type": "semantic",
                "token_count_est": current_tokens,
                "source_blocks": [b["chunk_id"] for b in current_chunk],
                "document_id": doc["document_id"],
                "structure_path": get_common_structure_path(current_chunk),
                "heading_context": get_nearest_heading(doc, current_chunk[0]["chunk_id"])
            })
            # Overlap: retain last N tokens worth of blocks
            current_chunk, current_tokens = apply_overlap(current_chunk, overlap_tokens)
        current_chunk.append(block)
        current_tokens += block["token_count_est"]
    return chunks
```

### 8.4 Embedding Generation

**Strategy:** Embed every `embedding_ready=True` block individually, with rich metadata for vector DB filtering.

```python
def derive_embedding_payloads(doc):
    payloads = []
    for cid in doc["ml_index"]["embedding_candidates"]:
        block = get_block(doc, cid)
        path = doc["ml_index"]["chunk_id_map"][cid]["structure_path"]
        payloads.append({
            "text": block["text_plain"],
            "document_id": doc["document_id"],
            "chunk_id": block["chunk_id"],
            "block_type": block["type"],
            "structure_path": path,
            "section_role": block["semantics"]["section_role"],
            "language": block["language"]["detected"],
            "page_number": block["provenance"]["page_number"],
            "token_count_est": block["token_count_est"],
            "metadata": {
                "title": doc["metadata"]["title"],
                "authors": [a["name"] for a in doc["metadata"]["authors"]],
                "doc_type": doc["metadata"]["classification"]["doc_type"]
            }
        })
    return payloads
```

### 8.5 Sequence-to-Sequence: Table-to-Text

**Strategy:** Use `table` blocks with dual representation to generate natural language descriptions.

```python
def derive_table_to_text_examples(doc):
    examples = []
    for block in doc["blocks"]:
        if block["type"] == "table":
            table = block["content"]
            # Input: records + caption
            input_text = f"Table: {table['caption']}\n" + "\n".join(
                [str(r) for r in table["records"]]
            )
            # Target: paragraph from same section describing the table
            related_para = find_nearest_paragraph(doc, block["chunk_id"])
            if related_para:
                examples.append({
                    "input": input_text,
                    "target": related_para["text_plain"],
                    "task": "table_to_text",
                    "table_id": table["table_id"]
                })
    return examples
```

### 8.6 Document Classification / Topic Modeling

**Strategy:** Use document-level metadata and aggregated block statistics.

```python
def derive_classification_features(doc):
    features = {
        "doc_type_label": doc["metadata"]["classification"]["doc_type"],
        "text_features": {
            "total_char_count": doc["metadata"]["statistics"]["total_char_count"],
            "total_token_count_est": doc["metadata"]["statistics"]["total_token_count_est"],
            "heading_count": len(doc["ml_index"]["heading_map"]),
            "table_count": doc["metadata"]["statistics"]["table_count"],
            "figure_count": doc["metadata"]["statistics"]["figure_count"],
            "footnote_count": doc["metadata"]["statistics"]["footnote_count"],
            "avg_paragraph_tokens": mean([b["token_count_est"] for b in doc["blocks"] if b["type"] == "paragraph"])
        },
        "language": doc["metadata"]["language"]["detected"],
        "keywords": doc["metadata"]["classification"]["keywords"],
        "topics_ml": doc["metadata"]["classification"]["topics_ml"],
        # Full text for embedding-based classification
        "full_text": " ".join([b["text_plain"] for b in doc["blocks"] if b["embedding_ready"]])[:8000]
    }
    return features
```

### 8.7 Question-Answering (Extractive)

**Strategy:** Use inline citations and cross-references as pseudo-QA pairs.

```python
def derive_qa_examples(doc):
    examples = []
    for ref in doc["cross_references"]:
        if ref["ref_type"] == "citation" and ref["resolved"]:
            source_block = get_block(doc, ref["source_chunk_id"])
            target_block = get_block(doc, ref["target_chunk_id"])
            if target_block:
                # Question: what does the source say about the cited work?
                question = f"According to this document, what is the contribution or finding of {ref['label']}?"
                answer = target_block["text_plain"]
                examples.append({
                    "question": question,
                    "answer": answer,
                    "context": source_block["context_window"]["surrounding_text_preview"],
                    "source_chunk_id": ref["source_chunk_id"],
                    "target_chunk_id": ref["target_chunk_id"]
                })
    return examples
```

### 8.8 Code LLM Fine-Tuning

**Strategy:** Extract `code_block` blocks with language metadata.

```python
def derive_code_examples(doc):
    examples = []
    for block in doc["blocks"]:
        if block["type"] == "code_block":
            # Context: preceding paragraph explaining the code
            prev_block = get_block(doc, block["context_window"]["prev_chunk_id"])
            explanation = prev_block["text_plain"] if prev_block and prev_block["type"] == "paragraph" else ""
            examples.append({
                "instruction": f"Write {block['content']['language']} code that does the following:",
                "input": explanation,
                "output": block["content"]["code"],
                "language": block["content"]["language"],
                "document_id": doc["document_id"]
            })
    return examples
```

---

## 9. Schema Validation JSON Schema (Draft 2020-12)

For programmatic validation, here is the JSON Schema that validates any `doc2ml-json` document.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://doc2ml.dev/schema/v0.6.2.json",
  "title": "Doc2MLDocument",
  "type": "object",
  "required": ["doc2ml_version", "document_id", "metadata", "structure", "blocks", "cross_references", "ml_index"],
  "properties": {
    "doc2ml_version": {
      "type": "string",
      "enum": ["0.6.2"]
    },
    "document_id": {
      "type": "string",
      "format": "uuid"
    },
    "metadata": {
      "$ref": "#/$defs/DocumentMetadata"
    },
    "structure": {
      "$ref": "#/$defs/StructureNode"
    },
    "blocks": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/Block"
      }
    },
    "cross_references": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/CrossReference"
      }
    },
    "ml_index": {
      "$ref": "#/$defs/MLIndex"
    },
    "custom": {
      "type": "object"
    }
  },
  "$defs": {
    "DocumentMetadata": {
      "type": "object",
      "required": ["title", "source", "ingestion", "language", "statistics"],
      "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "authors": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "orcid": {"type": "string"},
              "affiliation": {"type": "string"},
              "email": {"type": "string", "format": "email"}
            },
            "required": ["name"]
          }
        },
        "source": {
          "type": "object",
          "required": ["uri", "mime_type", "filename", "checksum_sha256", "file_size_bytes"],
          "properties": {
            "uri": {"type": "string"},
            "mime_type": {"type": "string"},
            "filename": {"type": "string"},
            "checksum_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "file_size_bytes": {"type": "integer", "minimum": 0}
          }
        },
        "ingestion": {
          "type": "object",
          "required": ["ingestion_date", "processing_version", "extractor", "extractor_version"],
          "properties": {
            "ingestion_date": {"type": "string", "format": "date-time"},
            "processing_version": {"type": "string"},
            "extractor": {"type": "string"},
            "extractor_version": {"type": "string"},
            "ingestion_pipeline": {"type": "array", "items": {"type": "string"}},
            "processing_duration_ms": {"type": "integer", "minimum": 0}
          }
        },
        "language": {
          "type": "object",
          "required": ["detected"],
          "properties": {
            "detected": {"type": "string", "pattern": "^[a-z]{2}(-[A-Z]{2})?$"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "declared": {"type": "string"}
          }
        },
        "statistics": {
          "type": "object",
          "properties": {
            "page_count": {"type": "integer", "minimum": 0},
            "chapter_count": {"type": "integer", "minimum": 0},
            "section_count": {"type": "integer", "minimum": 0},
            "block_count": {"type": "integer", "minimum": 0},
            "table_count": {"type": "integer", "minimum": 0},
            "figure_count": {"type": "integer", "minimum": 0},
            "footnote_count": {"type": "integer", "minimum": 0},
            "total_char_count": {"type": "integer", "minimum": 0},
            "total_token_count_est": {"type": "integer", "minimum": 0},
            "total_word_count": {"type": "integer", "minimum": 0}
          }
        },
        "classification": {
          "type": "object",
          "properties": {
            "doc_type": {"type": "string"},
            "genre": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "topics_ml": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "label": {"type": "string"},
                  "score": {"type": "number", "minimum": 0, "maximum": 1}
                }
              }
            }
          }
        },
        "dates": {
          "type": "object",
          "properties": {
            "created": {"type": "string", "format": "date-time"},
            "modified": {"type": "string", "format": "date-time"},
            "published": {"type": "string", "format": "date-time"}
          }
        },
        "rights": {
          "type": "object",
          "properties": {
            "license": {"type": "string"},
            "copyright": {"type": "string"},
            "open_access": {"type": "boolean"}
          }
        }
      }
    },
    "StructureNode": {
      "type": "object",
      "required": ["node_id", "node_type", "level"],
      "properties": {
        "node_id": {"type": "string"},
        "node_type": {
          "type": "string",
          "enum": ["document", "part", "chapter", "section", "subsection", "subsubsection", "appendix", "front_matter", "back_matter", "page"]
        },
        "title": {"type": "string"},
        "level": {"type": "integer", "minimum": 0},
        "chunk_ids": {"type": "array", "items": {"type": "string"}},
        "children": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/StructureNode"
          }
        },
        "page_start": {"type": "integer", "minimum": 1},
        "page_end": {"type": "integer", "minimum": 1},
        "custom": {"type": "object"}
      }
    },
    "Block": {
      "type": "object",
      "required": ["chunk_id", "type", "content", "text_plain", "char_count", "token_count_est", "embedding_ready", "context_window", "provenance", "language", "semantics", "relations"],
      "properties": {
        "chunk_id": {"type": "string"},
        "type": {
          "type": "string",
          "enum": [
            "heading", "paragraph", "table", "list", "code_block", "figure_caption",
            "footnote", "quote", "metadata", "equation", "abstract", "keyword_block",
            "reference_entry", "page_header", "page_footer", "image_placeholder",
            "annotation", "divider", "unknown"
          ]
        },
        "content": {"type": "object"},
        "text_plain": {"type": "string"},
        "char_count": {"type": "integer", "minimum": 0},
        "token_count_est": {"type": "integer", "minimum": 0},
        "embedding_ready": {"type": "boolean"},
        "context_window": {
          "type": "object",
          "properties": {
            "prev_chunk_id": {"type": ["string", "null"]},
            "next_chunk_id": {"type": ["string", "null"]},
            "parent_heading_chunk_id": {"type": ["string", "null"]},
            "parent_structure_node_id": {"type": ["string", "null"]},
            "surrounding_text_preview": {"type": "string"}
          }
        },
        "provenance": {
          "type": "object",
          "required": ["extraction_method", "confidence"],
          "properties": {
            "page_number": {"type": ["integer", "null"]},
            "page_range": {"type": "array", "items": {"type": "integer"}},
            "bounding_box": {
              "type": "object",
              "properties": {
                "x1": {"type": "number"},
                "y1": {"type": "number"},
                "x2": {"type": "number"},
                "y2": {"type": "number"}
              }
            },
            "source_location": {"type": "string"},
            "extraction_method": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
          }
        },
        "language": {
          "type": "object",
          "properties": {
            "detected": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
          }
        },
        "semantics": {
          "type": "object",
          "properties": {
            "heading_level": {"type": ["integer", "null"]},
            "is_first_paragraph": {"type": "boolean"},
            "is_last_paragraph": {"type": "boolean"},
            "section_role": {"type": ["string", "null"]},
            "sentiment_score": {"type": ["number", "null"]},
            "readability_flesch": {"type": ["number", "null"]}
          }
        },
        "relations": {
          "type": "object",
          "properties": {
            "part_of_table": {"type": ["string", "null"]},
            "part_of_list": {"type": ["string", "null"]},
            "part_of_figure": {"type": ["string", "null"]},
            "footnotes_for": {"type": "array", "items": {"type": "string"}},
            "references": {"type": "array", "items": {"type": "string"}}
          }
        },
        "custom": {"type": "object"}
      }
    },
    "CrossReference": {
      "type": "object",
      "required": ["ref_id", "ref_type", "source_chunk_id", "resolved"],
      "properties": {
        "ref_id": {"type": "string"},
        "ref_type": {
          "type": "string",
          "enum": ["citation", "internal_link", "figure_ref", "table_ref", "footnote_ref", "equation_ref", "external_link"]
        },
        "source_chunk_id": {"type": "string"},
        "target_chunk_id": {"type": ["string", "null"]},
        "target_structure_node_id": {"type": ["string", "null"]},
        "label": {"type": "string"},
        "context_text": {"type": "string"},
        "resolved": {"type": "boolean"}
      }
    },
    "MLIndex": {
      "type": "object",
      "properties": {
        "chunk_id_map": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "index": {"type": "integer"},
              "structure_path": {"type": "array", "items": {"type": "string"}}
            }
          }
        },
        "heading_map": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "chunk_id": {"type": "string"},
              "heading_text": {"type": "string"},
              "level": {"type": "integer"},
              "node_id": {"type": "string"}
            }
          }
        },
        "embedding_candidates": {"type": "array", "items": {"type": "string"}},
        "chunk_boundaries": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "start_chunk_id": {"type": "string"},
              "end_chunk_id": {"type": "string"},
              "boundary_type": {"type": "string"},
              "token_count_est": {"type": "integer"}
            }
          }
        }
      }
    }
  }
}
```

---

## Appendix: Field Reference Quick-Table

| Top-Level Key      | Cardinality | Purpose                                                 |
| ------------------ | ----------- | ------------------------------------------------------- |
| `doc2ml_version`   | 1           | Schema version for compatibility checks.                |
| `document_id`      | 1           | UUID for global uniqueness across datasets.             |
| `metadata`         | 1           | All bibliographic, ingestion, and statistical metadata. |
| `structure`        | 1           | Tree of `StructureNode`s preserving logical hierarchy.  |
| `blocks`           | 1..*        | Flat ordered array of all content blocks.               |
| `cross_references` | 0..*        | All internal and external references.                   |
| `ml_index`         | 1           | Pre-computed indices for ML pipeline acceleration.      |
| `custom`           | 0..1        | Extensible key-value store for format-specific data.    |

---

## Appendix B: Changelog

| Version | Date       | Changes                                                 |
| ------- | ---------- | ------------------------------------------------------- |
| 0.6.2   | 2026-05-04 | Added optional Tesseract OCR fallback for scanned PDFs. |

---

*End of `doc2ml-json` Canonical Schema v0.6.2*
