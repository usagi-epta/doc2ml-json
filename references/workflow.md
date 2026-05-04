# doc2ml-json: Document Ingestion & ML-Ready JSON Workflow

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Format Detection](#3-phase-1-format-detection)
4. [Phase 2: Content Extraction](#4-phase-2-content-extraction)
5. [Phase 3: Structure Understanding](#5-phase-3-structure-understanding)
6. [Phase 4: Normalization](#6-phase-4-normalization)
7. [Phase 5: JSON Generation](#7-phase-5-json-generation)
8. [Phase 6: Validation & Output](#8-phase-6-validation--output)
9. [Error Handling & Recovery](#9-error-handling--recovery)
10. [Edge Case Guide](#10-edge-case-guide)
11. [Tool Mapping](#11-tool-mapping)
12. [Decision Trees](#12-decision-trees)
13. [Appendix A: JSON Schema](#appendix-a-json-schema)
14. [Appendix B: Quick Reference](#appendix-b-quick-reference)

---

## 1. Executive Summary

`doc2ml-json` is an agent skill that transforms any document into a structured, ML-ready JSON representation. The workflow operates in six sequential phases: Format Detection → Content Extraction → Structure Understanding → Normalization → JSON Generation → Validation & Output.

**Design principles:**
- **Defensive parsing**: Expect malformed input; always have a fallback strategy.
- **Progressive enhancement**: Extract what you can; never fail entirely if partial extraction is possible.
- **Semantic preservation**: Maintain heading hierarchies, section boundaries, list nesting, and table structures.
- **Agent executability**: Every step must be runnable by an autonomous agent using available tools.

---

## 2. Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Input File     │───▶│  Phase 1: Format │───▶│  Phase 2: Extraction│
│  (any format)   │    │  Detection       │    │  (format-specific)  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌───────▼──────────────┐
│  Phase 6:       │◄───│  Phase 5: JSON   │◄───│  Phase 3: Structure  │
│  Validation &   │    │  Generation      │    │  Understanding       │
│  Output         │    │                  │    │                      │
└─────────────────┘    └──────────────────┘    └───────▲──────────────┘
                                                       │
                                             ┌─────────┴──────────┐
                                             │  Phase 4:          │
                                             │  Normalization     │
                                             └────────────────────┘
```

**Intermediate Representation (IR)**: All phases communicate via a normalized Python dictionary (the `DocumentIR`) with the following structure:

```python
DocumentIR = {
    "meta": {
        "source_file": str,
        "detected_format": str,
        "detection_confidence": float,  # 0.0-1.0
        "extraction_method": str,
        "extraction_timestamp": str,    # ISO 8601
        "processing_version": str,
    },
    "content": {
        "title": str | None,
        "authors": list[str],
        "language": str | None,
        "created_date": str | None,
        "modified_date": str | None,
        "pages": int | None,
        "word_count": int | None,
    },
    "structure": {
        "headings": list[HeadingNode],
        "sections": list[SectionNode],
        "tables": list[TableNode],
        "lists": list[ListNode],
        "footnotes": list[FootnoteNode],
        "references": list[ReferenceNode],
    },
    "body": list[BlockNode],  # Sequential content blocks
    "errors": list[ErrorRecord],
    "warnings": list[WarningRecord],
}
```

---

## 3. Phase 1: Format Detection

### 3.1 Detection Strategy (Three-Layer Defense)

The agent must determine format using a cascading approach:

**Layer 1: File Extension**
```python
EXTENSION_MAP = {
    ".pdf":  "application/pdf",
    ".epub": "application/epub+zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".html": "text/html",
    ".htm":  "text/html",
    ".rtf":  "application/rtf",
    ".xml":  "application/xml",
    ".json": "application/json",
    ".csv":  "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".odt":  "application/vnd.oasis.opendocument.text",
}
```

**Layer 2: Magic Number / Content Signature**
When extension is missing, wrong, or ambiguous, inspect the first bytes:

```python
def detect_by_signature(filepath: str) -> str | None:
    with open(filepath, "rb") as f:
        header = f.read(16)
    
    signatures = {
        b"%PDF-": "application/pdf",
        b"PK\x03\x04": "application/zip",  # DOCX, EPUB, XLSX are ZIP-based
        b"\x89PNG": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"{\xff\xd8": None,  # False positive guard
        b"<?xml": "application/xml",
        b"<html": "text/html",
        b"<!DOCT": "text/html",
    }
    
    for sig, mime in signatures.items():
        if header.startswith(sig):
            return mime
    
    # ZIP-based formats need deeper inspection
    if header.startswith(b"PK"):
        return _classify_zip(filepath)
    
    # Try UTF-8/ASCII text detection
    if _is_text_file(filepath):
        return _classify_text(filepath)
    
    return None
```

**Layer 3: Deep Content Inspection**
For ZIP-based files (DOCX, EPUB, XLSX), inspect internal structure:

```python
def _classify_zip(filepath: str) -> str:
    import zipfile
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            namelist = z.namelist()
            
            if "[Content_Types].xml" in namelist:
                # Could be DOCX, XLSX, PPTX
                if "word/document.xml" in namelist:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if "xl/workbook.xml" in namelist:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            if "mimetype" in namelist:
                mime = z.read("mimetype").decode("utf-8", errors="replace").strip()
                if "epub" in mime:
                    return "application/epub+zip"
            
            # EPUB fallback: look for .opf in OEBPS or OPS
            if any(".opf" in name for name in namelist):
                return "application/epub+zip"
            
            # ODT fallback
            if "content.xml" in namelist and "META-INF/manifest.xml" in namelist:
                return "application/vnd.oasis.opendocument.text"
                
    except zipfile.BadZipFile:
        pass
    
    return "application/zip"
```

**Text file classification** (for `.txt`, `.md`, `.csv`, `.html` without extension):
```python
def _classify_text(filepath: str) -> str:
    with open(filepath, "rb") as f:
        sample = f.read(8192)
    
    text = sample.decode("utf-8", errors="replace")
    
    # Check for markdown indicators
    if any(line.startswith(("# ", "## ", "### ", "- ", "* ", "| ")) for line in text.split("\n")[:50]):
        return "text/markdown"
    
    # Check for HTML tags
    if "<html" in text.lower() or "<!doctype html" in text.lower():
        return "text/html"
    
    # Check for CSV
    lines = text.strip().split("\n")
    if len(lines) > 1:
        first_line_commas = lines[0].count(",")
        second_line_commas = lines[1].count(",")
        if first_line_commas > 1 and second_line_commas == first_line_commas:
            return "text/csv"
    
    return "text/plain"
```

### 3.2 Confidence Scoring

```python
def compute_detection_confidence(
    extension_mime: str | None,
    signature_mime: str | None,
    deep_mime: str | None,
) -> tuple[str, float]:
    """
    Returns (final_mime, confidence_score)
    
    Scoring:
    - All three agree: 1.0
    - Extension + signature agree, no deep: 0.85
    - Signature + deep agree, extension wrong: 0.90
    - Only signature: 0.70
    - Only extension: 0.50
    - None agree: 0.30 (flag for manual review)
    """
    votes = [v for v in [extension_mime, signature_mime, deep_mime] if v is not None]
    
    if len(votes) >= 2 and len(set(votes)) == 1:
        return votes[0], 1.0
    
    if signature_mime == deep_mime and signature_mime is not None:
        return signature_mime, 0.90
    
    if extension_mime == signature_mime and extension_mime is not None:
        return extension_mime, 0.85
    
    if signature_mime is not None:
        return signature_mime, 0.70
    
    if extension_mime is not None:
        return extension_mime, 0.50
    
    if deep_mime is not None:
        return deep_mime, 0.60
    
    return "application/octet-stream", 0.10
```

### 3.3 Format-to-Strategy Router

```python
FORMAT_STRATEGY_MAP = {
    "application/pdf":                 Strategy.PDF,
    "application/epub+zip":            Strategy.EPUB,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Strategy.DOCX,
    "application/msword":              Strategy.DOCX_LEGACY,
    "text/plain":                      Strategy.TXT,
    "text/markdown":                   Strategy.MARKDOWN,
    "text/html":                       Strategy.HTML,
    "application/xhtml+xml":           Strategy.HTML,
    "application/rtf":                 Strategy.RTF,
    "application/vnd.oasis.opendocument.text": Strategy.ODT,
    "text/csv":                        Strategy.CSV,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Strategy.XLSX,
}
```

---

## 4. Phase 2: Content Extraction

### 4.1 PDF Extraction Strategy

**Decision tree for PDF:**
```
PDF file
├── Is it password-protected?
│   ├── YES → Try empty password → If fails, report error
│   └── NO → Continue
├── Is it a scanned PDF (image-only)?
│   ├── YES → OCR pipeline (see 10.1)
│   └── NO → Text-based extraction
├── Extraction approach:
│   ├── Layout-aware extraction (primary)
│   │   └── Use pdfplumber for tables + text with bbox
│   ├── Structured fallback
│   │   └── Use PyMuPDF (fitz) for text + metadata
│   └── Raw fallback
│       └── Use PyPDF2 for basic text
└── Post-processing:
    ├── Detect headings by font size/style
    ├── Detect columns by x-position clustering
    ├── Extract tables with pdfplumber
    └── Remove headers/footers by position
```

**Primary extraction recipe (pdfplumber + PyMuPDF):**
```python
import pdfplumber
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Iterator

@dataclass
class PDFPage:
    page_number: int
    width: float
    height: float
    text_blocks: list[dict]
    tables: list[list[list[str]]]
    images: list[dict]
    raw_text: str


def extract_pdf_pdfplumber(filepath: str) -> list[PDFPage]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract words with position info for layout analysis
            words = page.extract_words(
                keep_blank_chars=False,
                x_tolerance=3,
                y_tolerance=3,
            )
            
            # Extract tables
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                }
            )
            
            # Group words into text blocks by y-position (reading order)
            text_blocks = group_words_into_blocks(words)
            
            # Detect images (for OCR fallback if needed)
            images = page.images
            
            pages.append(PDFPage(
                page_number=i,
                width=page.width,
                height=page.height,
                text_blocks=text_blocks,
                tables=tables if tables else [],
                images=images,
                raw_text=page.extract_text() or "",
            ))
    return pages


def extract_pdf_pymupdf(filepath: str) -> list[dict]:
    """Fallback / supplementary extraction for metadata and structure."""
    doc = fitz.open(filepath)
    pages = []
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        structured_blocks = []
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        structured_blocks.append({
                            "text": span["text"],
                            "font": span["font"],
                            "size": span["size"],
                            "flags": span["flags"],  # bold, italic, etc.
                            "bbox": span["bbox"],
                            "color": span["color"],
                        })
        
        pages.append({
            "page_number": page.number + 1,
            "blocks": structured_blocks,
            "raw_text": page.get_text(),
        })
    
    metadata = doc.metadata
    doc.close()
    
    return {
        "pages": pages,
        "metadata": metadata,
        "toc": doc.get_toc() if hasattr(doc, "get_toc") else [],
    }


def group_words_into_blocks(words: list[dict], y_threshold: float = 5) -> list[dict]:
    """Group extracted words into reading-order text blocks."""
    if not words:
        return []
    
    # Sort by y-position, then x-position
    sorted_words = sorted(words, key=lambda w: (round(w["top"] / y_threshold), w["x0"]))
    
    blocks = []
    current_block = {"text": sorted_words[0]["text"], "top": sorted_words[0]["top"], "x0": sorted_words[0]["x0"]}
    
    for word in sorted_words[1:]:
        prev_y = current_block.get("bottom", current_block["top"])
        if abs(word["top"] - prev_y) <= y_threshold:
            current_block["text"] += " " + word["text"]
            current_block["bottom"] = word["bottom"]
        else:
            blocks.append(current_block)
            current_block = {"text": word["text"], "top": word["top"], "x0": word["x0"], "bottom": word["bottom"]}
    
    blocks.append(current_block)
    return blocks
```

**Font-based heading inference:**
```python
def infer_headings_from_font(spans: list[dict]) -> list[dict]:
    """
    Detect headings by analyzing font size distribution.
    Returns list of candidate heading blocks.
    """
    if not spans:
        return []
    
    # Compute size statistics
    sizes = [s["size"] for s in spans if s["text"].strip()]
    if not sizes:
        return []
    
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    
    headings = []
    for span in spans:
        text = span["text"].strip()
        size = span["size"]
        
        # Heuristic: heading if size is significantly above average
        # or if it's the largest size used for short text
        is_heading = False
        heading_level = None
        
        if size > avg_size * 1.3 and len(text) < 200:
            if size >= max_size * 0.95:
                heading_level = 1
            elif size >= max_size * 0.80:
                heading_level = 2
            elif size >= max_size * 0.65:
                heading_level = 3
            else:
                heading_level = 4
            is_heading = True
        
        # Also check bold flags (flag bit 4 = bold in PyMuPDF)
        if span.get("flags", 0) & 2**4 and size > avg_size and len(text) < 150:
            if not is_heading:
                heading_level = 3
                is_heading = True
        
        if is_heading:
            headings.append({
                "text": text,
                "level": heading_level,
                "bbox": span["bbox"],
                "page": span.get("page"),
                "font_size": size,
            })
    
    return headings
```

**Header/footer removal:**
```python
def remove_headers_footers(pages: list[PDFPage], margin_ratio: float = 0.10) -> list[PDFPage]:
    """
    Remove text blocks that appear in the same position across multiple pages
    and contain similar text (headers/footers/page numbers).
    """
    if len(pages) < 2:
        return pages
    
    # Collect text from top and bottom margins across all pages
    top_margin = pages[0].height * margin_ratio
    bottom_margin = pages[0].height * (1 - margin_ratio)
    
    header_candidates = []
    footer_candidates = []
    
    for page in pages:
        for block in page.text_blocks:
            if block["top"] < top_margin:
                header_candidates.append(block["text"])
            elif block.get("bottom", block["top"]) > bottom_margin:
                footer_candidates.append(block["text"])
    
    # Find repeating patterns (appearing on >50% of pages)
    from collections import Counter
    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)
    
    common_headers = {t for t, c in header_counts.items() if c > len(pages) * 0.5}
    common_footers = {t for t, c in footer_counts.items() if c > len(pages) * 0.5}
    
    # Filter pages
    for page in pages:
        page.text_blocks = [
            b for b in page.text_blocks
            if b["text"] not in common_headers and b["text"] not in common_footers
        ]
    
    return pages
```

### 4.2 EPUB Extraction Strategy

**Decision tree for EPUB:**
```
EPUB file
├── Validate ZIP structure
│   ├── Missing mimetype or META-INF → Report corruption, attempt repair
│   └── Valid → Continue
├── Read OPF package document
│   ├── Extract metadata (title, author, language, identifiers)
│   ├── Read spine (reading order)
│   └── Read manifest (resource mapping)
├── Navigate spine sequentially
│   ├── Parse each XHTML/HTML chapter
│   ├── Extract semantic structure (headings, sections, articles)
│   ├── Handle CSS (inline styles, class-based semantics)
│   └── Resolve internal links (fragment IDs)
└── Post-processing:
    ├── Merge split paragraphs across chapter boundaries
    ├── Detect chapter boundaries from heading patterns
    └── Extract table of contents from NCX/NavDoc
```

**Primary extraction recipe (ebooklib + BeautifulSoup):**
```python
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup

def extract_epub(filepath: str) -> dict:
    """Extract content and metadata from an EPUB file."""
    
    # Phase: Validate and inspect structure
    with zipfile.ZipFile(filepath, "r") as z:
        namelist = z.namelist()
        
        # Find OPF file
        opf_path = None
        if "META-INF/container.xml" in namelist:
            container_xml = z.read("META-INF/container.xml").decode("utf-8")
            container = ET.fromstring(container_xml)
            # Namespace handling
            ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = container.find(".//container:rootfile", ns)
            if rootfile is not None:
                opf_path = rootfile.get("full-path")
        
        if not opf_path:
            # Fallback: search for .opf file
            opf_candidates = [n for n in namelist if n.endswith(".opf")]
            if opf_candidates:
                opf_path = opf_candidates[0]
        
        # Parse OPF for metadata and spine
        opf_content = z.read(opf_path).decode("utf-8")
        opf = ET.fromstring(opf_content)
        
        # OPF namespaces
        opf_ns = {
            "opf": "http://www.idpf.org/2007/opf",
            "dc": "http://purl.org/dc/elements/1.1/",
        }
        
        # Extract metadata
        metadata = {
            "title": _get_dc_text(opf, "dc:title", opf_ns),
            "creator": _get_dc_text(opf, "dc:creator", opf_ns),
            "language": _get_dc_text(opf, "dc:language", opf_ns),
            "publisher": _get_dc_text(opf, "dc:publisher", opf_ns),
            "date": _get_dc_text(opf, "dc:date", opf_ns),
            "identifier": _get_dc_text(opf, "dc:identifier", opf_ns),
            "description": _get_dc_text(opf, "dc:description", opf_ns),
        }
        
        # Parse manifest
        manifest = {}
        manifest_elem = opf.find(".//opf:manifest", opf_ns)
        if manifest_elem is not None:
            for item in manifest_elem.findall("opf:item", opf_ns):
                manifest[item.get("id")] = {
                    "href": item.get("href"),
                    "media-type": item.get("media-type"),
                }
        
        # Parse spine (reading order)
        spine = []
        spine_elem = opf.find(".//opf:spine", opf_ns)
        if spine_elem is not None:
            for itemref in spine_elem.findall("opf:itemref", opf_ns):
                spine.append(itemref.get("idref"))
        
        # Determine base directory for relative hrefs
        opf_dir = str(Path(opf_path).parent) if "/" in opf_path else ""
        
        # Extract chapters
        chapters = []
        for item_id in spine:
            if item_id not in manifest:
                continue
            
            href = manifest[item_id]["href"]
            chapter_path = f"{opf_dir}/{href}" if opf_dir else href
            chapter_path = chapter_path.replace("//", "/")
            
            if chapter_path not in namelist:
                continue
            
            html_content = z.read(chapter_path).decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style tags
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            
            chapter_data = parse_epub_chapter(soup, chapter_path, href)
            chapters.append(chapter_data)
        
        # Extract TOC from NCX or NavDoc
        toc = extract_epub_toc(z, namelist, manifest, opf_dir)
    
    return {
        "metadata": metadata,
        "chapters": chapters,
        "toc": toc,
        "manifest": manifest,
        "spine": spine,
    }


def _get_dc_text(opf, tag, ns):
    elem = opf.find(f".//{tag}", ns)
    return elem.text if elem is not None else None


def parse_epub_chapter(soup: BeautifulSoup, full_path: str, href: str) -> dict:
    """Parse a single EPUB chapter into structured blocks."""
    
    blocks = []
    
    # Get chapter title from first h1-h6
    title = None
    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading:
        title = heading.get_text(strip=True)
    
    # Extract body or main content area
    content_root = soup.find("body") or soup.find("main") or soup.find("article") or soup
    
    # Walk the DOM and extract structure
    for elem in content_root.find_all(recursive=False):
        block = _convert_element_to_block(elem)
        if block:
            blocks.append(block)
    
    return {
        "href": href,
        "full_path": full_path,
        "title": title,
        "blocks": blocks,
        "raw_text": soup.get_text(separator="\n", strip=True),
    }


def _convert_element_to_block(elem) -> dict | None:
    """Convert a BeautifulSoup element to a structured block."""
    tag = elem.name
    
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return {
            "type": "heading",
            "level": int(tag[1]),
            "text": elem.get_text(strip=True),
            "id": elem.get("id"),
        }
    
    elif tag == "p":
        return {
            "type": "paragraph",
            "text": elem.get_text(strip=True),
            "id": elem.get("id"),
        }
    
    elif tag in ("ul", "ol"):
        return {
            "type": "list",
            "ordered": tag == "ol",
            "items": [_extract_list_item(li) for li in elem.find_all("li", recursive=False)],
        }
    
    elif tag == "table":
        return {
            "type": "table",
            "rows": _extract_table_rows(elem),
        }
    
    elif tag == "img":
        return {
            "type": "image",
            "src": elem.get("src"),
            "alt": elem.get("alt", ""),
        }
    
    elif tag in ("div", "section", "article"):
        # Recurse into containers
        inner_blocks = []
        for child in elem.find_all(recursive=False):
            block = _convert_element_to_block(child)
            if block:
                inner_blocks.append(block)
        if inner_blocks:
            return {
                "type": "container",
                "tag": tag,
                "id": elem.get("id"),
                "blocks": inner_blocks,
            }
        return None
    
    return None  # Skip unknown/unhandled tags


def _extract_list_item(li) -> dict:
    nested = li.find(["ul", "ol"], recursive=False)
    text = li.get_text(strip=True)
    if nested:
        text = text.replace(nested.get_text(strip=True), "").strip()
    
    item = {"text": text}
    if nested:
        item["nested"] = {
            "ordered": nested.name == "ol",
            "items": [_extract_list_item(nli) for nli in nested.find_all("li", recursive=False)],
        }
    return item


def _extract_table_rows(table) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            row.append(cell.get_text(strip=True))
        if row:
            rows.append(row)
    return rows


def extract_epub_toc(z, namelist, manifest, opf_dir):
    """Extract table of contents from NCX or HTML NavDoc."""
    toc = []
    
    # Try NCX first (EPUB2)
    ncx_items = [k for k, v in manifest.items() if v.get("media-type") == "application/x-dtbncx+xml"]
    if ncx_items:
        ncx_href = manifest[ncx_items[0]]["href"]
        ncx_path = f"{opf_dir}/{ncx_href}".replace("//", "/")
        if ncx_path in namelist:
            ncx_content = z.read(ncx_path).decode("utf-8")
            ncx = ET.fromstring(ncx_content)
            ncx_ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
            
            for navpoint in ncx.findall(".//ncx:navPoint", ncx_ns):
                label = navpoint.find(".//ncx:text", ncx_ns)
                content = navpoint.find("ncx:content", ncx_ns)
                toc.append({
                    "title": label.text if label is not None else "",
                    "src": content.get("src") if content is not None else "",
                })
    
    # Try NavDoc (EPUB3)
    if not toc:
        nav_items = [k for k, v in manifest.items() if v.get("properties") == "nav"]
        if nav_items:
            nav_href = manifest[nav_items[0]]["href"]
            nav_path = f"{opf_dir}/{nav_href}".replace("//", "/")
            if nav_path in namelist:
                nav_content = z.read(nav_path).decode("utf-8")
                soup = BeautifulSoup(nav_content, "html.parser")
                nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav")
                if nav:
                    for a in nav.find_all("a"):
                        toc.append({
                            "title": a.get_text(strip=True),
                            "src": a.get("href", ""),
                        })
    
    return toc
```

### 4.3 DOCX Extraction Strategy

**Decision tree for DOCX:**
```
DOCX file
├── Validate ZIP structure
│   ├── Check word/document.xml exists
│   └── If missing, check for older Word formats (fallback to antiword/catdoc)
├── Parse document.xml (main content)
│   ├── Handle w:p (paragraphs) with w:pPr (properties)
│   ├── Extract w:t (text runs) within w:r (runs)
│   ├── Detect headings from pStyle (Heading1, Heading2, ...)
│   ├── Handle tables (w:tbl with w:tr rows and w:tc cells)
│   ├── Handle lists (numPr with numId/ilvl)
│   └── Handle hyperlinks (w:hyperlink with rId)
├── Parse styles.xml
│   ├── Build style ID → style definition map
│   └── Resolve inherited styles
├── Parse numbering.xml
│   ├── Build numId → numbering definition map
│   └── Determine list type (bullet, decimal, etc.)
├── Parse relationships.xml
│   ├── Resolve hyperlink targets
│   └── Map image rIds to actual image files
├── Parse footnotes.xml / endnotes.xml
│   └── Extract footnote content with references
└── Post-processing:
    ├── Reconstruct heading hierarchy from styles
    ├── Resolve revision marks (accept all changes)
    └── Merge split runs into coherent paragraphs
```

**Primary extraction recipe (python-docx + manual XML):**
```python
from docx import Document
from docx.oxml.ns import qn
from lxml import etree
import zipfile

def extract_docx(filepath: str) -> dict:
    """Extract structured content from a DOCX file."""
    
    doc = Document(filepath)
    
    # Build style map for heading detection
    styles = {}
    for style in doc.styles:
        if style.style_id:
            styles[style.style_id] = {
                "name": style.name,
                "type": style.type,
                "based_on": style.base_style.style_id if style.base_style else None,
            }
    
    # Build numbering definitions from XML
    numbering_map = _extract_numbering_map(filepath)
    
    # Build hyperlink relationship map
    hyperlink_map = _extract_hyperlink_map(filepath)
    
    paragraphs = []
    for para in doc.paragraphs:
        para_data = parse_docx_paragraph(para, styles, numbering_map, hyperlink_map)
        if para_data:
            paragraphs.append(para_data)
    
    tables = []
    for table in doc.tables:
        tables.append(extract_docx_table(table))
    
    # Extract footnotes
    footnotes = extract_docx_footnotes(filepath)
    
    # Extract document properties
    core_props = doc.core_properties
    metadata = {
        "title": core_props.title,
        "author": core_props.author,
        "subject": core_props.subject,
        "created": str(core_props.created) if core_props.created else None,
        "modified": str(core_props.modified) if core_props.modified else None,
        "language": core_props.language,
    }
    
    return {
        "metadata": metadata,
        "paragraphs": paragraphs,
        "tables": tables,
        "footnotes": footnotes,
        "styles": styles,
    }


def parse_docx_paragraph(para, styles, numbering_map, hyperlink_map):
    """Parse a single DOCX paragraph into a structured block."""
    
    # Get paragraph style
    style_id = para.style.style_id if para.style else None
    style_info = styles.get(style_id, {}) if style_id else {}
    style_name = style_info.get("name", "").lower() if style_info else ""
    
    # Detect heading by style name
    heading_level = None
    if "heading" in style_name:
        try:
            heading_level = int(style_name.replace("heading", "").strip())
        except ValueError:
            heading_level = 1  # Default if parsing fails
    
    # Detect list item
    list_info = None
    pPr = para._p.find(qn("w:pPr"))
    if pPr is not None:
        numPr = pPr.find(qn("w:numPr"))
        if numPr is not None:
            numId = numPr.find(qn("w:numId"))
            ilvl = numPr.find(qn("w:ilvl"))
            if numId is not None:
                num_id_val = numId.get(qn("w:val"))
                ilvl_val = ilvl.get(qn("w:val")) if ilvl is not None else "0"
                list_type = numbering_map.get(num_id_val, "bullet")
                list_info = {
                    "num_id": num_id_val,
                    "level": int(ilvl_val),
                    "type": list_type,
                }
    
    # Extract text runs with formatting
    runs = []
    for run in para.runs:
        run_text = run.text
        if not run_text:
            continue
        
        run_data = {
            "text": run_text,
            "bold": run.bold,
            "italic": run.italic,
            "underline": run.underline,
            "font_size": run.font.size.pt if run.font.size else None,
        }
        runs.append(run_data)
    
    full_text = "".join(r["text"] for r in runs)
    if not full_text.strip():
        return None
    
    # Check for hyperlink in runs via XML
    links = []
    for hyperlink in para._p.findall(qn("w:hyperlink")):
        rId = hyperlink.get(qn("r:id"))
        if rId and rId in hyperlink_map:
            link_text = "".join(r.text for r in hyperlink.findall(".//" + qn("w:t")) if r.text)
            links.append({
                "text": link_text,
                "url": hyperlink_map[rId],
            })
    
    return {
        "type": "heading" if heading_level else ("list_item" if list_info else "paragraph"),
        "heading_level": heading_level,
        "list_info": list_info,
        "text": full_text,
        "runs": runs,
        "links": links,
        "alignment": para.alignment,
        "style": style_id,
    }


def extract_docx_table(table) -> dict:
    """Extract a DOCX table into structured rows/cells."""
    rows = []
    for row in table.rows:
        row_data = []
        for cell in row.cells:
            # Handle merged cells: check gridSpan
            cell_xml = cell._tc
            gridSpan = cell_xml.find(".//" + qn("w:gridSpan"))
            span = int(gridSpan.get(qn("w:val"))) if gridSpan is not None else 1
            
            cell_text = cell.text.strip()
            row_data.append({
                "text": cell_text,
                "span": span,
            })
        rows.append(row_data)
    
    return {
        "type": "table",
        "rows": rows,
        "row_count": len(rows),
        "column_count": max(len(r) for r in rows) if rows else 0,
    }


def _extract_numbering_map(filepath: str) -> dict:
    """Extract numbering definitions to determine list types."""
    numbering_map = {}
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            if "word/numbering.xml" in z.namelist():
                xml = z.read("word/numbering.xml")
                root = etree.fromstring(xml)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                
                for num in root.findall("w:num", ns):
                    num_id = num.get(qn("w:numId"))
                    abstract_num_id = num.find("w:abstractNumId", ns)
                    if abstract_num_id is not None:
                        abs_id = abstract_num_id.get(qn("w:val"))
                        # Simplified: default to bullet, could be extended
                        numbering_map[num_id] = "bullet"
    except Exception:
        pass
    
    return numbering_map


def _extract_hyperlink_map(filepath: str) -> dict:
    """Extract hyperlink relationships from relationships.xml."""
    hyperlink_map = {}
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            rels_path = "word/_rels/document.xml.rels"
            if rels_path in z.namelist():
                xml = z.read(rels_path)
                root = etree.fromstring(xml)
                ns = {
                    "rels": "http://schemas.openxmlformats.org/package/2006/relationships"
                }
                for rel in root.findall("rels:Relationship", ns):
                    if "hyperlink" in (rel.get("Type") or ""):
                        rId = rel.get("Id")
                        target = rel.get("Target")
                        hyperlink_map[rId] = target
    except Exception:
        pass
    
    return hyperlink_map


def extract_docx_footnotes(filepath: str) -> list[dict]:
    """Extract footnotes from DOCX."""
    footnotes = []
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            if "word/footnotes.xml" in z.namelist():
                xml = z.read("word/footnotes.xml")
                root = etree.fromstring(xml)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                
                for fn in root.findall("w:footnote", ns):
                    fn_id = fn.get(qn("w:id"))
                    if fn_id in ("0", "-1"):  # Separator footnotes
                        continue
                    text = " ".join(t.text for t in fn.findall(".//" + qn("w:t")) if t.text)
                    if text.strip():
                        footnotes.append({"id": fn_id, "text": text.strip()})
    except Exception:
        pass
    
    return footnotes
```

### 4.4 TXT / Markdown Extraction Strategy

**Decision tree for TXT/MD:**
```
TXT/MD file
├── Detect encoding (UTF-8 → Latin-1 → chardet fallback)
├── Check for frontmatter (YAML between --- delimiters)
│   └── Extract if present, parse with yaml.safe_load
├── For Markdown:
│   ├── Parse with markdown.parser or mistletoe
│   ├── Extract heading hierarchy (# → h1, ## → h2, etc.)
│   ├── Identify code blocks (``` or indented)
│   ├── Identify tables (GFM pipe syntax)
│   ├── Identify lists (-, *, 1.)
│   └── Identify blockquotes (>)
└── For TXT:
    ├── Line-based structure inference
    ├── Detect paragraphs (blank-line separated)
    ├── Detect possible headings (all-caps, underlined with =/-)
    └── Detect possible lists (leading bullets/numbers)
```

**Primary extraction recipe:**
```python
import re
import yaml
from markdown import Markdown
from markdown.blockprocessors import BlockProcessor
from markdown.extensions import Extension

def extract_text_file(filepath: str) -> dict:
    """Extract structured content from a plain text or markdown file."""
    
    # Phase: Detect encoding
    raw_bytes = open(filepath, "rb").read()
    encoding = detect_encoding(raw_bytes)
    text = raw_bytes.decode(encoding, errors="replace")
    
    # Phase: Check for frontmatter
    frontmatter = {}
    body_text = text
    
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                body_text = parts[2].strip()
            except yaml.YAMLError:
                pass  # Not valid YAML, treat as regular text
    
    # Determine if markdown by extension or content
    is_markdown = filepath.lower().endswith((".md", ".markdown", ".mdown"))
    if not is_markdown:
        # Content-based detection
        md_indicators = ["# ", "## ", "```", "| ", "- ["]
        sample_lines = body_text.split("\n")[:30]
        is_markdown = any(any(line.startswith(ind) for ind in md_indicators) for line in sample_lines)
    
    if is_markdown:
        return extract_markdown(body_text, frontmatter)
    else:
        return extract_plain_text(body_text, frontmatter)


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect file encoding with fallback chain."""
    # Try UTF-8 BOM first
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    
    # Try UTF-8
    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    
    # Try Latin-1 (never fails, but may be wrong)
    return "latin-1"


def extract_markdown(text: str, frontmatter: dict) -> dict:
    """Parse markdown into structured blocks."""
    
    blocks = []
    lines = text.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        
        # Heading: ATX style (# Heading)
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped.lstrip("#").strip()
            blocks.append({
                "type": "heading",
                "level": min(level, 6),
                "text": heading_text,
            })
            i += 1
            continue
        
        # Heading: Setext style (underlined with = or -)
        if i + 1 < len(lines) and lines[i + 1].rstrip() and all(c == "=" for c in lines[i + 1].rstrip()):
            blocks.append({
                "type": "heading",
                "level": 1,
                "text": stripped,
            })
            i += 2
            continue
        if i + 1 < len(lines) and lines[i + 1].rstrip() and all(c == "-" for c in lines[i + 1].rstrip()):
            blocks.append({
                "type": "heading",
                "level": 2,
                "text": stripped,
            })
            i += 2
            continue
        
        # Code block: fenced
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({
                "type": "code_block",
                "language": lang,
                "text": "\n".join(code_lines),
            })
            i += 1  # Skip closing ```
            continue
        
        # Code block: indented (4 spaces)
        if line.startswith("    ") or line.startswith("\t"):
            code_lines = []
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i] == ""):
                code_lines.append(lines[i])
                i += 1
            blocks.append({
                "type": "code_block",
                "language": None,
                "text": "\n".join(code_lines),
            })
            continue
        
        # Table: GFM pipe syntax
        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
            # Check if next line is separator (---|---|---)
            next_stripped = lines[i + 1].strip()
            if re.match(r"^\|?[\s\-:]+(\|[\s\-:]+)*\|?\s*$", next_stripped):
                table_rows = []
                # Header row
                table_rows.append([cell.strip() for cell in line.split("|") if cell.strip() or cell == ""])
                table_rows = [[c for c in row if c] for row in table_rows]  # Clean empty
                i += 2  # Skip separator line
                # Data rows
                while i < len(lines) and "|" in lines[i]:
                    cells = [cell.strip() for cell in lines[i].split("|")]
                    cells = [c for c in cells if c or c == ""]
                    if any(cells):
                        table_rows.append(cells)
                    i += 1
                blocks.append({
                    "type": "table",
                    "rows": table_rows,
                })
                continue
        
        # List item
        list_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", line)
        if list_match:
            indent = len(list_match.group(1))
            marker = list_match.group(2)
            is_ordered = marker[0].isdigit()
            
            list_items = []
            current_item = {
                "text": list_match.group(3),
                "level": indent // 2,
            }
            list_items.append(current_item)
            i += 1
            
            # Continue collecting list items
            while i < len(lines):
                next_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", lines[i])
                if next_match:
                    list_items.append({
                        "text": next_match.group(3),
                        "level": len(next_match.group(1)) // 2,
                    })
                    i += 1
                elif lines[i].strip() == "":
                    break
                elif lines[i].startswith("  ") or lines[i].startswith("\t"):
                    # Continuation of current item
                    list_items[-1]["text"] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            
            blocks.append({
                "type": "list",
                "ordered": is_ordered,
                "items": list_items,
            })
            continue
        
        # Blockquote
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and (lines[i].lstrip().startswith(">") or lines[i].strip() == ""):
                quote_lines.append(lines[i].lstrip().lstrip(">").strip())
                i += 1
            blocks.append({
                "type": "blockquote",
                "text": " ".join(quote_lines),
            })
            continue
        
        # Horizontal rule
        if stripped in ("---", "***", "___", "- - -", "* * *"):
            blocks.append({"type": "horizontal_rule"})
            i += 1
            continue
        
        # Paragraph
        if stripped:
            para_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip():
                para_lines.append(lines[i].strip())
                i += 1
            blocks.append({
                "type": "paragraph",
                "text": " ".join(para_lines),
            })
            continue
        
        i += 1
    
    return {
        "frontmatter": frontmatter,
        "blocks": blocks,
        "raw_text": text,
        "is_markdown": True,
    }


def extract_plain_text(text: str, frontmatter: dict) -> dict:
    """Extract structure from plain text using heuristics."""
    
    blocks = []
    lines = text.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.lstrip()
        
        # Skip empty lines
        if not stripped:
            i += 1
            continue
        
        # Detect heading: all caps short line
        if stripped.isupper() and len(stripped) < 100 and len(stripped) > 3:
            # Check if next line is blank or also caps
            if i + 1 < len(lines) and not lines[i + 1].strip():
                blocks.append({
                    "type": "heading",
                    "level": 1,  # Assume top-level
                    "text": stripped.title(),
                })
                i += 1
                continue
        
        # Detect heading: underlined with = or -
        if i + 1 < len(lines):
            next_line = lines[i + 1].rstrip()
            if next_line and all(c == "=" for c in next_line) and len(next_line) >= len(stripped):
                blocks.append({
                    "type": "heading",
                    "level": 1,
                    "text": stripped,
                })
                i += 2
                continue
            if next_line and all(c == "-" for c in next_line) and len(next_line) >= len(stripped):
                blocks.append({
                    "type": "heading",
                    "level": 2,
                    "text": stripped,
                })
                i += 2
                continue
        
        # Detect list item
        list_match = re.match(r"^(\s*)([-*•]|\d+[.):])\s+(.*)", line)
        if list_match:
            indent = len(list_match.group(1))
            marker = list_match.group(2)
            is_ordered = marker[0].isdigit()
            
            list_items = []
            list_items.append({
                "text": list_match.group(3),
                "level": indent // 2,
            })
            i += 1
            
            while i < len(lines):
                next_match = re.match(r"^(\s*)([-*•]|\d+[.):])\s+(.*)", lines[i])
                if next_match:
                    list_items.append({
                        "text": next_match.group(3),
                        "level": len(next_match.group(1)) // 2,
                    })
                    i += 1
                elif lines[i].strip() == "":
                    break
                else:
                    break
            
            blocks.append({
                "type": "list",
                "ordered": is_ordered,
                "items": list_items,
            })
            continue
        
        # Paragraph
        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip():
            para_lines.append(lines[i].strip())
            i += 1
        
        blocks.append({
            "type": "paragraph",
            "text": " ".join(para_lines),
        })
    
    return {
        "frontmatter": frontmatter,
        "blocks": blocks,
        "raw_text": text,
        "is_markdown": False,
    }
```

### 4.5 HTML Extraction Strategy

**Decision tree for HTML:**
```
HTML file
├── Parse with BeautifulSoup (lenient)
├── Detect semantic structure
│   ├── <main>, <article>, <section> → section boundaries
│   ├── <h1>-<h6> → heading hierarchy
│   ├── <nav> → skip (navigation, not content)
│   ├── <header>, <footer> → evaluate, may contain useful content
│   └── <aside> → optional inclusion
├── Extract content blocks
│   ├── <p> → paragraphs
│   ├── <ul>, <ol> → lists
│   ├── <table> → tables
│   ├── <pre>, <code> → code blocks
│   ├── <blockquote> → blockquotes
│   └── <img>, <figure> → images/figures
├── Handle special elements
│   ├── <a href="..."> → links (preserve URL)
│   ├── <strong>, <em> → inline formatting
│   ├── <br> → line breaks
│   └── <script>, <style> → remove
└── Post-processing:
    ├── Clean up whitespace
    ├── Resolve relative URLs (if base URL known)
    └── Extract metadata from <meta> tags and <title>
```

**Primary extraction recipe:**
```python
from bs4 import BeautifulSoup, NavigableString, Tag
import re

def extract_html(filepath_or_text: str, is_file: bool = True, base_url: str = None) -> dict:
    """Extract structured content from HTML."""
    
    if is_file:
        with open(filepath_or_text, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()
    else:
        raw_html = filepath_or_text
    
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # Remove script, style, nav tags
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    
    # Extract metadata
    metadata = {
        "title": soup.title.get_text(strip=True) if soup.title else None,
    }
    
    # Extract meta tags
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content")
        
        if name in ("description", "author", "keywords", "language"):
            metadata[name] = content
        if prop in ("og:title", "og:description"):
            metadata[prop.replace("og:", "")] = content
    
    # Determine content root
    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|article", re.I))
        or soup.find("body")
        or soup
    )
    
    blocks = []
    for elem in content_root.children:
        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            if text:
                blocks.append({"type": "text", "text": text})
        elif isinstance(elem, Tag):
            block = _convert_html_element(elem, base_url)
            if block:
                blocks.append(block)
    
    # Extract headings for structure
    headings = []
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append({
            "level": int(h.name[1]),
            "text": h.get_text(strip=True),
            "id": h.get("id"),
        })
    
    return {
        "metadata": metadata,
        "blocks": blocks,
        "headings": headings,
        "raw_text": content_root.get_text(separator="\n", strip=True),
    }


def _convert_html_element(elem: Tag, base_url: str | None) -> dict | None:
    """Convert an HTML element to a structured block."""
    tag = elem.name
    
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return {
            "type": "heading",
            "level": int(tag[1]),
            "text": _clean_html_text(elem),
            "id": elem.get("id"),
        }
    
    elif tag == "p":
        return {
            "type": "paragraph",
            "text": _clean_html_text(elem),
        }
    
    elif tag in ("ul", "ol"):
        return {
            "type": "list",
            "ordered": tag == "ol",
            "items": [_extract_html_list_item(li) for li in elem.find_all("li", recursive=False)],
        }
    
    elif tag == "table":
        return _extract_html_table(elem)
    
    elif tag in ("pre", "code"):
        if tag == "pre" and elem.find("code"):
            code_elem = elem.find("code")
            return {
                "type": "code_block",
                "language": _detect_code_language(code_elem),
                "text": code_elem.get_text(),
            }
        return {
            "type": "code_block",
            "language": None,
            "text": elem.get_text(),
        }
    
    elif tag == "blockquote":
        return {
            "type": "blockquote",
            "text": _clean_html_text(elem),
        }
    
    elif tag in ("div", "section", "article"):
        inner = []
        for child in elem.children:
            if isinstance(child, Tag):
                block = _convert_html_element(child, base_url)
                if block:
                    inner.append(block)
        if inner:
            return {
                "type": "container",
                "tag": tag,
                "id": elem.get("id"),
                "blocks": inner,
            }
        return None
    
    elif tag == "img":
        src = elem.get("src", "")
        if base_url and src.startswith(("http://", "https://")) is False:
            from urllib.parse import urljoin
            src = urljoin(base_url, src)
        return {
            "type": "image",
            "src": src,
            "alt": elem.get("alt", ""),
        }
    
    elif tag == "hr":
        return {"type": "horizontal_rule"}
    
    elif tag == "br":
        return None  # Skip standalone line breaks
    
    return None


def _clean_html_text(elem: Tag) -> str:
    """Extract clean text from an element, preserving inline formatting hints."""
    # Replace <br> and <br/> with newlines
    for br in elem.find_all("br"):
        br.replace_with("\n")
    
    text = elem.get_text(separator=" ", strip=True)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_html_list_item(li: Tag) -> dict:
    text = li.get_text(strip=True)
    nested = li.find(["ul", "ol"], recursive=False)
    
    item = {"text": text}
    if nested:
        # Remove nested text from parent text
        nested_text = nested.get_text(strip=True)
        text = text.replace(nested_text, "").strip()
        item["text"] = text
        item["nested"] = {
            "ordered": nested.name == "ol",
            "items": [_extract_html_list_item(nli) for nli in nested.find_all("li", recursive=False)],
        }
    return item


def _extract_html_table(table: Tag) -> dict:
    rows = []
    
    # Check for thead/tbody structure
    for row_elem in table.find_all("tr"):
        row = []
        for cell in row_elem.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            row.append({
                "text": cell.get_text(strip=True),
                "is_header": cell.name == "th",
                "colspan": colspan,
                "rowspan": rowspan,
            })
        if row:
            rows.append(row)
    
    return {
        "type": "table",
        "rows": rows,
    }


def _detect_code_language(code_elem: Tag) -> str | None:
    """Detect programming language from class attributes."""
    classes = code_elem.get("class", [])
    for cls in classes:
        if cls.startswith("language-") or cls.startswith("lang-"):
            return cls.split("-", 1)[1]
        if cls in ("python", "javascript", "java", "cpp", "c", "ruby", "go", "rust", "bash", "sql"):
            return cls
    return None
```

---

## 5. Phase 3: Structure Understanding

### 5.1 Heading Hierarchy Reconstruction

**Problem**: Documents often have inconsistent heading levels (h1 followed by h3), styled paragraphs masquerading as headings, or no headings at all.

**Algorithm**:
```python
def reconstruct_heading_hierarchy(blocks: list[dict]) -> list[dict]:
    """
    Normalize heading levels and insert inferred headings.
    Returns blocks with normalized heading levels.
    """
    
    # Step 1: Collect all headings
    headings = [(i, b) for i, b in enumerate(blocks) if b["type"] == "heading"]
    
    if not headings:
        # No headings at all: try to infer from paragraph patterns
        return _infer_headings_from_paragraphs(blocks)
    
    # Step 2: Determine actual heading level distribution
    levels = [b["level"] for _, b in headings]
    min_level = min(levels)
    max_level = max(levels)
    
    # Step 3: Handle skipped levels
    # If we see levels [1, 3, 3, 5], normalize to [1, 2, 2, 3]
    unique_levels = sorted(set(levels))
    level_map = {old: new for new, old in enumerate(unique_levels, start=min_level)}
    
    for idx, block in headings:
        blocks[idx]["level"] = level_map[block["level"]]
    
    # Step 4: Ensure h1 exists (document title)
    first_heading_level = headings[0][1]["level"]
    if first_heading_level > 1 and not any(b["level"] == 1 for _, b in headings):
        # Insert inferred h1 at beginning
        title_guess = _guess_document_title(blocks)
        blocks.insert(0, {
            "type": "heading",
            "level": 1,
            "text": title_guess,
            "inferred": True,
        })
    
    # Step 5: Detect paragraph-as-heading (short, bold, all-caps paragraphs)
    for i, block in enumerate(blocks):
        if block["type"] == "paragraph":
            text = block["text"].strip()
            if len(text) < 100 and (text.isupper() or block.get("style", "").lower() in ("title", "subtitle", "heading")):
                # Promote to heading
                # Determine level by font size if available, else level 2
                level = 2
                if "font_size" in block:
                    level = _font_size_to_level(block["font_size"])
                blocks[i] = {
                    "type": "heading",
                    "level": level,
                    "text": text.title() if text.isupper() else text,
                    "inferred": True,
                }
    
    return blocks


def _infer_headings_from_paragraphs(blocks: list[dict]) -> list[dict]:
    """When no headings exist, try to detect structure from paragraph patterns."""
    result = []
    
    for i, block in enumerate(blocks):
        if block["type"] != "paragraph":
            result.append(block)
            continue
        
        text = block["text"].strip()
        
        # Heuristic: very short all-caps line after blank space
        if text.isupper() and 10 < len(text) < 80:
            # Determine level by position (earlier = higher level)
            level = min(2, 1 + i // 10)  # First sections = h1/h2
            result.append({
                "type": "heading",
                "level": level,
                "text": text.title(),
                "inferred": True,
            })
        else:
            result.append(block)
    
    return result


def _guess_document_title(blocks: list[dict]) -> str:
    """Guess document title from first substantial text."""
    for block in blocks:
        if block["type"] in ("heading", "paragraph"):
            text = block.get("text", "").strip()
            if len(text) > 3:
                return text[:100]
    return "Untitled Document"
```

### 5.2 Section Boundary Detection

```python
def detect_section_boundaries(blocks: list[dict]) -> list[dict]:
    """
    Group blocks into sections based on heading hierarchy.
    Returns list of sections with nested structure.
    """
    sections = []
    current_section = None
    section_stack = []
    
    for block in blocks:
        if block["type"] == "heading":
            level = block["level"]
            new_section = {
                "heading": block,
                "level": level,
                "blocks": [],
                "subsections": [],
            }
            
            # Pop stack until we find parent
            while section_stack and section_stack[-1]["level"] >= level:
                section_stack.pop()
            
            if section_stack:
                section_stack[-1]["subsections"].append(new_section)
            else:
                sections.append(new_section)
            
            section_stack.append(new_section)
            current_section = new_section
        else:
            if current_section is None:
                # Content before first heading: create anonymous section
                anon_section = {
                    "heading": None,
                    "level": 0,
                    "blocks": [block],
                    "subsections": [],
                }
                sections.append(anon_section)
                current_section = anon_section
                section_stack = [current_section]
            else:
                current_section["blocks"].append(block)
    
    return sections
```

### 5.3 List Nesting Reconstruction

```python
def reconstruct_list_nesting(flat_items: list[dict]) -> list[dict]:
    """
    Reconstruct nested list structure from flat items with indentation levels.
    """
    root = []
    stack = [root]  # Stack of lists to append to
    
    for item in flat_items:
        level = item.get("level", 0)
        
        # Adjust stack depth
        while len(stack) > level + 1:
            stack.pop()
        
        if len(stack) <= level:
            # Create intermediate nesting levels
            while len(stack) <= level:
                new_list = []
                if stack:
                    parent = stack[-1][-1] if stack[-1] else None
                    if parent and "nested" not in parent:
                        parent["nested"] = {"ordered": False, "items": new_list}
                    stack.append(new_list)
        
        stack[level].append(item)
    
    return root
```

### 5.4 Table Structure Normalization

```python
def normalize_table(table: dict) -> dict:
    """
    Normalize table to rectangular structure, handling merged cells.
    """
    rows = table.get("rows", [])
    if not rows:
        return table
    
    # Find maximum column count
    max_cols = max(
        sum(cell.get("colspan", 1) for cell in row)
        for row in rows
    ) if rows else 0
    
    # Build normalized grid
    grid = []
    span_map = {}  # Tracks (row, col) -> source cell for rowspan
    
    for r_idx, row in enumerate(rows):
        grid_row = []
        c_idx = 0
        
        for cell in row:
            # Skip columns occupied by rowspan from above
            while (r_idx, c_idx) in span_map:
                grid_row.append(span_map[(r_idx, c_idx)])
                c_idx += 1
            
            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)
            
            # Fill the cell and its spans
            for cs in range(colspan):
                for rs in range(rowspan):
                    if rs == 0 and cs == 0:
                        grid_row.append(cell)
                    else:
                        span_key = (r_idx + rs, c_idx + cs)
                        span_map[span_key] = {
                            "text": cell["text"],
                            "is_header": cell.get("is_header", False),
                            "spanned": True,
                            "source": (r_idx, c_idx),
                        }
                c_idx += 1
        
        # Pad to max_cols
        while len(grid_row) < max_cols:
            grid_row.append({"text": "", "is_header": False, "padding": True})
        
        grid.append(grid_row[:max_cols])
    
    return {
        "type": "table",
        "rows": grid,
        "row_count": len(grid),
        "column_count": max_cols,
        "has_merged_cells": bool(span_map),
    }
```

### 5.5 Paragraph Grouping

```python
def group_paragraphs(blocks: list[dict]) -> list[dict]:
    """
    Group consecutive paragraphs and related blocks into logical units.
    """
    groups = []
    current_group = []
    
    for block in blocks:
        if block["type"] == "paragraph":
            current_group.append(block)
        elif block["type"] in ("list", "table", "code_block"):
            # Flush current paragraph group
            if current_group:
                if len(current_group) == 1:
                    groups.append(current_group[0])
                else:
                    groups.append({
                        "type": "paragraph_group",
                        "blocks": current_group,
                        "text": "\n\n".join(b["text"] for b in current_group),
                    })
                current_group = []
            groups.append(block)
        else:
            # Flush and pass through
            if current_group:
                if len(current_group) == 1:
                    groups.append(current_group[0])
                else:
                    groups.append({
                        "type": "paragraph_group",
                        "blocks": current_group,
                        "text": "\n\n".join(b["text"] for b in current_group),
                    })
                current_group = []
            groups.append(block)
    
    # Flush remaining
    if current_group:
        if len(current_group) == 1:
            groups.append(current_group[0])
        else:
            groups.append({
                "type": "paragraph_group",
                "blocks": current_group,
                "text": "\n\n".join(b["text"] for b in current_group),
            })
    
    return groups
```

---

## 6. Phase 4: Normalization

### 6.1 Unicode Normalization (NFKC)

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    """
    Apply NFKC normalization to handle:
    - Compatibility characters (fullwidth vs halfwidth)
    - Composed vs decomposed forms
    - Special spaces (non-breaking, em-space, etc.)
    """
    # NFKC: Compatibility decomposition followed by canonical composition
    return unicodedata.normalize("NFKC", text)
```

### 6.2 Whitespace Normalization

```python
import re

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving intentional structure:
    - Collapse multiple spaces/tabs to single space
    - Normalize line endings to \n
    - Preserve paragraph breaks (double newlines)
    - Trim leading/trailing whitespace per line
    - Remove zero-width characters
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove zero-width characters
    zws_chars = "\u200b\u200c\u200d\ufeff\u2060\u00ad"
    for char in zws_chars:
        text = text.replace(char, "")
    
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    
    # Collapse more than 2 consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Trim lines
    lines = [line.strip() for line in text.split("\n")]
    
    return "\n".join(lines)
```

### 6.3 Encoding Standardization

```python
def standardize_encoding(text: str, target_encoding: str = "utf-8") -> str:
    """
    Ensure text is valid UTF-8. All input should already be decoded,
    but this handles any remaining encoding issues.
    """
    # Re-encode and decode to catch any surrogates
    try:
        encoded = text.encode(target_encoding, errors="strict")
        return encoded.decode(target_encoding)
    except UnicodeEncodeError:
        # Fall back to replacing problematic characters
        encoded = text.encode(target_encoding, errors="replace")
        return encoded.decode(target_encoding)
```

### 6.4 Date/Time Format Standardization

```python
from datetime import datetime
import re

def standardize_dates(text: str) -> str:
    """
    Detect and standardize date formats to ISO 8601.
    Uses pattern matching for common formats.
    """
    # Common patterns
    patterns = [
        # MM/DD/YYYY or MM-DD-YYYY
        (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", "mdy"),
        # DD/MM/YYYY or DD-MM-YYYY
        (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", "dmy"),
        # YYYY/MM/DD or YYYY-MM-DD
        (r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", "ymd"),
        # Month DD, YYYY
        (r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", "text"),
    ]
    
    # Simplified: in practice, use dateparser or similar library
    # This is a placeholder for the normalization approach
    return text


def normalize_document_dates(doc: dict) -> dict:
    """Normalize all date fields in document metadata."""
    date_fields = ["created_date", "modified_date", "date"]
    
    for field in date_fields:
        if field in doc.get("content", {}) and doc["content"][field]:
            date_str = doc["content"][field]
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                doc["content"][field] = dt.isoformat()
            except ValueError:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y"]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        doc["content"][field] = dt.date().isoformat()
                        break
                    except ValueError:
                        continue
    
    return doc
```

### 6.5 Reference/Link Canonicalization

```python
from urllib.parse import urljoin, urlparse, urlunparse

def canonicalize_links(blocks: list[dict], base_url: str | None = None) -> list[dict]:
    """
    Normalize all URLs in the document:
    - Convert relative URLs to absolute (if base_url known)
    - Remove tracking parameters
    - Normalize protocol (https preferred)
    """
    tracking_params = ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                       "fbclid", "gclid", "ref", "source"]
    
    def clean_url(url: str) -> str:
        if not url:
            return url
        
        # Make absolute if relative
        if base_url and not bool(urlparse(url).netloc):
            url = urljoin(base_url, url)
        
        parsed = urlparse(url)
        
        # Remove tracking parameters
        if parsed.query:
            from urllib.parse import parse_qs, urlencode
            query = parse_qs(parsed.query)
            for param in tracking_params:
                query.pop(param, None)
            new_query = urlencode(query, doseq=True)
            parsed = parsed._replace(query=new_query)
        
        # Prefer https
        if parsed.scheme == "http":
            parsed = parsed._replace(scheme="https")
        
        return urlunparse(parsed)
    
    def process_block(block: dict):
        if "links" in block:
            for link in block["links"]:
                if "url" in link:
                    link["url"] = clean_url(link["url"])
        if "src" in block:
            block["src"] = clean_url(block["src"])
        if "blocks" in block:
            for child in block["blocks"]:
                process_block(child)
        if "items" in block:
            for item in block["items"]:
                if isinstance(item, dict):
                    process_block(item)
        if "subsections" in block:
            for sub in block["subsections"]:
                process_block(sub)
    
    for block in blocks:
        process_block(block)
    
    return blocks
```

### 6.6 Full Normalization Pipeline

```python
def normalize_document(ir: dict) -> dict:
    """Apply all normalization steps to the intermediate representation."""
    
    # 1. Unicode normalization on all text fields
    ir = _deep_normalize_unicode(ir)
    
    # 2. Whitespace normalization
    ir = _deep_normalize_whitespace(ir)
    
    # 3. Encoding standardization
    ir = _deep_standardize_encoding(ir)
    
    # 4. Date normalization
    ir = normalize_document_dates(ir)
    
    # 5. Link canonicalization
    if ir.get("structure", {}).get("sections"):
        for section in ir["structure"]["sections"]:
            if "blocks" in section:
                canonicalize_links(section["blocks"], ir.get("meta", {}).get("source_url"))
            if "subsections" in section:
                for sub in section["subsections"]:
                    if "blocks" in sub:
                        canonicalize_links(sub["blocks"], ir.get("meta", {}).get("source_url"))
    
    return ir


def _deep_normalize_unicode(obj):
    if isinstance(obj, str):
        return normalize_unicode(obj)
    elif isinstance(obj, dict):
        return {k: _deep_normalize_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_normalize_unicode(v) for v in obj]
    return obj


def _deep_normalize_whitespace(obj):
    if isinstance(obj, str):
        return normalize_whitespace(obj)
    elif isinstance(obj, dict):
        return {k: _deep_normalize_whitespace(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_normalize_whitespace(v) for v in obj]
    return obj


def _deep_standardize_encoding(obj):
    if isinstance(obj, str):
        return standardize_encoding(obj)
    elif isinstance(obj, dict):
        return {k: _deep_standardize_encoding(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_standardize_encoding(v) for v in obj]
    return obj
```

---

## 7. Phase 5: JSON Generation

### 7.1 Output Schema

The ML-ready JSON follows this schema:

```json
{
  "$schema": "doc2ml-json/1.0",
  "document": {
    "id": "uuid-generated",
    "source": {
      "filename": "example.pdf",
      "format": "application/pdf",
      "url": null
    },
    "metadata": {
      "title": "Document Title",
      "authors": ["Author Name"],
      "language": "en",
      "created": "2024-01-15T10:30:00Z",
      "modified": "2024-06-20T14:22:00Z",
      "page_count": 42,
      "word_count": 15000,
      "character_count": 95000
    },
    "structure": {
      "toc": [
        {"level": 1, "title": "Introduction", "section_id": "sec-001"},
        {"level": 2, "title": "Background", "section_id": "sec-002"}
      ],
      "headings": [
        {"level": 1, "text": "Introduction", "section_id": "sec-001", "position": 0}
      ],
      "sections": [
        {
          "id": "sec-001",
          "level": 1,
          "heading": "Introduction",
          "content_blocks": ["blk-001", "blk-002"],
          "subsections": ["sec-002"]
        }
      ]
    },
    "content": {
      "blocks": [
        {
          "id": "blk-001",
          "type": "paragraph",
          "text": "This is the first paragraph...",
          "section_id": "sec-001",
          "position": 0,
          "word_count": 42
        },
        {
          "id": "blk-002",
          "type": "table",
          "headers": ["Column 1", "Column 2"],
          "rows": [["A", "B"], ["C", "D"]],
          "section_id": "sec-001",
          "position": 1
        }
      ]
    },
    "chunks": [
      {
        "id": "chunk-001",
        "text": "Introduction\n\nThis is the first paragraph...",
        "token_count": 512,
        "block_ids": ["blk-001", "blk-002"],
        "section_ids": ["sec-001"],
        "metadata": {
          "heading_path": ["Introduction"],
          "position": 0
        }
      }
    ],
    "entities": {
      "links": [...],
      "footnotes": [...],
      "references": [...]
    },
    "quality": {
      "extraction_confidence": 0.95,
      "completeness_score": 0.98,
      "structure_score": 0.92
    }
  }
}
```

### 7.2 Chunking Strategy

```python
import re

def chunk_document(ir: dict, max_tokens: int = 512, overlap_tokens: int = 50) -> list[dict]:
    """
    Split document into ML-ready chunks while preserving context.
    
    Strategy:
    1. Respect section boundaries (never split across sections)
    2. Respect paragraph boundaries (never split mid-paragraph)
    3. Use sliding window with overlap for context
    4. Include heading path in each chunk for context
    """
    
    chunks = []
    chunk_id = 0
    
    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~0.75 tokens per word for English."""
        words = len(text.split())
        return int(words * 0.75)
    
    def create_chunk(blocks, section_path, start_pos):
        nonlocal chunk_id
        text_parts = []
        block_ids = []
        section_ids = set()
        
        for block in blocks:
            block_ids.append(block.get("id"))
            section_ids.add(block.get("section_id"))
            
            if block["type"] == "heading":
                text_parts.append("#" * block["level"] + " " + block["text"])
            elif block["type"] == "paragraph":
                text_parts.append(block["text"])
            elif block["type"] == "list":
                for item in block.get("items", []):
                    text_parts.append("- " + item.get("text", ""))
            elif block["type"] == "table":
                # Summarize table in text
                headers = block.get("headers", [])
                rows = block.get("rows", [])
                if headers:
                    text_parts.append(" | ".join(headers))
                for row in rows[:3]:  # Limit table rows in chunk
                    text_parts.append(" | ".join(str(c) for c in row))
            elif block["type"] == "code_block":
                text_parts.append("```\n" + block.get("text", "")[:500] + "\n```")
        
        chunk_text = "\n\n".join(text_parts)
        
        chunk_id += 1
        return {
            "id": f"chunk-{chunk_id:04d}",
            "text": chunk_text,
            "token_count": estimate_tokens(chunk_text),
            "block_ids": block_ids,
            "section_ids": list(section_ids),
            "metadata": {
                "heading_path": section_path,
                "position": start_pos,
            }
        }
    
    # Flatten sections into sequential blocks with context
    all_blocks = []
    
    def collect_blocks(sections, heading_path):
        for section in sections:
            current_path = heading_path + [section.get("heading", "")]
            section_id = section.get("id", "")
            
            for block in section.get("blocks", []):
                block_copy = dict(block)
                block_copy["section_id"] = section_id
                block_copy["heading_path"] = current_path
                all_blocks.append(block_copy)
            
            if section.get("subsections"):
                collect_blocks(section["subsections"], current_path)
    
    collect_blocks(ir.get("structure", {}).get("sections", []), [])
    
    # Chunk the blocks
    current_chunk_blocks = []
    current_token_count = 0
    position = 0
    
    for block in all_blocks:
        block_text = _block_to_text(block)
        block_tokens = estimate_tokens(block_text)
        
        if current_token_count + block_tokens > max_tokens and current_chunk_blocks:
            # Flush current chunk
            chunk = create_chunk(current_chunk_blocks, current_chunk_blocks[0].get("heading_path", []), position)
            chunks.append(chunk)
            position += 1
            
            # Start new chunk with overlap
            if overlap_tokens > 0:
                overlap_blocks = []
                overlap_count = 0
                for prev_block in reversed(current_chunk_blocks):
                    prev_text = _block_to_text(prev_block)
                    prev_tokens = estimate_tokens(prev_text)
                    if overlap_count + prev_tokens <= overlap_tokens:
                        overlap_blocks.insert(0, prev_block)
                        overlap_count += prev_tokens
                    else:
                        break
                current_chunk_blocks = overlap_blocks
                current_token_count = overlap_count
            else:
                current_chunk_blocks = []
                current_token_count = 0
        
        current_chunk_blocks.append(block)
        current_token_count += block_tokens
    
    # Flush remaining
    if current_chunk_blocks:
        chunk = create_chunk(current_chunk_blocks, current_chunk_blocks[0].get("heading_path", []), position)
        chunks.append(chunk)
    
    return chunks


def _block_to_text(block: dict) -> str:
    if block["type"] == "paragraph":
        return block.get("text", "")
    elif block["type"] == "heading":
        return block.get("text", "")
    elif block["type"] == "list":
        return "\n".join("- " + item.get("text", "") for item in block.get("items", []))
    elif block["type"] == "table":
        rows = block.get("rows", [])
        return "\n".join(" | ".join(str(c) for c in row) for row in rows)
    return ""
```

### 7.3 Metadata Enrichment

```python
def enrich_metadata(ir: dict) -> dict:
    """Add computed metadata to the document."""
    
    content = ir.get("content", {})
    body = ir.get("body", [])
    
    # Count words
    all_text = " ".join(
        b.get("text", "") for b in body if "text" in b
    )
    words = len(all_text.split())
    chars = len(all_text)
    
    content["word_count"] = words
    content["character_count"] = chars
    
    # Count blocks by type
    block_counts = {}
    for block in body:
        bt = block.get("type", "unknown")
        block_counts[bt] = block_counts.get(bt, 0) + 1
    
    content["block_counts"] = block_counts
    
    # Detect language (if not already set)
    if not content.get("language"):
        content["language"] = _detect_language(all_text[:1000])
    
    # Compute quality scores
    ir["quality"] = {
        "extraction_confidence": ir["meta"].get("detection_confidence", 0.5),
        "completeness_score": _compute_completeness(ir),
        "structure_score": _compute_structure_score(ir),
    }
    
    return ir


def _detect_language(text: str) -> str | None:
    """Simple language detection heuristic."""
    try:
        from langdetect import detect
        return detect(text)
    except ImportError:
        # Fallback heuristics
        # Check for common language markers
        if any(c in text for c in "äöüß"):
            return "de"
        if any(c in text for c in "éèêàù"):
            return "fr"
        if any(c in text for c in "ñáéíóú"):
            return "es"
        return "en"  # Default


def _compute_completeness(ir: dict) -> float:
    """Score how complete the extraction is (0-1)."""
    errors = len(ir.get("errors", []))
    warnings = len(ir.get("warnings", []))
    
    if errors == 0 and warnings == 0:
        return 1.0
    
    # Penalize by error/warning count
    penalty = min(0.5, (errors * 0.2) + (warnings * 0.05))
    return max(0.0, 1.0 - penalty)


def _compute_structure_score(ir: dict) -> float:
    """Score structural quality (0-1)."""
    sections = ir.get("structure", {}).get("sections", [])
    headings = ir.get("structure", {}).get("headings", [])
    
    if not sections:
        return 0.3  # No structure detected
    
    # Check for heading hierarchy consistency
    levels = [h["level"] for h in headings]
    if not levels:
        return 0.5
    
    # Check for skipped levels (h1 -> h3 is a skip)
    skips = 0
    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            skips += 1
    
    skip_rate = skips / len(levels) if levels else 0
    return max(0.0, 1.0 - (skip_rate * 0.5))
```

### 7.4 Cross-Reference Resolution

```python
def resolve_cross_references(ir: dict) -> dict:
    """
    Resolve internal cross-references:
    - Section links (#section-id)
    - Footnote references
    - Table/figure references
    """
    
    # Build ID lookup maps
    section_ids = {}
    block_ids = {}
    
    def index_sections(sections):
        for section in sections:
            sid = section.get("id")
            if sid:
                section_ids[sid] = section
            if section.get("subsections"):
                index_sections(section["subsections"])
    
    index_sections(ir.get("structure", {}).get("sections", []))
    
    for block in ir.get("body", []):
        bid = block.get("id")
        if bid:
            block_ids[bid] = block
    
    # Resolve references in block text
    for block in ir.get("body", []):
        if "references" in block:
            for ref in block["references"]:
                ref_id = ref.get("target_id")
                ref_type = ref.get("type")
                
                if ref_type == "section" and ref_id in section_ids:
                    ref["resolved"] = True
                    ref["target_heading"] = section_ids[ref_id].get("heading", "")
                elif ref_type == "block" and ref_id in block_ids:
                    ref["resolved"] = True
                    ref["target_type"] = block_ids[ref_id].get("type", "")
                else:
                    ref["resolved"] = False
    
    return ir
```

---

## 8. Phase 6: Validation & Output

### 8.1 Schema Compliance Check

```python
import jsonschema
from jsonschema import validate, ValidationError

SCHEMA = {
    "type": "object",
    "required": ["document"],
    "properties": {
        "document": {
            "type": "object",
            "required": ["id", "source", "metadata", "content"],
            "properties": {
                "id": {"type": "string"},
                "source": {
                    "type": "object",
                    "required": ["filename", "format"],
                    "properties": {
                        "filename": {"type": "string"},
                        "format": {"type": "string"},
                        "url": {"type": ["string", "null"]},
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "title": {"type": ["string", "null"]},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "language": {"type": ["string", "null"]},
                    }
                },
                "content": {
                    "type": "object",
                    "required": ["blocks"],
                    "properties": {
                        "blocks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "type"],
                            }
                        }
                    }
                },
            }
        }
    }
}


def validate_output(data: dict) -> tuple[bool, list[str]]:
    """
    Validate JSON output against schema.
    Returns (is_valid, list_of_errors).
    """
    try:
        validate(instance=data, schema=SCHEMA)
        return True, []
    except ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Validation exception: {e}"]
```

### 8.2 Roundtrip Sanity Check

```python
def roundtrip_sanity_check(original_ir: dict, generated_json: dict) -> dict:
    """
    Verify that the JSON output can reconstruct a logical document.
    """
    checks = {
        "text_coverage": 0.0,
        "structure_preservation": 0.0,
        "metadata_preservation": 0.0,
        "overall_score": 0.0,
    }
    
    # Check 1: Text coverage
    original_text = _extract_all_text(original_ir)
    json_text = _extract_all_text_from_json(generated_json)
    
    if original_text:
        # Compute similarity (simplified: length ratio + common substring)
        len_orig = len(original_text)
        len_json = len(json_text)
        coverage = min(len_json / len_orig, 1.0) if len_orig > 0 else 0.0
        checks["text_coverage"] = coverage
    
    # Check 2: Structure preservation
    orig_headings = len(original_ir.get("structure", {}).get("headings", []))
    json_headings = len(generated_json.get("document", {}).get("structure", {}).get("headings", []))
    
    if orig_headings > 0:
        checks["structure_preservation"] = min(json_headings / orig_headings, 1.0)
    elif json_headings == 0:
        checks["structure_preservation"] = 1.0
    
    # Check 3: Metadata preservation
    orig_meta = original_ir.get("content", {})
    json_meta = generated_json.get("document", {}).get("metadata", {})
    
    meta_fields = ["title", "author", "language"]
    preserved = sum(1 for f in meta_fields if json_meta.get(f) or json_meta.get(f) == orig_meta.get(f))
    checks["metadata_preservation"] = preserved / len(meta_fields)
    
    # Overall score
    checks["overall_score"] = (
        checks["text_coverage"] * 0.5 +
        checks["structure_preservation"] * 0.3 +
        checks["metadata_preservation"] * 0.2
    )
    
    return checks


def _extract_all_text(ir: dict) -> str:
    """Extract all text from IR for comparison."""
    texts = []
    for block in ir.get("body", []):
        if "text" in block:
            texts.append(block["text"])
    return " ".join(texts)


def _extract_all_text_from_json(data: dict) -> str:
    """Extract all text from JSON output."""
    texts = []
    for block in data.get("document", {}).get("content", {}).get("blocks", []):
        if "text" in block:
            texts.append(block["text"])
    for chunk in data.get("document", {}).get("chunks", []):
        if "text" in chunk:
            texts.append(chunk["text"])
    return " ".join(texts)
```

### 8.3 ML Readiness Check

```python
def check_ml_readiness(data: dict) -> dict:
    """
    Check if the output is suitable for ML consumption.
    """
    checks = {
        "chunk_size_ok": True,
        "token_counts_reasonable": True,
        "no_empty_chunks": True,
        "sufficient_context": True,
        "overall_ready": True,
        "issues": [],
    }
    
    chunks = data.get("document", {}).get("chunks", [])
    
    if not chunks:
        checks["issues"].append("No chunks generated")
        checks["overall_ready"] = False
        return checks
    
    for chunk in chunks:
        token_count = chunk.get("token_count", 0)
        chunk_text = chunk.get("text", "")
        
        # Check chunk size
        if token_count > 2048:
            checks["chunk_size_ok"] = False
            checks["issues"].append(f"Chunk {chunk['id']} exceeds 2048 tokens")
        
        if token_count == 0 or not chunk_text.strip():
            checks["no_empty_chunks"] = False
            checks["issues"].append(f"Chunk {chunk['id']} is empty")
        
        # Check for context (heading path)
        if not chunk.get("metadata", {}).get("heading_path"):
            checks["sufficient_context"] = False
    
    # Check if any chunk has too few tokens (might be noise)
    tiny_chunks = [c for c in chunks if c.get("token_count", 0) < 10]
    if len(tiny_chunks) > len(chunks) * 0.1:  # More than 10% are tiny
        checks["issues"].append(f"{len(tiny_chunks)} chunks are very small (<10 tokens)")
    
    checks["overall_ready"] = all([
        checks["chunk_size_ok"],
        checks["no_empty_chunks"],
    ])
    
    return checks
```

---

## 9. Error Handling & Recovery

### 9.1 Error Taxonomy

| Error Code    | Description                  | Severity | Recovery Strategy                           |
| ------------- | ---------------------------- | -------- | ------------------------------------------- |
| `DETECT_001`  | Extension/signature mismatch | Warning  | Use signature result, flag for review       |
| `DETECT_002`  | Unknown file format          | Error    | Try generic text extraction, fail if binary |
| `DETECT_003`  | Corrupted ZIP (DOCX/EPUB)    | Error    | Try to repair ZIP, extract readable parts   |
| `EXTRACT_001` | Password-protected PDF       | Error    | Report failure, request password            |
| `EXTRACT_002` | Scanned PDF without OCR text | Warning  | Trigger OCR pipeline                        |
| `EXTRACT_003` | Missing fonts in PDF         | Warning  | Use raw text extraction                     |
| `EXTRACT_004` | Broken XML in DOCX           | Error    | Parse with lenient XML parser               |
| `EXTRACT_005` | Encoding detection failure   | Warning  | Use UTF-8 with replacement                  |
| `STRUCT_001`  | No headings detected         | Warning  | Infer from paragraph patterns               |
| `STRUCT_002`  | Broken heading hierarchy     | Warning  | Normalize levels, insert inferred           |
| `STRUCT_003`  | Table extraction failed      | Warning  | Fallback to raw text                        |
| `NORM_001`    | Invalid date format          | Warning  | Keep original, flag unparseable             |
| `NORM_002`    | URL parse error              | Warning  | Keep original URL                           |

### 9.2 Recovery Strategies

```python
class ExtractionError(Exception):
    def __init__(self, code: str, message: str, severity: str, recoverable: bool):
        self.code = code
        self.message = message
        self.severity = severity
        self.recoverable = recoverable
        super().__init__(f"[{code}] {message}")


def handle_extraction_error(error: ExtractionError, ir: dict) -> dict:
    """Handle extraction error and attempt recovery."""
    
    error_record = {
        "code": error.code,
        "message": error.message,
        "severity": error.severity,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    ir.setdefault("errors", []).append(error_record)
    
    if not error.recoverable:
        ir["meta"]["extraction_status"] = "failed"
        return ir
    
    # Recovery strategies per error code
    recovery_strategies = {
        "DETECT_001": lambda ir: ir,  # Already handled in detection
        "DETECT_003": lambda ir: _recover_corrupted_zip(ir),
        "EXTRACT_001": lambda ir: _handle_password_protected(ir),
        "EXTRACT_002": lambda ir: _trigger_ocr_fallback(ir),
        "EXTRACT_004": lambda ir: _use_lenient_xml_parser(ir),
        "STRUCT_001": lambda ir: _infer_structure_from_text(ir),
    }
    
    strategy = recovery_strategies.get(error.code)
    if strategy:
        ir = strategy(ir)
        ir["meta"]["extraction_status"] = "partial"
    
    return ir


def _recover_corrupted_zip(ir: dict) -> dict:
    """Attempt to recover data from a corrupted ZIP-based file."""
    filepath = ir["meta"]["source_file"]
    recovered = []
    
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            for name in z.namelist():
                try:
                    content = z.read(name)
                    recovered.append({"name": name, "size": len(content)})
                except Exception:
                    pass
    except zipfile.BadZipFile:
        # Try to read as raw binary and extract text
        pass
    
    ir["warnings"].append({
        "code": "RECOVER_ZIP",
        "message": f"Recovered {len(recovered)} files from corrupted ZIP",
    })
    
    return ir


def _trigger_ocr_fallback(ir: dict) -> dict:
    """Trigger OCR pipeline for image-based PDFs."""
    ir["meta"]["ocr_triggered"] = True
    ir["warnings"].append({
        "code": "OCR_FALLBACK",
        "message": "Document appears image-based; OCR recommended",
    })
    return ir


def _infer_structure_from_text(ir: dict) -> dict:
    """Infer document structure when no explicit structure exists."""
    body = ir.get("body", [])
    if body:
        # Create a single anonymous section
        ir["structure"] = {
            "sections": [{
                "id": "sec-001",
                "level": 0,
                "heading": None,
                "blocks": body,
                "subsections": [],
            }],
            "headings": [],
        }
    return ir
```

---

## 10. Edge Case Guide

### 10.1 Scanned PDFs (Image-Only)

**Detection:**
- PDF with pages but `page.extract_text()` returns empty or minimal text
- High ratio of image objects to text objects
- Check using PyMuPDF: `page.get_text()` is empty but `page.get_images()` has entries

**Strategy:**
```python
def is_scanned_pdf(doc) -> bool:
    scanned_pages = 0
    for page in doc:
        text = page.get_text().strip()
        images = page.get_images()
        if len(text) < 50 and len(images) > 0:
            scanned_pages += 1
    return scanned_pages > len(doc) * 0.5


def handle_scanned_pdf(filepath: str) -> dict:
    """
    OCR strategy for scanned PDFs.
    
    Options (in order of preference):
    1. Use pytesseract with pdf2image (convert pages to images, OCR each)
    2. Use OCRmyPDF (wraps tesseract, produces searchable PDF)
    3. Use Tika (if available) with OCR enabled
    """
    from pdf2image import convert_from_path
    import pytesseract
    
    pages = convert_from_path(filepath, dpi=300)
    results = []
    
    for i, image in enumerate(pages, start=1):
        text = pytesseract.image_to_string(image, lang="eng")
        results.append({
            "page_number": i,
            "text": text,
            "source": "ocr",
        })
    
    return {"pages": results, "source": "ocr", "note": "Text extracted via OCR"}
```

### 10.2 Password-Protected Files

```python
def handle_password_protected_pdf(filepath: str, password: str | None = None) -> dict:
    import fitz
    
    doc = fitz.open(filepath)
    
    if doc.is_encrypted:
        if password:
            auth_result = doc.authenticate(password)
            if not auth_result:
                raise ExtractionError("EXTRACT_001", "Invalid password", "error", False)
        else:
            # Try common empty passwords
            for pwd in ["", "password", "123456"]:
                if doc.authenticate(pwd):
                    password = pwd
                    break
            if not password:
                raise ExtractionError("EXTRACT_001", "Password required", "error", False)
    
    # Continue with extraction...
    return extract_pdf_pymupdf(filepath)
```

### 10.3 Corrupted EPUBs

```python
def handle_corrupted_epub(filepath: str) -> dict:
    """
    Recovery strategy for corrupted EPUB files.
    1. Try standard ZIP read
    2. Try to repair with zipfile
    3. Extract raw HTML files directly
    """
    import zipfile
    
    # Attempt 1: Try with relaxed ZIP
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            return extract_epub(filepath)
    except zipfile.BadZipFile:
        pass
    
    # Attempt 2: Try to read individual files
    recovered_files = []
    try:
        with open(filepath, "rb") as f:
            data = f.read()
            # Search for local file headers (PK\x03\x04)
            pos = 0
            while True:
                pos = data.find(b"PK\x03\x04", pos)
                if pos == -1:
                    break
                # Extract filename and content
                # Simplified: in practice, parse ZIP local file headers
                pos += 1
    except Exception:
        pass
    
    # Attempt 3: Treat as raw HTML collection
    return {
        "metadata": {},
        "chapters": [],
        "errors": [{"code": "CORRUPTED_EPUB", "message": "EPUB corrupted, minimal recovery attempted"}],
    }
```

### 10.4 Tables-as-Images

**Detection:** PDF page contains a large image in the position where a table is expected, but no table is extracted.

**Strategy:**
1. Extract the image region
2. Run OCR on the image
3. Heuristic table reconstruction from OCR text

### 10.5 Merged Cells in Tables

Already handled in `normalize_table()` (Section 5.4). Key approach:
- Track `colspan` and `rowspan` attributes
- Fill spanned cells with references to source cell
- Mark as `spanned: True` to indicate duplication

### 10.6 Mixed-Layout PDFs (Text + Images + Tables)

**Strategy:** Process each page in layers:
1. Extract text blocks with positions
2. Extract images with positions
3. Extract tables with positions
4. Determine reading order by y-position (top-to-bottom), then x-position (left-to-right)
5. Interleave content blocks in reading order

### 10.7 Multi-Column Documents

**Detection:** Text blocks have x-positions clustered into 2+ groups.

**Strategy:**
```python
def detect_columns(page_width: float, text_blocks: list[dict]) -> int:
    """Detect number of columns based on x-position clustering."""
    if not text_blocks:
        return 1
    
    x_positions = [b["x0"] for b in text_blocks if "x0" in b]
    if not x_positions:
        return 1
    
    # Cluster x-positions
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for x in x_positions:
        # Round to nearest 50 pixels for clustering
        cluster_key = round(x / 50) * 50
        clusters[cluster_key].append(x)
    
    # Count significant clusters (more than a few blocks)
    significant = sum(1 for c in clusters.values() if len(c) > 3)
    return max(1, significant)


def sort_by_reading_order(blocks: list[dict], num_columns: int = 1) -> list[dict]:
    """Sort blocks by reading order (top-to-bottom, left-to-right)."""
    if num_columns == 1:
        return sorted(blocks, key=lambda b: (b.get("top", 0), b.get("x0", 0)))
    
    # Multi-column: group by row, then sort by column within row
    row_threshold = 20  # pixels
    
    # Sort by y first
    sorted_by_y = sorted(blocks, key=lambda b: b.get("top", 0))
    
    rows = []
    current_row = []
    current_y = None
    
    for block in sorted_by_y:
        y = block.get("top", 0)
        if current_y is None or abs(y - current_y) <= row_threshold:
            current_row.append(block)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            # Sort row by x-position
            rows.append(sorted(current_row, key=lambda b: b.get("x0", 0)))
            current_row = [block]
            current_y = y
    
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b.get("x0", 0)))
    
    # Flatten
    result = []
    for row in rows:
        result.extend(row)
    
    return result
```

---

## 11. Tool Mapping

### 11.1 Which Tool for Which Phase

| Phase                | Task                    | Preferred Tool                           | Fallback Tool                  |
| -------------------- | ----------------------- | ---------------------------------------- | ------------------------------ |
| **Format Detection** | Extension check         | `shell` (ls, file)                       | `read_file` (read first bytes) |
|                      | Magic number inspection | `ipython` (Python `open` in binary mode) | `shell` (`file`, `hexdump`)    |
|                      | Deep ZIP inspection     | `ipython` (`zipfile` module)             | `shell` (`unzip -l`)           |
|                      | Text classification     | `ipython` (Python heuristic)             | `read_file` (read sample)      |
| **Extraction**       | PDF text + layout       | `ipython` (`pdfplumber`, `PyMuPDF`)      | `read_file` (built-in PDF→MD)  |
|                      | PDF OCR fallback        | `ipython` (`pytesseract`, `pdf2image`)   | External OCR service           |
|                      | EPUB parsing            | `ipython` (`ebooklib`, `BeautifulSoup`)  | `read_file` (built-in EPUB→MD) |
|                      | DOCX parsing            | `ipython` (`python-docx`, `lxml`)        | `read_file` (built-in DOCX→MD) |
|                      | Markdown parsing        | `ipython` (custom parser)                | `read_file` (raw text)         |
|                      | HTML parsing            | `ipython` (`BeautifulSoup`)              | `read_file` (raw text)         |
|                      | Web-based docs          | `browser_visit`                          | `shell` (`curl`)               |
| **Structure**        | Heading reconstruction  | `ipython` (Python algorithm)             | N/A                            |
|                      | Section detection       | `ipython` (Python algorithm)             | N/A                            |
|                      | List nesting            | `ipython` (Python algorithm)             | N/A                            |
|                      | Table normalization     | `ipython` (Python algorithm)             | N/A                            |
| **Normalization**    | Unicode/whitespace      | `ipython` (`unicodedata`, `re`)          | N/A                            |
|                      | Date parsing            | `ipython` (`datetime`, `dateparser`)     | N/A                            |
|                      | Link canonicalization   | `ipython` (`urllib.parse`)               | N/A                            |
| **JSON Generation**  | Schema building         | `ipython` (Python dict assembly)         | N/A                            |
|                      | Chunking                | `ipython` (Python algorithm)             | N/A                            |
|                      | Metadata enrichment     | `ipython` (Python computation)           | N/A                            |
| **Validation**       | Schema validation       | `ipython` (`jsonschema`)                 | `ipython` (manual checks)      |
|                      | Roundtrip check         | `ipython` (Python comparison)            | N/A                            |
|                      | Token counting          | `ipython` (heuristic or `tiktoken`)      | N/A                            |
| **Output**           | File writing            | `shell` or `ipython`                     | N/A                            |

### 11.2 Agent Tool Selection Decision Tree

```
Need to read a file?
├── Is it a URL?
│   ├── YES → browser_visit
│   └── NO → read_file (handles PDF/EPUB/DOCX→MD auto-conversion)
│       ├── If read_file output is sufficient → use it
│       └── If need deeper structure → switch to ipython + specific library
├── Is it a binary file needing magic bytes?
│   ├── YES → ipython with Python file I/O
│   └── NO → read_file for text, ipython for structured data
└── Need to run extraction algorithm?
    └── ALWAYS ipython (Python has the libraries)
```

---

## 12. Decision Trees

### 12.1 Top-Level Format Detection Decision Tree

```
INPUT: file path
│
├─ Step 1: Check extension
│   ├─ .pdf → Route to PDF strategy
│   ├─ .epub → Route to EPUB strategy
│   ├─ .docx → Route to DOCX strategy
│   ├─ .doc → Route to DOCX_LEGACY strategy
│   ├─ .txt → Route to TXT strategy
│   ├─ .md, .markdown → Route to MARKDOWN strategy
│   ├─ .html, .htm → Route to HTML strategy
│   ├─ .rtf → Route to RTF strategy
│   ├─ .odt → Route to ODT strategy
│   ├─ .csv → Route to CSV strategy
│   └─ Unknown / no extension → Go to Step 2
│
├─ Step 2: Check magic bytes (first 16 bytes)
│   ├─ %PDF- → PDF strategy (confidence: 1.0)
│   ├─ PK\x03\x04 → ZIP-based → Go to Step 3
│   ├─ \x89PNG / \xff\xd8\xff → IMAGE → Report unsupported (or OCR if PDF)
│   ├─ <?xml / <html → Go to Step 4
│   └─ Unknown → Go to Step 5
│
├─ Step 3: Classify ZIP-based file
│   ├─ Contains word/document.xml → DOCX strategy
│   ├─ Contains mimetype with "epub" → EPUB strategy
│   ├─ Contains xl/workbook.xml → XLSX strategy (route to table strategy)
│   ├─ Contains META-INF/manifest.xml + content.xml → ODT strategy
│   └─ Generic ZIP → Report as unsupported archive
│
├─ Step 4: Classify XML/HTML
│   ├─ Starts with <!DOCTYPE html or <html → HTML strategy
│   ├─ Contains <html> tag somewhere → HTML strategy
│   └─ XML without HTML tags → XML strategy (generic parsing)
│
└─ Step 5: Content-based text detection
    ├─ Valid UTF-8 / ASCII text → TXT strategy (heuristic structure inference)
    └─ Binary / garbled → Report as unsupported, suggest manual inspection
```

### 12.2 Per-Format Extraction Decision Tree

#### PDF Extraction
```
PDF file
│
├─ Is encrypted?
│   ├─ YES → Try empty password → If fails, try common passwords → If fails, ERROR
│   └─ NO → Continue
│
├─ Is image-only (scanned)?
│   ├─ YES → OCR pipeline (Tesseract / OCRmyPDF)
│   │   ├─ OCR success → Text-based post-processing
│   │   └─ OCR failure → Report partial extraction
│   └─ NO → Continue with text-based extraction
│
├─ Primary extraction method
│   ├─ Use pdfplumber → Text + tables + bboxes
│   ├─ Supplement with PyMuPDF → Metadata + font info + heading detection
│   └─ Fallback to PyPDF2 → Raw text only
│
├─ Post-processing
│   ├─ Detect columns → Sort reading order
│   ├─ Detect headings → Font size + bold analysis
│   ├─ Remove headers/footers → Position-based + repetition analysis
│   ├─ Extract tables → pdfplumber table extraction
│   └─ Group into blocks → Reading order + semantic grouping
│
└─ Return structured IR
```

#### EPUB Extraction
```
EPUB file
│
├─ Validate ZIP structure
│   ├─ Missing critical files (mimetype, META-INF) → Corrupted recovery
│   └─ Valid → Continue
│
├─ Parse OPF
│   ├─ Find rootfile in container.xml
│   ├─ Extract metadata (DC elements)
│   ├─ Parse manifest (resource map)
│   └─ Parse spine (reading order)
│
├─ Navigate spine
│   ├─ For each chapter:
│   │   ├─ Parse XHTML with BeautifulSoup
│   │   ├─ Remove script/style/nav tags
│   │   ├─ Extract semantic structure
│   │   ├─ Handle CSS classes for styling hints
│   │   └─ Resolve internal links
│   └─ End for
│
├─ Extract TOC
│   ├─ Try NCX (EPUB2) first
│   └─ Try NavDoc (EPUB3) fallback
│
└─ Return structured IR
```

#### DOCX Extraction
```
DOCX file
│
├─ Validate ZIP
│   ├─ Missing word/document.xml → Try older format / report error
│   └─ Valid → Continue
│
├─ Parse XML components
│   ├─ document.xml → Main content (paragraphs, runs, tables, hyperlinks)
│   ├─ styles.xml → Style definitions for heading detection
│   ├─ numbering.xml → List definitions
│   ├─ relationships.xml → Link resolution
│   ├─ footnotes.xml → Footnote content
│   └─ endnotes.xml → Endnote content
│
├─ Build lookup maps
│   ├─ style_id → {name, type, base_style}
│   ├─ numId → list_type
│   └─ rId → URL (for hyperlinks)
│
├─ Process paragraphs
│   ├─ Detect heading by style name (Heading1, Heading2, ...)
│   ├─ Detect list by numPr
│   ├─ Extract runs with formatting
│   ├─ Extract hyperlinks
│   └─ Handle revisions (accept all changes)
│
├─ Process tables
│   ├─ Extract rows and cells
│   ├─ Handle merged cells (gridSpan)
│   └─ Preserve header row info
│
└─ Return structured IR
```

#### TXT/MD Extraction
```
Text file
│
├─ Detect encoding
│   ├─ UTF-8 BOM → utf-8-sig
│   ├─ UTF-8 valid → utf-8
│   └─ Fallback → latin-1 (with replacement)
│
├─ Check for frontmatter
│   ├─ Starts with --- → Parse YAML → Extract metadata
│   └─ No frontmatter → Body is entire file
│
├─ Determine if Markdown
│   ├─ Extension is .md/.markdown → Parse as Markdown
│   └─ Extension is .txt → Check content for Markdown indicators
│       ├─ Has # headings, ``` code, | tables → Parse as Markdown
│       └─ Plain text → Use heuristic structure inference
│
├─ If Markdown:
│   ├─ Parse ATX headings (# ## ###)
│   ├─ Parse Setext headings (underlined)
│   ├─ Parse fenced/indented code blocks
│   ├─ Parse GFM tables
│   ├─ Parse lists (-, *, 1.)
│   ├─ Parse blockquotes (>)
│   ├─ Parse horizontal rules
│   └─ Everything else → Paragraphs
│
├─ If plain text:
│   ├─ Detect all-caps headings
│   ├─ Detect underlined headings (=/-)
│   ├─ Detect list items
│   └─ Group into paragraphs (blank-line separated)
│
└─ Return structured IR
```

---

## Appendix A: JSON Schema

The complete JSON schema for `doc2ml-json` output:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "doc2ml-json Document",
  "type": "object",
  "required": ["document"],
  "properties": {
    "document": {
      "type": "object",
      "required": ["id", "source", "metadata", "content"],
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique document identifier (UUID)"
        },
        "source": {
          "type": "object",
          "required": ["filename", "format"],
          "properties": {
            "filename": {"type": "string"},
            "format": {"type": "string"},
            "url": {"type": ["string", "null"]}
          }
        },
        "metadata": {
          "type": "object",
          "properties": {
            "title": {"type": ["string", "null"]},
            "authors": {
              "type": "array",
              "items": {"type": "string"}
            },
            "language": {"type": ["string", "null"]},
            "created": {"type": ["string", "null"]},
            "modified": {"type": ["string", "null"]},
            "page_count": {"type": ["integer", "null"]},
            "word_count": {"type": ["integer", "null"]},
            "character_count": {"type": ["integer", "null"]}
          }
        },
        "structure": {
          "type": "object",
          "properties": {
            "toc": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "level": {"type": "integer"},
                  "title": {"type": "string"},
                  "section_id": {"type": "string"}
                }
              }
            },
            "headings": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "level": {"type": "integer"},
                  "text": {"type": "string"},
                  "section_id": {"type": "string"},
                  "position": {"type": "integer"}
                }
              }
            },
            "sections": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": {"type": "string"},
                  "level": {"type": "integer"},
                  "heading": {"type": ["string", "null"]},
                  "content_blocks": {
                    "type": "array",
                    "items": {"type": "string"}
                  },
                  "subsections": {
                    "type": "array",
                    "items": {"type": "string"}
                  }
                }
              }
            }
          }
        },
        "content": {
          "type": "object",
          "required": ["blocks"],
          "properties": {
            "blocks": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                  "id": {"type": "string"},
                  "type": {
                    "type": "string",
                    "enum": ["heading", "paragraph", "list", "table", "code_block", "blockquote", "image", "horizontal_rule", "paragraph_group", "container"]
                  },
                  "text": {"type": "string"},
                  "section_id": {"type": "string"},
                  "position": {"type": "integer"}
                }
              }
            }
          }
        },
        "chunks": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "text": {"type": "string"},
              "token_count": {"type": "integer"},
              "block_ids": {
                "type": "array",
                "items": {"type": "string"}
              },
              "section_ids": {
                "type": "array",
                "items": {"type": "string"}
              },
              "metadata": {
                "type": "object",
                "properties": {
                  "heading_path": {
                    "type": "array",
                    "items": {"type": "string"}
                  },
                  "position": {"type": "integer"}
                }
              }
            }
          }
        },
        "entities": {
          "type": "object",
          "properties": {
            "links": {"type": "array"},
            "footnotes": {"type": "array"},
            "references": {"type": "array"}
          }
        },
        "quality": {
          "type": "object",
          "properties": {
            "extraction_confidence": {"type": "number"},
            "completeness_score": {"type": "number"},
            "structure_score": {"type": "number"}
          }
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "code": {"type": "string"},
              "message": {"type": "string"},
              "severity": {"type": "string"},
              "timestamp": {"type": "string"}
            }
          }
        },
        "warnings": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "code": {"type": "string"},
              "message": {"type": "string"},
              "timestamp": {"type": "string"}
            }
          }
        }
      }
    }
  }
}
```

---

## Appendix B: Quick Reference

### B.1 Python Dependencies

```
# Core extraction
pdfplumber>=0.10.0       # PDF layout + table extraction
PyMuPDF>=1.23.0          # PDF text, metadata, structure (fitz)
PyPDF2>=3.0.0            # PDF fallback
python-docx>=1.1.0       # DOCX parsing
ebooklib>=0.18           # EPUB parsing (alternative to manual ZIP)
beautifulsoup4>=4.12.0   # HTML/XML parsing
lxml>=4.9.0              # Fast XML parsing
markdown>=3.5.0          # Markdown parsing
pyyaml>=6.0              # YAML frontmatter

# OCR (optional, for scanned PDFs)
pytesseract>=0.3.10      # OCR engine wrapper
pdf2image>=1.16.0        # PDF to image conversion
Pillow>=10.0.0           # Image processing

# Validation & quality
jsonschema>=4.19.0       # JSON schema validation
tiktoken>=0.5.0          # Accurate token counting (OpenAI)

# Utilities
chardet>=5.2.0           # Encoding detection
python-dateutil>=2.8.0   # Date parsing
dateparser>=1.2.0        # Flexible date parsing
langdetect>=1.0.9        # Language detection
```

### B.2 Complete Processing Pipeline (Single Function)

```python
from pathlib import Path
import uuid
from datetime import datetime

def doc2ml_json(filepath: str, output_path: str | None = None) -> dict:
    """
    Main entry point: convert any document to ML-ready JSON.
    
    Args:
        filepath: Path to input document
        output_path: Optional path to write JSON output
    
    Returns:
        ML-ready JSON dictionary
    """
    ir = {
        "meta": {
            "source_file": filepath,
            "id": str(uuid.uuid4()),
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "processing_version": "0.5.0",
        },
        "content": {},
        "structure": {},
        "body": [],
        "errors": [],
        "warnings": [],
    }
    
    try:
        # Phase 1: Format Detection
        format_result = detect_format(filepath)
        ir["meta"]["detected_format"] = format_result["mime"]
        ir["meta"]["detection_confidence"] = format_result["confidence"]
        ir["meta"]["extension_mime"] = format_result.get("extension_mime")
        ir["meta"]["signature_mime"] = format_result.get("signature_mime")
        ir["meta"]["deep_mime"] = format_result.get("deep_mime")
        
        # Phase 2: Content Extraction
        strategy = FORMAT_STRATEGY_MAP.get(format_result["mime"])
        
        if strategy == Strategy.PDF:
            extracted = extract_pdf(filepath)
        elif strategy == Strategy.EPUB:
            extracted = extract_epub(filepath)
        elif strategy == Strategy.DOCX:
            extracted = extract_docx(filepath)
        elif strategy in (Strategy.TXT, Strategy.MARKDOWN):
            extracted = extract_text_file(filepath)
        elif strategy == Strategy.HTML:
            extracted = extract_html(filepath)
        else:
            raise ExtractionError("DETECT_002", f"Unsupported format: {format_result['mime']}", "error", False)
        
        # Populate IR from extracted content
        ir["content"] = extracted.get("metadata", {})
        ir["body"] = extracted.get("blocks", extracted.get("paragraphs", []))
        
        # Phase 3: Structure Understanding
        ir["body"] = reconstruct_heading_hierarchy(ir["body"])
        ir["structure"]["sections"] = detect_section_boundaries(ir["body"])
        ir["structure"]["headings"] = [
            {"level": b["level"], "text": b["text"], "position": i}
            for i, b in enumerate(ir["body"]) if b["type"] == "heading"
        ]
        
        # Phase 4: Normalization
        ir = normalize_document(ir)
        
        # Phase 5: JSON Generation
        ir = enrich_metadata(ir)
        chunks = chunk_document(ir)
        
        # Build final output
        output = {
            "document": {
                "id": ir["meta"]["id"],
                "source": {
                    "filename": Path(filepath).name,
                    "format": ir["meta"]["detected_format"],
                    "url": None,
                },
                "metadata": {
                    "title": ir["content"].get("title"),
                    "authors": ir["content"].get("authors", []),
                    "language": ir["content"].get("language"),
                    "created": ir["content"].get("created_date"),
                    "modified": ir["content"].get("modified_date"),
                    "page_count": ir["content"].get("pages"),
                    "word_count": ir["content"].get("word_count"),
                    "character_count": ir["content"].get("character_count"),
                },
                "structure": {
                    "toc": _build_toc(ir["structure"]["sections"]),
                    "headings": ir["structure"]["headings"],
                    "sections": ir["structure"]["sections"],
                },
                "content": {
                    "blocks": _assign_block_ids(ir["body"]),
                },
                "chunks": chunks,
                "entities": {
                    "links": [],
                    "footnotes": ir.get("structure", {}).get("footnotes", []),
                    "references": [],
                },
                "quality": ir.get("quality", {}),
                "errors": ir.get("errors", []),
                "warnings": ir.get("warnings", []),
            }
        }
        
        # Phase 6: Validation
        is_valid, validation_errors = validate_output(output)
        if not is_valid:
            output["document"]["errors"].extend([
                {"code": "VALIDATION", "message": e, "severity": "warning"}
                for e in validation_errors
            ])
        
        ml_ready = check_ml_readiness(output)
        output["document"]["ml_ready"] = ml_ready["overall_ready"]
        if ml_ready["issues"]:
            output["document"]["warnings"].extend([
                {"code": "ML_READINESS", "message": issue}
                for issue in ml_ready["issues"]
            ])
        
        # Write output if path provided
        if output_path:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        
        return output
    
    except ExtractionError as e:
        ir = handle_extraction_error(e, ir)
        return {
            "document": {
                "id": ir["meta"]["id"],
                "source": {"filename": Path(filepath).name, "format": "unknown", "url": None},
                "metadata": {},
                "content": {"blocks": []},
                "chunks": [],
                "errors": ir.get("errors", []),
                "warnings": ir.get("warnings", []),
            }
        }
    except Exception as e:
        ir["errors"].append({
            "code": "UNEXPECTED",
            "message": str(e),
            "severity": "error",
        })
        return {
            "document": {
                "id": ir["meta"]["id"],
                "source": {"filename": Path(filepath).name, "format": "unknown", "url": None},
                "metadata": {},
                "content": {"blocks": []},
                "chunks": [],
                "errors": ir.get("errors", []),
                "warnings": ir.get("warnings", []),
            }
        }


def _build_toc(sections: list[dict], level: int = 1) -> list[dict]:
    toc = []
    for section in sections:
        heading = section.get("heading")
        if heading:
            toc.append({
                "level": heading.get("level", level),
                "title": heading.get("text", ""),
                "section_id": section.get("id", ""),
            })
        if section.get("subsections"):
            toc.extend(_build_toc(section["subsections"], level + 1))
    return toc


def _assign_block_ids(blocks: list[dict]) -> list[dict]:
    for i, block in enumerate(blocks):
        block["id"] = f"blk-{i:04d}"
    return blocks
```

### B.3 Performance Considerations

| Concern                    | Recommendation                                        |
| -------------------------- | ----------------------------------------------------- |
| Large PDFs (>100 pages)    | Stream pages, process in batches                      |
| Large EPUBs (>50 chapters) | Stream chapters, don't load all into memory           |
| Scanned PDFs               | Downsample images before OCR (150-200 DPI sufficient) |
| Memory limits              | Use generators for block/chunk iteration              |
| Speed                      | Cache parsed documents; reuse IR for multiple outputs |
| Token counting             | Use `tiktoken` for accuracy; heuristic for speed      |

### B.4 Security Considerations

| Threat                     | Mitigation                                         |
| -------------------------- | -------------------------------------------------- |
| Zip bomb (nested ZIP)      | Set max file size limit (100MB default)            |
| XML External Entity (XXE)  | Disable external entity resolution in lxml         |
| Malicious PDF (JavaScript) | Don't execute JS; extract text only                |
| Path traversal in ZIP      | Validate extracted paths against allowed directory |
| Excessive memory use       | Set resource limits; stream large files            |

```python
# Safe XML parsing (mitigates XXE)
from lxml import etree

def safe_xml_parse(xml_bytes: bytes):
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
    )
    return etree.fromstring(xml_bytes, parser=parser)
```

---

*End of doc2ml-json Workflow Design Document*

*Pre-Release Version: 0.5.0 | Last Updated: 05/2026*
