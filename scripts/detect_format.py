#!/usr/bin/env python3
"""Document format detection with three-layer cascade: extension, magic bytes, deep inspection."""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


EXTENSION_MAP = {
    ".pdf": "application/pdf",
    ".epub": "application/epub+zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".rtf": "application/rtf",
    ".xml": "application/xml",
    ".json": "application/json",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".odt": "application/vnd.oasis.opendocument.text",
}

SIGNATURES = {
    b"%PDF-": "application/pdf",
    b"PK\x03\x04": "application/zip",
    b"<?xml": "application/xml",
    b"<html": "text/html",
    b"<!DOCT": "text/html",
}


def detect_by_extension(filepath: str) -> str | None:
    """Layer 1: detect format from file extension."""
    ext = Path(filepath).suffix.lower()
    return EXTENSION_MAP.get(ext)


def detect_by_signature(filepath: str) -> str | None:
    """Layer 2: detect format from magic bytes."""
    with open(filepath, "rb") as f:
        header = f.read(16)
    for sig, mime in SIGNATURES.items():
        if header.startswith(sig):
            if mime == "application/zip":
                return _classify_zip(filepath)
            return mime
    # Try UTF-8/ASCII text detection
    if _is_text_file(filepath):
        return _classify_text(filepath)
    return None


def _classify_zip(filepath: str) -> str:
    """Deep inspection for ZIP-based formats."""
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            namelist = z.namelist()
            if "[Content_Types].xml" in namelist:
                if "word/document.xml" in namelist:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if "xl/workbook.xml" in namelist:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if "mimetype" in namelist:
                mime = z.read("mimetype").decode("utf-8", errors="replace").strip()
                if "epub" in mime:
                    return "application/epub+zip"
            if any(".opf" in name for name in namelist):
                return "application/epub+zip"
            if "content.xml" in namelist and "META-INF/manifest.xml" in namelist:
                return "application/vnd.oasis.opendocument.text"
    except zipfile.BadZipFile:
        pass
    return "application/zip"


def _is_text_file(filepath: str) -> bool:
    """Check if file is likely text-based."""
    with open(filepath, "rb") as f:
        sample = f.read(8192)
    if not sample:
        return True
    # Check for null bytes (binary indicator)
    if b"\x00" in sample:
        return False
    # Try UTF-8 decode
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    # Try Latin-1 (never fails)
    try:
        sample.decode("latin-1")
        # If high bytes ratio is low, likely text
        high_bytes = sum(1 for b in sample if b > 127)
        return high_bytes / len(sample) < 0.3
    except Exception:
        return False


def _classify_text(filepath: str) -> str:
    """Classify a text file by content inspection."""
    with open(filepath, "rb") as f:
        sample = f.read(8192)
    text = sample.decode("utf-8", errors="replace")
    lines = text.split("\n")[:50]
    # Markdown indicators
    md_indicators = ("# ", "## ", "### ", "- ", "* ", "| ", "```", "[", "![]")
    if any(line.startswith(md_indicators) for line in lines):
        return "text/markdown"
    # HTML check
    lower_text = text.lower()
    if "<html" in lower_text or "<!doctype html" in lower_text:
        return "text/html"
    # CSV check
    stripped_lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(stripped_lines) > 1:
        first = stripped_lines[0].count(",")
        second = stripped_lines[1].count(",")
        if first > 1 and second == first:
            return "text/csv"
    return "text/plain"


def compute_confidence(extension_mime: str | None, signature_mime: str | None) -> float:
    """Compute detection confidence from two signals."""
    votes = [v for v in [extension_mime, signature_mime] if v is not None]
    if len(votes) >= 2 and len(set(votes)) == 1:
        return 1.0
    if extension_mime == signature_mime and extension_mime is not None:
        return 0.85
    if signature_mime is not None:
        return 0.70
    if extension_mime is not None:
        return 0.50
    return 0.10


def detect_encoding(filepath: str) -> str:
    """Detect file encoding with fallback chain."""
    with open(filepath, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    return "latin-1"


def check_scanned_pdf(filepath: str) -> bool:
    """Check if PDF is image-only (scanned)."""
    try:
        import fitz
    except ImportError:
        return False
    try:
        doc = fitz.open(filepath)
        scanned = 0
        for page in doc:
            text = page.get_text().strip()
            images = page.get_images()
            if len(text) < 50 and len(images) > 0:
                scanned += 1
        doc.close()
        return scanned > len(doc) * 0.5 if len(doc) > 0 else False
    except Exception:
        return False


def detect_format(filepath: str) -> dict:
    """Run full detection pipeline on a file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ext_mime = detect_by_extension(filepath)
    sig_mime = detect_by_signature(filepath)
    final_mime = sig_mime or ext_mime or "application/octet-stream"
    confidence = compute_confidence(ext_mime, sig_mime)
    encoding = detect_encoding(filepath)
    is_scanned = False
    if final_mime == "application/pdf":
        is_scanned = check_scanned_pdf(filepath)
    return {
        "detected_format": final_mime,
        "confidence": confidence,
        "mime_type": final_mime,
        "encoding": encoding,
        "is_scanned": is_scanned,
        "extension_mime": ext_mime,
        "signature_mime": sig_mime,
        "filename": os.path.basename(filepath),
        "file_size_bytes": os.path.getsize(filepath),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect document format")
    parser.add_argument("filepath", help="Path to the document file")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: stdout)")
    args = parser.parse_args()
    try:
        result = detect_format(args.filepath)
        out = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Written to {args.output}")
        else:
            print(out)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
