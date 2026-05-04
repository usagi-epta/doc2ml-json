#!/usr/bin/env python3
"""
Validate a doc2ml-json output file against the canonical schema.

Produces a validation report with pass/fail status, per-field errors,
and ML readiness metrics.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Inline subset of the canonical JSON Schema for validation
SCHEMA = {
    "type": "object",
    "required": ["doc2ml_version", "document_id", "metadata", "structure", "blocks", "cross_references", "ml_index"],
    "properties": {
        "doc2ml_version": {"type": "string", "enum": ["0.6.2"]},
        "document_id": {"type": "string"},
        "metadata": {
            "type": "object",
            "required": ["title", "source", "ingestion", "language", "statistics"],
            "properties": {
                "title": {"type": "string"},
                "source": {
                    "type": "object",
                    "required": ["uri", "mime_type", "filename", "checksum_sha256", "file_size_bytes"],
                    "properties": {
                        "uri": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "filename": {"type": "string"},
                        "checksum_sha256": {"type": "string"},
                        "file_size_bytes": {"type": "integer", "minimum": 0},
                    },
                },
                "ingestion": {
                    "type": "object",
                    "required": ["ingestion_date", "processing_version", "extractor", "extractor_version"],
                    "properties": {
                        "ingestion_date": {"type": "string"},
                        "processing_version": {"type": "string"},
                        "extractor": {"type": "string"},
                        "extractor_version": {"type": "string"},
                        "ingestion_pipeline": {"type": "array", "items": {"type": "string"}},
                        "processing_duration_ms": {"type": "integer", "minimum": 0},
                    },
                },
                "language": {
                    "type": "object",
                    "required": ["detected"],
                    "properties": {
                        "detected": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "declared": {"type": "string"},
                    },
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
                        "total_word_count": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
        "structure": {
            "type": "object",
            "required": ["node_id", "node_type", "level"],
            "properties": {
                "node_id": {"type": "string"},
                "node_type": {"type": "string", "enum": [
                    "document", "part", "chapter", "section", "subsection", "subsubsection",
                    "appendix", "front_matter", "back_matter", "page"
                ]},
                "title": {"type": "string"},
                "level": {"type": "integer", "minimum": 0},
                "chunk_ids": {"type": "array", "items": {"type": "string"}},
                "children": {"type": "array"},
            },
        },
        "blocks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["chunk_id", "type", "content", "text_plain", "char_count", "token_count_est",
                             "embedding_ready", "context_window", "provenance", "language", "semantics", "relations"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "type": {"type": "string", "enum": [
                        "heading", "paragraph", "table", "list", "code_block", "figure_caption",
                        "footnote", "quote", "metadata", "equation", "abstract", "keyword_block",
                        "reference_entry", "page_header", "page_footer", "image_placeholder",
                        "annotation", "divider", "unknown"
                    ]},
                    "content": {"type": "object"},
                    "text_plain": {"type": "string"},
                    "char_count": {"type": "integer", "minimum": 0},
                    "token_count_est": {"type": "integer", "minimum": 0},
                    "embedding_ready": {"type": "boolean"},
                },
            },
        },
        "cross_references": {"type": "array"},
        "ml_index": {
            "type": "object",
            "properties": {
                "chunk_id_map": {"type": "object"},
                "heading_map": {"type": "array"},
                "embedding_candidates": {"type": "array", "items": {"type": "string"}},
                "chunk_boundaries": {"type": "array"},
            },
        },
        "custom": {"type": "object"},
    },
}


def check_required(data: dict, schema: dict, path: str = "") -> list[str]:
    """Recursively validate required fields and types."""
    errors = []
    stype = schema.get("type")
    if stype == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return errors
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required field '{req}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errors.extend(check_required(data[key], subschema, f"{path}.{key}"))
    elif stype == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return errors
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path}: array too short ({len(data)} < {schema['minItems']})")
        item_schema = schema.get("items", {})
        for i, item in enumerate(data):
            errors.extend(check_required(item, item_schema, f"{path}[{i}]"))
    elif stype == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
    elif stype == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data} below minimum {schema['minimum']}")
    elif stype == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data} below minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: value {data} above maximum {schema['maximum']}")
    elif stype == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")
    elif "enum" in schema:
        if data not in schema["enum"]:
            errors.append(f"{path}: value '{data}' not in enum {schema['enum']}")
    return errors


def validate_document(doc: dict) -> tuple[bool, list[str]]:
    """Validate a Doc2MLDocument against the inline schema."""
    errors = check_required(doc, SCHEMA)
    # Extra semantic checks
    blocks = doc.get("blocks", [])
    chunk_ids = [b["chunk_id"] for b in blocks]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("blocks: duplicate chunk_id values found")
    for i, b in enumerate(blocks):
        if b.get("char_count", 0) != len(b.get("text_plain", "")):
            errors.append(f"blocks[{i}].char_count does not match len(text_plain)")
    ml_index = doc.get("ml_index", {})
    for cid in ml_index.get("embedding_candidates", []):
        if cid not in chunk_ids:
            errors.append(f"ml_index.embedding_candidates: '{cid}' not found in blocks")
    return len(errors) == 0, errors


def check_ml_readiness(doc: dict) -> dict:
    """Check if the document is ready for ML consumption."""
    checks = {
        "has_blocks": False,
        "has_structure": False,
        "has_embedding_candidates": False,
        "no_empty_text": True,
        "token_counts_reasonable": True,
        "structure_covers_all_blocks": True,
        "overall_ready": False,
        "issues": [],
    }
    blocks = doc.get("blocks", [])
    if not blocks:
        checks["issues"].append("No blocks found")
    else:
        checks["has_blocks"] = True
    structure = doc.get("structure", {})
    if structure.get("children") or structure.get("chunk_ids"):
        checks["has_structure"] = True
    else:
        checks["issues"].append("No structure detected")
    candidates = doc.get("ml_index", {}).get("embedding_candidates", [])
    if candidates:
        checks["has_embedding_candidates"] = True
    else:
        checks["issues"].append("No embedding candidates found")
    for b in blocks:
        if not b.get("text_plain", "").strip():
            checks["no_empty_text"] = False
            checks["issues"].append(f"Block {b.get('chunk_id')} has empty text_plain")
        if b.get("token_count_est", 0) > 4096:
            checks["token_counts_reasonable"] = False
            checks["issues"].append(f"Block {b.get('chunk_id')} exceeds 4096 tokens")
    # Coverage
    all_struct_cids = set()
    def collect_cids(node):
        all_struct_cids.update(node.get("chunk_ids", []))
        for c in node.get("children", []):
            collect_cids(c)
    collect_cids(structure)
    block_cids = {b["chunk_id"] for b in blocks}
    missing = block_cids - all_struct_cids
    if missing:
        checks["structure_covers_all_blocks"] = False
        checks["issues"].append(f"{len(missing)} blocks not referenced in structure")
    checks["overall_ready"] = all([
        checks["has_blocks"],
        checks["has_structure"],
        checks["has_embedding_candidates"],
        checks["no_empty_text"],
    ])
    return checks


def validate_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        doc = json.load(f)
    valid, errors = validate_document(doc)
    ml_ready = check_ml_readiness(doc)
    stats = doc.get("metadata", {}).get("statistics", {})
    return {
        "file": filepath,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_valid": valid,
        "schema_errors": errors,
        "ml_ready": ml_ready["overall_ready"],
        "ml_readiness": ml_ready,
        "summary": {
            "block_count": stats.get("block_count", len(doc.get("blocks", []))),
            "total_token_count_est": stats.get("total_token_count_est", 0),
            "section_count": stats.get("section_count", 0),
            "table_count": stats.get("table_count", 0),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate doc2ml-json output")
    parser.add_argument("filepath", help="Path to the .doc2ml.json file")
    parser.add_argument("-o", "--output", help="Write report to file (default: stdout)")
    args = parser.parse_args()
    try:
        report = validate_file(args.filepath)
        out = json.dumps(report, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Report written to {args.output}")
        else:
            print(out)
        sys.exit(0 if report["schema_valid"] and report["ml_ready"] else 1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
