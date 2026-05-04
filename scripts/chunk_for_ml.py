#!/usr/bin/env python3
"""
Chunk a doc2ml-json document into ML-ready segments.

Respects section boundaries and heading context. Produces overlapping
chunks suitable for training, inference, and RAG retrieval.
"""

import argparse
import json
import sys
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 0.75))


def get_block(doc: dict, chunk_id: str) -> dict | None:
    for b in doc.get("blocks", []):
        if b["chunk_id"] == chunk_id:
            return b
    return None


def get_structure_path(doc: dict, chunk_id: str) -> list[str]:
    mapping = doc.get("ml_index", {}).get("chunk_id_map", {})
    entry = mapping.get(chunk_id, {})
    return entry.get("structure_path", ["root"])


def get_heading_path(doc: dict, chunk_id: str) -> list[str]:
    """Get the chain of heading texts for this chunk."""
    path = get_structure_path(doc, chunk_id)
    headings = []
    for node_id in path:
        if node_id == "root":
            continue
        title = _find_node_title(doc.get("structure", {}), node_id)
        if title:
            headings.append(title)
    return headings


def _find_node_title(node: dict, node_id: str) -> str | None:
    if node.get("node_id") == node_id:
        return node.get("title", "")
    for child in node.get("children", []):
        result = _find_node_title(child, node_id)
        if result is not None:
            return result
    return None


def block_to_text(block: dict) -> str:
    """Render a block to plain text for chunking."""
    btype = block.get("type", "paragraph")
    text = block.get("text_plain", "")
    if btype == "heading":
        level = block.get("content", {}).get("level", 1)
        return "#" * level + " " + text
    if btype == "code_block":
        lang = block.get("content", {}).get("language", "")
        return f"```{lang}\n{text}\n```"
    if btype == "list":
        items = block.get("content", {}).get("items", [])
        return "\n".join(f"- {it.get('text', '')}" for it in items)
    if btype == "table":
        rows = block.get("content", {}).get("rows", [])
        if rows:
            return "\n".join("| " + " | ".join(str(c or "") for c in row) + " |" for row in rows)
    return text


def chunk_document(doc: dict, max_tokens: int = 512, overlap_tokens: int = 50) -> list[dict]:
    """Split a doc2ml-json document into ML-ready chunks."""
    candidates = doc.get("ml_index", {}).get("embedding_candidates", [])
    blocks = [get_block(doc, cid) for cid in candidates if get_block(doc, cid)]
    if not blocks:
        # Fallback: use all embedding_ready blocks
        blocks = [b for b in doc.get("blocks", []) if b.get("embedding_ready")]
    chunks = []
    current_blocks = []
    current_tokens = 0
    chunk_counter = 0
    for block in blocks:
        block_text = block_to_text(block)
        block_tokens = estimate_tokens(block_text)
        # Respect oversized blocks: emit them as their own chunk if too large
        if block_tokens > max_tokens and current_blocks:
            chunk_counter += 1
            chunks.append(_make_chunk(doc, current_blocks, chunk_counter, max_tokens))
            current_blocks = []
            current_tokens = 0
        if block_tokens > max_tokens:
            chunk_counter += 1
            chunks.append(_make_chunk(doc, [block], chunk_counter, max_tokens))
            continue
        if current_tokens + block_tokens > max_tokens and current_blocks:
            chunk_counter += 1
            chunks.append(_make_chunk(doc, current_blocks, chunk_counter, max_tokens))
            # Overlap
            overlap_blocks = []
            overlap_count = 0
            for prev in reversed(current_blocks):
                pt = estimate_tokens(block_to_text(prev))
                if overlap_count + pt <= overlap_tokens:
                    overlap_blocks.insert(0, prev)
                    overlap_count += pt
                else:
                    break
            current_blocks = overlap_blocks
            current_tokens = overlap_count
        current_blocks.append(block)
        current_tokens += block_tokens
    if current_blocks:
        chunk_counter += 1
        chunks.append(_make_chunk(doc, current_blocks, chunk_counter, max_tokens))
    return chunks


def _make_chunk(doc: dict, chunk_blocks: list[dict], chunk_num: int, max_tokens: int) -> dict:
    texts = []
    block_ids = []
    structure_paths = []
    heading_paths = []
    total_tokens = 0
    for b in chunk_blocks:
        bt = block_to_text(b)
        texts.append(bt)
        block_ids.append(b["chunk_id"])
        total_tokens += estimate_tokens(bt)
        sp = get_structure_path(doc, b["chunk_id"])
        structure_paths.append(sp)
        heading_paths.append(get_heading_path(doc, b["chunk_id"]))
    # Common structure path prefix
    common_path = _common_prefix(structure_paths) if structure_paths else []
    common_headings = _common_prefix(heading_paths) if heading_paths else []
    chunk_text = "\n\n".join(texts)
    # Truncate if over max_tokens (last resort)
    words = chunk_text.split()
    while estimate_tokens(chunk_text) > max_tokens and len(words) > 10:
        words = words[:-10]
        chunk_text = " ".join(words)
    return {
        "chunk_id": f"chunk-{chunk_num:04d}",
        "text": chunk_text,
        "token_count_est": estimate_tokens(chunk_text),
        "block_ids": block_ids,
        "structure_path": common_path,
        "heading_path": common_headings,
        "document_id": doc.get("document_id"),
        "source": doc.get("metadata", {}).get("source", {}).get("uri"),
    }


def _common_prefix(lists: list[list]) -> list:
    if not lists:
        return []
    prefix = []
    for items in zip(*lists):
        if all(i == items[0] for i in items):
            prefix.append(items[0])
        else:
            break
    return prefix


def main():
    parser = argparse.ArgumentParser(description="Chunk doc2ml-json for ML context windows")
    parser.add_argument("filepath", help="Path to .doc2ml.json file")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk (default: 512)")
    parser.add_argument("--overlap-tokens", type=int, default=50, help="Overlap tokens between chunks (default: 50)")
    args = parser.parse_args()
    try:
        with open(args.filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunks = chunk_document(doc, args.max_tokens, args.overlap_tokens)
        out = {
            "doc2ml_version": doc.get("doc2ml_version", "0.5.0"),
            "document_id": doc.get("document_id"),
            "source": doc.get("metadata", {}).get("source", {}).get("uri"),
            "chunking_params": {"max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens},
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        out_path = args.output or f"{Path(args.filepath).stem}.chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Created {len(chunks)} chunks → {out_path}")
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
