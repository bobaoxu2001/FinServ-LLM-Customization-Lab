"""Load and chunk policy documents from the simulated servicing knowledge base."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_policy_docs(policy_dir: str | Path) -> list[dict]:
    """Read every markdown policy doc; title = first H1."""
    docs = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        text = path.read_text()
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        docs.append(
            {
                "doc_id": path.stem,
                "title": m.group(1).strip() if m else path.stem,
                "text": text,
            }
        )
    return docs


def chunk_document(doc: dict, max_words: int = 180) -> list[Chunk]:
    """Split on H2 sections, then window long sections by word count.

    Section-aware chunking keeps each snippet attributable to a specific
    policy clause, which is what makes citations auditable.
    """
    parts = re.split(r"^##\s+", doc["text"], flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        if lines[0].startswith("# "):
            section = "Overview"
            body = "\n".join(lines[1:]).strip()
        else:
            section = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        words = body.split()
        for i in range(0, len(words), max_words):
            window = " ".join(words[i : i + max_words])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::{section}::{i // max_words}",
                    doc_id=doc["doc_id"],
                    doc_title=doc["title"],
                    section=section,
                    text=window,
                )
            )
    return chunks


def build_chunks(policy_dir: str | Path, max_words: int = 180) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_policy_docs(policy_dir):
        chunks.extend(chunk_document(doc, max_words=max_words))
    return chunks
