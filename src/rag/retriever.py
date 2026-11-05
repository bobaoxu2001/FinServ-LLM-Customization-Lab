"""Policy retrieval over the chunked knowledge base.

Two interchangeable backends behind one interface:

- TfidfRetriever (default): sklearn TF-IDF + cosine similarity. Zero extra
  deps, fully deterministic, strong on the short keyword-dense KB used here.
- FaissRetriever (optional): sentence-transformers embeddings in a FAISS
  inner-product index. Enable with `pip install faiss-cpu sentence-transformers`.

The evaluation suite reports recall@k for whichever backend is active, so the
tradeoff is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunker import Chunk, build_chunks


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class TfidfRetriever:
    backend = "tfidf-cosine"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        order = np.argsort(sims)[::-1][:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in order]


class FaissRetriever:  # pragma: no cover - optional dependency path
    backend = "faiss-minilm"

    def __init__(self, chunks: list[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        emb = self.model.encode([c.text for c in chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        qv = self.model.encode([query], normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(qv, dtype="float32"), k)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], idx[0])
            if i >= 0
        ]


def build_retriever(policy_dir: str | Path, backend: str = "auto"):
    """Build the best available retriever over the policy KB."""
    chunks = build_chunks(policy_dir)
    if backend in ("faiss", "auto"):
        try:
            return FaissRetriever(chunks)
        except ImportError:
            if backend == "faiss":
                raise
    return TfidfRetriever(chunks)
