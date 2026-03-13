"""
models/embeddings.py
RAG embedding layer — HuggingFace sentence-transformers (free, local)
Handles: document chunking, FAISS vector store build/load, similarity search.

Fix notes vs v1:
  - Type hint was faiss.IndexFlatL2 but index was IndexFlatIP — unified to Any
  - add_documents now correctly re-uses the existing index dimension instead of
    recreating it, avoiding shape mismatches when adding docs incrementally
  - Graceful handling when FAISS index has 0 vectors at search time
  - Optional: pre-built knowledge base loaded automatically on first use
"""

import os
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from config.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
)

# ── Singleton encoder ─────────────────────────────────────────────────────────
_encoder: Optional[SentenceTransformer] = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Vector Store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed vector store using cosine similarity
    (inner-product on L2-normalised vectors).
    """

    def __init__(self):
        # Use Any to avoid faiss type-stub issues across platforms
        self.index: Any = None
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self._dim: Optional[int] = None   # embedding dimension, set on first add

    # ── internal helpers ──────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts and return float32 L2-normalised vectors."""
        encoder = _get_encoder()
        vecs = encoder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,   # ensures cosine via inner-product
            batch_size=64,
        )
        return np.array(vecs, dtype="float32")

    def _build_index(self, dim: int) -> Any:
        """Create a fresh IndexFlatIP of the given dimension."""
        return faiss.IndexFlatIP(dim)

    # ── public API ────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: List[str],
        source_name: str = "uploaded",
    ) -> int:
        """
        Embed and add documents to the store.
        Returns the number of chunks successfully added.
        Raises RuntimeError on failure.
        """
        try:
            all_chunks: List[str] = []
            for text in texts:
                if text and text.strip():
                    all_chunks.extend(chunk_text(text))

            if not all_chunks:
                return 0

            vectors = self._embed(all_chunks)

            # Initialise index on first call — dimension comes from the model
            if self.index is None:
                self._dim = vectors.shape[1]
                self.index = self._build_index(self._dim)

            # Safety: if somehow dim changed (shouldn't happen) raise early
            if vectors.shape[1] != self._dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._dim}, "
                    f"got {vectors.shape[1]}. Did you change EMBEDDING_MODEL?"
                )

            self.index.add(vectors)
            self.chunks.extend(all_chunks)
            self.sources.extend([source_name] * len(all_chunks))
            return len(all_chunks)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"add_documents failed: {e}") from e

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Tuple[str, str, float]]:
        """
        Return up to top_k (chunk, source, score) tuples ranked by relevance.
        Score is cosine similarity in [0, 1] (higher = more relevant).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            q_vec = self._embed([query])          # shape (1, dim)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(q_vec, k)

            results: List[Tuple[str, str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:                      # FAISS returns -1 for padding
                    results.append(
                        (self.chunks[idx], self.sources[idx], float(score))
                    )
            return results

        except Exception as e:
            raise RuntimeError(f"search failed: {e}") from e

    def save(self, path: str = VECTOR_STORE_PATH) -> None:
        """Persist FAISS index + metadata to disk."""
        if self.index is None:
            return   # nothing to save
        try:
            os.makedirs(path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            with open(os.path.join(path, "metadata.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "sources": self.sources,
                        "dim": self._dim,
                    },
                    f,
                )
        except Exception as e:
            raise RuntimeError(f"save failed: {e}") from e

    def load(self, path: str = VECTOR_STORE_PATH) -> bool:
        """
        Load FAISS index + metadata from disk.
        Returns True on success, False if no saved store exists.
        """
        idx_path  = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")

        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return False

        try:
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.chunks  = meta.get("chunks", [])
            self.sources = meta.get("sources", [])
            self._dim    = meta.get("dim")

            # Back-fill dim if loading an old pickle that didn't store it
            if self._dim is None and self.index is not None:
                self._dim = self.index.d

            return True
        except Exception:
            # Corrupt store — reset to empty
            self.index   = None
            self.chunks  = []
            self.sources = []
            self._dim    = None
            return False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def unique_sources(self) -> List[str]:
        return sorted(set(self.sources))


# ── Format retrieved chunks for the LLM prompt ────────────────────────────────

def format_kb_context(results: List[Tuple[str, str, float]]) -> str:
    if not results:
        return ""
    lines = []
    for i, (chunk, source, score) in enumerate(results, 1):
        lines.append(
            f"[{i}] (Source: {source} | relevance: {score:.3f})\n{chunk}"
        )
    return "\n\n".join(lines)