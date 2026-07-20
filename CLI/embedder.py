"""Embedding generation via Sentence Transformers (MPS on Apple Silicon)."""

from __future__ import annotations

import os
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from chunker import Chunk
from config import (
    BGE_QUERY_PREFIX,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    get_embedding_dimensions,
)


def _resolve_device() -> str:
    override = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """Generates dense vector embeddings using BGE via Sentence Transformers."""

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or EMBEDDING_MODEL
        self.dimensions = get_embedding_dimensions()
        self.device = _resolve_device()
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.total_tokens = 0
        self._use_query_prefix = "bge" in self.model_name.lower()
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def estimated_cost(self) -> float:
        return 0.0

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _format_query(self, query: str) -> str:
        if self._use_query_prefix:
            return f"{BGE_QUERY_PREFIX}{query}"
        return query

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        self.total_tokens += sum(self._estimate_tokens(t) for t in texts)
        payload = [self._format_query(t) if is_query else t for t in texts]
        vectors = self._model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in vectors]

    def embed_chunks(
        self, chunks: list[Chunk], *, show_progress: bool = True
    ) -> list[list[float]]:
        return self.embed_texts(
            [c.content for c in chunks],
            is_query=False,
            show_progress=show_progress,
        )

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], is_query=True)[0]
