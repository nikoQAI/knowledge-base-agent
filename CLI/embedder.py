"""Embedding generation — OpenAI (primary) with improved local fallback."""

from __future__ import annotations

import os
from typing import Sequence

from chunker import Chunk
from config import (
    BGE_QUERY_PREFIX,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_TOKEN,
    EMBEDDING_PRICES,
    LOCAL_EMBEDDING_MODEL,
    MODEL_DIMENSIONS,
    get_embedding_dimensions,
)


class Embedder:
    """Generates dense vector embeddings for text chunks."""

    BATCH_SIZE = 100

    def __init__(self, model: str | None = None) -> None:
        self.total_tokens = 0
        self._provider = "openai"
        self._local_model = None
        self._openai_client = None
        self._use_query_prefix = False

        provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        # Anthropic has no embedding API — use local models when configured.
        if provider == "local" or not api_key:
            self._init_local(model)
        elif api_key:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=api_key)
            self.model = model or EMBEDDING_MODEL
            self.dimensions = get_embedding_dimensions()
            self._price_per_token = EMBEDDING_PRICES.get(
                self.model, EMBEDDING_PRICE_PER_TOKEN
            )
        else:
            raise EnvironmentError(
                "No embedding provider configured. "
                "Set EMBEDDING_PROVIDER=local (free, runs locally) or OPENAI_API_KEY."
            )

    # Fallback chain if the preferred model isn't cached / can't be downloaded.
    _LOCAL_MODEL_FALLBACKS = (
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
    )

    def _init_local(self, model: str | None = None) -> None:
        """Local embedder — no API key required."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EnvironmentError(
                "Install fastembed for local embeddings: pip install fastembed"
            ) from exc

        preferred = model or LOCAL_EMBEDDING_MODEL
        candidates = [preferred, *self._LOCAL_MODEL_FALLBACKS]
        seen: set[str] = set()
        last_error: Exception | None = None

        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                self._local_model = TextEmbedding(model_name=candidate)
                self.model = candidate
                break
            except Exception as exc:
                last_error = exc
        else:
            raise EnvironmentError(
                f"Could not load a local embedding model. Tried: {', '.join(seen)}. "
                "Ensure fastembed is installed and you have network access for the first download."
            ) from last_error

        self._provider = "local"
        self.dimensions = MODEL_DIMENSIONS.get(self.model, 768)
        self._use_query_prefix = "bge" in self.model.lower()
        self._price_per_token = 0.0

    @property
    def estimated_cost(self) -> float:
        if self._provider == "local":
            return 0.0
        return self.total_tokens * self._price_per_token

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _local_embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert self._local_model is not None
        return [list(vec) for vec in self._local_model.embed(list(texts))]

    def _format_query(self, query: str) -> str:
        if self._use_query_prefix:
            return f"{BGE_QUERY_PREFIX}{query}"
        return query

    def embed_texts(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        if self._provider == "local":
            self.total_tokens += sum(self._estimate_tokens(t) for t in texts)
            payload = [self._format_query(t) if is_query else t for t in texts]
            return self._local_embed(payload)

        assert self._openai_client is not None
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = list(texts[i : i + self.BATCH_SIZE])
            kwargs: dict = {"model": self.model, "input": batch}
            # text-embedding-3-* supports optional dimension reduction
            if self.model.startswith("text-embedding-3-"):
                kwargs["dimensions"] = self.dimensions
            response = self._openai_client.embeddings.create(**kwargs)
            self.total_tokens += response.usage.total_tokens
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([c.content for c in chunks], is_query=False)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], is_query=True)[0]
