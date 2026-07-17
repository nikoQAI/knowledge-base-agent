"""RAG retrieval and answer generation."""

from __future__ import annotations

import os
from typing import Any

import anthropic
from anthropic import Anthropic

from config import (
    ANTHROPIC_BASE_URL,
    CHAT_MODEL,
    HYBRID_SEARCH,
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    TOP_K,
)
from embedder import Embedder
from store import VectorStore


class KnowledgeBaseChat:
    """Retrieval-augmented chat over the Q knowledge base."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is required for chat. Set it in .env."
            )
        self.store = store or VectorStore()
        self.embedder = embedder or Embedder()
        client_kwargs: dict[str, str] = {"api_key": api_key}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        self.client = Anthropic(**client_kwargs)
        self.model = CHAT_MODEL

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(
            query_embedding,
            top_k=top_k,
            query_text=query,
            hybrid=HYBRID_SEARCH,
        )

    def format_context(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No relevant knowledge base excerpts were found."

        parts: list[str] = []
        for i, hit in enumerate(results, 1):
            parts.append(
                f"### Excerpt {i} (similarity: {hit['similarity']:.3f})\n"
                f"**{hit['page_title']}** — {hit['breadcrumb']}\n"
                f"Section: {hit['section_heading']}\n\n"
                f"{hit['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def answer(self, query: str, top_k: int = TOP_K) -> dict[str, Any]:
        """Retrieve context and generate an answer."""
        results = self.retrieve(query, top_k=top_k)
        context = self.format_context(results)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT.format(context=context),
            messages=[{"role": "user", "content": query}],
        )

        answer_text = ""
        for block in response.content:
            if block.type == "text":
                answer_text += block.text

        return {
            "answer": answer_text,
            "sources": results,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def answer_stream(self, query: str, top_k: int = TOP_K):
        """Stream the answer tokens."""
        results = self.retrieve(query, top_k=top_k)
        context = self.format_context(results)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT.format(context=context),
            messages=[{"role": "user", "content": query}],
        ) as stream:
            for text in stream.text_stream:
                yield text

        yield {"__sources__": results}
