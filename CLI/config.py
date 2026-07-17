"""Configuration for the Q Knowledge Base Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://qkb:qkb_dev_password@localhost:5433/q_knowledge_base",
)

# BookStack
BOOKSTACK_BASE_URL = os.getenv("BOOKSTACK_BASE_URL", "https://kb.q.agency").rstrip("/")
BOOKSTACK_TOKEN_ID = os.getenv("BOOKSTACK_TOKEN_ID", "")
BOOKSTACK_TOKEN_SECRET = os.getenv("BOOKSTACK_TOKEN_SECRET", "")
BOOKSTACK_REQUEST_DELAY = float(os.getenv("BOOKSTACK_REQUEST_DELAY", "0.2"))

# Embeddings
# Default to local — Anthropic provides chat only, not embeddings.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.00000002,   # $0.02 / 1M tokens
    "text-embedding-3-large": 0.00000013,  # $0.13 / 1M tokens
}


def get_embedding_dimensions() -> int:
    """Return vector dimensions for the active embedding model."""
    if EMBEDDING_PROVIDER == "local":
        return MODEL_DIMENSIONS.get(LOCAL_EMBEDDING_MODEL, 768)
    override = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if override:
        return int(override)
    return MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1536)


EMBEDDING_DIMENSIONS = get_embedding_dimensions()

# Chat
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-5")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
MAX_OUTPUT_TOKENS = 1500

# Retrieval — BGE local models produce lower cosine scores than OpenAI.
_default_threshold = "0.15" if EMBEDDING_PROVIDER == "local" else "0.25"
TOP_K = int(os.getenv("TOP_K", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", _default_threshold))
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
RRF_K = int(os.getenv("RRF_K", "60"))

# Chunking (see docs/CHUNKING_STRATEGY.md)
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 64
CHUNK_MIN_TOKENS = 80

# Pricing (claude-sonnet-5) — per token
INPUT_PRICE_PER_TOKEN = 0.000003
OUTPUT_PRICE_PER_TOKEN = 0.000015

EMBEDDING_PRICE_PER_TOKEN = EMBEDDING_PRICES.get(
    EMBEDDING_MODEL, EMBEDDING_PRICES["text-embedding-3-small"]
)

# BGE models expect an instruction prefix on queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are the Q Agency internal knowledge base assistant. You help employees find accurate information about Q's internal processes, policies, and technical standards.

## Instructions
- Answer ONLY using the retrieved knowledge base excerpts provided below.
- Cite the source page title and breadcrumb path when referencing specific policies or procedures.
- If the retrieved context does not contain enough information to answer confidently, say so and suggest which KB section the user should check.
- Do NOT invent policies, deadlines, or technical requirements not present in the context.
- Be concise but complete. Use markdown for lists and emphasis when helpful.
- For procedural questions, provide step-by-step answers when the context supports it.

## Retrieved Knowledge Base Context
{context}
"""
