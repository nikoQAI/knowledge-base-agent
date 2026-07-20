# Embedding Model Selection

## Selected Model: BAAI `bge-large-en-v1.5` (1024 dimensions)

## Runtime: Sentence Transformers + Apple MPS

Embeddings run locally via **Sentence Transformers** with **MPS** (Apple GPU) when available. Typical throughput for ~5000 chunks: **~3–8 minutes** on Apple Silicon vs 30–60+ minutes on CPU-only fastembed.

## Decision Summary

| Criterion | `bge-large-en-v1.5` | Notes |
|-----------|----------------------|-------|
| Retrieval quality | Strong English retrieval | Top-tier open embedder for RAG |
| Cost | Free (local) | OpenAI embedding models blocked on Q proxy |
| Device | `mps` (auto) | Override with `EMBEDDING_DEVICE=cpu` |
| Dimensions | 1024 | Fits pgvector HNSW (max 2000) |
| Batch size | 64 (default) | Tune via `EMBEDDING_BATCH_SIZE` |

## Configuration

```bash
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DEVICE=mps          # optional; auto-detected on Mac
SIMILARITY_THRESHOLD=0.15
```

## Re-embedding Without Re-fetching

After the first API ingest, pages are cached to `data/cache/bookstack_pages.json`.

**Re-embed only (fastest after cache exists):**

```bash
python reembed.py --from-cache --clear
```

**Re-embed existing DB chunks (no re-chunking):**

```bash
python reembed.py
```

**Full ingest from cache (skip BookStack API):**

```bash
python ingest.py --source cache --clear
```

## Model Variants

| Model | Dims | Use case |
|-------|------|----------|
| `BAAI/bge-large-en-v1.5` | 1024 | **Default** — best English retrieval |
| `BAAI/bge-base-en-v1.5` | 768 | Faster, slightly lower quality |
| `BAAI/bge-small-en-v1.5` | 384 | Fastest, lowest quality |
| `BAAI/bge-m3` | 1024 | Multilingual (100+ languages) |

## Query Prefix

BGE uses an instruction prefix on **queries only** (handled automatically by the embedder):

```
Represent this sentence for searching relevant passages: <query>
```

## Migration Note

Switching models changes vector dimensions — use `--clear` and re-embed. First run downloads the model (~1.2 GB for `bge-large-en-v1.5`).
