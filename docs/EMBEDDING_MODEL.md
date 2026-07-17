# Embedding Model Selection

## Selected Model: OpenAI `text-embedding-3-small` (1536 dimensions)

## Decision Summary

| Criterion | `text-embedding-3-small` | Alternatives Considered |
|-----------|--------------------------|------------------------|
| Retrieval quality (MTEB avg) | ~62.3% | `text-embedding-3-large`: ~64.6%, `bge-small-en-v1.5`: ~51.6% |
| Cost per 1M tokens | $0.02 | `text-embedding-3-large`: $0.13, `voyage-3`: $0.06 |
| Latency (batch 100) | ~200ms | Local `bge-small`: ~2s (CPU) |
| Dimensions | 1536 (configurable) | `bge-small`: 384, `ada-002`: 1536 |
| Managed infra | Yes (API) | Local models require GPU/CPU provisioning |

## Engineering Rationale

### 1. Retrieval Quality vs. Cost

For an internal knowledge base with hundreds to low thousands of pages, `text-embedding-3-small` provides the best quality-to-cost ratio:

- **Quality gap to `text-embedding-3-large` is ~2.3 MTEB points** — meaningful but not transformative for a domain-specific KB where queries closely match document vocabulary (policy names, process terms, tool names).
- **Cost is 6.5× lower** than `text-embedding-3-large`. Full re-ingestion of 500 pages (~2000 chunks) costs approximately $0.004 in embedding tokens.
- The 1536-dimensional output captures sufficient semantic granularity for cosine similarity search over structured wiki content.

### 2. Dimensionality and pgvector Performance

- **1536 dimensions** aligns with pgvector HNSW index defaults and provides good recall without excessive storage.
- Each chunk embedding occupies ~6 KB (1536 × 4 bytes). For 5000 chunks, total vector storage is ~30 MB — negligible for PostgreSQL.
- HNSW index with `vector_cosine_ops` provides sub-10ms query latency at this scale.

The model supports dimension reduction (e.g., 512) via the `dimensions` parameter, but we use the full 1536 default because storage cost is negligible and quality is maximised.

### 3. Domain Fit

Q's knowledge base content is primarily:

- **Procedural** ("Submit expenses via Expensify within 30 days")
- **Policy-driven** ("Minimum 80% unit test coverage")
- **Entity-rich** (tool names, SLAs, thresholds, email addresses)

`text-embedding-3-small` handles entity-heavy factual text well because it was trained on diverse web and code content with strong proper-noun representations. Queries like "What is the P1 incident response time?" map reliably to chunks containing "P1", "15 min", and "Production down".

### 4. Operational Simplicity

- **No local model serving** — no GPU provisioning, model downloads, or version pinning for inference infrastructure.
- **Consistent with chat stack** — while Claude (Anthropic) generates answers, using OpenAI exclusively for embeddings keeps the embedding pipeline independent of the chat provider. If the chat model changes, embeddings remain stable.
- **Batch API support** — the ingestion pipeline batches 100 texts per API call, minimising round trips.

### 5. Why Not Local Models?

| Model | Pros | Cons for This Use Case |
|-------|------|----------------------|
| `BAAI/bge-small-en-v1.5` | Free, fast on GPU | 384 dims — lower precision for entity matching; requires sentence-transformers + torch (~2 GB); CPU inference too slow for batch ingestion |
| `all-MiniLM-L6-v2` | Very fast, lightweight | Lower MTEB scores; struggles with domain-specific terminology |
| `voyage-3` | Strong retrieval benchmarks | 3× cost of `text-embedding-3-small`; adds another API vendor dependency |

Local models make sense at >100K chunks with strict data residency requirements. Q's KB is internal but not at a scale where self-hosted embedding inference is justified.

### 6. Why Not `text-embedding-ada-002`?

OpenAI's previous-generation model. `text-embedding-3-small` outperforms it on all MTEB benchmarks at the same price point. No reason to use the legacy model for a greenfield implementation.

## Configuration

```python
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
```

Set via environment variable `EMBEDDING_MODEL` in `.env`.

## Monitoring

During ingestion, the pipeline logs total embedding tokens and estimated cost. For the sample KB (16 pages, ~45 chunks), embedding cost is < $0.001.

If retrieval quality degrades after significant KB growth (>5000 chunks), re-evaluate with `text-embedding-3-large` on the test set before switching — the evaluation script (`evaluate.py`) supports A/B comparison by re-ingesting with a different model.
