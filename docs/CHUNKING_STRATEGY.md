# Chunking Strategy

## Overview

The Q Knowledge Base Agent uses a **hierarchical, structure-aware chunking strategy** designed specifically for BookStack wiki content. Unlike naive fixed-size splitting, this approach preserves the semantic boundaries of documentation: shelves, books, chapters, pages, and section headings.

## Problem Statement

Wiki documentation has inherent structure that flat chunking destroys:

- A "Leave Policy" page contains distinct sections (Annual Leave, Sick Leave, Parental Leave) that should not bleed into each other.
- Procedural documents (Onboarding Checklist) have ordered steps that lose meaning when split mid-list.
- Retrieval queries often target a specific section ("What is the parental leave entitlement?") rather than the entire page.

Fixed-size chunking (e.g., 512 tokens with no structure) causes three failure modes:

1. **Context fragmentation** — a policy table split across two chunks loses its header row in the second chunk.
2. **Semantic dilution** — unrelated sections merged into one chunk reduce embedding specificity.
3. **Metadata loss** — without breadcrumb context, retrieved chunks cannot be attributed to their source.

## Strategy

### Step 1: Hierarchy-Aware Source Parsing

Each BookStack page carries metadata: shelf names, book name, chapter name, page title, and URL. This breadcrumb is prepended to every chunk as a non-negotiable prefix:

```
Source: Company Handbook > HR & People > Policies > Leave and Absence Policy
Page: Leave and Absence Policy
Section: Parental Leave
```

This ensures that even a small chunk retrieved in isolation carries enough context for both the embedding model and the LLM to understand where it belongs.

### Step 2: Section-Boundary Splitting

Pages are split at heading boundaries (H1–H4), not at arbitrary token counts:

- **Markdown pages**: split on ATX headings (`#`, `##`, etc.)
- **HTML pages**: parsed with BeautifulSoup, split at `<h1>`–`<h4>` tags

Each section becomes a candidate chunk containing its heading text and all content until the next heading of equal or higher level.

**Why headings?** Headings in wiki docs are authored deliberately — they mark topic boundaries that match how employees search ("sick leave policy", "PR approval process").

### Step 3: Size Normalisation

Candidate chunks are normalised to fit embedding model sweet spots:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target size | 512 tokens | Optimal for `text-embedding-3-small`; large enough for a complete policy paragraph, small enough for precise retrieval |
| Maximum size | 768 tokens | Hard ceiling before forced sub-splitting |
| Minimum size | 80 tokens | Chunks below this are merged with the next same-level section |
| Overlap | 64 tokens | Trailing paragraphs carried into the next chunk to preserve cross-boundary context |

Oversized sections are sub-split at paragraph boundaries (`\n\n`) with 64-token overlap from the previous chunk's trailing paragraphs. This prevents list items or multi-paragraph explanations from being cut mid-sentence.

### Step 4: Small-Chunk Merging

Consecutive sections below 80 tokens at the same heading level are merged if the combined size stays under 768 tokens. This prevents over-fragmentation of pages with many short subsections (e.g., a checklist with 10 one-line items under the same heading level).

## Configuration

All parameters are defined in `CLI/config.py`:

```python
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 64
CHUNK_MIN_TOKENS = 80
```

## Why Not Alternatives?

| Alternative | Why Not |
|-------------|---------|
| Fixed-size (256/512 tokens, no structure) | Breaks section boundaries; loses heading context |
| One chunk per page | Pages often exceed 2000 tokens; embedding quality degrades for long texts; retrieval lacks precision |
| Semantic chunking (embedding-based splits) | Adds latency and cost during ingestion; heading boundaries are already semantic in well-authored wikis |
| Sentence-level splitting | Too granular; retrieved chunks lack sufficient context to answer procedural questions |
| Recursive character splitting (LangChain default) | Ignores document structure; treats markdown headings as plain text |

## Expected Outcomes

On the 25-question evaluation set against the sample KB:

- **Page Recall@5**: measures whether the correct source page appears in the top 5 retrieved chunks
- **Answer Recall@5**: measures whether the expected answer keywords appear in retrieved content
- **MRR (Mean Reciprocal Rank)**: measures how highly the correct page ranks

The structured approach consistently outperforms fixed-size baselines on page recall because breadcrumb prefixes and section headings create embeddings that align with natural-language queries.

## Re-Ingestion

When a page is updated in BookStack, re-running ingestion with `--clear` or upserting by `chunk_id` (format: `{page_id}-{index}`) replaces stale chunks. The `ON CONFLICT` upsert in the vector store handles incremental updates without full re-indexing.
