# Q Knowledge Base Agent

An internal RAG (Retrieval-Augmented Generation) agent that ingests Q's BookStack wiki documentation into **pgvector** and provides a CLI chat interface for answering questions about internal processes, policies, and technical standards.

## Architecture

```
BookStack API / Local Export
        │
        ▼
  Structured Chunker  ──►  BGE Embeddings (local)  ──►  pgvector (PostgreSQL)
                                                        │
  User Question ──► Embed Query ──► Similarity Search ──┘
                                        │
                                        ▼
                              Claude (Answer Generation)
```

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 16+** with **pgvector** extension
- **Anthropic API key** (for Claude answer generation via proxy)
- **BookStack API token** (optional — for live ingestion from kb.q.agency)

Embeddings run locally via **Sentence Transformers + MPS** (Apple GPU) — no OpenAI key required.

## Quick Start

### 1. Start PostgreSQL with pgvector

**Option A — Docker (recommended):**

```bash
docker compose up -d
```

This starts PostgreSQL on port `5433` with pgvector pre-installed.

**Option B — Local PostgreSQL:**

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb q_knowledge_base
psql q_knowledge_base -c "CREATE EXTENSION vector;"
```

Update `DATABASE_URL` in `.env` accordingly (default local: `postgresql://$(whoami)@localhost:5432/q_knowledge_base`).

### 2. Install dependencies

```bash
cd "Q Knowledge Base Agent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Ingest the knowledge base

**From sample data (no API credentials needed):**

```bash
python ingest.py --source local --clear
```

**From live BookStack API:**

```bash
python ingest.py --source api --clear
```

Pages are cached to `data/cache/bookstack_pages.json` for fast re-runs.

**Re-embed only (skip BookStack API, uses MPS):**

```bash
python reembed.py --from-cache --clear
```

Or re-embed existing DB chunks without re-chunking:

```bash
python reembed.py
```

### 5. Chat

```bash
python chat.py
```

### 6. Run retrieval evaluation

```bash
python evaluate.py
```

## BookStack API Setup

The Q KB at [kb.q.agency](https://kb.q.agency) runs on BookStack. To ingest live documentation:

1. **Permission required:** Your BookStack user must have the **"Access System API"** role permission. Ask your KB admin to enable this on your role.

2. **Create an API token:**

   - Log in to https://kb.q.agency
   - Go to **My Account → Access & Security** (or `/my-account/auth`)
   - Under **API Tokens**, click **Create Token**
   - Copy the **Token ID** and **Token Secret** (shown once)

3. **Add to `.env`:**

   ```env
   BOOKSTACK_BASE_URL=https://kb.q.agency
   BOOKSTACK_TOKEN_ID=your-token-id
   BOOKSTACK_TOKEN_SECRET=your-token-secret
   ```

4. **Verify access:**

   ```bash
   curl --request GET \
     --url https://kb.q.agency/api/pages?count=1 \
     --header 'Authorization: Token YOUR_TOKEN_ID:YOUR_TOKEN_SECRET'
   ```

   A successful response returns JSON with a `data` array. A 401 means the token is missing/invalid; a 403 means your user lacks API access permission.

5. **Ingest:**

   ```bash
   python ingest.py --source api --clear
   ```

### Alternative: Local Export

If API access is not available, export pages from BookStack and place them in `data/sample_kb/pages.json` (see the included sample for format), then:

```bash
python ingest.py --source local --clear
```

## Project Structure

```
.
├── chat.py                 # CLI entry point
├── ingest.py               # Ingestion entry point
├── evaluate.py             # Evaluation entry point
├── docker-compose.yml      # PostgreSQL + pgvector
├── requirements.txt
├── CLI/
│   ├── config.py           # Configuration
│   ├── bookstack_client.py # BookStack API client
│   ├── chunker.py          # Structured chunking
│   ├── embedder.py         # BGE embeddings (fastembed)
│   ├── store.py            # pgvector storage
│   ├── chat.py             # RAG chat logic
│   ├── ingest.py           # Ingestion pipeline
│   ├── evaluate.py         # Retrieval evaluation
│   └── cli.py              # Interactive chat REPL
├── data/
│   └── sample_kb/          # Anonymised sample KB for development
├── docs/
│   ├── CHUNKING_STRATEGY.md
│   └── EMBEDDING_MODEL.md
└── eval/
    ├── test_questions.json # 25 ground-truth test questions
    └── results/            # Evaluation output
```

## Design Decisions

| Component    | Choice                                   | Documentation                                          |
| ------------ | ---------------------------------------- | ------------------------------------------------------ |
| Chunking     | Hierarchical, heading-aware with overlap | [docs/CHUNKING_STRATEGY.md](docs/CHUNKING_STRATEGY.md) |
| Embeddings   | BAAI `bge-large-en-v1.5` (1024d, MPS)    | [docs/EMBEDDING_MODEL.md](docs/EMBEDDING_MODEL.md)     |
| Vector store | pgvector with HNSW cosine index          | —                                                      |
| Chat model   | Claude Sonnet                            | —                                                      |
| Retrieval    | Top-5 cosine similarity, threshold 0.3   | —                                                      |

## Evaluation

The test set contains **25 questions** across HR, Engineering, Delivery, Finance, Infrastructure, and Compliance categories. Each question has an expected source page and ground-truth answer.

Metrics:

- **Page Recall@K** — correct source page in top-K results
- **Answer Recall@K** — expected answer keywords found in retrieved chunks
- **MRR** — mean reciprocal rank of the correct page

Run: `python evaluate.py --top-k 5`

Results are saved to `eval/results/`.

## Environment Variables

| Variable                 | Required   | Description                               |
| ------------------------ | ---------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY`      | Yes (chat) | Answer generation                         |
| `ANTHROPIC_BASE_URL`     | No         | Proxy gateway for Anthropic               |
| `DATABASE_URL`           | Yes        | PostgreSQL connection string              |
| `BOOKSTACK_TOKEN_ID`     | API ingest | BookStack API token ID                    |
| `BOOKSTACK_TOKEN_SECRET` | API ingest | BookStack API token secret                |
| `BOOKSTACK_BASE_URL`     | No         | Default: `https://kb.q.agency`            |
| `EMBEDDING_MODEL`        | No         | Default: `BAAI/bge-large-en-v1.5`         |
| `EMBEDDING_BATCH_SIZE`   | No         | Default: `64` (MPS batch size)            |
| `EMBEDDING_DEVICE`       | No         | Auto: `mps` on Mac, else `cpu`            |
| `CHAT_MODEL`             | No         | Default: `claude-sonnet-5`                |
| `TOP_K`                  | No         | Default: `8`                              |
| `SIMILARITY_THRESHOLD`   | No         | Default: `0.15` (tuned for BGE)           |

## Evaluation Results

See [docs/EVALUATION.md](docs/EVALUATION.md) for baseline metrics on the 25-question test set.

Internal / educational use.
