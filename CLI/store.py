"""PostgreSQL + pgvector storage layer with optional hybrid search."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from chunker import Chunk
from config import (
    DATABASE_URL,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K,
    get_embedding_dimensions,
)

_SEARCH_VECTOR_EXPR = """
    to_tsvector(
        'english',
        coalesce(page_title, '') || ' ' ||
        coalesce(breadcrumb, '') || ' ' ||
        coalesce(section_heading, '') || ' ' ||
        coalesce(content, '')
    )
"""


class VectorStore:
    """Manages chunk storage and similarity search in pgvector."""

    def __init__(self, dsn: str = DATABASE_URL) -> None:
        self.dsn = dsn

    @contextmanager
    def _connect(self) -> Generator[psycopg.Connection, None, None]:
        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        register_vector(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _current_embedding_dims(self, conn: psycopg.Connection) -> int | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.atttypmod AS dims
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = 'kb_documents'
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
            row = cur.fetchone()
            dims = row["dims"] if row else None
            return dims if dims and dims > 0 else None

    def _migrate_embedding_dims(self, conn: psycopg.Connection, dims: int) -> None:
        """Recreate the embedding column when the model dimension changes."""
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS idx_kb_documents_embedding")
            cur.execute("ALTER TABLE kb_documents DROP COLUMN IF EXISTS embedding")
            cur.execute(f"ALTER TABLE kb_documents ADD COLUMN embedding vector({dims})")
            cur.execute(
                """
                CREATE INDEX idx_kb_documents_embedding
                ON kb_documents
                USING hnsw (embedding vector_cosine_ops)
                """
            )

    def initialize(self) -> None:
        """Create extension, tables, and indexes."""
        dims = get_embedding_dimensions()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS kb_documents (
                        id          SERIAL PRIMARY KEY,
                        page_id     INTEGER NOT NULL,
                        page_title  TEXT NOT NULL,
                        breadcrumb  TEXT NOT NULL,
                        url         TEXT,
                        chunk_id    TEXT UNIQUE NOT NULL,
                        section_heading TEXT,
                        heading_level   INTEGER,
                        chunk_index INTEGER NOT NULL,
                        content     TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        metadata    JSONB DEFAULT '{{}}',
                        embedding   vector({dims}),
                        search_vector tsvector,
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_documents_page_id
                    ON kb_documents (page_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_documents_embedding
                    ON kb_documents
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )

            stored_dims = self._current_embedding_dims(conn)
            if stored_dims is not None and stored_dims != dims:
                self._migrate_embedding_dims(conn, dims)

            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_documents_search
                    ON kb_documents USING gin (search_vector)
                    """
                )
                cur.execute(
                    f"""
                    UPDATE kb_documents
                    SET search_vector = {_SEARCH_VECTOR_EXPR}
                    WHERE search_vector IS NULL
                    """
                )

    def clear(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE kb_documents RESTART IDENTITY")

    def _search_text(self, chunk: Chunk) -> str:
        return " ".join(
            part
            for part in (
                chunk.page_title,
                chunk.breadcrumb,
                chunk.section_heading or "",
                chunk.content,
            )
            if part
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO kb_documents (
                            page_id, page_title, breadcrumb, url, chunk_id,
                            section_heading, heading_level, chunk_index,
                            content, token_count, metadata, embedding, search_vector
                        ) VALUES (
                            %(page_id)s, %(page_title)s, %(breadcrumb)s, %(url)s, %(chunk_id)s,
                            %(section_heading)s, %(heading_level)s, %(chunk_index)s,
                            %(content)s, %(token_count)s, %(metadata)s, %(embedding)s,
                            to_tsvector('english', %(search_text)s)
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            token_count = EXCLUDED.token_count,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            search_vector = EXCLUDED.search_vector
                        """,
                        {
                            "page_id": chunk.page_id,
                            "page_title": chunk.page_title,
                            "breadcrumb": chunk.breadcrumb,
                            "url": chunk.url,
                            "chunk_id": chunk.chunk_id,
                            "section_heading": chunk.section_heading,
                            "heading_level": chunk.heading_level,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "token_count": chunk.token_count,
                            "metadata": json.dumps(chunk.metadata),
                            "embedding": embedding,
                            "search_text": self._search_text(chunk),
                        },
                    )
        return len(chunks)

    def upsert_chunks_without_embeddings(self, chunks: list[Chunk]) -> int:
        """Insert or update chunk content without touching embeddings."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO kb_documents (
                            page_id, page_title, breadcrumb, url, chunk_id,
                            section_heading, heading_level, chunk_index,
                            content, token_count, metadata, search_vector
                        ) VALUES (
                            %(page_id)s, %(page_title)s, %(breadcrumb)s, %(url)s, %(chunk_id)s,
                            %(section_heading)s, %(heading_level)s, %(chunk_index)s,
                            %(content)s, %(token_count)s, %(metadata)s,
                            to_tsvector('english', %(search_text)s)
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            page_id = EXCLUDED.page_id,
                            page_title = EXCLUDED.page_title,
                            breadcrumb = EXCLUDED.breadcrumb,
                            url = EXCLUDED.url,
                            section_heading = EXCLUDED.section_heading,
                            heading_level = EXCLUDED.heading_level,
                            chunk_index = EXCLUDED.chunk_index,
                            content = EXCLUDED.content,
                            token_count = EXCLUDED.token_count,
                            metadata = EXCLUDED.metadata,
                            search_vector = EXCLUDED.search_vector
                        """,
                        {
                            "page_id": chunk.page_id,
                            "page_title": chunk.page_title,
                            "breadcrumb": chunk.breadcrumb,
                            "url": chunk.url,
                            "chunk_id": chunk.chunk_id,
                            "section_heading": chunk.section_heading,
                            "heading_level": chunk.heading_level,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "token_count": chunk.token_count,
                            "metadata": json.dumps(chunk.metadata),
                            "search_text": self._search_text(chunk),
                        },
                    )
        return len(chunks)

    def update_embeddings(
        self, chunk_ids: list[str], embeddings: list[list[float]]
    ) -> int:
        """Batch-update embeddings for existing chunks."""
        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings must have the same length")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE kb_documents SET embedding = %s WHERE chunk_id = %s",
                    [(embedding, chunk_id) for chunk_id, embedding in zip(chunk_ids, embeddings)],
                )
        return len(chunk_ids)

    def load_all_chunks(self) -> list[Chunk]:
        """Load all stored chunks for re-embedding."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT page_id, page_title, breadcrumb, url, chunk_id,
                           section_heading, heading_level, chunk_index,
                           content, token_count, metadata
                    FROM kb_documents
                    ORDER BY page_id, chunk_index
                    """
                )
                rows = cur.fetchall()

        chunks: list[Chunk] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    page_id=row["page_id"],
                    page_title=row["page_title"],
                    breadcrumb=row["breadcrumb"],
                    section_heading=row["section_heading"] or "Introduction",
                    heading_level=row["heading_level"] or 1,
                    content=row["content"],
                    token_count=row["token_count"],
                    chunk_index=row["chunk_index"],
                    url=row["url"] or "",
                    metadata=metadata or {},
                )
            )
        return chunks

    def _vector_search(
        self,
        cur: psycopg.Cursor,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        cur.execute(
            """
            SELECT
                chunk_id, page_id, page_title, breadcrumb, url,
                section_heading, content, token_count, metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM kb_documents
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, threshold, query_embedding, top_k),
        )
        return list(cur.fetchall())

    def _keyword_search(
        self,
        cur: psycopg.Cursor,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        cur.execute(
            """
            SELECT
                chunk_id, page_id, page_title, breadcrumb, url,
                section_heading, content, token_count, metadata,
                ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS keyword_rank
            FROM kb_documents
            WHERE search_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY keyword_rank DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        return list(cur.fetchall())

    @staticmethod
    def _rrf_fuse(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        *,
        rrf_k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """Reciprocal rank fusion of vector and keyword result lists."""
        scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(vector_results):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            items[cid] = {**hit, "similarity": hit.get("similarity", 0.0)}

        for rank, hit in enumerate(keyword_results):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in items:
                items[cid] = {**hit, "similarity": 0.0}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [items[cid] for cid, _ in ranked]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        *,
        query_text: str | None = None,
        hybrid: bool = False,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search, optionally fused with keyword search."""
        fetch_k = top_k * 2 if hybrid and query_text else top_k

        with self._connect() as conn:
            with conn.cursor() as cur:
                vector_results = self._vector_search(
                    cur, query_embedding, fetch_k, threshold
                )

                if not hybrid or not query_text:
                    return vector_results[:top_k]

                keyword_results = self._keyword_search(cur, query_text, fetch_k)
                if not keyword_results:
                    return vector_results[:top_k]

                fused = self._rrf_fuse(vector_results, keyword_results)
                return fused[:top_k]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total_chunks FROM kb_documents")
                total = cur.fetchone()["total_chunks"]
                cur.execute("SELECT COUNT(DISTINCT page_id) AS total_pages FROM kb_documents")
                pages = cur.fetchone()["total_pages"]
                return {"total_chunks": total, "total_pages": pages}
