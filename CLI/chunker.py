"""Structured chunking for BookStack wiki pages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag

from bookstack_client import KBPage
from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)


@dataclass
class Chunk:
    """A single indexed chunk with metadata."""

    chunk_id: str
    page_id: int
    page_title: str
    breadcrumb: str
    section_heading: str
    heading_level: int
    content: str
    token_count: int
    chunk_index: int
    url: str
    metadata: dict = field(default_factory=dict)


class StructuredChunker:
    """
    Hierarchical chunker that respects wiki document structure.

    Strategy:
    1. Parse HTML into a heading tree (H1-H4).
    2. Each section becomes a candidate chunk with its heading as context prefix.
    3. Oversized sections are split at paragraph boundaries with overlap.
    4. Undersized adjacent sections at the same level are merged.
    5. Every chunk carries breadcrumb + section metadata for retrieval filtering.
    """

    HEADING_TAGS = {"h1", "h2", "h3", "h4"}
    BLOCK_TAGS = {"p", "li", "td", "th", "blockquote", "pre", "code"}

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk_page(self, page: KBPage) -> list[Chunk]:
        """Split a KB page into structured chunks."""
        source = page.markdown.strip() if page.markdown.strip() else self._html_to_text(page.html)
        if not source.strip():
            return []

        if page.markdown.strip():
            sections = self._split_markdown_sections(source)
        else:
            sections = self._split_html_sections(page.html)

        raw_chunks: list[tuple[str, int, str]] = []
        for heading, level, body in sections:
            prefix = self._build_prefix(page, heading)
            full_text = f"{prefix}\n\n{body}".strip()
            tokens = self.count_tokens(full_text)

            if tokens <= CHUNK_MAX_TOKENS:
                raw_chunks.append((heading, level, full_text))
            else:
                for sub in self._split_oversized(body, prefix):
                    raw_chunks.append((heading, level, sub))

        merged = self._merge_small_chunks(raw_chunks)
        return [
            Chunk(
                chunk_id=f"{page.page_id}-{i}",
                page_id=page.page_id,
                page_title=page.title,
                breadcrumb=page.breadcrumb,
                section_heading=heading,
                heading_level=level,
                content=text,
                token_count=self.count_tokens(text),
                chunk_index=i,
                url=page.url,
                metadata={
                    "book_name": page.book_name,
                    "chapter_name": page.chapter_name,
                    "shelf_names": page.shelf_names,
                    "updated_at": page.updated_at,
                },
            )
            for i, (heading, level, text) in enumerate(merged)
        ]

    def chunk_pages(self, pages: list[KBPage]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))
        return all_chunks

    def _build_prefix(self, page: KBPage, section_heading: str) -> str:
        return (
            f"Source: {page.breadcrumb}\n"
            f"Page: {page.title}\n"
            f"Section: {section_heading}"
        )

    def _split_markdown_sections(self, markdown: str) -> list[tuple[str, int, str]]:
        """Split markdown by ATX headings."""
        sections: list[tuple[str, int, str]] = []
        current_heading = "Introduction"
        current_level = 1
        current_lines: list[str] = []

        for line in markdown.split("\n"):
            match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if match:
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_heading, current_level, body))
                current_level = len(match.group(1))
                current_heading = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, current_level, body))

        if not sections:
            sections.append((page_title_fallback, 1, markdown.strip()))
        return sections

    def _split_html_sections(self, html: str) -> list[tuple[str, int, str]]:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()

        sections: list[tuple[str, int, str]] = []
        current_heading = "Introduction"
        current_level = 1
        current_parts: list[str] = []

        def flush() -> None:
            nonlocal current_parts
            body = "\n\n".join(p for p in current_parts if p.strip())
            if body.strip():
                sections.append((current_heading, current_level, body.strip()))
            current_parts = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "table"]):
            if element.name in self.HEADING_TAGS:
                flush()
                current_heading = element.get_text(strip=True)
                current_level = int(element.name[1])
            else:
                text = self._element_text(element)
                if text:
                    current_parts.append(text)

        flush()
        if not sections:
            text = soup.get_text(separator="\n", strip=True)
            if text:
                sections.append(("Introduction", 1, text))
        return sections

    def _element_text(self, element: Tag) -> str:
        if element.name == "li":
            return f"- {element.get_text(strip=True)}"
        if element.name == "pre":
            return f"```\n{element.get_text()}\n```"
        return element.get_text(separator=" ", strip=True)

    def _html_to_text(self, html: str) -> str:
        return "\n\n".join(body for _, _, body in self._split_html_sections(html))

    def _split_oversized(self, body: str, prefix: str) -> Iterator[str]:
        """Split long sections at paragraph boundaries with token overlap."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        if not paragraphs:
            yield f"{prefix}\n\n{body}".strip()
            return

        current_parts: list[str] = []
        current_tokens = self.count_tokens(prefix)

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            if current_tokens + para_tokens > CHUNK_TARGET_TOKENS and current_parts:
                text = f"{prefix}\n\n" + "\n\n".join(current_parts)
                yield text.strip()
                overlap = self._tail_overlap(current_parts)
                current_parts = overlap + [para]
                current_tokens = self.count_tokens(prefix) + self.count_tokens("\n\n".join(current_parts))
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            yield f"{prefix}\n\n" + "\n\n".join(current_parts).strip()

    def _tail_overlap(self, parts: list[str]) -> list[str]:
        """Keep trailing paragraphs that fit within overlap token budget."""
        overlap: list[str] = []
        tokens = 0
        for part in reversed(parts):
            pt = self.count_tokens(part)
            if tokens + pt > CHUNK_OVERLAP_TOKENS:
                break
            overlap.insert(0, part)
            tokens += pt
        return overlap

    def _merge_small_chunks(
        self, chunks: list[tuple[str, int, str]]
    ) -> list[tuple[str, int, str]]:
        """Merge consecutive undersized chunks at the same heading level."""
        if not chunks:
            return []

        merged: list[tuple[str, int, str]] = []
        buf_heading, buf_level, buf_text = chunks[0]

        for heading, level, text in chunks[1:]:
            combined_tokens = self.count_tokens(buf_text) + self.count_tokens(text)
            if (
                self.count_tokens(buf_text) < CHUNK_MIN_TOKENS
                and level == buf_level
                and combined_tokens <= CHUNK_MAX_TOKENS
            ):
                buf_text = f"{buf_text}\n\n{text}"
            else:
                merged.append((buf_heading, buf_level, buf_text))
                buf_heading, buf_level, buf_text = heading, level, text

        merged.append((buf_heading, buf_level, buf_text))
        return merged
