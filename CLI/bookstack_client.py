"""BookStack REST API client for fetching KB pages."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from config import (
    BOOKSTACK_BASE_URL,
    BOOKSTACK_REQUEST_DELAY,
    BOOKSTACK_TOKEN_ID,
    BOOKSTACK_TOKEN_SECRET,
)

_RETRYABLE_STATUS = {202, 429, 500, 502, 503, 504}
_MAX_RETRIES = 6


class BookStackAPIError(Exception):
    """Raised when the BookStack API returns an unexpected response."""


@dataclass
class KBPage:
    """A knowledge base page with hierarchy metadata."""

    page_id: int
    title: str
    html: str
    markdown: str
    book_id: int
    book_name: str
    chapter_id: int | None
    chapter_name: str | None
    shelf_names: list[str]
    url: str
    updated_at: str

    @property
    def breadcrumb(self) -> str:
        parts = self.shelf_names + [self.book_name]
        if self.chapter_name:
            parts.append(self.chapter_name)
        parts.append(self.title)
        return " > ".join(parts)


class BookStackClient:
    """Client for the BookStack REST API."""

    def __init__(
        self,
        base_url: str = BOOKSTACK_BASE_URL,
        token_id: str = BOOKSTACK_TOKEN_ID,
        token_secret: str = BOOKSTACK_TOKEN_SECRET,
        request_delay: float = BOOKSTACK_REQUEST_DELAY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api"
        self.token_id = token_id
        self.token_secret = token_secret
        self.request_delay = request_delay
        self._last_request_at = 0.0
        self._client = httpx.Client(
            headers={
                "Authorization": f"Token {token_id}:{token_secret}",
                "Accept": "application/json",
            },
            timeout=60.0,
            follow_redirects=True,
        )

    def is_configured(self) -> bool:
        return bool(self.token_id and self.token_secret)

    def _throttle(self) -> None:
        if self.request_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """GET with throttling and retry for rate-limited responses."""
        url = path if path.startswith("http") else f"{self.api_url}/{path.lstrip('/')}"
        last_resp: httpx.Response | None = None

        for attempt in range(_MAX_RETRIES):
            self._throttle()
            last_resp = self._client.get(url, params=params)
            self._last_request_at = time.monotonic()

            if last_resp.status_code == 200 and last_resp.text.strip():
                try:
                    return last_resp.json()
                except json.JSONDecodeError as exc:
                    raise BookStackAPIError(
                        f"Non-JSON response from {url} "
                        f"(content-type: {last_resp.headers.get('content-type')})"
                    ) from exc

            if last_resp.status_code in _RETRYABLE_STATUS or not last_resp.text.strip():
                time.sleep(min(30.0, (2**attempt) * 0.75))
                continue

            last_resp.raise_for_status()

        status = last_resp.status_code if last_resp else "unknown"
        raise BookStackAPIError(
            f"BookStack rate-limited or unavailable for {url} "
            f"(status {status} after {_MAX_RETRIES} retries). "
            "Try again in a minute or increase BOOKSTACK_REQUEST_DELAY in .env."
        )

    def test_connection(self) -> dict[str, Any]:
        """Verify API credentials by listing one page."""
        return self._get_json("pages", params={"count": 1})

    def _paginate(self, endpoint: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Iterate all items from a paginated BookStack endpoint."""
        offset = 0
        count = 100
        base_params = dict(params or {})
        while True:
            payload = self._get_json(
                endpoint,
                params={**base_params, "count": count, "offset": offset},
            )
            items = payload.get("data", [])
            if not items:
                break
            yield from items
            total = payload.get("total", 0)
            offset += count
            if offset >= total:
                break

    def fetch_all_pages(
        self,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[KBPage]:
        """Fetch all pages with full content and hierarchy metadata."""
        books = {b["id"]: b["name"] for b in self._paginate("books")}
        chapters = {
            c["id"]: {"name": c["name"], "book_id": c["book_id"]}
            for c in self._paginate("chapters")
        }
        shelf_map: dict[int, list[str]] = {}
        for shelf in self._paginate("shelves"):
            for book_id in shelf.get("books", []):
                shelf_map.setdefault(book_id, []).append(shelf["name"])

        summaries = list(self._paginate("pages"))
        total = len(summaries)
        pages: list[KBPage] = []
        skipped: list[str] = []

        for index, summary in enumerate(summaries, 1):
            page_id = summary["id"]
            title = summary.get("name", "Untitled")
            if on_progress:
                on_progress(index, total, title)

            try:
                detail = self._get_json(f"pages/{page_id}")
            except BookStackAPIError:
                skipped.append(f"{page_id} ({title})")
                continue

            book_id = detail.get("book_id", summary.get("book_id", 0))
            chapter_id = detail.get("chapter_id") or summary.get("chapter_id")
            chapter_info = chapters.get(chapter_id) if chapter_id else None

            pages.append(
                KBPage(
                    page_id=page_id,
                    title=detail.get("name", summary.get("name", "Untitled")),
                    html=detail.get("html", ""),
                    markdown=detail.get("markdown", ""),
                    book_id=book_id,
                    book_name=books.get(book_id, "Unknown Book"),
                    chapter_id=chapter_id,
                    chapter_name=chapter_info["name"] if chapter_info else None,
                    shelf_names=shelf_map.get(book_id, []),
                    url=f"{self.base_url}/books/{book_id}/page/{page_id}",
                    updated_at=detail.get("updated_at", summary.get("updated_at", "")),
                )
            )

        if skipped:
            import warnings

            warnings.warn(
                f"Skipped {len(skipped)} page(s) after retries: "
                f"{', '.join(skipped[:5])}"
                + (" …" if len(skipped) > 5 else "")
            )

        return pages


def page_to_dict(page: KBPage) -> dict[str, Any]:
    return {
        "page_id": page.page_id,
        "title": page.title,
        "html": page.html,
        "markdown": page.markdown,
        "book_id": page.book_id,
        "book_name": page.book_name,
        "chapter_id": page.chapter_id,
        "chapter_name": page.chapter_name,
        "shelf_names": page.shelf_names,
        "url": page.url,
        "updated_at": page.updated_at,
    }


def page_from_dict(item: dict[str, Any]) -> KBPage:
    return KBPage(
        page_id=item["page_id"],
        title=item["title"],
        html=item.get("html", ""),
        markdown=item.get("markdown", ""),
        book_id=item.get("book_id", 0),
        book_name=item.get("book_name", "Unknown"),
        chapter_id=item.get("chapter_id"),
        chapter_name=item.get("chapter_name"),
        shelf_names=item.get("shelf_names", []),
        url=item.get("url", ""),
        updated_at=item.get("updated_at", ""),
    )


def save_pages_cache(pages: list[KBPage], path: Path) -> None:
    """Persist fetched BookStack pages for fast re-ingest without API calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([page_to_dict(page) for page in pages], f)


def load_pages_cache(path: Path) -> list[KBPage]:
    """Load pages previously saved by save_pages_cache."""
    if not path.exists():
        raise FileNotFoundError(f"No page cache found at {path}")
    with path.open(encoding="utf-8") as f:
        raw_pages = json.load(f)
    return [page_from_dict(item) for item in raw_pages]


def load_local_export(export_dir: Path) -> list[KBPage]:
    """Load pages from a local JSON export (BookStack format or our sample format)."""
    pages: list[KBPage] = []
    manifest = export_dir / "pages.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No pages.json found in {export_dir}")

    with manifest.open(encoding="utf-8") as f:
        raw_pages = json.load(f)

    for item in raw_pages:
        pages.append(
            page_from_dict(item)
        )
    return pages
