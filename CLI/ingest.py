#!/usr/bin/env python3
"""Ingest knowledge base pages into pgvector."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from config import BOOKSTACK_REQUEST_DELAY
from bookstack_client import BookStackClient, load_local_export
from chunker import StructuredChunker
from embedder import Embedder
from store import VectorStore

console = Console()


@click.command()
@click.option(
    "--source",
    type=click.Choice(["api", "local"], case_sensitive=False),
    default="local",
    help="Ingest from BookStack API or local JSON export.",
)
@click.option(
    "--export-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to local export directory (default: data/sample_kb).",
)
@click.option("--clear", is_flag=True, help="Clear existing chunks before ingest.")
def main(source: str, export_dir: Path | None, clear: bool) -> None:
    """Fetch KB pages, chunk, embed, and store in pgvector."""
    store = VectorStore()
    console.print("[cyan]Initializing database...[/cyan]")
    store.initialize()

    if clear:
        console.print("[yellow]Clearing existing chunks...[/yellow]")
        store.clear()

    if source == "api":
        client = BookStackClient()
        if not client.is_configured():
            console.print(
                "[red]BookStack API credentials not configured.[/red]\n"
                "Set BOOKSTACK_TOKEN_ID and BOOKSTACK_TOKEN_SECRET in .env.\n"
                "See README.md for setup instructions."
            )
            sys.exit(1)
        console.print(
            f"[cyan]Fetching pages from BookStack API "
            f"({BOOKSTACK_REQUEST_DELAY}s delay between requests)...[/cyan]"
        )
        try:
            probe = client.test_connection()
            console.print(f"[dim]Found {probe.get('total', '?')} pages in BookStack[/dim]")

            def _progress(current: int, total: int, title: str) -> None:
                if current == 1 or current == total or current % 25 == 0:
                    console.print(f"[dim]  {current}/{total}: {title[:60]}[/dim]")

            pages = client.fetch_all_pages(on_progress=_progress)
            if len(pages) < int(probe.get("total", 0)):
                console.print(
                    f"[yellow]Warning: fetched {len(pages)}/{probe.get('total')} pages. "
                    "Some were skipped due to rate limits — increase "
                    "BOOKSTACK_REQUEST_DELAY and re-run.[/yellow]"
                )
        except Exception as exc:
            console.print(f"[red]API error:[/red] {exc}")
            sys.exit(1)
    else:
        export_dir = export_dir or Path(__file__).resolve().parent.parent / "data" / "sample_kb"
        console.print(f"[cyan]Loading local export from {export_dir}...[/cyan]")
        pages = load_local_export(export_dir)

    console.print(f"[green]Loaded {len(pages)} pages[/green]")

    chunker = StructuredChunker()
    chunks = chunker.chunk_pages(pages)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    if not chunks:
        console.print("[yellow]No chunks to ingest.[/yellow]")
        return

    console.print("[cyan]Generating embeddings...[/cyan]")
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    console.print(
        f"[dim]Embedding tokens: {embedder.total_tokens:,} "
        f"(est. ${embedder.estimated_cost:.4f})[/dim]"
    )

    console.print("[cyan]Storing in pgvector...[/cyan]")
    count = store.upsert_chunks(chunks, embeddings)
    stats = store.stats()

    table = Table(title="Ingestion Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Pages ingested", str(len(pages)))
    table.add_row("Chunks stored", str(count))
    table.add_row("Total chunks in DB", str(stats["total_chunks"]))
    table.add_row("Total pages in DB", str(stats["total_pages"]))
    console.print(table)


if __name__ == "__main__":
    main()
