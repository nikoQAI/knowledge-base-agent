#!/usr/bin/env python3
"""Re-embed existing chunks in pgvector (no BookStack fetch)."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from embedder import Embedder
from store import VectorStore

console = Console()

PAGES_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "bookstack_pages.json"


@click.command()
@click.option(
    "--from-cache",
    is_flag=True,
    help="Chunk cached BookStack pages and embed (skip API + skip DB read).",
)
@click.option("--clear", is_flag=True, help="Clear existing chunks before re-embed.")
def main(from_cache: bool, clear: bool) -> None:
    """Re-generate embeddings for stored chunks or cached pages."""
    store = VectorStore()
    console.print("[cyan]Initializing database...[/cyan]")
    store.initialize()

    if clear:
        console.print("[yellow]Clearing existing chunks...[/yellow]")
        store.clear()

    embedder = Embedder()
    console.print(
        f"[dim]Model: {embedder.model} | Device: {embedder.device} | "
        f"Batch size: {embedder.batch_size}[/dim]"
    )

    if from_cache:
        from bookstack_client import load_pages_cache
        from chunker import StructuredChunker

        if not PAGES_CACHE.exists():
            console.print(
                f"[red]No page cache at {PAGES_CACHE}[/red]\n"
                "Run [bold]python ingest.py --source api[/bold] once to create it."
            )
            sys.exit(1)

        console.print(f"[cyan]Loading cached pages from {PAGES_CACHE}...[/cyan]")
        pages = load_pages_cache(PAGES_CACHE)
        console.print(f"[green]Loaded {len(pages)} pages[/green]")

        chunker = StructuredChunker()
        chunks = chunker.chunk_pages(pages)
        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        if not chunks:
            console.print("[yellow]No chunks to embed.[/yellow]")
            return

        console.print("[cyan]Storing chunk content...[/cyan]")
        store.upsert_chunks_without_embeddings(chunks)
    else:
        chunks = store.load_all_chunks()
        if not chunks:
            console.print(
                "[yellow]No chunks in database.[/yellow] Options:\n"
                "  python reembed.py --from-cache   (embed from saved page cache)\n"
                "  python ingest.py --source api  (full ingest)"
            )
            sys.exit(1)
        console.print(f"[green]Loaded {len(chunks)} chunks from database[/green]")

    console.print("[cyan]Generating embeddings...[/cyan]")
    embeddings = embedder.embed_chunks(chunks, show_progress=True)
    console.print(
        f"[dim]Embedding tokens: {embedder.total_tokens:,} "
        f"(est. ${embedder.estimated_cost:.4f})[/dim]"
    )

    console.print("[cyan]Updating embeddings in pgvector...[/cyan]")
    chunk_ids = [c.chunk_id for c in chunks]
    store.update_embeddings(chunk_ids, embeddings)
    stats = store.stats()

    table = Table(title="Re-embed Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Chunks embedded", str(len(chunks)))
    table.add_row("Total chunks in DB", str(stats["total_chunks"]))
    table.add_row("Total pages in DB", str(stats["total_pages"]))
    table.add_row("Device", embedder.device)
    console.print(table)


if __name__ == "__main__":
    main()
