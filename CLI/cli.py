#!/usr/bin/env python3
"""Interactive CLI for Q Knowledge Base Q&A."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from chat import KnowledgeBaseChat
from store import VectorStore

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

console = Console()
error_console = Console(stderr=True)

EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


def print_welcome(stats: dict) -> None:
    banner = (
        "[bold cyan]Q Knowledge Base Agent[/bold cyan]\n"
        "[dim]Ask questions about Q's internal processes, policies, and technical standards[/dim]"
    )
    console.print()
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
    console.print(
        f"[dim]Indexed: {stats['total_pages']} pages, "
        f"{stats['total_chunks']} chunks[/dim]\n"
    )
    console.print("[dim]Type 'exit', 'quit', or Ctrl+C to leave.[/dim]\n")


def print_sources(sources: list) -> None:
    if not sources:
        return
    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Page", style="green")
    table.add_column("Section")
    table.add_column("Score", justify="right")

    for i, src in enumerate(sources, 1):
        table.add_row(
            str(i),
            src["page_title"],
            src.get("section_heading", ""),
            f"{src['similarity']:.3f}",
        )
    console.print()
    console.print(table)


def main() -> None:
    store = VectorStore()
    try:
        store.initialize()
        stats = store.stats()
    except Exception as exc:
        error_console.print(f"[red]Database error:[/red] {exc}")
        error_console.print(
            "\nEnsure PostgreSQL with pgvector is running.\n"
            "Run: docker compose up -d   (or see README.md)"
        )
        sys.exit(1)

    if stats["total_chunks"] == 0:
        error_console.print(
            "[yellow]No indexed content found.[/yellow] Run ingestion first:\n"
            "  python ingest.py --source local\n"
            "  python ingest.py --source api   (requires BookStack API token)"
        )
        sys.exit(1)

    print_welcome(stats)

    try:
        chat = KnowledgeBaseChat(store=store)
    except EnvironmentError as exc:
        error_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            console.print("[dim]Goodbye![/dim]")
            break

        turn += 1
        console.print()
        console.print("[bold green]Assistant[/bold green]")

        try:
            results = chat.retrieve(query)
            sources = results

            with console.status("[dim]Generating answer...[/dim]"):
                response_text = ""
                for token in chat.answer_stream(query):
                    if isinstance(token, dict) and "__sources__" in token:
                        sources = token["__sources__"]
                        continue
                    response_text += token
                    console.print(token, end="")

            console.print()
            print_sources(sources)

        except KeyboardInterrupt:
            console.print("\n[dim]Response cancelled.[/dim]")
        except Exception as exc:
            error_console.print(f"[red]Error:[/red] {exc}")

        console.print()


if __name__ == "__main__":
    main()
