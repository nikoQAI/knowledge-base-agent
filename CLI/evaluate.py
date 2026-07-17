#!/usr/bin/env python3
"""Retrieval quality evaluation against a ground-truth test set."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from config import HYBRID_SEARCH
from embedder import Embedder
from store import VectorStore

console = Console()

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
TEST_QUESTIONS_PATH = EVAL_DIR / "test_questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def check_answer_match(expected: str, retrieved_contents: list[str]) -> bool:
    """Check if expected answer keywords appear in any retrieved chunk."""
    expected_norm = normalize(expected)
    keywords = [w for w in expected_norm.split() if len(w) > 3]
    if not keywords:
        return expected_norm in normalize(" ".join(retrieved_contents))

    for content in retrieved_contents:
        content_norm = normalize(content)
        matches = sum(1 for kw in keywords if kw in content_norm)
        if matches / len(keywords) >= 0.6:
            return True
    return False


def check_page_match(expected_page: str, results: list[dict]) -> bool:
    expected_norm = normalize(expected_page)
    for hit in results:
        if expected_norm in normalize(hit.get("page_title", "")):
            return True
        if expected_norm in normalize(hit.get("breadcrumb", "")):
            return True
    return False


@click.command()
@click.option("--top-k", default=5, help="Number of chunks to retrieve per query.")
@click.option("--output", default=None, help="Output JSON path for detailed results.")
def main(top_k: int, output: str | None) -> None:
    """Evaluate retrieval quality on the test question set."""
    store = VectorStore()
    try:
        store.initialize()
        stats = store.stats()
    except Exception as exc:
        console.print(f"[red]Database error:[/red] {exc}")
        sys.exit(1)

    if stats["total_chunks"] == 0:
        console.print("[red]No indexed content. Run ingest.py first.[/red]")
        sys.exit(1)

    with TEST_QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)

    embedder = Embedder()

    page_hits = 0
    answer_hits = 0
    mrr_sum = 0.0
    detailed: list[dict] = []

    for q in questions:
        query = q["question"]
        embedding = embedder.embed_query(query)
        results = store.search(
            embedding,
            top_k=top_k,
            query_text=query,
            hybrid=HYBRID_SEARCH,
        )

        page_match = check_page_match(q["expected_page"], results)
        contents = [r["content"] for r in results]
        answer_match = check_answer_match(q["expected_answer"], contents)

        reciprocal_rank = 0.0
        expected_page_norm = normalize(q["expected_page"])
        for rank, hit in enumerate(results, 1):
            if expected_page_norm in normalize(hit.get("page_title", "")):
                reciprocal_rank = 1.0 / rank
                break

        if page_match:
            page_hits += 1
        if answer_match:
            answer_hits += 1
        mrr_sum += reciprocal_rank

        detailed.append({
            "id": q["id"],
            "question": query,
            "category": q.get("category", ""),
            "expected_page": q["expected_page"],
            "expected_answer": q["expected_answer"],
            "page_hit": page_match,
            "answer_hit": answer_match,
            "reciprocal_rank": reciprocal_rank,
            "top_result": {
                "page_title": results[0]["page_title"] if results else None,
                "section": results[0].get("section_heading") if results else None,
                "similarity": results[0]["similarity"] if results else 0,
            },
        })

    n = len(questions)
    page_recall = page_hits / n
    answer_recall = answer_hits / n
    mrr = mrr_sum / n

    table = Table(title=f"Retrieval Evaluation ({n} questions, top_k={top_k})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Page Recall@K", f"{page_recall:.1%} ({page_hits}/{n})")
    table.add_row("Answer Recall@K", f"{answer_recall:.1%} ({answer_hits}/{n})")
    table.add_row("MRR", f"{mrr:.3f}")
    console.print(table)

    failures = [d for d in detailed if not d["page_hit"]]
    if failures:
        console.print(f"\n[yellow]Missed page retrievals ({len(failures)}):[/yellow]")
        fail_table = Table(show_header=True)
        fail_table.add_column("ID")
        fail_table.add_column("Question")
        fail_table.add_column("Expected Page")
        fail_table.add_column("Got")
        for f in failures:
            fail_table.add_row(
                f["id"],
                f["question"][:60] + "…" if len(f["question"]) > 60 else f["question"],
                f["expected_page"],
                f["top_result"]["page_title"] or "—",
            )
        console.print(fail_table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "total_questions": n,
        "page_recall_at_k": page_recall,
        "answer_recall_at_k": answer_recall,
        "mrr": mrr,
        "details": detailed,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[dim]Detailed results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
