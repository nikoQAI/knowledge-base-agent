#!/usr/bin/env python3
"""Batch-run the RAG agent on test questions and export CSV + Markdown."""

import csv
import json
import sys
from pathlib import Path

# Allow imports from CLI/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CLI"))

from chat import KnowledgeBaseChat  # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "test_questions.json"
OUT_DIR = EVAL_DIR / "results"


def main(limit: int | None = None, fmt: str = "both") -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]

    chat = KnowledgeBaseChat()
    chat.store.initialize()

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        result = chat.answer(q["question"])
        top_source = result["sources"][0]["page_title"] if result["sources"] else ""
        rows.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "expected_page": q["expected_page"],
            "actual_answer": result["answer"].strip(),
            "top_source_page": top_source,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"agent_answers_{len(rows)}q"

    if fmt in ("csv", "both"):
        csv_path = OUT_DIR / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {csv_path}")

    if fmt in ("md", "both"):
        md_path = OUT_DIR / f"{stem}.md"
        lines = [
            f"# Agent evaluation ({len(rows)} questions)\n",
            "| ID | Category | Question | Expected Answer | Actual Answer | Expected Page | Top Source |",
            "|----|----------|----------|-----------------|---------------|---------------|------------|",
        ]
        for r in rows:
            def esc(s: str) -> str:
                return s.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {r['id']} | {r['category']} | {esc(r['question'])} | "
                f"{esc(r['expected_answer'])} | {esc(r['actual_answer'])} | "
                f"{r['expected_page']} | {r['top_source_page']} |"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Markdown: {md_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20, help="Number of questions (default: 20)")
    p.add_argument("--format", choices=["csv", "md", "both"], default="both")
    args = p.parse_args()
    main(limit=args.limit, fmt=args.format)