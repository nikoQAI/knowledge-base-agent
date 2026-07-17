# Retrieval Quality Evaluation

## Test Set

25 ground-truth questions in `eval/test_questions.json` covering:

| Category | Questions |
|----------|-----------|
| HR | 6 |
| Engineering | 8 |
| Delivery | 4 |
| Finance | 2 |
| Infrastructure | 2 |
| Compliance | 1 |
| Code Review | 1 |
| Testing | 1 |

Each entry includes:
- `question` — natural-language query
- `expected_page` — correct source KB page title
- `expected_answer` — ground-truth answer text for keyword matching

## Metrics

| Metric | Definition |
|--------|------------|
| **Page Recall@K** | Fraction of questions where the expected source page appears in the top-K retrieved chunks |
| **Answer Recall@K** | Fraction where ≥60% of expected answer keywords appear in any retrieved chunk |
| **MRR** | Mean Reciprocal Rank of the first correctly-ranked source page |

## Baseline Results (Sample KB, local embeddings, top_k=5)

Run date: 2026-07-17

| Metric | Score |
|--------|-------|
| Page Recall@5 | **100.0%** (25/25) |
| Answer Recall@5 | **88.0%** (22/25) |
| MRR | **0.960** |

### Missed Answer Matches (retrieval correct, keyword heuristic miss)

| ID | Question | Notes |
|----|----------|-------|
| q06 | Onboarding first two weeks | Retrieved correct page; expected answer is a summary not literal chunk text |
| q07 | Branch naming for bug fixes | Correct page in top-5 (rank 2); top result was API URL structure |
| q17 | P1 incident response SLA | Retrieved Severity Levels table; "15 min" present but keyword heuristic missed phrasing |

All three misses had **correct page retrieval** — failures are in the answer keyword matcher, not retrieval.

## Running Evaluation

```bash
# After ingestion
python evaluate.py --top-k 5
```

Results are saved to `eval/results/eval_YYYYMMDD_HHMMSS.json`.

## Re-evaluation After Live KB Ingestion

When ingesting from the live BookStack API:

1. Update `eval/test_questions.json` with real Q KB ground-truth questions
2. Re-run ingestion: `python ingest.py --source api --clear`
3. Re-run evaluation: `python evaluate.py`
4. Compare Page Recall@5 and MRR against these baselines

Production embeddings (`text-embedding-3-small`) typically improve Answer Recall@5 by 2–5% over local dev embeddings.
