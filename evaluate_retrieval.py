#!/usr/bin/env python3
"""Evaluate ColQwen2 retrieval at document level on a labeled query set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("retrieval_eval_queries.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_validation_512.pt")
    )
    parser.add_argument("--model", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/retrieval_evaluation.json")
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network checks; default is cache-only loading.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_id", "domain", "query", "expected_pdf"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Query file must contain columns: {sorted(required)}")
    return rows


def summarize(ranks: list[int]) -> dict:
    count = len(ranks)
    return {
        "queries": count,
        "recall_at_1": round(sum(rank <= 1 for rank in ranks) / count, 4),
        "recall_at_3": round(sum(rank <= 3 for rank in ranks) / count, 4),
        "recall_at_5": round(sum(rank <= 5 for rank in ranks) / count, 4),
        "mrr": round(sum(1.0 / rank for rank in ranks) / count, 4),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    records = read_jsonl(args.metadata)
    queries = read_queries(args.queries)
    corpus_documents = {record["pdf_filename"] for record in records}
    missing_labels = sorted(
        {row["expected_pdf"] for row in queries} - corpus_documents
    )
    if missing_labels:
        raise RuntimeError(f"Expected PDFs are absent from the corpus: {missing_labels}")

    payload = torch.load(args.index, map_location="cpu", weights_only=True)
    expected_page_ids = [record["page_id"] for record in records]
    if payload.get("model_name") != args.model:
        raise RuntimeError("Model name does not match the cached index.")
    if payload.get("page_ids") != expected_page_ids:
        raise RuntimeError("Metadata does not match the cached index.")

    print(
        f"Evaluating {len(queries)} queries over "
        f"{len(records)} pages / {len(corpus_documents)} PDFs"
    )
    started = time.perf_counter()
    local_only = not args.allow_network
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=local_only,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(
        args.model,
        local_files_only=local_only,
    )

    query_inputs = processor.process_queries(
        [row["query"] for row in queries]
    ).to(model.device)
    page_embeddings = [embedding.to(model.device) for embedding in payload["embeddings"]]
    with torch.inference_mode():
        query_embeddings = model(**query_inputs).embeddings
        page_scores = processor.score_retrieval(
            query_embeddings,
            page_embeddings,
            output_dtype=torch.float32,
        ).cpu()

    results = []
    for query_index, row in enumerate(queries):
        document_scores: dict[str, float] = defaultdict(lambda: float("-inf"))
        best_pages: dict[str, tuple[int, float]] = {}
        for page_index, record in enumerate(records):
            score = float(page_scores[query_index, page_index])
            document = record["pdf_filename"]
            if score > document_scores[document]:
                document_scores[document] = score
                best_pages[document] = (record["page_number"], score)

        ranked_documents = sorted(
            document_scores,
            key=document_scores.get,
            reverse=True,
        )
        expected = row["expected_pdf"]
        expected_rank = ranked_documents.index(expected) + 1
        top_documents = []
        for rank, document in enumerate(ranked_documents[:5], start=1):
            page_number, score = best_pages[document]
            top_documents.append(
                {
                    "rank": rank,
                    "pdf_filename": document,
                    "best_page": page_number,
                    "score": round(score, 4),
                }
            )

        passed = expected_rank == 1
        print(
            f"{row['query_id']}  expected_rank={expected_rank:<2} "
            f"top1={ranked_documents[0]}  {'PASS' if passed else 'CHECK'}"
        )
        results.append(
            {
                **row,
                "expected_rank": expected_rank,
                "reciprocal_rank": round(1.0 / expected_rank, 6),
                "top_documents": top_documents,
            }
        )

    all_ranks = [result["expected_rank"] for result in results]
    metrics = {
        **summarize(all_ranks),
        "pages": len(records),
        "documents": len(corpus_documents),
        "elapsed_seconds_including_model_load": round(time.perf_counter() - started, 2),
    }

    by_difficulty = {}
    for difficulty in sorted({row.get("difficulty", "unspecified") for row in results}):
        ranks = [
            row["expected_rank"]
            for row in results
            if row.get("difficulty", "unspecified") == difficulty
        ]
        by_difficulty[difficulty] = summarize(ranks)

    by_domain = {}
    for domain in sorted({row["domain"] for row in results}):
        ranks = [
            row["expected_rank"] for row in results if row["domain"] == domain
        ]
        by_domain[domain] = summarize(ranks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "metrics_by_difficulty": by_difficulty,
                "metrics_by_domain": by_domain,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nMetrics")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print("\nMetrics by difficulty")
    for difficulty, values in by_difficulty.items():
        print(f"  {difficulty}: {values}")
    print("\nMetrics by domain")
    for domain, values in by_domain.items():
        print(f"  {domain}: {values}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
