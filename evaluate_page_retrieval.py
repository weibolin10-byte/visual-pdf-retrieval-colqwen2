#!/usr/bin/env python3
"""Evaluate ColQwen2 retrieval against strict page-level gold labels."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("page_retrieval_queries.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_final_2312.pt")
    )
    parser.add_argument("--model", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/page_retrieval_evaluation.json")
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
    required = {"query_id", "domain", "query", "expected_pdf", "expected_pages"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Query file must contain columns: {sorted(required)}")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Query IDs must be unique.")
    for row in rows:
        try:
            pages = {int(value.strip()) for value in row["expected_pages"].split(";")}
        except ValueError as error:
            raise RuntimeError(
                f"Invalid expected_pages for {row['query_id']}: {row['expected_pages']}"
            ) from error
        if not pages or min(pages) < 1:
            raise RuntimeError(f"Expected pages must be positive for {row['query_id']}")
        row["gold_pages"] = pages
    return rows


def summarize(ranks: list[int]) -> dict:
    count = len(ranks)
    return {
        "queries": count,
        "page_recall_at_1": round(sum(rank <= 1 for rank in ranks) / count, 4),
        "page_recall_at_3": round(sum(rank <= 3 for rank in ranks) / count, 4),
        "page_recall_at_5": round(sum(rank <= 5 for rank in ranks) / count, 4),
        "page_recall_at_10": round(sum(rank <= 10 for rank in ranks) / count, 4),
        "page_mrr": round(sum(1.0 / rank for rank in ranks) / count, 4),
        "median_gold_page_rank": statistics.median(ranks),
    }


def grouped_metrics(results: list[dict], field: str) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for result in results:
        grouped[result.get(field, "unspecified")].append(result["gold_page_rank"])
    return {name: summarize(ranks) for name, ranks in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    records = read_jsonl(args.metadata)
    queries = read_queries(args.queries)
    page_lookup = {
        (record["pdf_filename"], int(record["page_number"])): index
        for index, record in enumerate(records)
    }
    missing_gold = []
    for row in queries:
        for page_number in row["gold_pages"]:
            key = (row["expected_pdf"], page_number)
            if key not in page_lookup:
                missing_gold.append(f"{row['query_id']}:{key[0]}:p{key[1]}")
    if missing_gold:
        raise RuntimeError(f"Gold pages are absent from metadata: {missing_gold}")

    payload = torch.load(args.index, map_location="cpu", weights_only=True)
    expected_page_ids = [record["page_id"] for record in records]
    if payload.get("model_name") != args.model:
        raise RuntimeError("Model name does not match the cached index.")
    if payload.get("page_ids") != expected_page_ids:
        raise RuntimeError("Metadata does not match the cached index.")

    print(
        f"Evaluating {len(queries)} page-level queries over "
        f"{len(records)} pages / {len({r['pdf_filename'] for r in records})} PDFs"
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

    query_inputs = processor.process_queries([row["query"] for row in queries]).to(
        model.device
    )
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
        ranked_indices = torch.argsort(page_scores[query_index], descending=True).tolist()
        gold_indices = {
            page_lookup[(row["expected_pdf"], page_number)]
            for page_number in row["gold_pages"]
        }
        gold_rank = next(
            rank
            for rank, page_index in enumerate(ranked_indices, start=1)
            if page_index in gold_indices
        )
        top_pages = []
        for rank, page_index in enumerate(ranked_indices[:10], start=1):
            record = records[page_index]
            top_pages.append(
                {
                    "rank": rank,
                    "pdf_filename": record["pdf_filename"],
                    "page_number": int(record["page_number"]),
                    "page_id": record["page_id"],
                    "score": round(float(page_scores[query_index, page_index]), 4),
                }
            )

        top1 = top_pages[0]
        status = "PASS" if gold_rank == 1 else "CHECK"
        print(
            f"{row['query_id']}  gold_rank={gold_rank:<4} "
            f"top1={top1['pdf_filename']}:p{top1['page_number']}  {status}"
        )
        serializable_row = {key: value for key, value in row.items() if key != "gold_pages"}
        results.append(
            {
                **serializable_row,
                "gold_pages": sorted(row["gold_pages"]),
                "gold_page_rank": gold_rank,
                "reciprocal_rank": round(1.0 / gold_rank, 6),
                "top_pages": top_pages,
            }
        )

    metrics = {
        **summarize([result["gold_page_rank"] for result in results]),
        "pages": len(records),
        "documents": len({record["pdf_filename"] for record in records}),
        "elapsed_seconds_including_model_load": round(time.perf_counter() - started, 2),
    }
    output = {
        "metrics": metrics,
        "metrics_by_difficulty": grouped_metrics(results, "difficulty"),
        "metrics_by_domain": grouped_metrics(results, "domain"),
        "metrics_by_visual_type": grouped_metrics(results, "visual_type"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMetrics")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print("\nMetrics by difficulty")
    for name, values in output["metrics_by_difficulty"].items():
        print(f"  {name}: {values}")
    print("\nMetrics by domain")
    for name, values in output["metrics_by_domain"].items():
        print(f"  {name}: {values}")
    print("\nMetrics by visual type")
    for name, values in output["metrics_by_visual_type"].items():
        print(f"  {name}: {values}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
