#!/usr/bin/env python3
"""Evaluate a PDF text-layer BM25 baseline with strict page-level labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import fitz


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "which", "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", type=Path, default=Path("page_retrieval_queries_v1_1.csv")
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument(
        "--text-cache", type=Path, default=Path("data/index/page_text_cache.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/bm25_page_retrieval_evaluation.json")
    )
    parser.add_argument("--baseline-name", default="pdf_text_layer_bm25")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--rebuild-cache", action="store_true")
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
    for row in rows:
        row["gold_pages"] = {
            int(value.strip()) for value in row["expected_pages"].split(";")
        }
    return rows


def tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    latin = [
        token
        for token in re.findall(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", lowered)
        if token not in ENGLISH_STOPWORDS
    ]
    cjk_tokens = []
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered):
        characters = list(sequence)
        cjk_tokens.extend(characters)
        cjk_tokens.extend(
            characters[index] + characters[index + 1]
            for index in range(len(characters) - 1)
        )
    return latin + cjk_tokens


def extract_texts(records: list[dict], pdf_dir: Path) -> list[str]:
    texts = [""] * len(records)
    record_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        record_indices[record["pdf_filename"]].append(index)

    processed = 0
    for pdf_filename, indices in sorted(record_indices.items()):
        pdf_path = pdf_dir / pdf_filename
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        with fitz.open(pdf_path) as document:
            for index in indices:
                page_number = int(records[index]["page_number"])
                texts[index] = document[page_number - 1].get_text("text", sort=True)
                processed += 1
                if processed % 250 == 0 or processed == len(records):
                    print(f"Extracted text from {processed}/{len(records)} pages")
    return texts


def load_or_build_text_cache(args: argparse.Namespace, records: list[dict]) -> list[str]:
    expected_ids = [record["page_id"] for record in records]
    if args.text_cache.exists() and not args.rebuild_cache:
        cached = read_jsonl(args.text_cache)
        cached_ids = [row["page_id"] for row in cached]
        if cached_ids != expected_ids:
            raise RuntimeError(
                "Text cache does not match metadata; rerun with --rebuild-cache."
            )
        print(f"Loaded text cache: {args.text_cache}")
        return [row["text"] for row in cached]

    texts = extract_texts(records, args.pdf_dir)
    args.text_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.text_cache.open("w", encoding="utf-8") as handle:
        for page_id, text in zip(expected_ids, texts):
            handle.write(json.dumps({"page_id": page_id, "text": text}, ensure_ascii=False) + "\n")
    print(f"Saved text cache: {args.text_cache}")
    return texts


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
    if args.k1 <= 0 or not 0 <= args.b <= 1:
        raise ValueError("Require k1 > 0 and 0 <= b <= 1")

    started = time.perf_counter()
    records = read_jsonl(args.metadata)
    queries = read_queries(args.queries)
    page_lookup = {
        (record["pdf_filename"], int(record["page_number"])): index
        for index, record in enumerate(records)
    }
    missing_gold = [
        f"{row['query_id']}:{row['expected_pdf']}:p{page_number}"
        for row in queries
        for page_number in row["gold_pages"]
        if (row["expected_pdf"], page_number) not in page_lookup
    ]
    if missing_gold:
        raise RuntimeError(f"Gold pages are absent from metadata: {missing_gold}")

    texts = load_or_build_text_cache(args, records)
    tokenized_pages = [tokenize(text) for text in texts]
    term_frequencies = [Counter(tokens) for tokens in tokenized_pages]
    document_lengths = [len(tokens) for tokens in tokenized_pages]
    average_length = sum(document_lengths) / max(1, len(document_lengths))
    document_frequency = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())

    corpus_size = len(records)
    idf = {
        term: math.log(1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    print(
        f"Evaluating {args.baseline_name} on {len(queries)} queries over "
        f"{corpus_size} pages / "
        f"{len({record['pdf_filename'] for record in records})} PDFs"
    )

    results = []
    for row in queries:
        query_terms = list(dict.fromkeys(tokenize(row["query"])))
        scores = [0.0] * corpus_size
        for page_index, frequencies in enumerate(term_frequencies):
            length = document_lengths[page_index]
            normalization = args.k1 * (
                1.0 - args.b + args.b * length / max(average_length, 1.0)
            )
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency:
                    score += idf[term] * (
                        term_frequency * (args.k1 + 1.0)
                        / (term_frequency + normalization)
                    )
            scores[page_index] = score

        ranked_indices = sorted(range(corpus_size), key=scores.__getitem__, reverse=True)
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
                    "score": round(scores[page_index], 6),
                }
            )
        top1 = top_pages[0]
        print(
            f"{row['query_id']}  gold_rank={gold_rank:<4} "
            f"top1={top1['pdf_filename']}:p{top1['page_number']}  "
            f"{'PASS' if gold_rank == 1 else 'CHECK'}"
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

    ranks = [result["gold_page_rank"] for result in results]
    text_nonempty_rate = round(
        sum(bool(text.strip()) for text in texts) / corpus_size, 4
    )
    metrics = {
        **summarize(ranks),
        "pages": corpus_size,
        "documents": len({record["pdf_filename"] for record in records}),
        "text_nonempty_rate": text_nonempty_rate,
        "k1": args.k1,
        "b": args.b,
        "elapsed_seconds_including_text_cache": round(time.perf_counter() - started, 2),
    }
    if args.baseline_name == "pdf_text_layer_bm25":
        metrics["text_layer_nonempty_rate"] = text_nonempty_rate
    output = {
        "baseline": args.baseline_name,
        "metrics": metrics,
        "metrics_by_difficulty": grouped_metrics(results, "difficulty"),
        "metrics_by_domain": grouped_metrics(results, "domain"),
        "metrics_by_visual_type": grouped_metrics(results, "visual_type"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
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
