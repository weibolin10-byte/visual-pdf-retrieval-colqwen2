#!/usr/bin/env python3
"""Aggregate manually checked generation-quality scores with automatic metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path("outputs/generation_quality_raw_results.json"),
    )
    parser.add_argument(
        "--manual-scores",
        type=Path,
        default=Path("outputs/generation_quality_manual_scores.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/generation_quality_final_evaluation.json"),
    )
    return parser.parse_args()


def read_manual_scores(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Manual score CSV is empty.")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("Manual score CSV contains duplicate query IDs.")
    return {row["query_id"]: row for row in rows}


def parse_integer(value: str, field: str, query_id: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise RuntimeError(f"{query_id}: {field} must be an integer.") from error
    return number


def main() -> None:
    args = parse_args()
    payload = json.loads(args.raw_results.read_text(encoding="utf-8"))
    results = payload["results"]
    manual = read_manual_scores(args.manual_scores)

    expected_ids = {row["query_id"] for row in results}
    if set(manual) != expected_ids:
        missing = sorted(expected_ids - set(manual))
        extra = sorted(set(manual) - expected_ids)
        raise RuntimeError(f"Manual score IDs mismatch; missing={missing}, extra={extra}")

    answerable = [row for row in results if row["is_answerable"]]
    total_facts = 0
    correct_facts = 0
    completeness_scores = []
    page_support_scores = []
    checked_results = []

    for result in results:
        score = manual[result["query_id"]]
        checked = dict(result)
        checked["manual_notes"] = score["notes"].strip()
        if result["is_answerable"]:
            fact_total = len(result["required_fact_list"])
            fact_correct = parse_integer(
                score["correct_fact_count"], "correct_fact_count", result["query_id"]
            )
            completeness = parse_integer(
                score["completeness_score_0_to_2"],
                "completeness_score_0_to_2",
                result["query_id"],
            )
            page_support = parse_integer(
                score["page_support_score_0_to_2"],
                "page_support_score_0_to_2",
                result["query_id"],
            )
            if not 0 <= fact_correct <= fact_total:
                raise RuntimeError(
                    f"{result['query_id']}: correct facts must be within 0..{fact_total}."
                )
            if completeness not in {0, 1, 2} or page_support not in {0, 1, 2}:
                raise RuntimeError(
                    f"{result['query_id']}: completeness and page support must be 0, 1, or 2."
                )
            total_facts += fact_total
            correct_facts += fact_correct
            completeness_scores.append(completeness)
            page_support_scores.append(page_support)
            checked.update(
                {
                    "correct_fact_count": fact_correct,
                    "total_required_facts": fact_total,
                    "completeness_score_0_to_2": completeness,
                    "page_support_score_0_to_2": page_support,
                }
            )
        checked_results.append(checked)

    manual_metrics = {
        "required_facts": total_facts,
        "correct_facts": correct_facts,
        "key_fact_accuracy": round(correct_facts / total_facts, 4),
        "mean_completeness_score_0_to_2": round(
            sum(completeness_scores) / len(completeness_scores), 4
        ),
        "answer_completeness_rate": round(
            sum(completeness_scores) / (2 * len(completeness_scores)), 4
        ),
        "mean_page_support_score_0_to_2": round(
            sum(page_support_scores) / len(page_support_scores), 4
        ),
        "page_support_rate": round(
            sum(page_support_scores) / (2 * len(page_support_scores)), 4
        ),
    }
    final_payload = {
        "automatic_metrics": payload["metrics"],
        "manually_checked_answer_metrics": manual_metrics,
        "results": checked_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Final generation-quality metrics")
    for key, value in payload["metrics"].items():
        print(f"  {key}: {value}")
    for key, value in manual_metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
