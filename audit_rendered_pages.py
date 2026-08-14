#!/usr/bin/env python3
"""Audit rendered page images against page metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument("--page-dir", type=Path, default=Path("data/pages"))
    parser.add_argument("--expected-pages", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = []
    page_ids = [record["page_id"] for record in records]
    image_paths = [Path(record["image_path"]) for record in records]

    if len(page_ids) != len(set(page_ids)):
        problems.append("duplicate page_id values")
    normalized_paths = [path.as_posix() for path in image_paths]
    if len(normalized_paths) != len(set(normalized_paths)):
        problems.append("duplicate image_path values")

    pages_by_doc = defaultdict(list)
    pages_by_domain = Counter()
    pages_by_language = Counter()
    missing = []
    invalid = []
    dimension_mismatches = []

    for index, (record, image_path) in enumerate(zip(records, image_paths), start=1):
        pages_by_doc[record["pdf_filename"]].append(int(record["page_number"]))
        pages_by_domain[record["domain"]] += 1
        pages_by_language[record["language"]] += 1
        if not image_path.exists():
            missing.append(image_path.as_posix())
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (int(record["width"]), int(record["height"])):
                    dimension_mismatches.append(image_path.as_posix())
        except Exception as error:
            invalid.append(f"{image_path}: {error}")
        if index % 250 == 0 or index == len(records):
            print(f"Checked {index}/{len(records)} images")

    for pdf_filename, page_numbers in pages_by_doc.items():
        expected = list(range(1, len(page_numbers) + 1))
        if sorted(page_numbers) != expected:
            problems.append(f"non-contiguous pages: {pdf_filename}")

    expected_images = set(normalized_paths)
    actual_images = {
        path.as_posix() for path in args.page_dir.rglob("*.jpg") if path.is_file()
    }
    extras = sorted(actual_images - expected_images)

    print("\nSummary")
    print(f"  metadata_records: {len(records)}")
    print(f"  unique_documents: {len(pages_by_doc)}")
    print(f"  expected_images: {len(expected_images)}")
    print(f"  actual_images: {len(actual_images)}")
    print(f"  pages_by_domain: {dict(pages_by_domain)}")
    print(f"  pages_by_language: {dict(pages_by_language)}")
    print(f"  missing_images: {missing or 'None'}")
    print(f"  invalid_images: {invalid or 'None'}")
    print(f"  dimension_mismatches: {dimension_mismatches or 'None'}")
    print(f"  extra_images: {extras or 'None'}")
    print(f"  metadata_problems: {problems or 'None'}")

    expected_count_ok = args.expected_pages is None or len(records) == args.expected_pages
    if not expected_count_ok:
        print(
            f"  expected-page mismatch: expected {args.expected_pages}, "
            f"found {len(records)}"
        )
    passed = not any(
        [missing, invalid, dimension_mismatches, extras, problems]
    ) and expected_count_ok
    print(f"\nRendered-page audit: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
