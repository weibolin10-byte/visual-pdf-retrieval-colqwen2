#!/usr/bin/env python3
"""Bundle selected full-resolution page images for manual evaluation design."""

from __future__ import annotations

import argparse
import csv
import tarfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("page_eval_expansion_candidates.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/page_eval_expansion_pages.tar.gz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.selection.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "candidate_id",
        "domain",
        "pdf_filename",
        "page_number",
        "page_id",
        "image_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Selection CSV must contain: {sorted(required)}")

    candidate_ids = [row["candidate_id"] for row in rows]
    page_ids = [row["page_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("Candidate IDs must be unique.")
    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Page IDs must be unique.")

    domains = Counter(row["domain"] for row in rows)
    if len(rows) != 20 or set(domains.values()) != {5}:
        raise RuntimeError(
            f"Expected 20 pages with 5 per domain; got {len(rows)}, {dict(domains)}"
        )

    images = []
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append((row, path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        archive.add(args.selection, arcname=args.selection.name)
        for row, path in images:
            archive.add(path, arcname=f"pages/{row['page_id']}{path.suffix.lower()}")

    print(f"Selected pages: {len(images)}")
    print(f"Pages by domain: {dict(sorted(domains.items()))}")
    print(f"Bundle: {args.output}")


if __name__ == "__main__":
    main()
