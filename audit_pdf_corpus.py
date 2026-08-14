#!/usr/bin/env python3
"""Audit the selected PDF corpus before rendering and indexing."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import fitz


STAGE_ORDER = {"smoke": 0, "validation": 1, "final": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("pdf_sources.csv"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--stage", choices=STAGE_ORDER, default="final")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if STAGE_ORDER[row["stage"]] <= STAGE_ORDER[args.stage]
        ]

    selected_names = {row["filename"] for row in rows}
    problems = []
    page_totals = Counter()
    language_totals = Counter()
    hashes = defaultdict(list)
    total_pages = 0

    print(f"Auditing {len(rows)} selected PDFs for stage={args.stage}\n")
    for index, row in enumerate(rows, start=1):
        path = args.pdf_dir / row["filename"]
        if not path.exists():
            problems.append(f"MISSING: {path}")
            print(f"[{index:02d}/{len(rows):02d}] MISSING  {path.name}")
            continue
        try:
            with fitz.open(path) as document:
                if document.needs_pass:
                    raise ValueError("password protected")
                pages = len(document)
                if pages <= 0:
                    raise ValueError("zero pages")
        except Exception as error:
            problems.append(f"INVALID: {path}: {error}")
            print(f"[{index:02d}/{len(rows):02d}] INVALID  {path.name}: {error}")
            continue

        size_mib = path.stat().st_size / 1024**2
        digest = sha256(path)
        hashes[digest].append(path.name)
        total_pages += pages
        page_totals[row["domain"]] += pages
        language_totals[row["language"]] += pages
        print(
            f"[{index:02d}/{len(rows):02d}] OK  {path.name:<46} "
            f"pages={pages:4d}  size={size_mib:7.1f} MiB"
        )

    actual_pdfs = {path.name for path in args.pdf_dir.glob("*.pdf")}
    extras = sorted(actual_pdfs - selected_names)
    partials = sorted(path.name for path in args.pdf_dir.glob("*.part"))
    duplicate_groups = [names for names in hashes.values() if len(names) > 1]

    print("\nSummary")
    print(f"  selected_documents: {len(rows)}")
    print(f"  valid_selected_documents: {len(rows) - len(problems)}")
    print(f"  total_pages: {total_pages}")
    print(f"  pages_by_domain: {dict(page_totals)}")
    print(f"  pages_by_language: {dict(language_totals)}")
    print(f"  extra_pdfs: {extras or 'None'}")
    print(f"  partial_files: {partials or 'None'}")
    print(f"  duplicate_groups: {duplicate_groups or 'None'}")

    if problems:
        print("\nProblems")
        for problem in problems:
            print(f"  - {problem}")

    page_target_ok = 1500 <= total_pages <= 2500
    passed = (
        not problems
        and page_target_ok
        and not duplicate_groups
        and not extras
        and not partials
    )
    print(f"\nPage target 1500-2500: {'PASS' if page_target_ok else 'FAIL'}")
    print(f"Corpus audit: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
