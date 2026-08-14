#!/usr/bin/env python3
"""Download the curated PDF corpus directly on AutoDL."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path


STAGE_ORDER = {"smoke": 0, "validation": 1, "final": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        default="smoke",
        help="smoke=3 PDFs, validation=first 10 PDFs, final=all 30 PDFs",
    )
    parser.add_argument("--manifest", type=Path, default=Path("pdf_sources.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def is_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return b"%PDF" in handle.read(1024)


def download(url: str, destination: Path, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 visual-document-rag-reproduction/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total_header = response.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        last_report = 0
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 1024 * 1024:
                    downloaded_mb = downloaded / 1024**2
                    if total_bytes:
                        total_mb = total_bytes / 1024**2
                        print(f"    downloaded {downloaded_mb:.1f}/{total_mb:.1f} MiB")
                    else:
                        print(f"    downloaded {downloaded_mb:.1f} MiB")
                    last_report = downloaded
    if not is_pdf(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("downloaded content is not a valid PDF")
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if STAGE_ORDER[row["stage"]] <= STAGE_ORDER[args.stage]:
                selected.append(row)

    print(f"Selected {len(selected)} PDFs for stage={args.stage}")
    failures = []
    for index, row in enumerate(selected, start=1):
        destination = args.output_dir / row["filename"]
        if is_pdf(destination):
            print(f"[{index:02d}/{len(selected):02d}] skip  {destination.name}")
            continue
        print(f"[{index:02d}/{len(selected):02d}] fetch {destination.name}")
        error = None
        for attempt in range(1, args.retries + 1):
            try:
                download(row["download_url"], destination, args.timeout)
                error = None
                break
            except Exception as exc:
                error = exc
                print(
                    f"  attempt {attempt}/{args.retries} failed: {exc}",
                    file=sys.stderr,
                )
                if attempt < args.retries:
                    time.sleep(2**attempt)
        if error is not None:  # continue so one blocked host does not lose the batch
            failures.append((row["id"], row["download_url"], str(error)))

    valid_count = sum(is_pdf(args.output_dir / row["filename"]) for row in selected)
    print(f"Valid PDFs: {valid_count}/{len(selected)}")
    if failures:
        print("Failed downloads:", file=sys.stderr)
        for item_id, url, error in failures:
            print(f"- {item_id}: {error}\n  {url}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
