#!/usr/bin/env python3
"""Render PDF pages into JPEG images and write page-level metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fitz
from PIL import Image


STAGE_ORDER = {"smoke": 0, "validation": 1, "final": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("pdf_sources.csv"))
    parser.add_argument("--stage", choices=STAGE_ORDER, default="final")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--page-dir", type=Path, default=Path("data/pages"))
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def render_page(page: fitz.Page, output_path: Path, dpi: int, quality: int) -> tuple[int, int]:
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    image.save(output_path, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return image.size


def main() -> int:
    args = parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        selected_rows = [
            row
            for row in csv.DictReader(handle)
            if STAGE_ORDER[row["stage"]] <= STAGE_ORDER[args.stage]
        ]
    if not selected_rows:
        raise ValueError(f"No PDFs selected from {args.manifest}")
    missing = [
        row["filename"]
        for row in selected_rows
        if not (args.pdf_dir / row["filename"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing selected PDFs: {missing}")

    args.page_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for pdf_index, row in enumerate(selected_rows, start=1):
        pdf_path = args.pdf_dir / row["filename"]
        doc_id = pdf_path.stem
        document_dir = args.page_dir / doc_id
        document_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as document:
            print(
                f"[{pdf_index:02d}/{len(selected_rows):02d}] "
                f"{pdf_path.name}: {len(document)} pages"
            )
            for zero_based_page, page in enumerate(document):
                page_number = zero_based_page + 1
                image_path = document_dir / f"page_{page_number:04d}.jpg"
                if image_path.exists():
                    with Image.open(image_path) as existing:
                        width, height = existing.size
                else:
                    width, height = render_page(page, image_path, args.dpi, args.quality)

                records.append(
                    {
                        "page_id": f"{doc_id}__p{page_number:04d}",
                        "doc_id": doc_id,
                        "source_id": row["id"],
                        "pdf_filename": pdf_path.name,
                        "title": row["title"],
                        "domain": row["domain"],
                        "language": row["language"],
                        "stage": row["stage"],
                        "page_number": page_number,
                        "image_path": image_path.as_posix(),
                        "width": width,
                        "height": height,
                        "dpi": args.dpi,
                    }
                )

    with args.metadata.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Rendered pages: {len(records)}")
    print(f"Metadata: {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
