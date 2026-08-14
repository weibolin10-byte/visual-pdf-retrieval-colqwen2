#!/usr/bin/env python3
"""Run real image OCR over rendered PDF pages and build a resumable text cache."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/index/easyocr_page_text.jsonl")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("data/cache/easyocr")
    )
    parser.add_argument("--languages", nargs="+", default=["ch_sim", "en"])
    parser.add_argument(
        "--page-id",
        action="append",
        dest="page_ids",
        help="Process only this page_id; repeat for a representative smoke test.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow EasyOCR model downloads. Use on the first run only.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose_gpu(device: str) -> bool:
    import torch

    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if device == "cpu":
        return False
    return available


def validate_existing(
    existing: list[dict], records: list[dict], languages: list[str]
) -> None:
    if len(existing) > len(records):
        raise RuntimeError("OCR cache contains more records than page metadata.")
    expected_prefix = [record["page_id"] for record in records[: len(existing)]]
    actual_prefix = [record.get("page_id") for record in existing]
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            "OCR cache is not an exact metadata prefix. Move it aside and rebuild."
        )
    expected_languages = list(languages)
    mismatched = [
        row.get("page_id")
        for row in existing
        if row.get("ocr_engine") != "easyocr"
        or row.get("languages") != expected_languages
    ]
    if mismatched:
        raise RuntimeError(
            f"OCR cache engine/language mismatch; first affected page: {mismatched[0]}"
        )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.min_size < 1:
        raise ValueError("Invalid batch-size, workers, or min-size value.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "EasyOCR is not installed. Run: python -m pip install easyocr==1.7.2"
        ) from error

    records = read_jsonl(args.metadata)
    if not records:
        raise RuntimeError(f"No metadata records found: {args.metadata}")
    if args.page_ids:
        by_id = {record["page_id"]: record for record in records}
        missing_ids = [page_id for page_id in args.page_ids if page_id not in by_id]
        if missing_ids:
            raise RuntimeError(f"Requested page IDs are absent from metadata: {missing_ids}")
        if len(args.page_ids) != len(set(args.page_ids)):
            raise RuntimeError("Repeated --page-id values are not allowed.")
        records = [by_id[page_id] for page_id in args.page_ids]
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    existing = read_jsonl(args.output) if args.output.exists() else []
    validate_existing(existing, records, args.languages)
    target_count = min(args.limit or len(records), len(records))
    if len(existing) >= target_count:
        print(f"OCR cache already contains {len(existing)} pages; target={target_count}")
        return 0

    use_gpu = choose_gpu(args.device)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading EasyOCR languages={args.languages} device={'cuda' if use_gpu else 'cpu'} "
        f"download_enabled={args.allow_download}"
    )
    reader = easyocr.Reader(
        args.languages,
        gpu=use_gpu,
        model_storage_directory=str(args.model_dir),
        download_enabled=args.allow_download,
        verbose=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    page_seconds = []
    line_counts = []
    char_counts = []
    mode = "a" if existing else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index in range(len(existing), target_count):
            record = records[index]
            started = time.perf_counter()
            detections = reader.readtext(
                record["image_path"],
                decoder="greedy",
                batch_size=args.batch_size,
                workers=args.workers,
                detail=1,
                paragraph=False,
                min_size=args.min_size,
            )
            accepted = [
                (str(text).strip(), float(confidence))
                for _, text, confidence in detections
                if str(text).strip() and float(confidence) >= args.confidence_threshold
            ]
            text = "\n".join(value for value, _ in accepted)
            confidences = [confidence for _, confidence in accepted]
            elapsed = time.perf_counter() - started
            output_record = {
                "page_id": record["page_id"],
                "pdf_filename": record["pdf_filename"],
                "page_number": int(record["page_number"]),
                "ocr_engine": "easyocr",
                "languages": list(args.languages),
                "text": text,
                "line_count": len(accepted),
                "character_count": len(text),
                "mean_confidence": (
                    round(statistics.fmean(confidences), 6) if confidences else None
                ),
                "ocr_seconds": round(elapsed, 4),
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

            page_seconds.append(elapsed)
            line_counts.append(len(accepted))
            char_counts.append(len(text))
            completed = index + 1
            if (
                completed == 1
                or completed % args.log_every == 0
                or completed == target_count
            ):
                recent = page_seconds[-min(len(page_seconds), args.log_every) :]
                print(
                    f"OCR {completed}/{target_count}  {record['page_id']}  "
                    f"lines={len(accepted)} chars={len(text)} "
                    f"page={elapsed:.2f}s recent_avg={statistics.fmean(recent):.2f}s"
                )

    elapsed_total = time.perf_counter() - total_started
    print("\nOCR cache summary")
    print(f"  cache: {args.output}")
    print(f"  cached_pages: {target_count}/{len(records)}")
    print(f"  pages_processed_this_run: {len(page_seconds)}")
    print(f"  nonempty_this_run: {sum(count > 0 for count in char_counts)}")
    print(
        f"  mean_seconds_per_page_this_run: "
        f"{statistics.fmean(page_seconds):.3f}"
    )
    print(f"  elapsed_seconds_this_run: {elapsed_total:.2f}")
    if target_count < len(records):
        print("  status: PARTIAL (rerun without --limit to resume)")
    else:
        print("  status: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
