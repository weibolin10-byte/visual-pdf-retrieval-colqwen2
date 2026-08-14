#!/usr/bin/env python3
"""Select visually rich PDF pages for manual page-level retrieval annotation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps


DOMAIN_TERMS = {
    "paper": (
        "figure", "table", "architecture", "framework", "benchmark",
        "experiment", "ablation", "results", "performance", "comparison",
    ),
    "financial_report": (
        "revenue", "operating income", "net income", "segment", "fiscal",
        "cash flow", "balance sheet", "year ended", "millions", "billions",
    ),
    "public_report": (
        "figure", "table", "indicator", "trend", "forecast", "growth",
        "percentage", "percent", "distribution", "progress", "图", "表",
    ),
    "technical_manual": (
        "specification", "block diagram", "pin", "interface", "memory map",
        "electrical characteristics", "dimensions", "peripheral", "typical",
        "maximum", "minimum",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/page_eval_candidates.csv"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/page_eval_candidates.html"),
    )
    parser.add_argument(
        "--contact-sheet-dir",
        type=Path,
        default=Path("outputs/page_eval_contact_sheets"),
    )
    parser.add_argument("--per-domain", type=int, default=12)
    parser.add_argument("--max-per-document", type=int, default=3)
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def page_signals(page: fitz.Page, domain: str) -> dict:
    text = page.get_text("text", sort=True).strip()
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.casefold()
    terms = DOMAIN_TERMS.get(domain, ())
    term_hits = sum(lowered.count(term.casefold()) for term in terms)
    figure_refs = len(re.findall(r"\b(?:fig(?:ure)?\.?)[\s:]*\d+", lowered))
    table_refs = len(re.findall(r"\btable[\s:]*\d+", lowered))
    chinese_refs = len(re.findall(r"(?:图|表)\s*\d+", normalized))
    numeric_tokens = len(re.findall(r"(?<!\w)[+-]?\d[\d,.%]*(?!\w)", normalized))
    word_count = max(1, len(normalized.split()))
    numeric_density = min(numeric_tokens / word_count, 0.5)
    try:
        image_count = len(page.get_images(full=True))
    except Exception:
        image_count = 0
    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0

    score = (
        term_hits * 2.5
        + figure_refs * 4.0
        + table_refs * 4.0
        + chinese_refs * 4.0
        + min(image_count, 5) * 2.0
        + min(drawing_count, 100) / 10.0
        + numeric_density * 20.0
    )
    if 120 <= len(normalized) <= 5000:
        score += 1.0
    if page.number == 0:
        score -= 3.0
    if len(normalized) > 8000:
        score -= 2.0

    labels = []
    if figure_refs or chinese_refs:
        labels.append(f"figure_refs={figure_refs + chinese_refs}")
    if table_refs:
        labels.append(f"table_refs={table_refs}")
    if image_count:
        labels.append(f"images={image_count}")
    if drawing_count:
        labels.append(f"drawings={drawing_count}")
    if term_hits:
        labels.append(f"term_hits={term_hits}")
    labels.append(f"numeric_density={numeric_density:.2f}")

    return {
        "score": score,
        "text_preview": normalized[:500],
        "signals": "; ".join(labels),
    }


def select_diverse(candidates: list[dict], limit: int, max_per_document: int) -> list[dict]:
    by_document: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_document[candidate["pdf_filename"]].append(candidate)
    for values in by_document.values():
        values.sort(key=lambda item: item["score"], reverse=True)

    document_order = sorted(
        by_document,
        key=lambda name: by_document[name][0]["score"],
        reverse=True,
    )
    selected = []
    for round_index in range(max_per_document):
        for document in document_order:
            values = by_document[document]
            if round_index < len(values):
                selected.append(values[round_index])
                if len(selected) == limit:
                    return selected
    return selected


def relative_url(target: Path, html_path: Path) -> str:
    return Path(os.path.relpath(target.resolve(), html_path.parent.resolve())).as_posix()


def write_html(rows: list[dict], output_path: Path, pdf_dir: Path) -> None:
    cards = []
    for row in rows:
        image_path = Path(row["image_path"])
        image_url = relative_url(image_path, output_path)
        pdf_url = relative_url(pdf_dir / row["pdf_filename"], output_path)
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(pdf_url)}#page={row['page_number']}">
                <img src="{html.escape(image_url)}" loading="lazy" alt="{row['candidate_id']}">
              </a>
              <h3>{row['candidate_id']} · {html.escape(row['domain'])}</h3>
              <p><strong>{html.escape(row['pdf_filename'])}</strong> · PDF p.{row['page_number']} · score {row['score']}</p>
              <p class="signals">{html.escape(row['signals'])}</p>
              <p>{html.escape(row['text_preview'])}</p>
            </article>
            """
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Page-level evaluation candidates</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#17202a}
h1{margin-bottom:6px}.note{color:#536471;margin-top:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
.card{background:white;border:1px solid #dce2e8;border-radius:12px;padding:12px;box-shadow:0 2px 8px #0000000a}
.card img{width:100%;height:420px;object-fit:contain;background:#eef1f4;border-radius:7px}
.card h3{margin:10px 0 4px}.card p{font-size:13px;line-height:1.45}.signals{color:#52606d}
</style></head><body>
<h1>Page-level retrieval candidates</h1>
<p class="note">Candidates are selected without running the retriever. Click a page to open its source PDF.</p>
<section class="grid">"""
        + "\n".join(cards)
        + "</section></body></html>\n",
        encoding="utf-8",
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def write_contact_sheets(rows: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)

    font = load_font(22)
    small_font = load_font(18)
    outputs = []
    columns, cell_width, cell_height = 3, 520, 720
    for domain, domain_rows in sorted(grouped.items()):
        row_count = (len(domain_rows) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height), "#e9edf1")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(domain_rows):
            column = index % columns
            grid_row = index // columns
            left = column * cell_width
            top = grid_row * cell_height
            image = Image.open(row["image_path"]).convert("RGB")
            thumb = ImageOps.contain(image, (cell_width - 24, cell_height - 92))
            image.close()
            x = left + (cell_width - thumb.width) // 2
            y = top + 10
            sheet.paste(thumb, (x, y))
            label_y = top + cell_height - 72
            draw.rectangle((left, label_y, left + cell_width, top + cell_height), fill="white")
            draw.text((left + 10, label_y + 6), f"{row['candidate_id']}  PDF p.{row['page_number']}  score={row['score']}", fill="#111827", font=font)
            name = row["pdf_filename"]
            if len(name) > 54:
                name = name[:51] + "..."
            draw.text((left + 10, label_y + 38), name, fill="#475569", font=small_font)
        output_path = output_dir / f"{domain}.jpg"
        sheet.save(output_path, quality=90, optimize=True)
        outputs.append(output_path)
    return outputs


def main() -> int:
    args = parse_args()
    if args.per_domain < 1 or args.max_per_document < 1:
        raise ValueError("--per-domain and --max-per-document must be positive")

    records = read_metadata(args.metadata)
    metadata_by_document: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        metadata_by_document[record["pdf_filename"]].append(record)

    candidates_by_domain: dict[str, list[dict]] = defaultdict(list)
    processed = 0
    for pdf_filename, document_records in sorted(metadata_by_document.items()):
        pdf_path = args.pdf_dir / pdf_filename
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        by_page = {int(record["page_number"]): record for record in document_records}
        with fitz.open(pdf_path) as document:
            if len(document) != len(document_records):
                raise RuntimeError(
                    f"Page-count mismatch for {pdf_filename}: PDF={len(document)} metadata={len(document_records)}"
                )
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                record = by_page[page_number]
                candidate = {**record, **page_signals(page, record["domain"])}
                candidates_by_domain[record["domain"]].append(candidate)
                processed += 1
                if processed % 250 == 0 or processed == len(records):
                    print(f"Scanned {processed}/{len(records)} pages")

    selected = []
    for domain in sorted(candidates_by_domain):
        domain_selected = select_diverse(
            candidates_by_domain[domain],
            args.per_domain,
            args.max_per_document,
        )
        print(f"Selected {len(domain_selected)} candidates for {domain}")
        selected.extend(domain_selected)

    for index, row in enumerate(selected, start=1):
        row["candidate_id"] = f"C{index:03d}"
        row["score"] = round(float(row["score"]), 2)

    fieldnames = [
        "candidate_id", "domain", "pdf_filename", "title", "page_number",
        "page_id", "score", "signals", "image_path", "text_preview",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in selected)

    write_html(selected, args.output_html, args.pdf_dir)
    sheets = write_contact_sheets(selected, args.contact_sheet_dir)
    print(f"Candidates: {args.output_csv}")
    print(f"Browser view: {args.output_html}")
    for sheet in sheets:
        print(f"Contact sheet: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
