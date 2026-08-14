#!/usr/bin/env python3
"""Build a ColQwen2 page index and run a small retrieval smoke test."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import ColQwen2ForRetrieval, ColQwen2Processor


DEFAULT_QUERIES = [
    "How does ColPali use late interaction to retrieve visually rich document pages?",
    "What is the global progress on the Sustainable Development Goals in 2024?",
    "What are the connectors, interfaces, and mechanical specifications of Raspberry Pi 5?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_83pages.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/retrieval_smoke_hf")
    )
    parser.add_argument("--model", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image on metadata line {line_number} not found: {image_path}"
                )
            records.append(record)

    if not records:
        raise RuntimeError("No page records found in metadata.")
    return records


def load_model(model_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke test.")

    print(f"Loading model: {model_name}")
    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(model_name)
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded on {model.device}; allocated VRAM: {allocated:.2f} GiB")
    return model, processor


def build_index(
    records: list[dict],
    model,
    processor,
    index_path: Path,
    model_name: str,
    batch_size: int,
) -> list[torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    started = time.perf_counter()

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = []
        for record in batch_records:
            with Image.open(record["image_path"]) as image:
                images.append(image.convert("RGB"))

        batch = processor.process_images(images).to(model.device)
        with torch.inference_mode():
            batch_embeddings = model(**batch).embeddings

        embeddings.extend(
            embedding.contiguous().cpu()
            for embedding in torch.unbind(batch_embeddings)
        )
        completed = min(start + batch_size, len(records))
        print(f"Encoded {completed:>3}/{len(records)} pages")

        del batch, batch_embeddings, images

    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "page_ids": [record["page_id"] for record in records],
        "embeddings": embeddings,
    }
    torch.save(payload, index_path)
    elapsed = time.perf_counter() - started
    size_mb = index_path.stat().st_size / 1024**2
    print(f"Index saved: {index_path} ({size_mb:.1f} MiB, {elapsed:.1f} s)")
    return embeddings


def load_index(index_path: Path, records: list[dict], model_name: str):
    print(f"Loading cached index: {index_path}")
    payload = torch.load(index_path, map_location="cpu", weights_only=True)
    expected_page_ids = [record["page_id"] for record in records]
    if payload.get("model_name") != model_name:
        raise RuntimeError("Cached index was created with a different model; use --rebuild.")
    if payload.get("page_ids") != expected_page_ids:
        raise RuntimeError("Page metadata changed after indexing; use --rebuild.")
    embeddings = payload["embeddings"]
    print(f"Loaded {len(embeddings)} page embeddings")
    return embeddings


def search(
    queries: list[str],
    records: list[dict],
    page_embeddings: list[torch.Tensor],
    model,
    processor,
    top_k: int,
) -> list[dict]:
    query_batch = processor.process_queries(queries).to(model.device)
    with torch.inference_mode():
        query_embeddings = model(**query_batch).embeddings

    gpu_page_embeddings = [embedding.to(model.device) for embedding in page_embeddings]
    with torch.inference_mode():
        scores = processor.score_retrieval(query_embeddings, gpu_page_embeddings)
    scores = scores.float().cpu()

    all_results = []
    top_k = min(top_k, len(records))
    for query_index, query in enumerate(queries):
        values, indices = torch.topk(scores[query_index], k=top_k)
        hits = []
        for rank, (value, page_index) in enumerate(zip(values, indices), start=1):
            record = records[int(page_index)]
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(value), 4),
                    "page_id": record["page_id"],
                    "pdf_filename": record["pdf_filename"],
                    "page_number": record["page_number"],
                    "image_path": record["image_path"],
                }
            )
        all_results.append({"query": query, "hits": hits})

    del gpu_page_embeddings, scores, query_embeddings, query_batch
    torch.cuda.empty_cache()
    return all_results


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for query_number, result in enumerate(results, start=1):
        query_dir = output_dir / f"query_{query_number:02d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        for hit in result["hits"]:
            source = Path(hit["image_path"])
            target = query_dir / (
                f"rank_{hit['rank']:02d}_{source.parent.name}_{source.name}"
            )
            shutil.copy2(source, target)

    print(f"Results saved: {result_path}")


def print_results(results: list[dict]) -> None:
    print("\nRetrieval results")
    for query_number, result in enumerate(results, start=1):
        print(f"\nQ{query_number}: {result['query']}")
        for hit in result["hits"]:
            print(
                f"  #{hit['rank']} score={hit['score']:.4f}  "
                f"{hit['pdf_filename']}  page={hit['page_number']}"
            )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    records = load_metadata(args.metadata)
    counts = Counter(record["pdf_filename"] for record in records)
    print(f"Pages: {len(records)} across {len(counts)} PDFs")
    for filename, count in counts.items():
        print(f"  {filename}: {count}")

    model, processor = load_model(args.model)
    if args.index.is_file() and not args.rebuild:
        page_embeddings = load_index(args.index, records, args.model)
    else:
        page_embeddings = build_index(
            records,
            model,
            processor,
            args.index,
            args.model,
            args.batch_size,
        )

    queries = args.queries or DEFAULT_QUERIES
    results = search(
        queries,
        records,
        page_embeddings,
        model,
        processor,
        args.top_k,
    )
    print_results(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
