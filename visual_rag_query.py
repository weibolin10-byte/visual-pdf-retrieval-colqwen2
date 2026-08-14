#!/usr/bin/env python3
"""Run one end-to-end Visual RAG query with ColQwen2 and Qwen2.5-VL."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    ColQwen2ForRetrieval,
    ColQwen2Processor,
    Qwen2_5_VLForConditionalGeneration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default=(
            "How does ColPali perform visual document retrieval, "
            "and what role does late interaction play?"
        ),
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_83pages.pt")
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/visual_rag_answer_v2.json")
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No page metadata found in {path}")
    return records


def retrieve(args: argparse.Namespace, records: list[dict]) -> tuple[list[dict], float]:
    payload = torch.load(args.index, map_location="cpu", weights_only=True)
    expected_ids = [record["page_id"] for record in records]
    if payload.get("model_name") != args.retriever:
        raise RuntimeError("Retriever name does not match the cached index.")
    if payload.get("page_ids") != expected_ids:
        raise RuntimeError("Page metadata does not match the cached index.")

    started = time.perf_counter()
    print(f"Loading retriever: {args.retriever}")
    model = ColQwen2ForRetrieval.from_pretrained(
        args.retriever,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.retriever)

    query_inputs = processor.process_queries([args.query]).to(model.device)
    with torch.inference_mode():
        query_embedding = model(**query_inputs).embeddings

    page_embeddings = [embedding.to(model.device) for embedding in payload["embeddings"]]
    with torch.inference_mode():
        scores = processor.score_retrieval(query_embedding, page_embeddings)[0].float()

    values, indices = torch.topk(scores, k=min(args.top_k, len(records)))
    hits = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        record = records[index]
        hits.append(
            {
                "rank": rank,
                "score": round(value, 4),
                "pdf_filename": record["pdf_filename"],
                "page_number": record["page_number"],
                "image_path": record["image_path"],
            }
        )

    elapsed = time.perf_counter() - started
    del page_embeddings, scores, query_embedding, query_inputs, processor, model, payload
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Retriever released; allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    return hits, elapsed


def build_messages(query: str, hits: list[dict]) -> list[dict]:
    content = []
    for hit in hits:
        image_uri = Path(hit["image_path"]).resolve().as_uri()
        source_id = f"S{hit['rank']}"
        content.append(
            {
                "type": "text",
                "text": (
                    f"BEGIN SOURCE [{source_id}]. This source is exactly "
                    f"{hit['pdf_filename']}, PDF page {hit['page_number']}.\n"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": image_uri,
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }
        )
        content.append(
            {
                "type": "text",
                "text": f"END SOURCE [{source_id}].\n",
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Answer the question using only the supplied document-page images. "
                "Write a concise Chinese answer with 2 to 4 bullet points and avoid repetition. "
                "Cite every factual bullet using only the source tokens [S1], [S2], or [S3]. "
                "Do not write filenames or page numbers yourself. Do not use a page number "
                "printed inside an image as a citation. If the supplied pages do not contain "
                "enough evidence, state that explicitly.\n\n"
                f"Question: {query}"
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": "You are a careful multimodal document question-answering assistant.",
        },
        {"role": "user", "content": content},
    ]


def ground_citations(raw_answer: str, hits: list[dict]) -> tuple[str, list[str]]:
    """Validate model-selected source IDs and map them to exact PDF metadata."""
    mapping = {
        f"S{hit['rank']}": (
            f"[{hit['pdf_filename']}, PDF p.{hit['page_number']}]"
        )
        for hit in hits
    }
    cited_ids = re.findall(r"\[S(\d+)\]", raw_answer)
    cited_tokens = [f"S{number}" for number in cited_ids]
    invalid = sorted(set(cited_tokens) - set(mapping))
    if invalid:
        raise RuntimeError(f"Generator returned invalid source IDs: {invalid}")
    if not cited_tokens:
        raise RuntimeError("Generator returned no source IDs; citation validation failed.")

    grounded = re.sub(
        r"\[S(\d+)\]",
        lambda match: mapping[f"S{match.group(1)}"],
        raw_answer,
    )
    return grounded, sorted(set(cited_tokens))


def generate_answer(args: argparse.Namespace, hits: list[dict]) -> tuple[str, str, list[str], float]:
    started = time.perf_counter()
    print(f"Loading generator: {args.generator}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.generator,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.generator)
    print(f"Generator loaded; allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    messages = build_messages(args.query, hits)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_answer = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    answer, cited_source_ids = ground_citations(raw_answer, hits)
    elapsed = time.perf_counter() - started
    return answer, raw_answer, cited_source_ids, elapsed


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if not args.index.is_file():
        raise FileNotFoundError(f"Index not found: {args.index}")

    records = load_records(args.metadata)
    hits, retrieval_seconds = retrieve(args, records)

    print("\nRetrieved pages")
    for hit in hits:
        print(
            f"  #{hit['rank']} score={hit['score']:.4f}  "
            f"{hit['pdf_filename']}  page={hit['page_number']}"
        )

    answer, raw_answer, cited_source_ids, generation_seconds = generate_answer(args, hits)
    result = {
        "query": args.query,
        "retriever": args.retriever,
        "generator": args.generator,
        "hits": hits,
        "raw_answer": raw_answer,
        "answer": answer,
        "citation_validation": {
            "passed": True,
            "cited_source_ids": cited_source_ids,
        },
        "timing_seconds": {
            "retrieval_including_model_load": round(retrieval_seconds, 2),
            "generation_including_model_load": round(generation_seconds, 2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nAnswer")
    print(answer)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
