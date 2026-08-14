#!/usr/bin/env python3
"""Hybrid RAG evaluation: visual retrieval plus native PDF text grounding."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import fitz
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluate_visual_rag import (
    generate_once,
    parse_validate_and_ground,
    read_jsonl,
    read_queries,
    retrieve_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("generation_eval_queries_v2.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_validation_512.pt")
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars-per-page", type=int, default=16000)
    parser.add_argument("--evidence-max-new-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/hybrid_rag_evaluation.json")
    )
    return parser.parse_args()


def extract_native_text(pdf_dir: Path, hit: dict, max_chars: int) -> dict:
    pdf_path = pdf_dir / hit["pdf_filename"]
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    with fitz.open(pdf_path) as document:
        page_index = int(hit["page_number"]) - 1
        if not 0 <= page_index < len(document):
            raise IndexError(f"Invalid page {hit['page_number']} for {pdf_path}")
        text = document[page_index].get_text("text", sort=True).strip()
    original_chars = len(text)
    if len(text) > max_chars:
        text = text[:max_chars]
    return {
        **hit,
        "source_id": f"S{hit['rank']}",
        "native_text": text,
        "native_text_chars": original_chars,
        "text_truncated": original_chars > max_chars,
        "has_text_layer": original_chars >= 80,
    }


def extract_query_evidence(
    generator,
    processor,
    query: str,
    page: dict,
    max_new_tokens: int,
) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "你是PDF文本证据抽取器。只抽取与问题直接相关且页面中明确出现的"
                "事实，不回答问题，不使用外部知识。逐字核对数字、单位、增加或下降"
                "方向、字段名称、公式和阶段组成。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：{query}\n来源：{page['source_id']}，{page['pdf_filename']}，"
                f"PDF第{page['page_number']}页。\n页面原生文本：\n{page['native_text']}"
                "\n请输出最多8条与问题直接相关的事实，每行一条，保留原始数值和"
                "增减方向，不要总结或推断。若本页无相关证据，只输出NOT_RELEVANT。"
            ),
        },
    ]
    raw, elapsed = generate_once(generator, processor, messages, max_new_tokens)
    normalized = raw.strip()
    relevant = normalized.upper() != "NOT_RELEVANT" and len(normalized) >= 4
    return {
        **page,
        "query_evidence": normalized if relevant else "",
        "evidence_relevant": relevant,
        "evidence_generation_seconds": round(elapsed, 2),
    }


def build_messages(query: str, pages: list[dict], retry_reason: str | None = None):
    sources = []
    for page in pages:
        if not page["evidence_relevant"]:
            continue
        sources.append(
            {
                "source": page["source_id"],
                "file": page["pdf_filename"],
                "pdf_page": page["page_number"],
                "evidence": page["query_evidence"],
            }
        )
    allowed_sources = "、".join(item["source"] for item in sources) or "无"
    retry_text = f"\n上一次校验失败：{retry_reason}。请修正。" if retry_reason else ""
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的文档问答助手。只能使用给定PDF页面的原生文本，不使用"
                "外部知识。先在内部逐项核对数字、单位、增加或下降方向、字段名称、"
                "公式及阶段关系。必须直接回答，不能复述问题；证据不足时明确说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                "问题："
                + query
                + "\n逐页相关证据："
                + json.dumps(sources, ensure_ascii=False)
                + "\n只输出一个合法JSON对象，不要Markdown或其他说明。格式："
                + '{"points":[{"text":"直接回答问题的简体中文事实",'
                + '"sources":["S1"]}]}'
                + "。写1至6个不重复要点，每个要点都必须有来源。只能使用："
                + allowed_sources
                + "。若问题要求变化原因，必须分别说明增长项与抵消项；若问题要求"
                "阶段路径，必须逐阶段说明组成；若问题要求规格，必须区分容量、接口"
                "宽度、带宽、接口代际、功耗和互连，禁止把bits写成带宽。"
                + retry_text
            ),
        },
    ]


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.metadata)
    queries = read_queries(args.queries)
    all_hits = retrieve_all(args, queries, records)

    grounded_pages = [
        [extract_native_text(args.pdf_dir, hit, args.max_chars_per_page) for hit in hits]
        for hits in all_hits
    ]
    total_pages = sum(len(pages) for pages in grounded_pages)
    text_pages = sum(page["has_text_layer"] for pages in grounded_pages for page in pages)
    print(f"Native text available: {text_pages}/{total_pages} retrieved pages")

    print(f"Loading generator: {args.generator}")
    generator = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.generator,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.generator, local_files_only=True)
    print(
        "Generator loaded; allocated VRAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    results = []
    for row, hits, pages in zip(queries, all_hits, grounded_pages):
        started = time.perf_counter()
        pages = [
            extract_query_evidence(
                generator,
                processor,
                row["query"],
                page,
                args.evidence_max_new_tokens,
            )
            for page in pages
        ]
        attempts = []
        retry_reason = None
        for attempt_number in (1, 2):
            messages = build_messages(row["query"], pages, retry_reason)
            raw, generation_seconds = generate_once(
                generator, processor, messages, args.max_new_tokens
            )
            answer, citations_valid, chinese_valid, cited_sources, reason, strict_json = (
                parse_validate_and_ground(raw, hits)
            )
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_answer": raw,
                    "answer": answer,
                    "citations_valid": citations_valid,
                    "chinese_valid": chinese_valid,
                    "cited_source_ids": cited_sources,
                    "strict_json_output": strict_json,
                    "validation_error": reason,
                    "generation_seconds": round(generation_seconds, 2),
                }
            )
            if citations_valid and chinese_valid:
                break
            retry_reason = reason or "回答未满足中文或来源要求"

        final = attempts[-1]
        cited_sources = final["cited_source_ids"]
        expected_cited = any(
            hit["pdf_filename"] == row["expected_pdf"]
            for hit in hits
            if f"S{hit['rank']}" in cited_sources
        )
        retrieval_hit = any(hit["pdf_filename"] == row["expected_pdf"] for hit in hits)
        elapsed = time.perf_counter() - started
        print(
            f"{row['query_id']} retrieval={'PASS' if retrieval_hit else 'FAIL'} "
            f"attempts={len(attempts)} "
            f"citations={'PASS' if final['citations_valid'] else 'FAIL'} "
            f"expected_source={'PASS' if expected_cited else 'FAIL'} "
            f"chinese={'PASS' if final['chinese_valid'] else 'FAIL'} total={elapsed:.2f}s"
        )
        print(f"  {final['answer']}\n")
        results.append(
            {
                **row,
                "retrieval_hit_at_k": retrieval_hit,
                "pages": pages,
                "answer": final["answer"],
                "attempts": attempts,
                "citation_validation": final["citations_valid"],
                "expected_document_cited": expected_cited,
                "chinese_validation": final["chinese_valid"],
                "strict_json_output": final["strict_json_output"],
                "relevant_evidence_pages": sum(
                    page["evidence_relevant"] for page in pages
                ),
                "total_seconds": round(elapsed, 2),
            }
        )

    count = len(results)
    metrics = {
        "queries": count,
        "top_k": args.top_k,
        "retrieval_hit_at_k": round(
            sum(item["retrieval_hit_at_k"] for item in results) / count, 4
        ),
        "retrieved_page_text_layer_rate": round(text_pages / total_pages, 4),
        "valid_citation_rate": round(
            sum(item["citation_validation"] for item in results) / count, 4
        ),
        "expected_document_citation_rate": round(
            sum(item["expected_document_cited"] for item in results) / count, 4
        ),
        "chinese_answer_rate": round(
            sum(item["chinese_validation"] for item in results) / count, 4
        ),
        "strict_json_output_rate": round(
            sum(item["strict_json_output"] for item in results) / count, 4
        ),
        "retry_rate": round(sum(len(item["attempts"]) > 1 for item in results) / count, 4),
        "mean_total_seconds": round(
            sum(item["total_seconds"] for item in results) / count, 2
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nMetrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved: {args.output}")

    del generator, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
