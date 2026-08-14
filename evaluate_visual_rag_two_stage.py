#!/usr/bin/env python3
"""Two-stage Visual RAG evaluation: page evidence extraction, then synthesis."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

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
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_validation_512.pt")
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-max-new-tokens", type=int, default=512)
    parser.add_argument("--answer-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/visual_rag_two_stage_evaluation.json")
    )
    return parser.parse_args()


def first_json_object(text: str) -> tuple[dict | None, bool]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, consumed = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed_text = text[match.start() : match.start() + consumed]
            return value, text.strip() == parsed_text.strip()
    return None, False


def build_evidence_messages(
    query: str,
    hit: dict,
    source_id: str,
    retry_reason: str | None = None,
) -> list[dict]:
    retry_text = ""
    if retry_reason:
        retry_text = f"上一次未通过校验：{retry_reason}。请重新输出。"
    return [
        {
            "role": "system",
            "content": (
                "你是文档证据抽取器。只记录页面中直接可见的事实，不补充常识，"
                "不回答问题，不推断缺失信息。特别核对数字单位、增加或下降方向、"
                "公式、字段名称和阶段关系。"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"来源编号：{source_id}；文件：{hit['pdf_filename']}；"
                        f"PDF第{hit['page_number']}页。"
                    ),
                },
                {
                    "type": "image",
                    "image": Path(hit["image_path"]).resolve().as_uri(),
                    "min_pixels": 512 * 28 * 28,
                    "max_pixels": 1792 * 28 * 28,
                },
                {
                    "type": "text",
                    "text": (
                        "围绕下列问题抽取本页证据：\n"
                        + query
                        + "\n只输出合法JSON，不要Markdown："
                        + '{"source":"'
                        + source_id
                        + '","relevant":true,"evidence":["页面中直接可见的事实"]}'
                        + "。若无相关证据，relevant设为false且evidence为空数组。"
                        + retry_text
                    ),
                },
            ],
        },
    ]


def validate_evidence(raw: str, expected_source: str) -> tuple[dict | None, str, bool]:
    payload, strict_json = first_json_object(raw)
    if payload is None:
        return None, "没有合法JSON对象", False
    if payload.get("source") != expected_source:
        return None, "source与当前页面不一致", strict_json
    if not isinstance(payload.get("relevant"), bool):
        return None, "relevant必须是布尔值", strict_json
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        return None, "evidence必须是字符串数组", strict_json
    if payload["relevant"] and not evidence:
        return None, "相关页面的evidence不能为空", strict_json
    return {
        "source": expected_source,
        "relevant": payload["relevant"],
        "evidence": evidence,
    }, "", strict_json


def extract_page_evidence(
    generator,
    processor,
    query: str,
    hit: dict,
    max_new_tokens: int,
) -> dict:
    source_id = f"S{hit['rank']}"
    attempts = []
    retry_reason = None
    parsed = None
    for attempt_number in (1, 2):
        messages = build_evidence_messages(query, hit, source_id, retry_reason)
        raw, elapsed = generate_once(generator, processor, messages, max_new_tokens)
        parsed, reason, strict_json = validate_evidence(raw, source_id)
        attempts.append(
            {
                "attempt": attempt_number,
                "raw_output": raw,
                "parsed": parsed is not None,
                "strict_json_output": strict_json,
                "validation_error": reason,
                "generation_seconds": round(elapsed, 2),
            }
        )
        if parsed is not None:
            break
        retry_reason = reason
    return {
        **hit,
        "source_id": source_id,
        "parsed": parsed is not None,
        "relevant": parsed["relevant"] if parsed else False,
        "evidence": parsed["evidence"] if parsed else [],
        "attempts": attempts,
    }


def build_synthesis_messages(query: str, evidence_pages: list[dict]) -> list[dict]:
    evidence_payload = []
    allowed_sources = []
    for page in evidence_pages:
        if not page["parsed"] or not page["relevant"]:
            continue
        allowed_sources.append(page["source_id"])
        evidence_payload.append(
            {
                "source": page["source_id"],
                "file": page["pdf_filename"],
                "pdf_page": page["page_number"],
                "evidence": page["evidence"],
            }
        )
    allowed_text = "、".join(allowed_sources) or "无"
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的文档问答助手。只能使用提供的逐页证据，不使用外部知识。"
                "必须直接回答，不能复述问题；严格区分字段名称与单位、增加与下降、"
                "事实与推断。证据不足时明确说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                "问题："
                + query
                + "\n逐页证据："
                + json.dumps(evidence_payload, ensure_ascii=False)
                + "\n只输出合法JSON，不要Markdown或额外解释。结构必须为："
                + '{"points":[{"text":"直接回答问题的简体中文事实",'
                + '"sources":["S1"]}]}'
                + "。写1至6个不重复要点；每项必须引用真实支持它的来源；只能使用："
                + allowed_text
                + "。如果问题要求多个阶段、变化方向或规格类别，必须逐项覆盖，"
                "不要用一句话复述问题代替答案。"
            ),
        },
    ]


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.metadata)
    queries = read_queries(args.queries)
    all_hits = retrieve_all(args, queries, records)

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
    for row, hits in zip(queries, all_hits):
        query_started = time.perf_counter()
        evidence_pages = []
        for hit in hits:
            evidence_pages.append(
                extract_page_evidence(
                    generator,
                    processor,
                    row["query"],
                    hit,
                    args.evidence_max_new_tokens,
                )
            )

        synthesis_messages = build_synthesis_messages(row["query"], evidence_pages)
        raw_answer, synthesis_seconds = generate_once(
            generator,
            processor,
            synthesis_messages,
            args.answer_max_new_tokens,
        )
        answer, citations_valid, chinese_valid, cited_sources, reason, strict_json = (
            parse_validate_and_ground(raw_answer, hits)
        )
        expected_cited = any(
            hit["pdf_filename"] == row["expected_pdf"]
            for hit in hits
            if f"S{hit['rank']}" in cited_sources
        )
        retrieval_hit = any(hit["pdf_filename"] == row["expected_pdf"] for hit in hits)
        evidence_parse_rate = sum(page["parsed"] for page in evidence_pages) / len(
            evidence_pages
        )
        elapsed = time.perf_counter() - query_started

        print(
            f"{row['query_id']} retrieval={'PASS' if retrieval_hit else 'FAIL'} "
            f"evidence_parse={evidence_parse_rate:.0%} "
            f"citations={'PASS' if citations_valid else 'FAIL'} "
            f"expected_source={'PASS' if expected_cited else 'FAIL'} "
            f"chinese={'PASS' if chinese_valid else 'FAIL'} total={elapsed:.2f}s"
        )
        print(f"  {answer}\n")
        results.append(
            {
                **row,
                "retrieval_hit_at_k": retrieval_hit,
                "hits": hits,
                "evidence_pages": evidence_pages,
                "evidence_parse_rate": round(evidence_parse_rate, 4),
                "raw_answer": raw_answer,
                "answer": answer,
                "citation_validation": citations_valid,
                "cited_source_ids": cited_sources,
                "expected_document_cited": expected_cited,
                "chinese_validation": chinese_valid,
                "strict_json_output": strict_json,
                "synthesis_validation_error": reason,
                "synthesis_seconds": round(synthesis_seconds, 2),
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
        "evidence_parse_rate": round(
            sum(item["evidence_parse_rate"] for item in results) / count, 4
        ),
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
