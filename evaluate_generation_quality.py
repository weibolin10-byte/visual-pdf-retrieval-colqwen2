#!/usr/bin/env python3
"""Run first-pass visual RAG generation for manually scored answer quality."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluate_visual_rag import read_jsonl, retrieve_all


REFUSAL_TEXT = "证据不足，无法根据给定页面回答。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("generation_quality_queries_v1_16.csv"),
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/index/colqwen2_v1_hf_final_2312.pt"),
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/generation_quality_raw_results.json"),
    )
    parser.add_argument(
        "--manual-template",
        type=Path,
        default=Path("outputs/generation_quality_manual_scores.csv"),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access; default is cache-only.",
    )
    return parser.parse_args()


def read_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_id",
        "difficulty",
        "domain",
        "answerable",
        "query",
        "expected_pdf",
        "expected_pages",
        "reference_answer",
        "required_facts",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Query CSV must contain: {sorted(required)}")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("Query IDs must be unique.")
    for row in rows:
        if row["answerable"] not in {"true", "false"}:
            raise RuntimeError(f"Invalid answerable value for {row['query_id']}")
        row["is_answerable"] = row["answerable"] == "true"
        row["gold_pages"] = (
            {int(value.strip()) for value in row["expected_pages"].split(";")}
            if row["expected_pages"].strip()
            else set()
        )
        row["fact_list"] = (
            [value.strip() for value in row["required_facts"].split("|") if value.strip()]
            if row["is_answerable"]
            else []
        )
        if row["is_answerable"] and (
            not row["expected_pdf"] or not row["gold_pages"] or not row["fact_list"]
        ):
            raise RuntimeError(f"Answerable query lacks gold data: {row['query_id']}")
    return rows


def build_messages(query: str, hits: list[dict]) -> list[dict]:
    content = []
    for hit in hits:
        source_id = f"S{hit['rank']}"
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"来源[{source_id}]开始；元数据：{hit['pdf_filename']}，"
                        f"PDF第{hit['page_number']}页。"
                    ),
                },
                {
                    "type": "image",
                    "image": Path(hit["image_path"]).resolve().as_uri(),
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": f"来源[{source_id}]结束。"},
            ]
        )
    allowed = "、".join(f"[S{hit['rank']}]" for hit in hits)
    content.append(
        {
            "type": "text",
            "text": (
                "只根据以上页面图像回答，不使用外部知识。使用简洁中文，完整回答问题中"
                "要求的数字、单位和比较关系。每条事实后必须添加支持它的来源标记，"
                f"只允许使用{allowed}。如果这些页面没有足够证据，只输出这一句话："
                f"{REFUSAL_TEXT}\n问题：{query}"
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": "你是只依据给定页面证据回答问题的多模态文档助手。",
        },
        {"role": "user", "content": content},
    ]


def generate_answer(generator, processor, messages: list[dict], max_new_tokens: int):
    import time

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
    ).to(generator.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, elapsed


def analyze_answer(answer: str, hits: list[dict], row: dict) -> dict:
    source_ids = re.findall(r"\[S(\d+)\]", answer)
    allowed = {str(hit["rank"]) for hit in hits}
    invalid = sorted(set(source_ids) - allowed)
    cited = sorted(set(source_ids), key=int) if not invalid else []
    refusal = REFUSAL_TEXT in answer

    cited_hits = [
        hit for hit in hits if str(hit["rank"]) in set(cited)
    ]
    expected_document_cited = row["is_answerable"] and any(
        hit["pdf_filename"] == row["expected_pdf"] for hit in cited_hits
    )
    expected_page_cited = row["is_answerable"] and any(
        hit["pdf_filename"] == row["expected_pdf"]
        and int(hit["page_number"]) in row["gold_pages"]
        for hit in cited_hits
    )
    correct_refusal = (not row["is_answerable"]) and refusal and not source_ids
    citation_format_valid = not invalid and (
        (row["is_answerable"] and bool(source_ids) and not refusal)
        or ((not row["is_answerable"]) and correct_refusal)
    )

    mapping = {
        str(hit["rank"]): f"[{hit['pdf_filename']}, PDF p.{hit['page_number']}]"
        for hit in hits
    }
    grounded_answer = re.sub(
        r"\[S(\d+)\]",
        lambda match: mapping.get(match.group(1), match.group(0)),
        answer,
    )
    return {
        "source_ids": [f"S{value}" for value in cited],
        "invalid_source_ids": [f"S{value}" for value in invalid],
        "citation_format_valid": citation_format_valid,
        "expected_document_cited": expected_document_cited,
        "expected_page_cited": expected_page_cited,
        "refusal_detected": refusal,
        "correct_refusal": correct_refusal,
        "grounded_answer": grounded_answer,
    }


def write_manual_template(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "answerable",
        "total_required_facts",
        "correct_fact_count",
        "completeness_score_0_to_2",
        "page_support_score_0_to_2",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "query_id": result["query_id"],
                    "answerable": str(result["is_answerable"]).lower(),
                    "total_required_facts": len(result["fact_list"]),
                    "correct_fact_count": "",
                    "completeness_score_0_to_2": "",
                    "page_support_score_0_to_2": "",
                    "notes": "",
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    rows = read_queries(args.queries)
    records = read_jsonl(args.metadata)
    all_hits = retrieve_all(args, rows, records)

    local_only = not args.allow_network
    print(f"Loading generator: {args.generator}")
    generator = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.generator,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=local_only,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.generator, local_files_only=local_only
    )
    print(
        f"Generator loaded; allocated VRAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    results = []
    for row, hits in zip(rows, all_hits):
        retrieval_hit = row["is_answerable"] and any(
            hit["pdf_filename"] == row["expected_pdf"]
            and int(hit["page_number"]) in row["gold_pages"]
            for hit in hits
        )
        answer, elapsed = generate_answer(
            generator,
            processor,
            build_messages(row["query"], hits),
            args.max_new_tokens,
        )
        checks = analyze_answer(answer, hits, row)
        result = {
            **row,
            "gold_pages": sorted(row["gold_pages"]),
            "retrieval_hit_at_k": retrieval_hit,
            "hits": hits,
            "raw_answer": answer,
            **checks,
            "generation_seconds": round(elapsed, 2),
        }
        results.append(result)
        if row["is_answerable"]:
            status = (
                f"retrieval={'PASS' if retrieval_hit else 'FAIL'} "
                f"page_citation={'PASS' if checks['expected_page_cited'] else 'FAIL'}"
            )
        else:
            status = f"refusal={'PASS' if checks['correct_refusal'] else 'FAIL'}"
        print(
            f"{row['query_id']} {status} generation={elapsed:.2f}s\n"
            f"  {checks['grounded_answer']}\n"
        )

    answerable = [row for row in results if row["is_answerable"]]
    unanswerable = [row for row in results if not row["is_answerable"]]
    metrics = {
        "queries": len(results),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "top_k": args.top_k,
        "retrieval_hit_at_k_answerable": round(
            sum(row["retrieval_hit_at_k"] for row in answerable) / len(answerable), 4
        ),
        "valid_citation_format_rate_answerable": round(
            sum(row["citation_format_valid"] for row in answerable) / len(answerable),
            4,
        ),
        "correct_document_citation_rate_answerable": round(
            sum(row["expected_document_cited"] for row in answerable) / len(answerable),
            4,
        ),
        "correct_page_citation_rate_answerable": round(
            sum(row["expected_page_cited"] for row in answerable) / len(answerable),
            4,
        ),
        "correct_refusal_rate_unanswerable": round(
            sum(row["correct_refusal"] for row in unanswerable) / len(unanswerable), 4
        ),
        "unsupported_answer_rate_unanswerable": round(
            sum(not row["correct_refusal"] for row in unanswerable) / len(unanswerable),
            4,
        ),
        "mean_generation_seconds": round(
            sum(row["generation_seconds"] for row in results) / len(results), 2
        ),
    }
    serializable_results = []
    for result in results:
        serializable_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"gold_pages", "fact_list"}
            }
            | {
                "gold_pages": result["gold_pages"],
                "required_fact_list": result["fact_list"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"metrics": metrics, "results": serializable_results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manual_template(args.manual_template, results)

    print("\nAutomatic metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Raw results: {args.output}")
    print(f"Manual score template: {args.manual_template}")


if __name__ == "__main__":
    main()
