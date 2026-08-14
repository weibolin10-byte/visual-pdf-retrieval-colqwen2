#!/usr/bin/env python3
"""Evaluate retrieval, grounded citations, and generation across four domains."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--queries", type=Path, default=Path("generation_eval_queries.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_validation_512.pt")
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/visual_rag_evaluation.json")
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def retrieve_all(args, queries: list[dict], records: list[dict]) -> list[list[dict]]:
    payload = torch.load(args.index, map_location="cpu", weights_only=True)
    if payload.get("model_name") != args.retriever:
        raise RuntimeError("Retriever does not match the cached index.")
    if payload.get("page_ids") != [record["page_id"] for record in records]:
        raise RuntimeError("Metadata does not match the cached index.")

    print(f"Loading retriever: {args.retriever}")
    model = ColQwen2ForRetrieval.from_pretrained(
        args.retriever,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(
        args.retriever, local_files_only=True
    )

    query_inputs = processor.process_queries(
        [row["query"] for row in queries]
    ).to(model.device)
    page_embeddings = [embedding.to(model.device) for embedding in payload["embeddings"]]
    with torch.inference_mode():
        query_embeddings = model(**query_inputs).embeddings
        scores = processor.score_retrieval(
            query_embeddings,
            page_embeddings,
            output_dtype=torch.float32,
        ).cpu()

    all_hits = []
    for query_index in range(len(queries)):
        values, indices = torch.topk(scores[query_index], k=args.top_k)
        hits = []
        for rank, (value, page_index) in enumerate(
            zip(values.tolist(), indices.tolist()), start=1
        ):
            record = records[page_index]
            hits.append(
                {
                    "rank": rank,
                    "score": round(value, 4),
                    "pdf_filename": record["pdf_filename"],
                    "page_number": record["page_number"],
                    "image_path": record["image_path"],
                }
            )
        all_hits.append(hits)

    del model, processor, query_inputs, query_embeddings, page_embeddings, scores, payload
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Retriever released; allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    return all_hits


def build_messages(
    query: str, hits: list[dict], retry_reason: str | None = None
) -> list[dict]:
    allowed_sources = [f"S{hit['rank']}" for hit in hits]
    allowed_text = "、".join(allowed_sources)
    content = []
    for hit in hits:
        source_id = f"S{hit['rank']}"
        content.append(
            {
                "type": "text",
                "text": (
                    f"来源[{source_id}]开始；准确元数据：{hit['pdf_filename']}，"
                    f"PDF第{hit['page_number']}页。"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": Path(hit["image_path"]).resolve().as_uri(),
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }
        )
        content.append({"type": "text", "text": f"来源[{source_id}]结束。"})

    retry_text = ""
    if retry_reason:
        retry_text = (
            "\n上一次输出未通过自动校验，原因是："
            + retry_reason
            + "。请重新检查格式、语言和来源字段。"
        )
    content.append(
        {
            "type": "text",
            "text": (
                "请仅根据上述页面回答问题，证据不足时明确说明。只输出一个合法JSON"
                "对象，不要输出Markdown代码块、标题、解释或公式环境。JSON必须严格"
                "采用以下结构："
                '{"points":[{"text":"简体中文事实要点","sources":["S1"]}]}'
                "。points必须有1至6项，以完整覆盖问题且避免重复；text必须是简体"
                "中文；每项sources必须至少包含一个支持该事实的来源编号，并且只能"
                f"使用{allowed_text}。不要自行书写文件名或页码。"
                + retry_text
                + "\n问题："
                + query
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的多模态文档问答助手。无论问题使用哪种语言，"
                "都必须使用简体中文回答并严格引用给定来源编号。"
            ),
        },
        {"role": "user", "content": content},
    ]


def parse_validate_and_ground(
    raw_answer: str, hits: list[dict]
) -> tuple[str, bool, bool, list[str], str, bool]:
    mapping = {
        f"S{hit['rank']}": f"[{hit['pdf_filename']}, PDF p.{hit['page_number']}]"
        for hit in hits
    }
    payload = None
    strict_json_output = False
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_answer):
        try:
            candidate, consumed = decoder.raw_decode(raw_answer[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            json_text = raw_answer[match.start() : match.start() + consumed]
            strict_json_output = raw_answer.strip() == json_text.strip()
            break
    if payload is None:
        return raw_answer, False, is_chinese_answer(raw_answer), [], "没有找到JSON对象", False

    points = payload.get("points") if isinstance(payload, dict) else None
    if not isinstance(points, list) or not 1 <= len(points) <= 6:
        return raw_answer, False, is_chinese_answer(raw_answer), [], "points必须包含1至6项", strict_json_output

    rendered = []
    cited = []
    errors = []
    texts = []
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            errors.append(f"第{index}项不是对象")
            continue
        text_value = point.get("text")
        sources = point.get("sources")
        if not isinstance(text_value, str) or not text_value.strip():
            errors.append(f"第{index}项缺少text")
            continue
        if not isinstance(sources, list) or not sources:
            errors.append(f"第{index}项缺少sources")
            continue
        normalized_sources = list(dict.fromkeys(str(source) for source in sources))
        invalid_sources = set(normalized_sources) - set(mapping)
        if invalid_sources:
            errors.append(f"第{index}项包含非法来源：{sorted(invalid_sources)}")
            continue
        texts.append(text_value.strip())
        cited.extend(normalized_sources)
        source_text = " ".join(mapping[source] for source in normalized_sources)
        rendered.append(f"- {text_value.strip()} {source_text}")

    answer = "\n".join(rendered) if rendered else raw_answer
    chinese_valid = is_chinese_answer("\n".join(texts))
    valid = not errors and len(rendered) == len(points)
    reason_parts = list(errors)
    if not chinese_valid:
        reason_parts.append("中文字符不足")
    reason = "；".join(reason_parts)
    return answer, valid, chinese_valid, sorted(set(cited)), reason, strict_json_output


def is_chinese_answer(text: str) -> bool:
    return len(re.findall(r"[\u4e00-\u9fff]", text)) >= 8


def generate_once(generator, processor, messages: list[dict], max_new_tokens: int):
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
        generated_ids = generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del inputs, generated_ids, trimmed
    torch.cuda.empty_cache()
    return raw_answer, elapsed


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
    generator_processor = AutoProcessor.from_pretrained(
        args.generator, local_files_only=True
    )
    print(f"Generator loaded; allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    results = []
    for row, hits in zip(queries, all_hits):
        retrieval_hit = row["expected_pdf"] in {
            hit["pdf_filename"] for hit in hits
        }
        attempts = []
        retry_reason = None
        for attempt_number in (1, 2):
            messages = build_messages(row["query"], hits, retry_reason=retry_reason)
            raw_answer, elapsed = generate_once(
                generator,
                generator_processor,
                messages,
                args.max_new_tokens,
            )
            answer, citations_valid, chinese_valid, cited_source_ids, reason, strict_json = (
                parse_validate_and_ground(raw_answer, hits)
            )
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_answer": raw_answer,
                    "answer": answer,
                    "citation_validation": citations_valid,
                    "chinese_validation": chinese_valid,
                    "cited_source_ids": cited_source_ids,
                    "validation_error": reason,
                    "strict_json_output": strict_json,
                    "generation_seconds": round(elapsed, 2),
                }
            )
            if citations_valid and chinese_valid:
                break
            retry_reason = reason or "输出不符合JSON、中文或来源要求"

        first_attempt = attempts[0]
        final_attempt = attempts[-1]
        answer = final_attempt["answer"]
        citations_valid = final_attempt["citation_validation"]
        chinese_valid = final_attempt["chinese_validation"]
        cited_source_ids = final_attempt["cited_source_ids"]
        generation_seconds = sum(item["generation_seconds"] for item in attempts)
        cited_expected_document = any(
            hit["pdf_filename"] == row["expected_pdf"]
            for hit in hits
            if f"S{hit['rank']}" in cited_source_ids
        )

        print(
            f"{row['query_id']} retrieval={'PASS' if retrieval_hit else 'FAIL'} "
            f"attempts={len(attempts)} "
            f"citations={'PASS' if citations_valid else 'FAIL'} "
            f"expected_source={'PASS' if cited_expected_document else 'FAIL'} "
            f"chinese={'PASS' if chinese_valid else 'FAIL'} "
            f"generation={generation_seconds:.2f}s"
        )
        print(f"  {answer}\n")
        results.append(
            {
                **row,
                "retrieval_hit_at_k": retrieval_hit,
                "hits": hits,
                "answer": answer,
                "attempts": attempts,
                "retried": len(attempts) > 1,
                "first_pass": {
                    "citation_validation": first_attempt["citation_validation"],
                    "chinese_validation": first_attempt["chinese_validation"],
                    "strict_json_output": first_attempt["strict_json_output"],
                },
                "citation_validation": {
                    "passed": citations_valid,
                    "cited_source_ids": cited_source_ids,
                    "cited_expected_document": cited_expected_document,
                },
                "chinese_validation": chinese_valid,
                "strict_json_output": final_attempt["strict_json_output"],
                "generation_seconds": round(generation_seconds, 2),
            }
        )

    count = len(results)
    metrics = {
        "queries": count,
        "top_k": args.top_k,
        "retrieval_hit_at_k": round(
            sum(row["retrieval_hit_at_k"] for row in results) / count, 4
        ),
        "first_pass_valid_citation_rate": round(
            sum(row["first_pass"]["citation_validation"] for row in results)
            / count,
            4,
        ),
        "first_pass_chinese_answer_rate": round(
            sum(row["first_pass"]["chinese_validation"] for row in results)
            / count,
            4,
        ),
        "first_pass_strict_json_rate": round(
            sum(row["first_pass"]["strict_json_output"] for row in results)
            / count,
            4,
        ),
        "retry_rate": round(sum(row["retried"] for row in results) / count, 4),
        "valid_citation_rate": round(
            sum(row["citation_validation"]["passed"] for row in results) / count, 4
        ),
        "expected_document_citation_rate": round(
            sum(
                row["citation_validation"]["cited_expected_document"]
                for row in results
            )
            / count,
            4,
        ),
        "chinese_answer_rate": round(
            sum(row["chinese_validation"] for row in results) / count, 4
        ),
        "strict_json_output_rate": round(
            sum(row["strict_json_output"] for row in results) / count, 4
        ),
        "mean_generation_seconds": round(
            sum(row["generation_seconds"] for row in results) / count, 2
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


if __name__ == "__main__":
    main()
