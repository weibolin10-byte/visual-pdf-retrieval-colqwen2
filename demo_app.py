#!/usr/bin/env python3
"""Gradio demo for ColQwen2 page retrieval with optional Qwen2.5-VL answering."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path

import gradio as gr
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
    parser.add_argument("--metadata", type=Path, default=Path("data/page_metadata.jsonl"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/index/colqwen2_v1_hf_final_2312.pt")
    )
    parser.add_argument("--retriever", default="vidore/colqwen2-v1.0-hf")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No page metadata found: {path}")
    for record in records:
        if not Path(record["image_path"]).is_file():
            raise FileNotFoundError(record["image_path"])
    return records


class VisualRAG:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required.")
        self.args = args
        self.records = load_records(args.metadata)
        payload = torch.load(args.index, map_location="cpu", weights_only=True)
        if payload.get("model_name") != args.retriever:
            raise RuntimeError("Retriever name does not match the index.")
        if payload.get("page_ids") != [record["page_id"] for record in self.records]:
            raise RuntimeError("Metadata does not match the index.")

        local_only = not args.allow_network
        print(f"Loading retriever: {args.retriever}")
        self.retriever = ColQwen2ForRetrieval.from_pretrained(
            args.retriever,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=local_only,
        ).eval()
        self.retriever_processor = ColQwen2Processor.from_pretrained(
            args.retriever,
            local_files_only=local_only,
        )
        self.page_embeddings = [
            embedding.to(self.retriever.device) for embedding in payload["embeddings"]
        ]
        self.generator = None
        self.generator_processor = None
        self.lock = threading.Lock()
        print(
            f"Ready: {len(self.records)} pages; "
            f"VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
        )

    def retrieve(self, query: str, top_k: int) -> tuple[list[dict], float]:
        started = time.perf_counter()
        inputs = self.retriever_processor.process_queries([query]).to(
            self.retriever.device
        )
        with torch.inference_mode():
            query_embedding = self.retriever(**inputs).embeddings
            scores = self.retriever_processor.score_retrieval(
                query_embedding,
                self.page_embeddings,
                output_dtype=torch.float32,
            )[0]
        values, indices = torch.topk(scores, k=min(top_k, len(self.records)))
        hits = []
        for rank, (score, index) in enumerate(
            zip(values.cpu().tolist(), indices.cpu().tolist()), start=1
        ):
            record = self.records[index]
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "pdf_filename": record["pdf_filename"],
                    "page_number": int(record["page_number"]),
                    "image_path": record["image_path"],
                }
            )
        return hits, time.perf_counter() - started

    def load_generator(self) -> None:
        if self.generator is not None:
            return
        local_only = not self.args.allow_network
        print(f"Loading generator: {self.args.generator}")
        self.generator = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.generator,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=local_only,
        ).eval()
        self.generator_processor = AutoProcessor.from_pretrained(
            self.args.generator,
            local_files_only=local_only,
        )
        print(f"Generator ready; VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    @staticmethod
    def build_messages(query: str, hits: list[dict]) -> list[dict]:
        content = []
        for hit in hits:
            source_id = f"S{hit['rank']}"
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"BEGIN SOURCE [{source_id}]. Exact metadata: "
                            f"{hit['pdf_filename']}, PDF page {hit['page_number']}."
                        ),
                    },
                    {
                        "type": "image",
                        "image": Path(hit["image_path"]).resolve().as_uri(),
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": f"END SOURCE [{source_id}]."},
                ]
            )
        valid_tokens = ", ".join(f"[S{hit['rank']}]" for hit in hits)
        content.append(
            {
                "type": "text",
                "text": (
                    "仅根据以上页面图像回答。使用简洁中文，并为每条事实引用对应来源标记。"
                    f"只允许使用这些标记：{valid_tokens}。证据不足时明确说明。\n\n"
                    f"问题：{query}"
                ),
            }
        )
        return [
            {"role": "system", "content": "You are a grounded visual document QA assistant."},
            {"role": "user", "content": content},
        ]

    def answer(self, query: str, hits: list[dict]) -> tuple[str, float]:
        self.load_generator()
        started = time.perf_counter()
        messages = self.build_messages(query, hits)
        prompt = self.generator_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.generator_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.generator.device)
        with torch.inference_mode():
            generated = self.generator.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        raw_answer = self.generator_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        mapping = {
            f"S{hit['rank']}": f"[{hit['pdf_filename']}, PDF p.{hit['page_number']}]"
            for hit in hits
        }
        cited = {f"S{number}" for number in re.findall(r"\[S(\d+)\]", raw_answer)}
        invalid = cited - set(mapping)
        if invalid:
            return f"生成器返回了非法来源标记：{sorted(invalid)}", time.perf_counter() - started
        if not cited:
            return "生成器没有返回可验证的来源标记。", time.perf_counter() - started
        grounded = re.sub(
            r"\[S(\d+)\]",
            lambda match: mapping[f"S{match.group(1)}"],
            raw_answer,
        )
        return grounded, time.perf_counter() - started

    def run(self, query: str, top_k: int, generate_answer: bool):
        query = (query or "").strip()
        if not query:
            raise gr.Error("请输入检索问题。")
        with self.lock:
            hits, retrieval_seconds = self.retrieve(query, int(top_k))
            gallery = [
                (
                    hit["image_path"],
                    f"#{hit['rank']} · {hit['pdf_filename']} · PDF p.{hit['page_number']} · score={hit['score']:.4f}",
                )
                for hit in hits
            ]
            status = (
                f"检索 {len(self.records)} 页，返回 Top-{len(hits)}，"
                f"耗时 {retrieval_seconds:.3f} 秒。"
            )
            if generate_answer:
                answer, generation_seconds = self.answer(query, hits)
                status += f" 回答生成耗时 {generation_seconds:.2f} 秒。"
            else:
                answer = "未启用答案生成；当前仅展示 ColQwen2 页面检索结果。"
        return status, gallery, answer


def build_demo(engine: VisualRAG):
    with gr.Blocks(title="ColQwen2 Visual PDF Retrieval") as demo:
        gr.Markdown(
            "# ColQwen2 Visual PDF Retrieval\n"
            "在论文、财报、公共报告和技术手册中直接检索 PDF 页面图像。"
        )
        query = gr.Textbox(
            label="问题",
            value="Find the page with the Raspberry Pi board physical dimensions.",
            lines=2,
        )
        with gr.Row():
            top_k = gr.Slider(1, 5, value=5, step=1, label="Top-k pages")
            generate_answer = gr.Checkbox(
                value=False,
                label="生成中文答案（首次使用会加载 Qwen2.5-VL）",
            )
        submit = gr.Button("检索", variant="primary")
        status = gr.Markdown()
        gallery = gr.Gallery(
            label="Retrieved pages",
            columns=5,
            rows=1,
            height="auto",
            object_fit="contain",
        )
        answer = gr.Markdown(label="Answer")
        submit.click(
            engine.run,
            inputs=[query, top_k, generate_answer],
            outputs=[status, gallery, answer],
        )
    return demo


def main() -> None:
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if not args.index.is_file():
        raise FileNotFoundError(args.index)
    engine = VisualRAG(args)
    demo = build_demo(engine)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
