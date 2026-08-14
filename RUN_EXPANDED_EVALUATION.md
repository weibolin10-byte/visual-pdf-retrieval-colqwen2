# 扩充评测运行说明

把压缩包上传到 AutoDL 项目目录 `/root/autodl-tmp/visual-rag`，解压后在该目录运行。

## 1. 40 条严格页面检索

### ColQwen2

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1

python evaluate_page_retrieval.py \
  --queries page_retrieval_queries_v2_40.csv \
  --metadata data/page_metadata.jsonl \
  --index data/index/colqwen2_v1_hf_final_2312.pt \
  --output outputs/page_retrieval_evaluation_v2_40.json
```

### PDF 原生文字层 + BM25

```bash
python evaluate_bm25_page_retrieval.py \
  --queries page_retrieval_queries_v2_40.csv \
  --metadata data/page_metadata.jsonl \
  --pdf-dir data/pdfs \
  --text-cache data/index/page_text_cache.jsonl \
  --baseline-name pdf_text_layer_bm25 \
  --output outputs/bm25_page_retrieval_evaluation_v2_40.json
```

### EasyOCR + BM25

```bash
python evaluate_bm25_page_retrieval.py \
  --queries page_retrieval_queries_v2_40.csv \
  --metadata data/page_metadata.jsonl \
  --pdf-dir data/pdfs \
  --text-cache data/index/easyocr_page_text.jsonl \
  --baseline-name easyocr_bm25 \
  --output outputs/easyocr_bm25_page_retrieval_evaluation_v2_40.json
```

这里复用已有的文字层和 EasyOCR 缓存，不需要再次对 2312 页做 OCR。

## 2. 16 条答案生成质量评测

```bash
python evaluate_generation_quality.py \
  --queries generation_quality_queries_v1_16.csv \
  --metadata data/page_metadata.jsonl \
  --index data/index/colqwen2_v1_hf_final_2312.pt \
  --top-k 5 \
  --output outputs/generation_quality_raw_results.json \
  --manual-template outputs/generation_quality_manual_scores.csv

unset HF_HUB_OFFLINE
```

这一步包括 12 个可由语料回答的问题和 4 个语料中没有答案的问题。程序自动检查检索命中、引用页码和拒答情况；事实是否正确、回答是否完整，需要查看原始答案后逐条核对。

## 3. 打包运行结果

```bash
tar -czf outputs/expanded_evaluation_raw_bundle.tar.gz \
  outputs/page_retrieval_evaluation_v2_40.json \
  outputs/bm25_page_retrieval_evaluation_v2_40.json \
  outputs/easyocr_bm25_page_retrieval_evaluation_v2_40.json \
  outputs/generation_quality_raw_results.json \
  outputs/generation_quality_manual_scores.csv
```

下载并上传 `outputs/expanded_evaluation_raw_bundle.tar.gz`。收到后将逐条核对 12 个可回答问题，填写事实正确数、完整度和页面支持度，再运行：

```bash
python score_generation_quality.py \
  --raw-results outputs/generation_quality_raw_results.json \
  --manual-scores outputs/generation_quality_manual_scores.csv \
  --output outputs/generation_quality_final_evaluation.json
```
