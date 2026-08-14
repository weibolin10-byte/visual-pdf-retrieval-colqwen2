# Retrieval and Answer Generation Evaluation Report

## 1. Experiment snapshot

This report evaluates zero-shot ColQwen2 visual page retrieval on a fixed multi-domain PDF corpus and compares it with both native PDF text-layer BM25 and true EasyOCR + BM25 baselines.

| Item | Value |
|---|---:|
| Documents | 30 |
| Pages | 2312 |
| Domains | papers, financial reports, public reports, technical manuals |
| Languages | English and Chinese |
| ColQwen2 index | 426.6 MiB |
| Index build time | 200.5 s |
| Page-level queries | 40 |
| Document-level queries | 40 |
| Generation-quality queries | 16 (12 answerable, 4 unanswerable) |

No model training or fine-tuning was performed.

## 2. Evaluation protocol

### Document level

Every query is scored against all 2312 page embeddings. A document receives the maximum score of any page belonging to it. The benchmark contains 40 questions whose target documents were decided before evaluation.

This protocol measures closed-set document routing. It does not require the highest-ranked page to be the exact answer page.

### Page level

Forty visual candidates were manually selected before running the page-level evaluator. They include charts, financial tables, architecture figures, pin diagrams, mechanical drawings, comics, and specification sheets. The set contains 10 questions per domain and 20 questions at each difficulty level. A hit is counted only when the exact annotated PDF page appears in the top-k results; adjacent pages are not accepted.

The first 20-question version contained one page-label error in V01. Direct page inspection showed that PDF page 52, originally returned at rank 1, was the revenue table described by the query, while the original target page 41 was a cost-of-revenue table. Version 1.1 corrected only that page label. The additional V21-V40 questions and exact page labels were then written before their model results were seen.

### BM25 baselines

Both BM25 systems use the same 40 questions, exact page labels, tokenizer, and BM25 parameters (`k1=1.5`, `b=0.75`). The first uses text extracted from each PDF's native text layer and is nonempty for 98.75% of pages.

The second is a true OCR baseline: EasyOCR 1.7.2 reads the 150-DPI rendered JPG pages with `ch_sim` and `en`; it never accesses the native PDF text layer. OCR text is nonempty for 2291 of 2312 pages (99.09%). No confidence filtering, text cleanup, query-specific tuning, or OCR fine-tuning was applied.

## 3. Results

### Document-level retrieval

| Queries | Recall@1 | Recall@3 | Recall@5 | MRR |
|---:|---:|---:|---:|---:|
| 40 | 1.00 | 1.00 | 1.00 | 1.00 |

Batch evaluation including model loading took 3.29 seconds. This timing is not a production per-query latency measurement.

### Page-level retrieval

| Method | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| ColQwen2 | **0.90** | **0.975** | **1.00** | **1.00** | **0.9437** |
| Text-layer BM25 | 0.775 | 0.90 | 0.925 | 0.925 | 0.8325 |
| EasyOCR + BM25 | 0.75 | 0.85 | 0.925 | 0.925 | 0.8163 |
| ColQwen2 minus text-layer BM25 | **+0.125** | **+0.075** | **+0.075** | **+0.075** | **+0.1112** |
| ColQwen2 minus EasyOCR + BM25 | **+0.15** | **+0.125** | **+0.075** | **+0.075** | **+0.1274** |

### Page Recall@1 by domain

| Domain | ColQwen2 | Text-layer BM25 | EasyOCR + BM25 |
|---|---:|---:|---:|
| Financial reports | **0.90** | 0.60 | 0.50 |
| Papers | **0.90** | 0.90 | 0.90 |
| Public reports | **0.80** | 0.80 | 0.80 |
| Technical manuals | **1.00** | 0.80 | 0.80 |

The strongest visual evidence appears in technical manuals, where spatial layout, pin placement, and mechanical drawings carry information that is poorly represented by either extracted or OCR text order. EasyOCR's slightly higher text coverage does not translate into higher retrieval quality: recognition noise and layout loss reduce R@1 below the native-text baseline.

## 4. Paired query comparison

Both systems used identical queries and gold labels.

| Outcome | Count |
|---|---:|
| ColQwen2 obtains a better gold-page rank | 9 |
| BM25 obtains a better gold-page rank | 3 |
| Same gold-page rank | 28 |

| Query | ColQwen2 rank | BM25 rank | Interpretation |
|---|---:|---:|---|
| V01 Tesla revenue table | 1 | 3 | Visual table structure helps distinguish revenue from nearby cost tables. |
| V04 Apple regional segment table | 1 | 2 | Visual table and regional layout improve exact-page localization. |
| V05 NVIDIA three-year segment table | 2 | 1 | Exact financial terminology favors BM25. |
| V08 MInference plots and heatmap | 4 | 5 | Both systems are distracted by the visually related title page. |
| V11 South Asia multi-panel chart | 1 | 3 | ColQwen2 better matches the requested four-panel composition. |
| V15 WHO Triple Billion trends | 2 | 1 | Exact section terms allow BM25 to locate the target page. |
| V17 CC1352P pin diagram | 1 | 31 | Strong visual-retrieval win; extracted text loses pin geometry. |
| V19 camera mechanical drawings | 1 | 15 | Strong visual-retrieval win; side-by-side drawings carry the query intent. |
| V21 NVIDIA compensation tables | 1 | 3 | Visual grouping distinguishes stacked tables with similar terminology. |
| V25 Apple product revenue table | 1 | 2 | Table structure improves exact-page localization. |
| V31 four-panel commuter comic | 1 | 1401 | Native text contains little of the visual sequence semantics. |
| V33 AI publication trend chart | 2 | 1 | Exact regional terms favor BM25; ColQwen2 returns the adjacent citation chart first. |

ColQwen2 wins nine of the twelve non-tied comparisons, but the test set is too small for a statistical-significance claim.

Against EasyOCR + BM25, ColQwen2 obtains a better gold-page rank on 9 queries, EasyOCR + BM25 is better on 2, and 29 are tied. Three representative visual cases are especially stable across both textual baselines:

| Query | ColQwen2 | Text-layer BM25 | EasyOCR + BM25 |
|---|---:|---:|---:|
| V17 CC1352P pin diagram | **1** | 31 | 29 |
| V19 camera mechanical drawings | **1** | 15 | 16 |
| V31 four-panel commuter comic | **1** | 1401 | 45 |

## 5. ColQwen2 error analysis

Four strict top-1 errors remain:

- **V05, gold rank 2:** an NVIDIA business-summary page contains the same segment names and fiscal-2025 values as the three-year statement page. Score margin to the gold page was 0.25.
- **V08, gold rank 4:** the MInference title page contains a Needle in a Haystack heatmap and latency plot, creating genuine partial relevance to a query targeting the experiment page.
- **V15, gold rank 2:** a WHO table-of-contents page contains exact Triple Billion section terminology and narrowly outranks the page containing the three requested charts. Score margin was 0.125.
- **V33, gold rank 2:** the adjacent regional citation chart shares the same regions and year, so it outranks the target publication-share chart.

All four errors remain within the correct PDF and all target pages appear within top 5.

## 6. Answer generation quality

Qwen2.5-VL answered from the ColQwen2 top-5 page images. The 16 questions contain 12 answerable cases and 4 deliberately unanswerable cases. Reference answers and expected pages were written before generation. The 12 answerable cases were split into 61 required facts and checked one by one against the source pages.

| Metric | Result |
|---|---:|
| Answerable-question retrieval hit@5 | 0.9167 |
| Key-fact accuracy | 0.7541 (46/61) |
| Answer completeness | 0.9167 |
| Correct exact-page citation rate | 0.6667 |
| Page-support rate after manual checking | 0.6667 |
| Correct refusal rate on unanswerable questions | 1.00 (4/4) |
| Mean generation time | 2.36 s |

The most important failures are not output formatting failures. One answerable Microsoft question was missed because the correct page did not enter top 5. Other errors came from selecting the wrong row in a nearby table, confusing RAG-Token with RAG-Sequence, reading citation-share values as publication-share values, and attaching an answer to a page that did not support it. Therefore retrieval quality and answer quality are reported separately.

## 7. Timing interpretation

- ColQwen2 page-index construction: 200.5 seconds for 2312 pages on RTX 5090, producing a 426.6 MiB index.
- ColQwen2 page evaluation: 3.43 seconds for a batch of 40 queries including model loading, with the visual index already built.
- BM25 first evaluation: 34.89 seconds including native PDF text extraction and cache creation.
- EasyOCR preprocessing: 11,542.39 seconds (3 h 12 min) for 2312 page images, averaging 4.99 seconds per page.
- EasyOCR + BM25 evaluation: 0.77 seconds for the 40-query batch with the OCR text cache already built.

ColQwen2's page-index construction was about 57.6 times faster than this EasyOCR preprocessing run on the same corpus. Query-time figures are not directly comparable because model loading, tokenization, scoring, and cache boundaries differ. A production benchmark should separately measure cold start, warm query encoding, scoring, and storage cost.

## 8. Limitations

- The page-level benchmark contains 40 manually authored queries; the generation-quality set contains 16.
- Query types are balanced across four domains but are not sampled from a public benchmark.
- The OCR baseline operates on rendered pages from native PDFs; robustness to low-resolution scans, skew, compression artifacts, and handwriting is not established.
- The document-level set is easier and should not be conflated with strict page retrieval.
- No fine-tuning was performed, so results reflect the pretrained model and this corpus only.
- The answer-quality scores are based on a small manually checked set and reveal substantial complex-table reading errors; they should not be generalized as model-wide accuracy.
- The evaluation supports a project-level engineering conclusion, not a general SOTA claim.

## 9. Reproducibility artifacts

- `retrieval_final_queries.csv`
- `page_retrieval_queries.csv` — frozen v1
- `page_retrieval_queries_v1_1.csv` — one audited label correction
- `page_retrieval_queries_v2_40.csv` — 40 exact-page questions
- `generation_quality_queries_v1_16.csv` — 12 answerable and 4 unanswerable questions
- `page_eval_annotation_audit.md`
- `evaluate_retrieval.py`
- `evaluate_page_retrieval.py`
- `evaluate_bm25_page_retrieval.py`
- `evaluate_generation_quality.py`
- `score_generation_quality.py`
- `build_easyocr_cache.py`
- `outputs/retrieval_final_evaluation.json`
- `outputs/page_retrieval_evaluation.json`
- `outputs/page_retrieval_evaluation_v1_1.json`
- `outputs/bm25_page_retrieval_evaluation.json`
- `outputs/easyocr_bm25_page_retrieval_evaluation.json`
- `outputs/generation_quality_final_evaluation.json`
