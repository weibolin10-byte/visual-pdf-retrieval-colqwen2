# Page-level retrieval annotation audit

## Benchmark status

- The original `page_retrieval_queries.csv` and its evaluation JSON remain unchanged as benchmark v1.
- Visual review of every non-top-1 case found one objective label error in V01.
- Benchmark v1.1 changes only V01 `expected_pages` from PDF page 41 to PDF page 52.
- No query text, model, index, retrieval parameter, or other gold label was changed.

## V01 correction

The V01 query asks for the blue Tesla table comparing automotive sales, regulatory credits, leasing, energy generation and storage, and services across 2024 and 2023. PDF page 52 contains that consolidated revenue table. PDF page 41 instead contains narrative discussion followed by a cost-of-revenues and gross-margin table. The model's original top-1 page 52 is therefore the correct gold page.

## Audited failure analysis

| Query | v1 gold rank | Finding |
|---|---:|---|
| V01 | 3 | Annotation error. Top-1 PDF page 52 exactly matches the query; v1.1 corrects the gold label. |
| V05 | 2 | Valid model error. NVIDIA PDF page 18 is a visually prominent fiscal-2025 summary with the same segment names and values, while gold page 170 is the requested three-year financial-statement detail. Score margin: 0.25. |
| V08 | 4 | Valid model error with partial relevance. The title page contains a Needle in a Haystack heatmap and latency plot; gold page 8 contains the requested group of line plots and heatmap. |
| V15 | 2 | Valid model error. WHO PDF page 6 is a table of contents containing exact Triple Billion section terms; gold page 64 contains the requested three trend charts. Score margin: 0.125. |

## Metrics

| Metric | Frozen v1 | Annotation-audited v1.1 |
|---|---:|---:|
| Page Recall@1 | 0.80 | 0.85 |
| Page Recall@3 | 0.95 | 0.95 |
| Page Recall@5 | 1.00 | 1.00 |
| Page MRR | 0.8792 | 0.9125 |

The v1.1 metrics are derived from the same frozen ranking after correcting V01. A reproducibility run should use `page_retrieval_queries_v1_1.csv` and keep the original v1 artifacts for provenance.
