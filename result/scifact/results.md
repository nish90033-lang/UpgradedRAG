# scifact ablation

Test queries: 150. Tuned on 150 dev queries: rrf_k=10, w_bm25=0.5. Seed 0.

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 vs v1 (95% CI) | nDCG@10 vs BM25 (95% CI) | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| v1 lexical (hashed TF + BM25Okapi, 70/30) | 0.568 | 0.693 | 0.850 | 0.533 | n/a | n/a | 44.2 | 85.7 |
| BM25 (v2 tokeniser) | 0.655 | 0.803 | 0.930 | 0.617 | +0.087 [+0.044, +0.131] | n/a | 3.4 | 6.0 |
| Dense (BAAI/bge-small-en-v1.5) | 0.685 | 0.796 | 0.953 | 0.655 | +0.117 [+0.062, +0.171] | +0.030 [-0.023, +0.082] | 53.0 | 65.9 |
| BM25 + dense (RRF) | 0.706 | 0.836 | 0.980 | 0.670 | +0.138 [+0.089, +0.189] | +0.051 [+0.007, +0.095] | 48.9 | 59.1 |
| BM25 + dense (RRF) + rerank top 50 | 0.686 | 0.825 | 0.980 | 0.649 | +0.118 [+0.069, +0.168] | +0.032 [-0.008, +0.072] | 4159.4 | 5317.6 |

Pairwise comparisons, nDCG@10 (paired bootstrap, 95% interval, not corrected for multiple comparisons):

| A | B | A minus B (95% CI) | p |
|---|---|---|---|
| BM25 + dense (RRF) | Dense (BAAI/bge-small-en-v1.5) | +0.021 [+0.000, +0.043] | 0.044 |
| BM25 + dense (RRF) + rerank top 50 | BM25 + dense (RRF) | -0.020 [-0.057, +0.018] | 0.313 |

Embedding 8184 chunks took 1283.7s (cached on later runs).
Sanitiser flagged 0 of 5183 corpus documents (0.00%).
