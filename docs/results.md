# Results ledger

**Provenance legend.** `PAPER-REPORTED` = transcribed from the IEEE paper (rounded to the paper's precision). `MEASURED` = produced by
`runs/scifact-nomic-embed/` (artifacts: per-query rankings, per-query CSVs, `run.json`, `metrics.json`). `NOT RUN` = no data/resources.
Synthetic demo outputs are excluded from this ledger.

**Measured run:** SciFact (BEIR format), 5,183 docs, released test split, 300 queries; train queries (used only to pick α and train the
policy). Encoder `ollama:nomic-embed-text` (digest `0a109f42…`, a transformer; **not BERT**), CPU, Windows, 12 cores, concurrency 1 for latency,
single process. Fixed α = 0.7, selected on train NDCG@10 (curve in `alpha_selection.json`).

## Metric conventions
Relevant = qrels grade ≥ 1; unjudged = non-relevant. P@10 = hits/10. **Recall cutoff = 10** (the paper's unqualified "Recall" has no stated cutoff,
so none is assumed for it). F1 = macro mean of per-query F1@10 (pooled F1 also in `metrics.json`). NDCG@10: linear gain, log2(i+1) discount,
zero-ideal queries undefined (none here: 300/300 defined).

> **Why our P@10 is ~0.09 and the paper's is ~0.9:** SciFact has ≈1.13 relevant documents per query (339 judgments / 300 queries), so P@10
> cannot exceed ≈0.11 regardless of system. The paper's P@10 values are not comparable to any SciFact number. Compare ranking order and NDCG, not magnitude.

## Table I: retrieval accuracy
| System | Paper P@10 / Recall / F1 | **MEASURED** P@10 | Recall@10 | F1@10 |
|---|---|---|---|---|
| TF-IDF | 0.71 / 0.66 / 0.68 (PAPER-REPORTED) | 0.0867 | 0.7835 | 0.1540 |
| BM25 | 0.78 / 0.72 / 0.75 | 0.0877 | 0.7889 | 0.1557 |
| Dense ("BERT retrieval" in paper) | 0.86 / 0.81 / 0.83 | 0.0950 | 0.8436 | 0.1681 |
| Knowledge-graph search | 0.88 / 0.84 / 0.86 | **NOT RUN** (SciFact has no graph) | – | – |
| Hybrid (α=0.7) | 0.94 / 0.91 / 0.92 | 0.0963 | 0.8586 | 0.1706 |

Order TF-IDF < BM25 < dense < hybrid agrees with the paper. Magnitudes are not comparable (see note above). No confidence intervals are claimed.

## Table II: NDCG@10
| System | Paper | **MEASURED** |
|---|---|---|
| BM25 | 0.74 | 0.6626 |
| Dense | 0.82 (BERT) | 0.7041 |
| RAG | 0.87 | 0.7041 (**identical to dense by construction**: extractive CPU fallback cites the dense top passages; a generator-backed RAG was **NOT RUN** at 300 queries) |
| Hybrid (α=0.7, fixed) | 0.93 | 0.7283 |
| Hybrid + frozen Q-policy | – | 0.7363 (in the ablation "full" row; graph unavailable) |

## Table III: latency (retrieval only, warm, concurrency 1, 100 queries)
| System | Paper mean ms | **MEASURED** mean / p50 / p95 ms |
|---|---|---|
| TF-IDF | 145 | 62.3 / 52.8 / 120.6 |
| BM25 | 172 | 59.8 / 50.3 / 121.3 |
| Dense | 310 | 28.6 / 23.5 / 63.6 |
| RAG (extractive, retrieval only) | 355 | 76.8 / 61.2 / 161.5 |
| Hybrid | 198 | 105.2 / 85.9 / 202.8 |

**Where the paper's pattern fails here:** hybrid is *slower* than every single retriever we ran (it computes both and fuses); the dense matmul over
5,183 vectors is the cheapest component. Pure-python BM25/TF-IDF are slow in our implementation, unlike the paper's. Latency excludes the
Ollama query-embedding call (query vectors are memoised, so it is not repeated per system) and excludes generation.

## Table IV: throughput
Paper: 420 / 395 / 372 / 351 queries/s at 100K / 300K / 600K / 1M (PAPER-REPORTED). **MEASURED** is a *worker sweep on the fixed 5,183-doc corpus*
(a different question): 1 worker 9.08 q/s, 2 workers 10.66 q/s, 4 workers 10.71 q/s (300 completed each, 0 errors; Python threads, GIL-bound).
**100K / 300K / 600K / 1M: NOT RUN** (no manifest of that size). Do not read 10 q/s as comparable with 351–420 q/s.

## Table V: knowledge discovery
Qualitative in the paper; trend-prediction accuracy 81% → 92% (PAPER-REPORTED). **NOT RUN**: no dated topic gold labels, adjudicated cross-domain
links or directed-path truth exist for SciFact. `discovery.forecast_topics` is tested for cutoff correctness on synthetic data only.

## Table VI: ablation (P@10 paper vs measured P@10 / NDCG@10)
| Configuration | Paper P@10 | **MEASURED** P@10 / NDCG@10 | Replacement used |
|---|---|---|---|
| Without knowledge graph | 0.88 | **NOT RUN** (graph does not exist here; the `no_graph` row equals `full`: 0.0960 / 0.7363) | – |
| Without hybrid retrieval | 0.86 | 0.0950 / 0.7041 | dense-only retriever |
| Without reinforcement ranking | 0.90 | 0.0963 / 0.7283 | fixed α = 0.7 |
| Full | 0.94 | 0.0960 / 0.7363 | Q-policy chooses α ∈ {0.2,0.5,0.8} per query |

The frozen policy (trained on 800 simulated train episodes, rewards from train qrels) adds +0.008 NDCG@10 over fixed α=0.7; its P@10 is marginally
*lower* (0.0960 vs 0.0963). With one run and no per-query significance test, treat this as unproven.

## Reproduction gaps (outstanding)
1. Original corpus, queries, judgments, logs, checkpoints, ontology and cluster config are unavailable, so no paper value was re-created.
2. Graph search, graph ablation, KG-based expansion: unevaluated on SciFact (demo-only, synthetic).
3. Table V and 100K–1M scaling: not run.
4. RAG with a real generator on the full query set: not run (extractive fallback only).
5. PySpark backend: coded, not executed (no JVM). Multiprocessing + Parquet backend executed in tests.
6. Q-learning rewards are simulated from qrels; no real clicks exist.
7. Run recorded `git_commit: no-git` and `encoder.dim: null`; both are corrected/annotated in `run.json` (`post_run_note`).
