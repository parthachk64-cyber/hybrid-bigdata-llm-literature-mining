
### Table I analogue: retrieval accuracy

> **MEASURED** — run `scifact-nomic-embed`, SciFact test, 5183 docs, 300 test queries, encoder `ollama:nomic-embed-text` (transformer: True). Not comparable with the paper's values.

| System | P@10 | Recall@10 | F1@10 (macro) |
|---|---|---|---|
| TF-IDF | 0.0867 | 0.7835 | 0.1540 |
| BM25 | 0.0877 | 0.7889 | 0.1557 |
| Dense retrieval | 0.0950 | 0.8436 | 0.1681 |
| Hybrid (fixed alpha) | 0.0963 | 0.8586 | 0.1706 |

### Table II analogue: ranking quality

> **MEASURED** — run `scifact-nomic-embed`, SciFact test, 5183 docs, 300 test queries, encoder `ollama:nomic-embed-text` (transformer: True). Not comparable with the paper's values.

| System | NDCG@10 | queries defined |
|---|---|---|
| BM25 | 0.6626 | 300 |
| Dense retrieval | 0.7041 | 300 |
| RAG | 0.7041 | 300 |
| Hybrid (fixed alpha) | 0.7283 | 300 |

### Table III analogue: latency (concurrency 1, warm)

> **MEASURED** — run `scifact-nomic-embed`, SciFact test, 5183 docs, 300 test queries, encoder `ollama:nomic-embed-text` (transformer: True). Not comparable with the paper's values. Retrieval-only unless marked (+generation).

| System | mean ms | p50 ms | p95 ms | n |
|---|---|---|---|---|
| tfidf | 62.2721 | 52.8094 | 120.6227 | 100 |
| bm25 | 59.8477 | 50.2654 | 121.3361 | 100 |
| dense | 28.6402 | 23.4553 | 63.5986 | 100 |
| hybrid_fixed | 105.2191 | 85.9236 | 202.8206 | 100 |
| rag | 76.8164 | 61.1962 | 161.5374 | 100 |

### Throughput (fixed corpus, worker sweep)

> **MEASURED** — run `scifact-nomic-embed`, SciFact test, 5183 docs, 300 test queries, encoder `ollama:nomic-embed-text` (transformer: True). Not comparable with the paper's values. Fixed corpus; a worker sweep, not a corpus-size sweep.

| workers | completed | errors | elapsed s | queries/s |
|---|---|---|---|---|
| 1 | 300 | 0 | 33.0308 | 9.0824 |
| 2 | 300 | 0 | 28.1558 | 10.6550 |
| 4 | 300 | 0 | 28.0180 | 10.7074 |

### Table VI analogue: ablation (graph row n/a if no graph)

> **MEASURED** — run `scifact-nomic-embed`, SciFact test, 5183 docs, 300 test queries, encoder `ollama:nomic-embed-text` (transformer: True). Not comparable with the paper's values.

| Configuration | P@10 | NDCG@10 |
|---|---|---|
| full | 0.0960 | 0.7363 |
| no_graph | 0.0960 | 0.7363 |
| no_hybrid | 0.0950 | 0.7041 |
| no_rl | 0.0963 | 0.7283 |
