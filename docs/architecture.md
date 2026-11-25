# Architecture

```
JSONL / exports ─► data.ingest ─► snapshot (corpus.jsonl + hash) ─► distributed.publish_generation ─► shards (Parquet) + CURRENT
                                         │                                   │
                                         ▼                                   ▼
                              synthesis.extract_claims ─► graph.KnowledgeGraph      encoders ─► DenseIndex     BM25Index
                                         │                       │                       └────────┬───────────┘
                                         ▼                       ▼                                ▼
                                   workbench ◄──────── retrieval.HybridSearcher ◄── feedback.QPolicy (alpha profile)
                                         │
                      report / synthesis / citation path / trends  ─►  experiments (runs/<id>/) ─► plots, ledger
```

| Layer (paper Fig 1) | File | Input → output | Learned? |
|---|---|---|---|
| Ingestion | `data.py` | JSONL → `Work`, `Passage` | no |
| Distributed storage/processing | `distributed.py` | works → Parquet shards, manifest, CURRENT | no |
| LLM intelligence | `encoders.py`, `synthesis.py` | text → vectors; passages → claims/answers | pretrained |
| Knowledge graph | `graph.py` | works + claims → typed graph | no (rule/lexicon extraction) |
| Adaptive retrieval | `retrieval.py`, `feedback.py` | query → ranked works + score breakdown | Q-table only |
| Decision support | `workbench.py`, `plots.py` | → report, figures, ledger | no |

## Implementation choices the paper leaves open (all changeable in `configs/`)
* **Score calibration**: raw BM25 and cosine are min-max normalised over the *union of lexical and dense candidates*; both raw
  scores are computed for every candidate, so there is no "missing candidate". A constant-score component normalises to 0.0 (it
  adds no ranking information). Raw scores and `candidate_depth` are stored with every hit.
* **Passage→document**: dense document score = max cosine over its passages. BM25 runs on title+abstract+body.
* **Zero vector**: empty text → zero vector → cosine 0.0 with everything.
* **Ties**: descending score, then ascending id. Deterministic.
* **Lexical stop words** only in the BM25/TF-IDF branch, and negations are kept; the encoder input is the raw passage.
* **Q-learning**: actions = α ∈ {0.2, 0.5, 0.8}; state = (query length bucket, lexical/dense top-5 agreement); reward = NDCG@10 from
  *training* qrels (simulated feedback); single-query episodes are terminal; ε-greedy in training, greedy + frozen at evaluation.
  Real clicks would need exposure/position-bias correction before use as rewards. Policies are per user; default is neutral α=0.5.
* **Graph-derived feature**: only query expansion — terms from method/topic nodes that co-occur with nodes named in the query,
  appended to the BM25 query and recorded on every hit. `graph=None` ⇒ no expansion terms (tested).
* **Trend**: year bins from the record's own date; `T_k = Σ w_t F_kt`, `w_t = decay^(t_end−t)`; description only.
  Forecast removes all works after the cutoff and scores a predeclared top-n target against a held-out year.
* **Backends**: every run records backend, workers and mode. `local` = single process. `multiprocessing-parquet` = N workers,
  one Parquet shard each, global BM25 statistics merged from shard stats, shard-wise scoring merged. `pyspark` code exists but
  needs a JVM and was **not executed**.
* **Substitutes, labelled**: `hashing` encoder (deterministic, not a transformer), `extractive-cpu-fallback` generator.
  Real transformer: `ollama:nomic-embed-text` (digest recorded). Real generator: `ollama:<model>`.
* **Untrusted documents**: passages matching injection patterns are excluded from synthesis; the generator prompt marks passages as
  data; generated sentences must cite a supplied passage and be lexically supported by it or they are rejected. No tool is ever
  invoked from document text.
