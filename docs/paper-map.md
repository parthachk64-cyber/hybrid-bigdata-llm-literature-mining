# Paper map: published specification → companion → code → test → run status

Paper: *Hybrid Big Data-LLM Framework for Intelligent Scientific Literature Mining*, IEEE I3CTCON 2026, DOI
[10.1109/I3CTCON68242.2026.11507315](https://doi.org/10.1109/I3CTCON68242.2026.11507315). The PDF is not redistributed.

**Available from the authors' paper:** aggregate comparison tables, equations, architecture figure.
**Not available (so not reproduced):** corpus snapshot, query set, relevance judgments, feedback sessions, model checkpoints,
graph ontology, cluster configuration, source code, licence. Table values therefore cannot be re-created from the PDF.
Referenced works ([1]–[18]) are citations, not data sources or dependencies of this repository.

"Run status": **executed** = run in this repo with artifacts; **fixture** = unit-tested on synthetic/fixture data only;
**not executed** = code exists but could not be run here.

| Paper element | Published specification | Companion choice | Implemented in | Test | Run status |
|---|---|---|---|---|---|
| Fig 1 / III-B ingestion | harvest, parse, normalise, dedup, noise filter | JSONL + metadata, version-aware dedup, spans | `data.py` | `test_data_pipeline.py` | executed (demo) |
| Fig 1 / III-C storage & parallel | distributed index, batch + stream, fault tolerance | multiprocessing + Parquet shards, atomic generation publish; PySpark path coded | `distributed.py` | partial-restart, parity | executed (multiprocessing); Spark **not executed** (no JVM) |
| III-D LLM layer | summarisation, embeddings, topic, method, citation intent | Ollama transformer encoder; extractive CPU fallback + Ollama generator; rule-based extraction/intent | `encoders.py`, `synthesis.py` | `test_graph_synthesis.py` | encoder + generator executed; hashing = offline substitute |
| III-E graph | entities, relations, multi-hop | typed directed graph, evidence per edge, inferred ≠ cited | `graph.py` | edge direction/temporal tests | executed (demo only) |
| III-F retrieval | dense + contextual ranking, expansion | BM25 + cosine, graph expansion with recorded terms | `retrieval.py` | alpha endpoints, ties | executed |
| IV-A `D={d_i}`, `d_i'=P(d_i)` | corpus, preprocessing | `normalize_record`, `segment`, lexical stop words only in BM25 branch | `data.py` | spans, negation kept | executed |
| IV-B `e_i=f_θ(d_i')`, cosine | transformer embedding, cosine | pinned encoder card; zero-vector policy | `encoders.py` | shape/zero tests | executed |
| IV-C `R(q)=TopK`, hybrid score | `α·Sim+(1-α)·BM25` | min-max over candidate union; depth recorded | `retrieval.fuse` | endpoints, constants | executed |
| IV-D Q-learning | `Q←Q+η[r+γ max Q'−Q]` | 3 α-profiles × (length, agreement) states, simulated reward | `feedback.py` | hand-computed update | executed (sandbox) |
| IV-E `G=(V,E)`, `T_k=Σ w_t F_kt` | graph, time-weighted trend | year bins, `w_t=decay^(t_end−t)` | `graph.py`, `discovery.py` | weights, cutoff | executed (demo) |
| IV-F `L=λ1 L_embed+λ2 L_rank+λ3 L_graph` | loss terms not defined | InfoNCE, pairwise margin, link-prediction BCE on a linear adapter | `objective.py` | gradient flow | fixture (trains an adapter on random vectors) |
| Table I / Fig 2 | P@10, Recall, F1 | recall cutoff = 10, macro F1 | `experiments.py` | metric hand tests | executed on SciFact (see results.md) |
| Table II / Fig 3 | NDCG@10 | graded linear gain | `evaluation.py` | hand-computed | executed |
| Table III / Fig 4 | mean latency ms | mean + p50 + p95, warm, c=1, retrieval vs +generation | `evaluation.bench_latency` | percentile test | executed |
| Table IV / Fig 5 | queries/s at 100K–1M | worker sweep on fixed corpus; size plan | `bench.py`, `evaluation.bench_throughput` | plan test | worker sweep executed; sizes **not executed** |
| Table V / Fig 6 | qualitative capabilities; trend accuracy 81%→92% | forecast with predeclared target | `discovery.py` | cutoff tests | demo only; **no ground truth → unevaluated** |
| Table VI / Fig 7 | ablation P@10 | four genuine configurations | `evaluation.ablation_systems` | graph-removal test | executed (graph row N/A on SciFact) |

Learned vs not learned: the encoder and (optionally) the generator are learned/pretrained; BM25, TF-IDF, indexes, shards,
ranking fusion, graph construction and orchestration are deterministic code; the Q-table is learned from (simulated) rewards;
`objective.py` adapter is trained only in the optional experiment.
