# Reproduction

All commands run from the repo root after `pip install -e ".[dev,full]"`. CPU only. No token, no paid API.

| Task | Command |
|---|---|
| Offline walkthrough | `litmine demo` |
| Ingest | `litmine ingest --input src/litmine/demo/corpus.jsonl --out work/snap` |
| Index (partitioned, atomic generation) | `litmine index --snapshot work/snap --index work/idx --workers 2 --shards 4` |
| Search with score breakdown | `litmine search "kernel two-sample test for drift" --snapshot work/snap --k 5` |
| Graph expansion | `litmine search "Bayesian changepoint streamflow" --expand` |
| Verified citation chain | `litmine path S30 S05` |
| Synthesis (extractive) | `litmine synthesize "citation intent classification"` |
| Synthesis (local LLM) | `litmine synthesize "..." --backend ollama` (needs `ollama serve` + `llama3.1:8b`) |
| Feedback training (simulated) | `litmine feedback-train --out policy.json` |
| Baselines + ablation + latency + throughput | `litmine eval --dataset demo --run-id demo` or `python scripts/run_scifact.py --encoder ollama:nomic-embed-text --run-id scifact-nomic-embed` |
| Plots | `litmine plot --run runs/scifact-nomic-embed` |
| Results ledger | `litmine ledger` |
| Scale plan | `litmine scale --manifest M.jsonl --state runs/scale.state` |
| Fetch SciFact | `python scripts/fetch_scifact.py` |

## Full profile
`ollama pull nomic-embed-text && ollama pull llama3.1:8b`, then pass `--encoder ollama:nomic-embed-text`. The model digest is recorded in
`run.json` (`encoder.revision`). Hugging Face encoders need an explicit pinned `revision` in config (`encoder: {name: "hf:<id>", revision: "<sha>"}`).

## Spark
`litmine.distributed.build_shards_spark` needs `pip install pyspark` and a JVM. It was **not executed** in this repo.

## Known reproduction gaps
1. The paper's corpus, queries, judgments, logs, checkpoints, ontology and cluster setup are unavailable: no value in Tables I–VI can be re-created.
2. SciFact has no citation graph or ontology: graph search and the graph ablation are unevaluated; Table V (trend accuracy) has no ground truth here.
3. 100K–1M scale: no manifest; no size executed.
4. Q-learning evaluated only with simulated rewards from qrels; no real user feedback exists.
5. The paper's "BERT retrieval" is represented by a different transformer (nomic-embed-text), not BERT.
6. `objective.py` is demonstrated on a linear adapter over frozen vectors, not an end-to-end fine-tune.
7. Per-query confidence intervals are not derivable from the paper's rounded averages; ours come only from our own per-query files.
