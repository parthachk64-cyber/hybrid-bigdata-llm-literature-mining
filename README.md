# litmine — evidence-linked companion to *Hybrid Big Data-LLM Framework for Intelligent Scientific Literature Mining*

**Paper:** Alam, Rashed, Chakraborty, Mohonta, Imam, Bhuiyan. IEEE I3CTCON 2026, Lonavala, pp. 1–9.
DOI [10.1109/I3CTCON68242.2026.11507315](https://doi.org/10.1109/I3CTCON68242.2026.11507315).
Maintainer of this independent companion: Partha Chakraborty ([@parthachk64-cyber](https://github.com/parthachk64-cyber)).

> This repo is **not** the authors' original code or data (none was published). It implements the paper's specification from scratch, runs it on
> independent data, and reports **measured** numbers next to **paper-reported** ones without ever presenting one as the other.
> See [results ledger](docs/results.md) for every value and its provenance.

## Quick start (offline, CPU, no token)
```bash
pip install git+https://github.com/parthachk64-cyber/hybrid-bigdata-llm-literature-mining && litmine demo
```
From a clone: `pip install -e ".[dev]" && litmine demo && pytest -q`.

## What a result looks like
`litmine demo` on the bundled **synthetic** corpus answers *"Which methods have hydrology and manufacturing used to detect abrupt shifts in sensor streams?"*:

```
## Ranked papers and why
| # | paper | score | semantic raw | BM25 raw | alpha |
| 3 | S22 (2019) Transferring changepoint methods between river gauges and assembly lines | 0.810 | 0.194 | 7.749 | 0.5 |
| 4 | S05 (2012) Bayesian online changepoint detection for river gauge streams | 0.682 | 0.286 | 3.469 | 0.5 |
Evidence: S22 `…#abstract#0` chars 0-221: "We transfer Bayesian online changepoint detection from river gauges and cumulative sum monitoring ..."
hydrology: Bayesian online changepoint detection, CUSUM, Wavelet decomposition
manufacturing: Bayesian online changepoint detection, CUSUM, Kernel two-sample test, Page-Hinkley test
### Verified citation chain
- S30 (2023) cites S22 (2019) (marker [1], intent extends): "Unlike fixed-threshold charts, our monitor extends the transferred method of [1]. ..."
- S22 (2019) cites S05 (2012) (marker [1], intent uses_method): "We apply the Bayesian run-length method from river gauge monitoring [1] ..."
```
Note rank 2 for this query is a deliberate hard negative (a traffic CUSUM paper): the hashing substitute encoder is lexical, so this lure works on it.
`litmine synthesize "effect of lunar gravity on yeast fermentation"` returns *No supporting passage found*.

## Commands
| Task | Command |
|---|---|
| ingest / index / search | `litmine ingest …` · `litmine index …` · `litmine search "q" --expand` |
| synthesis, citation chain | `litmine synthesize "q" [--backend ollama]` · `litmine path S30 S05` |
| feedback training (simulated) | `litmine feedback-train` |
| baselines + ablation + latency + throughput | `litmine eval --dataset demo\|scifact --run-id ID` |
| plots, ledger, scale plan | `litmine plot --run runs/ID` · `litmine ledger` · `litmine scale --manifest M.jsonl` |

Details: [architecture](docs/architecture.md) · [paper map](docs/paper-map.md) · [data](docs/data.md) · [reproduction](docs/reproduction.md).

## Headline measured result (SciFact test, 300 queries, nomic-embed-text transformer encoder)
| System | NDCG@10 |
|---|---|
| BM25 | 0.6626 |
| Dense | 0.7041 |
| Hybrid (α=0.7 from train) | 0.7283 |
| Hybrid + frozen Q-policy | 0.7363 |

The paper's ordering (lexical < dense < hybrid) is reproduced; its absolute values are **not** (different data, no original inputs). Several things are
**not run**: graph search/ablation on SciFact, trend accuracy, 100K–1M scale, real-generator RAG, Spark. See the ledger.

## Honesty rules enforced in code and tests
No invented DOI/reference/edge/year · inferred edges never labelled citations · synthetic demo output excluded from benchmark reports ·
held-out queries never used for training · documents treated as untrusted data (injection patterns excluded, generated sentences must be
supported by a cited passage) · the offline `hashing` encoder and extractive fallback are labelled substitutes, not transformers or LLMs.

License: MIT (original code). The IEEE PDF and restricted full texts are not included.
