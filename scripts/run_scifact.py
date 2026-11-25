"""SciFact (BEIR format) measured run. Usage: python scripts/run_scifact.py --encoder ollama:nomic-embed-text --run-id scifact-nomic"""
import argparse, json, sys
from pathlib import Path
from litmine.bench import beir_corpus, load_beir, sha256_file
from litmine.config import load_config
from litmine.encoders import make_encoder
from litmine.experiments import run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/scifact")
ap.add_argument("--encoder", default="hashing")
ap.add_argument("--run-id", required=True)
ap.add_argument("--runs", default="runs")
a = ap.parse_args()
d = Path(a.data)
cfg = load_config(overrides={"encoder": {"name": a.encoder}})
train_ids = {l.split("\t")[0] for l in open(d / "train.tsv") if not l.startswith("query-id")}
test_ids = {l.split("\t")[0] for l in open(d / "test.tsv") if not l.startswith("query-id")}
import litmine.evaluation as ev
docs, _, _ = load_beir(d / "corpus.parquet", d / "queries.parquet", d / "test.tsv")
_, test_q, test_rel = load_beir(d / "corpus.parquet", d / "queries.parquet", d / "test.tsv", test_ids)
_, train_q, train_rel = load_beir(d / "corpus.parquet", d / "queries.parquet", d / "train.tsv", train_ids)
enc = make_encoder(cfg)
corpus = beir_corpus(docs, cfg, enc, cache_dir=d / "cache")
meta = {"name": "SciFact (BEIR format, HF mirror BeIR/scifact rev b3b53356)", "split": "released test (300 q) / train for tuning",
        "n_train_queries": len(train_q), "files_sha256": {f: sha256_file(d / f) for f in ("corpus.parquet", "queries.parquet", "test.tsv", "train.tsv")},
        "limits": "no click log, no citation network, no 1M-scale evidence; relevance != truth"}
qrels = {**train_rel, **test_rel}
out = run_experiment(a.runs, a.run_id, "SciFact test", corpus, None, test_q, qrels, train_q=train_q, dataset_meta=meta, latency_queries=100)
print(json.dumps({k: out["metrics"][k] for k in out["metrics"]}, indent=1, default=str)[:3000])
