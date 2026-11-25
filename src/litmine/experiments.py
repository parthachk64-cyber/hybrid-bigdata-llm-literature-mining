"""Orchestrates a full measured run: comparators, ablation, latency, throughput. Every table is built from saved artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from .config import config_hash
from .evaluation import (
    ablation_systems,
    bench_latency,
    bench_throughput,
    build_systems,
    evaluate_rankings,
    md_table,
    new_run,
    provenance,
    run_system,
    save_per_query,
    save_rankings,
)
from .feedback import QPolicy, train_policy
from .retrieval import HybridSearcher


def ndcg_reward(ranked: list[str], rel: dict) -> float:
    from .evaluation import ndcg_at_k
    v = ndcg_at_k(ranked, rel, 10)
    return 0.0 if v is None else v


def select_alpha(corpus, queries: dict, qrels: dict, grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> dict:
    """Pick the fixed alpha on TRAIN queries only (mean NDCG@10). Returned with the whole curve for audit."""
    hy = HybridSearcher(corpus.bm25, corpus.dense, corpus.encoder, corpus.cfg, None)
    curve = {}
    for a in grid:
        rk = {q: [h.work_id for h in hy.search(t, alpha=a, k=10, expand=False)] for q, t in queries.items()}
        curve[a] = evaluate_rankings(rk, qrels, 10)["ndcg"]
    best = max(curve, key=lambda a: (curve[a] if curve[a] is not None else -1, -abs(a - 0.5)))
    return {"alpha": best, "curve": curve, "selected_on": "train queries (mean NDCG@10)"}


def run_experiment(run_root: str | Path, run_id: str, label: str, corpus, graph, test_q: dict, qrels: dict,
                   train_q: dict | None = None, train_sessions: list | None = None, generator=None, k: int = 10,
                   dataset_meta: dict | None = None, bench_workers=(1, 2, 4), latency_queries: int | None = None,
                   banner: str = "") -> dict:
    cfg = corpus.cfg
    meta = {"label": label, "banner": banner, "config": cfg, "config_hash": config_hash(cfg), "encoder": corpus.encoder.card(),
            "corpus_run_info": corpus.run_info, "n_docs": len(corpus.works), "dataset": dataset_meta or {},
            "k": k, "n_test_queries": len(test_q), "seed": cfg["feedback"]["seed"], "generator": getattr(generator, "model", None)}
    run_dir = new_run(run_root, run_id, **meta)
    pol = None
    sel = None
    if train_q:
        sel = select_alpha(corpus, train_q, qrels)
        cfg["retrieval"]["alpha"] = sel["alpha"]
        fb = cfg["feedback"]
        hy_train = HybridSearcher(corpus.bm25, corpus.dense, corpus.encoder, cfg, graph)
        pol = train_policy(hy_train, train_q, qrels, ndcg_reward, epochs=3, sessions=train_sessions,
                           policy=QPolicy(fb["eta"], fb["gamma"], fb["epsilon"], fb["seed"])).freeze()
        (run_dir / "policy.json").write_text(json.dumps(pol.to_json(), indent=2), encoding="utf-8")
        (run_dir / "alpha_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    systems = build_systems(corpus, graph, pol, generator)
    abl = ablation_systems(corpus, graph, pol) if pol else {}
    metrics: dict = {}
    for name, fn in systems.items():
        rk = run_system(fn, test_q)
        save_rankings(run_dir, name, rk)
        m = evaluate_rankings({q: [d for d, _ in r] for q, r in rk.items()}, qrels, k)
        save_per_query(run_dir, name, m)
        metrics[name] = {x: m[x] for x in m if x != "per_query"}
    ablation: dict = {}
    for name, fn in abl.items():
        rk = run_system(fn, test_q)
        save_rankings(run_dir, f"ablation.{name}", rk)
        m = evaluate_rankings({q: [d for d, _ in r] for q, r in rk.items()}, qrels, k)
        ablation[name] = {x: m[x] for x in m if x != "per_query"}
    qs = list(test_q.values())[: latency_queries or len(test_q)]
    lat = {n: bench_latency(fn, qs) for n, fn in systems.items() if n != "rag" or generator is None}
    if generator is not None and "rag" in systems:
        lat["rag(+generation)"] = bench_latency(systems["rag"], qs[: min(5, len(qs))], warmup=1)
    thr = {w: bench_throughput(systems["hybrid_fixed"], qs, w) for w in bench_workers}
    out = {"metrics": metrics, "ablation": ablation, "latency": lat, "throughput": thr, "alpha_selection": sel,
           "policy_updates": pol.n_updates if pol else None}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "tables" / "tables.md").write_text(render_tables(out, meta, run_id), encoding="utf-8")
    return {"run_dir": str(run_dir), **out, "meta": meta}


NAMES = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense retrieval", "graph_search": "Graph search", "rag": "RAG",
         "hybrid_fixed": "Hybrid (fixed alpha)", "hybrid_full": "Hybrid + graph + Q-policy"}


def render_tables(out: dict, meta: dict, run_id: str) -> str:
    enc = meta["encoder"]
    prov = provenance("MEASURED", f"run `{run_id}`, {meta['label']}, {meta['n_docs']} docs, {meta['n_test_queries']} test queries, "
                      f"encoder `{enc['name']}` (transformer: {enc['is_transformer']}). Not comparable with the paper's values.")
    m, k = out["metrics"], meta["k"]
    t = [f"{meta['banner']}\n" if meta["banner"] else ""]
    rows = [[NAMES[s], m[s]["p"], m[s]["recall"], m[s]["f1"]] for s in ("tfidf", "bm25", "dense", "graph_search", "hybrid_fixed", "hybrid_full") if s in m]
    t.append(md_table(["System", f"P@{k}", f"Recall@{k}", f"F1@{k} (macro)"], rows, prov, "Table I analogue: retrieval accuracy"))
    rows = [[NAMES[s], m[s]["ndcg"], m[s]["n_defined_ndcg"]] for s in ("bm25", "dense", "rag", "hybrid_fixed", "hybrid_full") if s in m]
    t.append(md_table(["System", f"NDCG@{k}", "queries defined"], rows, prov, "Table II analogue: ranking quality"))
    rows = [[n, v["mean_ms"], v["p50_ms"], v["p95_ms"], v["n"]] for n, v in out["latency"].items()]
    t.append(md_table(["System", "mean ms", "p50 ms", "p95 ms", "n"], rows, prov + " Retrieval-only unless marked (+generation).", "Table III analogue: latency (concurrency 1, warm)"))
    rows = [[w, v["completed"], v["errors"], v["elapsed_s"], v["queries_per_s"]] for w, v in out["throughput"].items()]
    t.append(md_table(["workers", "completed", "errors", "elapsed s", "queries/s"], rows, prov + " Fixed corpus; a worker sweep, not a corpus-size sweep.", "Throughput (fixed corpus, worker sweep)"))
    if out["ablation"]:
        rows = [[n, v["p"], v["ndcg"]] for n, v in out["ablation"].items()]
        t.append(md_table(["Configuration", f"P@{k}", f"NDCG@{k}"], rows, prov, "Table VI analogue: ablation (graph row n/a if no graph)"))
    return "\n".join(t)
