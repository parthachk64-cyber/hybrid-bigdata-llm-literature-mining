"""Metrics, comparator systems, ablations, latency/throughput benchmarks, run artifacts and provenance-labelled tables.

Metric conventions (stated once, used everywhere; see docs/results.md)
  relevance  : qrels grade g >= 1 is relevant (binary for P/R/F1); NDCG uses the graded value as linear gain.
  unjudged   : a retrieved document absent from the qrels is treated as NON-relevant (grade 0).
  P@k        : relevant in top k / k  (k fixed even when fewer than k results are returned).
  Recall@k   : relevant in top k / number of relevant docs for the query. Undefined if a query has no relevant doc.
  F1         : harmonic mean of per-query P@k and Recall@k, then macro-averaged over defined queries.
               A pooled (micro) F1 from summed counts is also reported.
  NDCG@k     : DCG = sum gain_i / log2(i + 1), i = 1-based rank; ideal = qrels sorted by grade. If ideal DCG is 0
               (no relevant doc) the query is UNDEFINED and excluded from the mean; the exclusion count is reported.
Macro averages are always accompanied by n_defined / n_queries.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import Work
from .retrieval import BM25Index, HybridSearcher, TfidfIndex, top_k

# ------------------------------------------------------------------ metrics

def precision_at_k(ranked: list[str], rel: dict, k: int) -> float:
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked: list[str], rel: dict, k: int) -> float | None:
    n_rel = sum(1 for g in rel.values() if g > 0)
    return None if n_rel == 0 else sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel


def f1(p: float, r: float | None) -> float | None:
    if r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rel: dict, k: int) -> float | None:
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg([max(rel.get(d, 0), 0) for d in ranked[:k]]) / idcg


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict], k: int = 10) -> dict:
    per_q, hits_sum, rel_sum, ret_sum = {}, 0, 0, 0
    for qid, ranked in rankings.items():
        rel = qrels.get(qid, {})
        p, r, nd = precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k), ndcg_at_k(ranked, rel, k)
        per_q[qid] = {"p": p, "recall": r, "f1": f1(p, r), "ndcg": nd}
        if r is not None:
            hits_sum += sum(1 for d in ranked[:k] if rel.get(d, 0) > 0)
            rel_sum += sum(1 for g in rel.values() if g > 0)
            ret_sum += k

    def mean(key):
        v = [m[key] for m in per_q.values() if m[key] is not None]
        return (sum(v) / len(v) if v else None), len(v)

    out = {"k": k, "n_queries": len(per_q)}
    for key in ("p", "recall", "f1", "ndcg"):
        out[key], out[f"n_defined_{key}"] = mean(key)
    pp, rr = (hits_sum / ret_sum if ret_sum else 0.0), (hits_sum / rel_sum if rel_sum else None)
    out["pooled_f1"] = f1(pp, rr)
    out["per_query"] = per_q
    return out


# ------------------------------------------------------------------ systems

def doc_text(w: Work) -> str:
    return w.full_text()


def build_systems(corpus, graph=None, policy=None, generator=None) -> dict:
    """Comparators. Each is query -> list[(work_id, score)]. Built fresh per call so ablations cannot share caches."""
    works = corpus.works
    texts = {i: doc_text(w) for i, w in works.items()}
    tfidf = TfidfIndex(texts, corpus.cfg["bm25"]["remove_stopwords"])
    hy = HybridSearcher(corpus.bm25, corpus.dense, corpus.encoder, corpus.cfg, graph)
    alpha0 = corpus.cfg["retrieval"]["alpha"]
    systems = {
        "tfidf": lambda q: top_k(tfidf.scores(q), 100),
        "bm25": lambda q: top_k(corpus.bm25.scores(q), 100),
        "dense": lambda q: top_k(corpus.dense.doc_scores(corpus.encoder.encode([q], "query")[0]), 100),
        "hybrid_fixed": lambda q: [(h.work_id, h.score) for h in hy.search(q, alpha=alpha0, k=100, expand=False)],
    }
    if graph is not None:
        systems["graph_search"] = lambda q: graph_search(q, graph, corpus.bm25, 100)
        systems["hybrid_full"] = _full(hy, policy, corpus, expand=True, adaptive=True)
    systems["rag"] = lambda q: rag_ranking(q, corpus, generator)
    return systems


def _full(hy, policy, corpus, expand: bool, adaptive: bool, alpha: float | None = None):
    from .feedback import state_for

    def run(q: str):
        a = alpha if alpha is not None else corpus.cfg["retrieval"]["alpha"]
        if adaptive and policy is not None:
            a = policy.act(state_for(hy, q), explore=False)
        return [(h.work_id, h.score) for h in hy.search(q, alpha=a, k=100, expand=expand)]
    return run


def graph_search(query: str, graph, bm25: BM25Index, k: int) -> list[tuple[str, float]]:
    """Graph-only retrieval: works linked to entities named in the query (+0.5 for works they cite or are cited by).

    BM25 is used ONLY as a sub-0.01 tie-breaker so that works the graph cannot reach still receive a deterministic order.
    """
    from .retrieval import lex_tokens
    ql = " " + " ".join(lex_tokens(query, stop=False)) + " "
    matched = {nid for s, nid in graph.aliases.items() if f" {s} " in ql}
    score: dict[str, float] = {}
    for e in graph.edges:
        if e["dst"] in matched and e["type"] in ("uses_method", "about_topic", "uses_dataset"):
            score[e["src"]] = score.get(e["src"], 0.0) + 1.0
    for e in graph.edges:
        if e["type"] == "cites":
            if e["src"] in score:
                score[e["dst"]] = score.get(e["dst"], 0.0) + 0.5
            if e["dst"] in score:
                score[e["src"]] = score.get(e["src"], 0.0) + 0.5
    lex = bm25.scores(query)
    mx = max(lex.values()) or 1.0
    full = {w: score.get(w, 0.0) + 0.009 * lex[w] / mx for w in lex}
    return top_k(full, k)


def rag_ranking(query: str, corpus, generator=None, n_ctx: int = 5) -> list[tuple[str, float]]:
    """RAG comparator.

    Retriever : dense passage retrieval (top n_ctx passages by cosine from the configured encoder).
    Generator : `generator` (an OllamaGenerator) if given, else the extractive CPU fallback that quotes the retrieved
                passages. Both must cite passages; the extractive one cites by construction.
    Ranking   : documents are ordered by (number of generator-cited passages desc, retriever rank); documents the
                generator did not cite follow in retriever order. So the ranking is the answer's evidence, then the rest.
    """
    q_vec = corpus.encoder.encode([query], "query")[0]
    ps = corpus.dense.passage_scores(q_vec)
    order = sorted(range(len(ps)), key=lambda i: (-ps[i], corpus.dense.passage_ids[i]))
    ctx_rows = order[:n_ctx]
    cited: dict[str, int] = {}
    if generator is None:
        for r in ctx_rows:
            w = corpus.dense.work_ids[r]
            cited[w] = cited.get(w, 0) + 1
    else:
        ctx = [(corpus.dense.passage_ids[r], corpus.passages[corpus.dense.passage_ids[r]]) for r in ctx_rows]
        ctx_text = "\n".join(f"[P{i + 1}] {p.text}" for i, (_, p) in enumerate(ctx))
        import re

        from .synthesis import PROMPT
        raw = generator.generate(PROMPT.format(q=query, ctx=ctx_text), num_predict=200)
        for m in re.findall(r"\[P(\d+)\]", raw):
            i = int(m) - 1
            if 0 <= i < len(ctx):
                cited[ctx[i][1].work_id] = cited.get(ctx[i][1].work_id, 0) + 1
    dense = corpus.dense.doc_scores(q_vec)
    scores = {w: (1000.0 * cited.get(w, 0) + s) for w, s in dense.items()}
    return top_k(scores, 100)


def ablation_systems(corpus, graph, policy) -> dict:
    """Four genuine configurations, each with its own searcher; the disabled part is replaced, not zeroed.

    full         : graph expansion + hybrid fusion + frozen Q-policy alpha
    no_graph     : graph=None (no expansion terms exist), hybrid + policy
    no_hybrid    : fusion replaced by the dense-only retriever
    no_rl        : policy replaced by the fixed neutral alpha from config, graph + hybrid kept
    """
    def mk(g):
        return HybridSearcher(corpus.bm25, corpus.dense, corpus.encoder, corpus.cfg, g)
    full, nog, norl = mk(graph), mk(None), mk(graph)
    return {
        "full": _full(full, policy, corpus, expand=True, adaptive=True),
        "no_graph": _full(nog, policy, corpus, expand=False, adaptive=True),
        "no_hybrid": lambda q: top_k(corpus.dense.doc_scores(corpus.encoder.encode([q], "query")[0]), 100),
        "no_rl": _full(norl, None, corpus, expand=True, adaptive=False),
    }


def run_system(fn, queries: dict[str, str]) -> dict[str, list[tuple[str, float]]]:
    return {qid: fn(q) for qid, q in sorted(queries.items())}


# --------------------------------------------------------------- benchmarks

def _pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def bench_latency(fn, queries: list[str], warmup: int = 3, repeats: int = 1) -> dict:
    for q in queries[:warmup]:
        fn(q)
    ms = []
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            fn(q)
            ms.append((time.perf_counter() - t) * 1000)
    return {"n": len(ms), "mean_ms": float(np.mean(ms)), "p50_ms": _pct(ms, 50), "p95_ms": _pct(ms, 95),
            "warmup_calls": min(warmup, len(queries)), "concurrency": 1,
            "cache_state": "warm process; no result cache; fresh scoring per call"}


def bench_throughput(fn, queries: list[str], workers: int, rounds: int = 3) -> dict:
    """Completed queries per elapsed wall-clock second under `workers` concurrent threads. Results are validated."""
    work = [q for _ in range(rounds) for q in queries]
    errors, empty = 0, 0

    def one(q):
        nonlocal errors, empty
        try:
            r = fn(q)
            if not r:
                empty += 1
            return 1
        except Exception:  # noqa: BLE001 - counted and reported, never swallowed silently
            errors += 1
            return 0
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = sum(ex.map(one, work))
    el = time.perf_counter() - t
    return {"workers": workers, "completed": done, "errors": errors, "empty_results": empty, "elapsed_s": el,
            "queries_per_s": done / el if el else None, "executor": "threads (GIL-bound for pure-python scoring)"}


# ----------------------------------------------------------- run artifacts

def git_commit(cwd: str | Path = ".") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "no-git"


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "machine": platform.machine(), "git_commit": git_commit(Path(__file__).resolve().parent)}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    return info


def new_run(root: str | Path, run_id: str, **meta) -> Path:
    d = Path(root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "env": env_info(), **meta}, indent=2, default=str), encoding="utf-8")
    return d


def save_rankings(run_dir: Path, system: str, rankings: dict, k: int = 100) -> None:
    with open(run_dir / f"rankings.{system}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps({"qid": qid, "ranking": [[d, round(float(s), 8)] for d, s in rankings[qid][:k]]}) + "\n" for qid in sorted(rankings))


def load_rankings(path: str | Path) -> dict[str, list[str]]:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["qid"]] = [d for d, _ in r["ranking"]]
    return out


def save_per_query(run_dir: Path, system: str, metrics: dict) -> None:
    with open(run_dir / f"per_query.{system}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "p_at_k", "recall_at_k", "f1", "ndcg_at_k"])
        for qid, m in sorted(metrics["per_query"].items()):
            w.writerow([qid] + ["" if m[x] is None else f"{m[x]:.6f}" for x in ("p", "recall", "f1", "ndcg")])


def load_qrels(path: str | Path) -> dict[str, dict]:
    q: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and parts[-1].lower() == "score":
                continue
            qid, did, sc = parts[0], parts[1], int(float(parts[2]))
            q.setdefault(qid, {})[did] = sc
    return q


# ------------------------------------------------------------------ tables

def provenance(kind: str, detail: str) -> str:
    assert kind in {"PAPER-REPORTED", "MEASURED", "PAPER-REPORTED vs MEASURED"}
    return f"> **{kind}** — {detail}"


def md_table(header: list[str], rows: list[list], prov: str, title: str = "") -> str:
    def fmt(x):
        return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines = ([f"### {title}", ""] if title else []) + [prov, "", "| " + " | ".join(header) + " |",
                                                      "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


