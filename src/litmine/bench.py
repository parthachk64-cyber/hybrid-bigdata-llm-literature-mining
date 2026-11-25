"""Independent retrieval evaluation (SciFact, BEIR format) and the paper-scale workload runner.

SciFact is an independent scientific claim-to-document benchmark. Retrieval relevance does not entail truth (a cited
abstract may SUPPORT or REFUTE a claim). It has no click log, no million-document scaling evidence and no citation network.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .data import Passage, Work
from .distributed import Corpus
from .encoders import make_encoder
from .evaluation import load_qrels
from .retrieval import BM25Index, DenseIndex


def load_beir(corpus: str | Path, queries: str | Path, qrels: str | Path, split_qids: set | None = None):
    """BEIR layout: corpus (jsonl or parquet with _id,title,text), queries (same), qrels tsv. IDs are preserved verbatim."""
    def rows(p):
        p = Path(p)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(p).to_pydict()
            return [dict(zip(t, v)) for v in zip(*t.values())]
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    docs = {str(r["_id"]): r for r in rows(corpus)}
    qs = {str(r["_id"]): r["text"] for r in rows(queries)}
    rel = load_qrels(qrels)
    if split_qids is None:
        split_qids = set(rel)
    return docs, {q: qs[q] for q in sorted(split_qids) if q in qs and q in rel}, {q: rel[q] for q in split_qids if q in rel}


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def beir_corpus(docs: dict, cfg: dict, encoder=None, cache_dir: str | Path | None = None, max_chars: int = 2000) -> Corpus:
    """One passage per document (title + abstract). work_id == the BEIR document id. Embeddings cached on disk."""
    t0 = time.perf_counter()
    encoder = encoder or make_encoder(cfg)
    ids = sorted(docs)
    texts = [(docs[i].get("title") or "") + ". " + (docs[i].get("text") or "") for i in ids]
    texts = [t.strip(". ")[:max_chars] for t in texts]
    key = hashlib.sha256((json.dumps(encoder.card(), sort_keys=True, default=str) + "|" + hashlib.sha256("\n".join(ids).encode()).hexdigest()).encode()).hexdigest()[:16]
    vecs = None
    cache = Path(cache_dir) / f"vecs-{key}.npy" if cache_dir else None
    if cache and cache.exists():
        vecs = np.load(cache)
    if vecs is None:
        vecs = encoder.encode(texts, "document")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)
    works = {i: Work(i, i, None, 1, (docs[i].get("title") or ""), [], None, None, docs[i].get("text") or "", "", [], None) for i in ids}
    passages = {f"{i}#doc#0": Passage(f"{i}#doc#0", i, "abstract", 0, len(t), t) for i, t in zip(ids, texts)}
    dense = DenseIndex(vecs, [f"{i}#doc#0" for i in ids], ids)
    b = cfg["bm25"]
    bm25 = BM25Index({i: t for i, t in zip(ids, texts)}, b["k1"], b["b"], b["remove_stopwords"])
    info = {"backend": "local", "workers": 1, "mode": "single-process", "seconds": time.perf_counter() - t0}
    return Corpus(works, passages, bm25, dense, encoder, cfg, info)


# ----------------------------------------------------- paper-scale workload

SIZES = (100_000, 300_000, 600_000, 1_000_000)


def nested_sample(manifest_ids: list[str], sizes=SIZES, seed: int = 13) -> dict[int, list[str]]:
    """Nested reproducible samples of DISTINCT works: sample(100K) ⊂ sample(300K) ⊂ ... Sizes above the corpus are skipped."""
    distinct = sorted(set(manifest_ids))
    order = list(distinct)
    random.Random(seed).shuffle(order)
    return {s: sorted(order[:s]) for s in sizes if s <= len(distinct)}


def resource_estimate(n_works: int, dim: int = 768, passages_per_work: float = 4.0, embed_ms_per_passage: float = 25.0) -> dict:
    ps = n_works * passages_per_work
    return {"works": n_works, "passages_est": int(ps), "vector_store_gb": ps * dim * 4 / 1e9,
            "embed_hours_single_cpu": ps * embed_ms_per_passage / 1000 / 3600,
            "assumptions": f"{passages_per_work} passages/work, {dim}-d float32, {embed_ms_per_passage} ms/passage (CPU, measure yours)"}


def plan_scale_run(manifest_ids: list[str], sizes=SIZES, state_path: str | Path | None = None) -> dict:
    """Which sizes are runnable from this manifest, which are not, plus resource estimates. Resumable via state file."""
    samples = nested_sample(manifest_ids, sizes)
    done = set()
    if state_path and Path(state_path).exists():
        done = set(json.loads(Path(state_path).read_text())["completed"])
    plan = {"distinct_works_in_manifest": len(set(manifest_ids)), "sizes": []}
    for s in sizes:
        plan["sizes"].append({"size": s, "status": ("completed" if s in done else "runnable") if s in samples else "NOT RUNNABLE: manifest has fewer distinct works",
                              "actual_distinct": len(samples[s]) if s in samples else None, "estimate": resource_estimate(s)})
    return plan


def mark_done(state_path: str | Path, size: int) -> None:
    p = Path(state_path)
    st = json.loads(p.read_text()) if p.exists() else {"completed": []}
    st["completed"] = sorted(set(st["completed"]) | {size})
    p.write_text(json.dumps(st), encoding="utf-8")


def parity_tolerance(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
