"""Index building: a local path and a partitioned path that must agree.

Partitioned backend  = multiprocessing workers writing one Parquet shard each (pyarrow). A PySpark backend with the
same shard format is provided (`build_shards_spark`) but is only usable where pyspark + a JVM exist; it is reported as
NOT EXECUTED when unavailable. Every run records backend, worker count and execution mode.

Publish protocol: shards are written to a temporary generation directory; only after every shard is present and
its checksum matches the manifest is the CURRENT pointer replaced (atomic os.replace). A failed shard leaves the
previous generation untouched and visible.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import Passage, Work, segment
from .encoders import make_encoder
from .retrieval import BM25Index, BM25Stats, DenseIndex, HybridSearcher, fuse, lex_tokens, top_k


@dataclass
class Corpus:
    works: dict[str, Work]
    passages: dict[str, Passage]
    bm25: BM25Index
    dense: DenseIndex
    encoder: object
    cfg: dict
    run_info: dict


def work_shard(work_id: str, n_shards: int) -> int:
    return int(hashlib.sha1(work_id.encode()).hexdigest(), 16) % n_shards


def _doc_text(w: Work) -> str:
    return w.full_text()


# ------------------------------------------------------------------ local ---

def build_local(works: list[Work], cfg: dict, encoder=None) -> Corpus:
    t0 = time.perf_counter()
    encoder = encoder or make_encoder(cfg)
    ing = cfg["ingest"]
    passages = {p.passage_id: p for w in works for p in segment(w, ing["segment_max_chars"], ing["segment_sentences"])}
    order = sorted(passages)
    vecs = encoder.encode([passages[i].text for i in order], "document") if order else np.zeros((0, 1), np.float32)
    dense = DenseIndex(vecs, order, [passages[i].work_id for i in order])
    b = cfg["bm25"]
    bm25 = BM25Index({w.work_id: _doc_text(w) for w in works}, b["k1"], b["b"], b["remove_stopwords"])
    info = {"backend": "local", "workers": 1, "mode": "single-process", "seconds": time.perf_counter() - t0}
    return Corpus({w.work_id: w for w in works}, passages, bm25, dense, encoder, cfg, info)


def searcher(c: Corpus, graph=None) -> HybridSearcher:
    return HybridSearcher(c.bm25, c.dense, c.encoder, c.cfg, graph)


# ------------------------------------------------------------ partitioned ---

def _shard_task(args) -> dict:
    """Worker: build one shard. Returns its manifest entry. Raises on failure (publish then aborts)."""
    shard_id, works_json, cfg, out_dir, fail = args
    if fail:
        raise RuntimeError(f"injected failure in shard {shard_id}")
    works = [Work(**w) for w in works_json]
    enc = make_encoder(cfg)
    ing = cfg["ingest"]
    ps = [p for w in works for p in segment(w, ing["segment_max_chars"], ing["segment_sentences"])]
    vecs = enc.encode([p.text for p in ps], "document") if ps else np.zeros((0, 1), np.float32)
    tab = pa.table({"passage_id": [p.passage_id for p in ps], "work_id": [p.work_id for p in ps],
                    "field": [p.field for p in ps], "start": [p.start for p in ps], "end": [p.end for p in ps],
                    "text": [p.text for p in ps], "vector": [v.tolist() for v in vecs]})
    path = Path(out_dir) / f"shard-{shard_id:03d}.parquet"
    pq.write_table(tab, path)
    b = cfg["bm25"]
    toks = {w.work_id: lex_tokens(_doc_text(w), b["remove_stopwords"]) for w in works}
    st = BM25Stats.of(toks)
    docs = {w.work_id: _doc_text(w) for w in works}
    (Path(out_dir) / f"shard-{shard_id:03d}.docs.json").write_text(json.dumps(docs), encoding="utf-8")
    return {"shard": shard_id, "n_works": len(works), "n_passages": len(ps), "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stats": {"n_docs": st.n_docs, "total_len": st.total_len, "df": dict(st.df)}}


def build_shards(works: list[Work], cfg: dict, out_dir: str | Path, fail_shard: int | None = None) -> list[dict]:
    d = cfg["distributed"]
    n_sh, n_w = d["shards"], d["workers"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    parts: list[list[Work]] = [[] for _ in range(n_sh)]
    for w in works:
        parts[work_shard(w.work_id, n_sh)].append(w)
    tasks = [(i, [w.to_json() for w in sorted(p, key=lambda x: x.work_id)], cfg, str(out_dir), i == fail_shard)
             for i, p in enumerate(parts)]
    if n_w <= 1:
        return [_shard_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        return list(ex.map(_shard_task, tasks))


def build_shards_spark(jsonl_path: str, cfg: dict, out_dir: str):  # pragma: no cover - needs pyspark + JVM
    """Same shard layout via PySpark. Not executed in this repository's CI (no JVM); see docs/reproduction.md."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("litmine").master(f"local[{cfg['distributed']['workers']}]").getOrCreate()
    df = spark.read.json(jsonl_path).repartition(cfg["distributed"]["shards"])
    df.write.mode("overwrite").parquet(str(Path(out_dir) / "works.parquet"))
    return {"backend": "pyspark", "master": spark.sparkContext.master, "partitions": df.rdd.getNumPartitions()}


def spark_available() -> bool:
    try:
        import shutil as _s

        import pyspark  # noqa: F401
        return _s.which("java") is not None
    except ImportError:
        return False


def publish_generation(index_dir: str | Path, works: list[Work], cfg: dict, fail_shard: int | None = None) -> dict:
    """Build shards into gen-N.tmp, verify, then flip CURRENT. All-or-nothing."""
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    cur = root / "CURRENT"
    n = 1 + max([int(p.name[4:]) for p in root.glob("gen-[0-9]*") if p.is_dir()] + [0])
    tmp = root / f"gen-{n:04d}.tmp"
    final = root / f"gen-{n:04d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        entries = build_shards(works, cfg, tmp, fail_shard)
        for e in entries:  # verify before exposing
            if hashlib.sha256((tmp / e["file"]).read_bytes()).hexdigest() != e["sha256"]:
                raise RuntimeError(f"checksum mismatch in {e['file']}")
        manifest = {"generation": n, "shards": entries, "backend": cfg["distributed"]["backend"],
                    "workers": cfg["distributed"]["workers"], "n_works": len(works)}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    ptr = root / "CURRENT.tmp"
    ptr.write_text(final.name, encoding="utf-8")
    os.replace(ptr, cur)
    return manifest


def current_generation(index_dir: str | Path) -> Path | None:
    cur = Path(index_dir) / "CURRENT"
    return Path(index_dir) / cur.read_text(encoding="utf-8").strip() if cur.exists() else None


class ShardedSearcher:
    """Scores each shard separately with GLOBAL BM25 statistics, then merges. Independent code path from local."""

    def __init__(self, gen_dir: str | Path, cfg: dict, works: dict[str, Work], encoder=None, graph=None):
        self.cfg, self.graph, self.works = cfg, graph, works
        self.encoder = encoder or make_encoder(cfg)
        man = json.loads((Path(gen_dir) / "manifest.json").read_text(encoding="utf-8"))
        gstats = BM25Stats()
        for e in man["shards"]:
            s = e["stats"]
            gstats = gstats.merge(BM25Stats(s["n_docs"], s["total_len"], Counter(s["df"])))
        b = cfg["bm25"]
        self.shards = []
        self.passages: dict[str, Passage] = {}
        for e in man["shards"]:
            docs = json.loads((Path(gen_dir) / e["file"].replace(".parquet", ".docs.json")).read_text(encoding="utf-8"))
            bm = BM25Index(docs, b["k1"], b["b"], b["remove_stopwords"], stats=gstats)
            t = pq.read_table(Path(gen_dir) / e["file"]).to_pydict()
            vecs = np.asarray(t["vector"], dtype=np.float32) if t["vector"] else np.zeros((0, 1), np.float32)
            self.shards.append((bm, DenseIndex(vecs, t["passage_id"], t["work_id"])))
            for i, pid in enumerate(t["passage_id"]):
                self.passages[pid] = Passage(pid, t["work_id"][i], t["field"][i], t["start"][i], t["end"][i], t["text"][i])
        self.manifest = man

    def scores(self, query: str, expand: bool | None = None):
        terms: list[str] = []
        extra: list[str] = []
        want = self.cfg["expansion"]["enabled"] if expand is None else expand
        if want and self.graph is not None:
            terms = self.graph.expansion_terms(query, self.cfg["expansion"]["max_terms"])
            extra = [t for term in terms for t in lex_tokens(term, self.cfg["bm25"]["remove_stopwords"])]
        q_vec = self.encoder.encode([query], "query")[0]
        lex, sem = {}, {}
        for bm, dn in self.shards:
            lex.update(bm.scores(query, extra))
            sem.update(dn.doc_scores(q_vec))
        return lex, sem, terms

    def search(self, query: str, alpha: float | None = None, k: int | None = None, expand: bool | None = None,
               depth: int | None = None):
        r = self.cfg["retrieval"]
        alpha = r["alpha"] if alpha is None else alpha
        k = r["top_k"] if k is None else k
        depth = depth or r["candidate_depth"]
        lex, sem, terms = self.scores(query, expand)
        cands = {w for w, _ in top_k(lex, depth)} | {w for w, _ in top_k(sem, depth)}
        fused = fuse({w: sem[w] for w in cands}, {w: lex[w] for w in cands}, alpha, r["normalization"])
        return [(w, s) for w, s in top_k({w: v["score"] for w, v in fused.items()}, k)]
