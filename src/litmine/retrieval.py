"""Sparse (BM25, TF-IDF), dense (cosine) and hybrid retrieval with score breakdowns.

Section IV score:  Score(d) = alpha * Sim(e_q, e_d) + (1 - alpha) * BM25(q, d)
Both raw scores are computed for every candidate in the union of lexical and dense candidates, then
min-max normalised *over that candidate set* (implementation choice, documented in docs/architecture.md).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .encoders import l2_normalize

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# negations and quantity words are deliberately NOT stop words
STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "which", "with", "these", "those", "into", "over", "under", "between", "across", "than", "then", "so", "such", "can", "may", "also", "using", "used", "use", "we", "our", "us", "how", "what", "which", "when", "where", "who", "whom", "why", "does", "do", "did", "been", "being"])


def lex_tokens(text: str, stop: bool = True) -> list[str]:
    toks = WORD_RE.findall(text.lower())
    return [t for t in toks if t not in STOP] if stop else toks


# ---------------------------------------------------------------- BM25 ---

@dataclass
class BM25Stats:
    """Collection statistics. Additive, so shards can be merged exactly (local == partitioned)."""
    n_docs: int = 0
    total_len: int = 0
    df: Counter = field(default_factory=Counter)

    @property
    def avgdl(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 0.0

    def merge(self, other: BM25Stats) -> BM25Stats:
        return BM25Stats(self.n_docs + other.n_docs, self.total_len + other.total_len, self.df + other.df)

    @staticmethod
    def of(tokenized: dict[str, list[str]]) -> BM25Stats:
        s = BM25Stats()
        for toks in tokenized.values():
            s.n_docs += 1
            s.total_len += len(toks)
            s.df.update(set(toks))
        return s


class BM25Index:
    def __init__(self, texts: dict[str, str], k1: float = 1.5, b: float = 0.75, stop: bool = True,
                 stats: BM25Stats | None = None):
        self.k1, self.b, self.stop = k1, b, stop
        self.tf = {i: Counter(lex_tokens(t, stop)) for i, t in texts.items()}
        self.dl = {i: sum(c.values()) for i, c in self.tf.items()}
        self.stats = stats or BM25Stats.of({i: list(c.elements()) for i, c in self.tf.items()})

    def idf(self, term: str) -> float:
        n, df = self.stats.n_docs, self.stats.df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_doc(self, q_terms: list[str], doc_id: str) -> float:
        tf, dl, avg = self.tf[doc_id], self.dl[doc_id], self.stats.avgdl or 1.0
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
        return s

    def scores(self, query: str, extra_terms: list[str] | None = None) -> dict[str, float]:
        q = lex_tokens(query, self.stop) + list(extra_terms or [])
        return {d: self.score_doc(q, d) for d in self.tf}


class TfidfIndex:
    def __init__(self, texts: dict[str, str], stop: bool = True):
        self.stop = stop
        toks = {i: lex_tokens(t, stop) for i, t in texts.items()}
        self.n = len(toks)
        df = Counter(t for ts in toks.values() for t in set(ts))
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vec = {i: self._vec(ts) for i, ts in toks.items()}

    def _vec(self, toks: list[str]) -> dict[str, float]:
        c = Counter(toks)
        v = {t: (1 + math.log(f)) * self.idf.get(t, 0.0) for t, f in c.items()}
        n = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / n for t, x in v.items()} if n else {}

    def scores(self, query: str) -> dict[str, float]:
        q = self._vec(lex_tokens(query, self.stop))
        return {d: sum(w * dv.get(t, 0.0) for t, w in q.items()) for d, dv in self.vec.items()}


# --------------------------------------------------------------- dense ---

class DenseIndex:
    """Passage vectors; a document's score is the max cosine over its passages (recorded in the encoder card).

    Vectors are L2-normalised once at build time (zero vectors stay zero, so their cosine is 0.0), so scoring is one matmul.
    """

    def __init__(self, vectors: np.ndarray, passage_ids: list[str], work_ids: list[str]):
        assert len(vectors) == len(passage_ids) == len(work_ids)
        self.vectors = l2_normalize(np.asarray(vectors, dtype=np.float32)) if len(vectors) else vectors
        self.passage_ids, self.work_ids = passage_ids, work_ids
        self.rows: dict[str, list[int]] = {}
        for r, w in enumerate(work_ids):
            self.rows.setdefault(w, []).append(r)
        self._docs = list(self.rows)
        self._idx = [np.asarray(self.rows[w]) for w in self._docs]
        self._single = all(len(i) == 1 for i in self._idx)

    def passage_scores(self, q_vec: np.ndarray) -> np.ndarray:
        if len(self.vectors) == 0:
            return np.zeros(0)
        q = l2_normalize(np.asarray(q_vec, dtype=np.float32).reshape(1, -1))[0]
        return (self.vectors @ q).astype(np.float64)

    def doc_scores(self, q_vec: np.ndarray) -> dict[str, float]:
        ps = self.passage_scores(q_vec)
        if self._single:
            order = np.concatenate(self._idx) if self._idx else np.zeros(0, int)
            return dict(zip(self._docs, ps[order].tolist()))
        return {w: float(ps[i].max()) for w, i in zip(self._docs, self._idx)}


# ---------------------------------------------------------- fusion etc ---

def top_k(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Stable top-k: descending score, ties broken by ascending id (deterministic)."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def normalize(scores: dict[str, float], mode: str = "minmax") -> dict[str, float]:
    """Calibrate raw scores to [0,1] over the candidate set.

    minmax : (s - min) / (max - min). Constant-score case (max == min) -> all 0.0, i.e. the component carries
             no ranking information instead of being inflated to 1.0.
    none   : raw scores unchanged.
    """
    if mode == "none" or not scores:
        return dict(scores)
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(sem: dict[str, float], lex: dict[str, float], alpha: float, mode: str = "minmax") -> dict[str, dict]:
    """alpha * semantic + (1 - alpha) * lexical over the union; both raw scores must cover every id."""
    ids = sorted(set(sem) | set(lex))
    raw_s = {i: sem.get(i, 0.0) for i in ids}
    raw_l = {i: lex.get(i, 0.0) for i in ids}
    ns, nl = normalize(raw_s, mode), normalize(raw_l, mode)
    return {i: {"score": alpha * ns[i] + (1 - alpha) * nl[i], "semantic_raw": raw_s[i], "lexical_raw": raw_l[i],
                "semantic_norm": ns[i], "lexical_norm": nl[i], "alpha": alpha} for i in ids}


@dataclass
class Hit:
    work_id: str
    rank: int
    score: float
    breakdown: dict
    expansion_terms: list = field(default_factory=list)


class HybridSearcher:
    """Local (single-process) hybrid retriever over a Corpus. Used as the reference for partitioned parity."""

    def __init__(self, bm25: BM25Index, dense: DenseIndex, encoder, cfg: dict, graph=None):
        self.bm25, self.dense, self.encoder, self.cfg, self.graph = bm25, dense, encoder, cfg, graph

    def lexical_scores(self, query: str, expand: bool | None = None) -> tuple[dict[str, float], list[str]]:
        want = self.cfg["expansion"]["enabled"] if expand is None else expand
        terms: list[str] = []
        if want and self.graph is not None:
            terms = self.graph.expansion_terms(query, self.cfg["expansion"]["max_terms"])
            toks = [t for term in terms for t in lex_tokens(term, self.bm25.stop)]
            return self.bm25.scores(query, toks), terms
        return self.bm25.scores(query), terms

    def search(self, query: str, alpha: float | None = None, k: int | None = None, expand: bool | None = None,
               depth: int | None = None) -> list[Hit]:
        r = self.cfg["retrieval"]
        alpha = r["alpha"] if alpha is None else alpha
        k = r["top_k"] if k is None else k
        depth = depth or r["candidate_depth"]
        lex_all, terms = self.lexical_scores(query, expand)
        q_vec = self.encoder.encode([query], "query")[0]
        sem_all = self.dense.doc_scores(q_vec)
        cands = {w for w, _ in top_k(lex_all, depth)} | {w for w, _ in top_k(sem_all, depth)}
        fused = fuse({w: sem_all[w] for w in cands}, {w: lex_all[w] for w in cands}, alpha, r["normalization"])
        ranked = top_k({w: v["score"] for w, v in fused.items()}, k)
        return [Hit(w, i + 1, s, {**fused[w], "candidate_depth": depth, "n_candidates": len(cands)}, terms)
                for i, (w, s) in enumerate(ranked)]

    def lexical_only(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return top_k(self.lexical_scores(query, False)[0], k)

    def dense_only(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return top_k(self.dense.doc_scores(self.encoder.encode([query], "query")[0]), k)


def best_passages(query: str, dense: DenseIndex, encoder, passages_by_id: dict, work_id: str, n: int = 2) -> list[dict]:
    """Passages of one work ranked by dense similarity plus lexical overlap with the query (for evidence display)."""
    q_vec = encoder.encode([query], "query")[0]
    ps = dense.passage_scores(q_vec)
    q_terms = set(lex_tokens(query))
    rows = []
    for r in dense.rows.get(work_id, []):
        pid = dense.passage_ids[r]
        p = passages_by_id[pid]
        overlap = len(q_terms & set(lex_tokens(p.text)))
        rows.append((float(ps[r]), overlap, pid))
    rows.sort(key=lambda x: (-x[1], -x[0], x[2]))
    return [{"passage_id": pid, "dense": s, "term_overlap": ov} for s, ov, pid in rows[:n]]
