import math

import pytest

from litmine.evaluation import (
    bench_latency,
    bench_throughput,
    dcg,
    evaluate_rankings,
    f1,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from litmine.feedback import PolicyStore, QPolicy, q_update, train_policy
from litmine.retrieval import fuse, normalize, top_k

RANK = ["a", "b", "c", "d", "e"]
REL = {"a": 2, "c": 1, "f": 1}  # f never retrieved; b,d,e unjudged => non-relevant


def test_metrics_by_hand():
    assert precision_at_k(RANK, REL, 5) == pytest.approx(0.4)
    assert recall_at_k(RANK, REL, 5) == pytest.approx(2 / 3)
    assert f1(0.4, 2 / 3) == pytest.approx(0.5)
    assert dcg([2, 0, 1]) == pytest.approx(2 + 0.5)
    ideal = 2 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(RANK, REL, 5) == pytest.approx(2.5 / ideal)


def test_zero_ideal_gain_is_undefined_not_zero():
    assert ndcg_at_k(RANK, {}, 5) is None and recall_at_k(RANK, {}, 5) is None
    out = evaluate_rankings({"q1": RANK, "q2": RANK}, {"q1": REL, "q2": {}})
    assert out["n_defined_ndcg"] == 1 and out["n_queries"] == 2


def test_tie_break_is_deterministic():
    assert top_k({"b": 1.0, "a": 1.0, "c": 2.0}, 3) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


def test_alpha_endpoints_match_pure_retrievers(wb):
    q = "kernel two-sample test for drift in production sensors"
    sem = [d for d, _ in wb.searcher.dense_only(q, 10)]
    lex = [d for d, _ in wb.searcher.lexical_only(q, 10)]
    assert [h.work_id for h in wb.search(q, 10, alpha=1.0)][:3] == sem[:3]
    assert [h.work_id for h in wb.search(q, 10, alpha=0.0)][:3] == lex[:3]


def test_constant_scores_normalise_to_zero_and_missing_fused():
    assert normalize({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}
    f = fuse({"a": 0.9}, {"b": 5.0}, 0.5)  # id present in only one list still gets both raw fields
    assert f["a"]["lexical_raw"] == 0.0 and f["b"]["semantic_raw"] == 0.0


def test_hybrid_candidate_depth_recorded(wb):
    h = wb.search("abrupt shifts", 3)[0]
    assert h.breakdown["candidate_depth"] == 50 and h.breakdown["n_candidates"] >= 3


def test_q_update_arithmetic():
    # Q=0.5, r=1, gamma=0.9, max next=0.4, eta=0.2 -> 0.5 + 0.2*(1 + 0.36 - 0.5) = 0.672
    assert q_update(0.5, 1.0, 0.4, 0.2, 0.9) == pytest.approx(0.672)
    assert q_update(0.5, 1.0, None, 0.2, 0.9) == pytest.approx(0.6)  # terminal: no bootstrap


def test_frozen_policy_cannot_learn():
    p = QPolicy().freeze()
    with pytest.raises(RuntimeError):
        p.update(("short", "low"), 0.5, 1.0)


def test_policy_isolated_per_user_and_neutral_default():
    st = PolicyStore()
    st.get("u1").update(("short", "low"), 0.8, 1.0)
    assert st.get("u2").q == {} and st.get("u2").greedy(("short", "low")) == 0.5


def test_training_never_touches_test_queries(wb):
    seen = []
    orig = wb.searcher.search

    class Spy:
        def __init__(self):
            self.__dict__.update(wb.searcher.__dict__)

        def search(self, q, **kw):
            seen.append(q)
            return orig(q, **kw)

        lexical_only = wb.searcher.lexical_only
        dense_only = wb.searcher.dense_only
    train = {"t1": "control chart for tool wear drift"}
    train_policy(Spy(), train, {"t1": {}}, lambda r, rel: 0.0, epochs=1)
    assert set(seen) == set(train.values())


def test_graph_removal_eliminates_expansion_effect(wb):
    from litmine.retrieval import HybridSearcher
    q = "Bayesian changepoint streamflow"
    with_g = wb.searcher.search(q, 10, expand=True)
    assert with_g[0].expansion_terms  # graph contributed recorded terms
    no_g = HybridSearcher(wb.corpus.bm25, wb.corpus.dense, wb.corpus.encoder, wb.cfg, None)
    a = [(h.work_id, h.score) for h in no_g.search(q, 10, expand=True)]
    b = [(h.work_id, h.score) for h in no_g.search(q, 10, expand=False)]
    assert a == b and not no_g.search(q, 1, expand=True)[0].expansion_terms


def test_benchmarks_record_percentiles_and_errors():
    r = bench_latency(lambda q: [q], ["a", "b", "c", "d"], warmup=1)
    assert r["n"] == 4 and r["p95_ms"] >= r["p50_ms"] and r["concurrency"] == 1
    t = bench_throughput(lambda q: [q] if q != "bad" else (_ for _ in ()).throw(ValueError()), ["a", "bad"], 2, rounds=2)
    assert t["errors"] == 2 and t["completed"] == 2
