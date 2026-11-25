"""litmine command line. Every command prints plain text/JSON; nothing needs a network or token in the offline profile."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PAPER_DOI
from .config import load_config
from .data import ingest, load_snapshot, read_jsonl
from .demo_data import (
    SYNTHETIC_BANNER,
    demo_lexicon_path,
    demo_qrels,
    demo_queries,
    demo_sessions,
    demo_split,
    demo_works,
)
from .workbench import Workbench, render_report

QUESTION = "Which methods have hydrology and manufacturing used to detect abrupt shifts in sensor streams?"


def _cfg(a) -> dict:
    ov = {}
    if getattr(a, "encoder", None):
        ov["encoder"] = {"name": a.encoder}
    if getattr(a, "alpha", None) is not None:
        ov["retrieval"] = {"alpha": a.alpha}
    return load_config(getattr(a, "config", None), ov)


def _wb(a) -> Workbench:
    cfg = _cfg(a)
    lex = getattr(a, "lexicon", None) or demo_lexicon_path()
    if getattr(a, "snapshot", None):
        return Workbench(load_snapshot(a.snapshot), cfg, lex, label="user snapshot")
    works, _ = demo_works()
    return Workbench(works, cfg, lex, label="demo")


def cmd_demo(a):
    wb = _wb(a)
    text = render_report(wb, a.question, SYNTHETIC_BANNER if wb.label == "demo" else "", k=5,
                         backend=a.backend, field_a="hydrology", field_b="manufacturing")
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")
    print(text)


def cmd_ingest(a):
    print(json.dumps(ingest(a.input, a.out), indent=2))


def cmd_index(a):
    from .distributed import current_generation, publish_generation
    cfg = _cfg(a)
    cfg["distributed"]["workers"], cfg["distributed"]["shards"] = a.workers, a.shards
    man = publish_generation(a.index, load_snapshot(a.snapshot), cfg)
    print(json.dumps({"generation": man["generation"], "shards": len(man["shards"]), "backend": man["backend"],
                      "workers": man["workers"], "current": str(current_generation(a.index))}, indent=2))


def cmd_search(a):
    wb = _wb(a)
    for e in wb.explain(a.query, a.k, a.alpha, True if a.expand else None):
        b = e["breakdown"]
        print(f"{e['rank']:>2}. {e['source_id']} ({e['year']}) {e['title']}\n    score={e['score']:.3f} semantic={b['semantic_raw']:.3f}"
              f" bm25={b['lexical_raw']:.3f} alpha={b['alpha']} expansion={e['expansion_terms']}")
        if e["evidence"]:
            ev = e["evidence"]
            print(f"    evidence {ev['passage_id']} [{ev['start']}:{ev['end']}]: {ev['text']}")


def cmd_synthesize(a):
    print(json.dumps(_wb(a).answer(a.query, a.backend), indent=2, default=str))


def cmd_path(a):
    wb = _wb(a)
    by_src = {w.source_id: w.work_id for w in wb.works.values()}
    chain = wb.citation_chain(by_src[a.a], by_src[a.b], a.max_hops)
    print(json.dumps(chain, indent=2) if chain else "No citation chain found")


def cmd_feedback(a):
    from .experiments import ndcg_reward
    from .feedback import QPolicy, train_policy
    wb = _wb(a)
    works = list(wb.works.values())
    qs, rel, split, sess = demo_queries(), demo_qrels(works), demo_split(), demo_sessions()
    train = {q: qs[q] for q in split["train"]}
    pol = train_policy(wb.searcher, train, rel, ndcg_reward, epochs=a.epochs, sessions=sess["user_a"] + sess["user_b"],
                       policy=QPolicy(seed=wb.cfg["feedback"]["seed"]))
    Path(a.out).write_text(json.dumps(pol.to_json(), indent=2), encoding="utf-8")
    print(f"SIMULATED feedback sandbox (rewards from training qrels, not clicks). updates={pol.n_updates} -> {a.out}")


def cmd_eval(a):
    from .bench import beir_corpus, load_beir
    from .experiments import run_experiment
    cfg = _cfg(a)
    if a.dataset == "demo":
        wb = _wb(a)
        works = list(wb.works.values())
        qs, rel, sp = demo_queries(), demo_qrels(works), demo_split()
        out = run_experiment(a.runs, a.run_id, "SYNTHETIC DEMO (regression only)", wb.corpus, wb.graph, {q: qs[q] for q in sp["test"]}, rel,
                             {q: qs[q] for q in sp["train"]}, banner=SYNTHETIC_BANNER, dataset_meta={"synthetic": True})
    else:
        d = Path(a.data)
        docs, tq, trel = load_beir(d / "corpus.parquet", d / "queries.parquet", d / "test.tsv")
        _, trq, rrel = load_beir(d / "corpus.parquet", d / "queries.parquet", d / "train.tsv")
        out = run_experiment(a.runs, a.run_id, "SciFact test", beir_corpus(docs, cfg, cache_dir=d / "cache"), None, tq, {**rrel, **trel}, trq)
    print(out["run_dir"])
    print((Path(out["run_dir"]) / "tables" / "tables.md").read_text(encoding="utf-8"))


def cmd_scale(a):
    from .bench import plan_scale_run
    ids = [r["id"] if "id" in r else r.get("doi") or r.get("source_id") for r in read_jsonl(a.manifest)]
    print(json.dumps(plan_scale_run(ids, tuple(int(x) for x in a.sizes.split(",")), a.state), indent=2))


def cmd_plot(a):
    from .plots import plot_run
    print("\n".join(plot_run(a.run)))


def cmd_ledger(a):
    from .plots import write_ledger
    print(write_ledger(a.runs, a.out))


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser("litmine", description=f"Companion to DOI {PAPER_DOI}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        s.add_argument("--config")
        s.add_argument("--encoder", help="hashing (offline substitute) | ollama:<model> | hf:<id> (needs revision in config)")
        return s
    s = add("demo", cmd_demo); s.add_argument("--question", default=QUESTION); s.add_argument("--report")
    s.add_argument("--backend", default="extractive", choices=["extractive", "ollama"]); s.add_argument("--snapshot")
    s = add("ingest", cmd_ingest); s.add_argument("--input", nargs="+", required=True); s.add_argument("--out", required=True)
    s = add("index", cmd_index); s.add_argument("--snapshot", required=True); s.add_argument("--index", required=True)
    s.add_argument("--workers", type=int, default=2); s.add_argument("--shards", type=int, default=4)
    s = add("search", cmd_search); s.add_argument("query"); s.add_argument("--k", type=int, default=5)
    s.add_argument("--alpha", type=float); s.add_argument("--expand", action="store_true"); s.add_argument("--snapshot")
    s = add("synthesize", cmd_synthesize); s.add_argument("query"); s.add_argument("--backend", default="extractive", choices=["extractive", "ollama"])
    s.add_argument("--snapshot")
    s = add("path", cmd_path); s.add_argument("a"); s.add_argument("b"); s.add_argument("--max-hops", type=int, default=4); s.add_argument("--snapshot")
    s = add("feedback-train", cmd_feedback); s.add_argument("--out", default="policy.json"); s.add_argument("--epochs", type=int, default=3); s.add_argument("--snapshot")
    s = add("eval", cmd_eval); s.add_argument("--dataset", choices=["demo", "scifact"], default="demo"); s.add_argument("--data", default="data/scifact")
    s.add_argument("--runs", default="runs"); s.add_argument("--run-id", required=True); s.add_argument("--snapshot")
    s = add("scale", cmd_scale); s.add_argument("--manifest", required=True); s.add_argument("--sizes", default="100000,300000,600000,1000000"); s.add_argument("--state")
    s = add("plot", cmd_plot); s.add_argument("--run", required=True)
    s = add("ledger", cmd_ledger); s.add_argument("--runs", default="runs"); s.add_argument("--out", default="docs/results.md")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
