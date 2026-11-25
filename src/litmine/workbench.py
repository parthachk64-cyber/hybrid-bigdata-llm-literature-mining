"""Workbench: ties ingestion, indexes, graph, retrieval, synthesis and reporting into one object for the CLI and notebooks."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import config_hash
from .data import Work, load_snapshot, segment
from .discovery import topic_year_counts, trend_scores
from .distributed import Corpus, build_local, searcher
from .graph import KnowledgeGraph
from .retrieval import Hit, best_passages
from .synthesis import (
    OllamaGenerator,
    citation_intent,
    extract_claims,
    extractive_answer,
    generative_answer,
    load_lexicon,
)


class Workbench:
    def __init__(self, works: list[Work], cfg: dict, lexicon_path: str | Path, encoder=None, label: str = ""):
        self.cfg, self.label = cfg, label
        self.corpus: Corpus = build_local(works, cfg, encoder)
        self.lexicon = load_lexicon(lexicon_path)
        ing = cfg["ingest"]
        self.claims: list[dict] = []
        for w in works:
            self.claims += extract_claims(w, segment(w, ing["segment_max_chars"], ing["segment_sentences"]), self.lexicon)
        self.graph = KnowledgeGraph.build(works, self.claims)
        self.searcher = searcher(self.corpus, self.graph)
        self.rows = {w: [self.corpus.dense.passage_ids[r] for r in rs] for w, rs in self.corpus.dense.rows.items()}

    @classmethod
    def from_snapshot(cls, snap_dir: str | Path, cfg: dict, lexicon_path: str | Path, encoder=None) -> Workbench:
        return cls(load_snapshot(snap_dir), cfg, lexicon_path, encoder)

    @property
    def works(self) -> dict[str, Work]:
        return self.corpus.works

    def tag(self, wid: str) -> str:
        w = self.works[wid]
        return f"{w.source_id} ({w.year})"

    def search(self, q: str, k: int | None = None, alpha: float | None = None, expand: bool | None = None) -> list[Hit]:
        return self.searcher.search(q, alpha=alpha, k=k, expand=expand)

    def explain(self, q: str, k: int = 5, alpha: float | None = None, expand: bool | None = None) -> list[dict]:
        out = []
        for h in self.search(q, k, alpha, expand):
            w = self.works[h.work_id]
            ps = best_passages(q, self.corpus.dense, self.corpus.encoder, self.corpus.passages, h.work_id, 1)
            ev = None
            if ps:
                p = self.corpus.passages[ps[0]["passage_id"]]
                ev = {"passage_id": p.passage_id, "start": p.start, "end": p.end, "field": p.field, "text": p.text}
            out.append({"rank": h.rank, "work_id": h.work_id, "source_id": w.source_id, "year": w.year, "title": w.title,
                        "score": h.score, "breakdown": h.breakdown, "expansion_terms": h.expansion_terms, "evidence": ev})
        return out

    def methods_of(self, wid: str) -> list[dict]:
        return [c for c in self.claims if c["work_id"] == wid and c["kind"] == "method"]

    def citation_chain(self, a: str, b: str, max_hops: int = 4) -> dict | None:
        path = self.graph.find_chain_between(a, b, max_hops)
        if not path:
            return None
        return {"verified": self.graph.verify_path(path), "hops": [{
            "citing": self.tag(e["src"]), "cited": self.tag(e["dst"]), "marker": e["marker"],
            "intent": citation_intent(e["evidence"]["text"]), "evidence": e["evidence"]} for e in path]}

    def compare_fields(self, question: str, field_a: str, field_b: str, k: int = 12) -> dict:
        hits = self.search(question, k=k)
        groups: dict[str, list[Hit]] = defaultdict(list)
        for h in hits:
            for s in self.works[h.work_id].subjects:
                if s in (field_a, field_b):
                    groups[s].append(h)
        report: dict = {"question": question, "fields": {}, "note": None}
        for f in (field_a, field_b):
            ms: dict[str, dict] = {}
            for h in groups.get(f, []):
                for c in self.methods_of(h.work_id):
                    ms.setdefault(c["label"], {"label": c["label"], "evidence": c, "work": self.tag(h.work_id)})
            report["fields"][f] = sorted(ms.values(), key=lambda x: x["label"])
        chain = None
        only = lambda f, hs: [h for h in hs if self.works[h.work_id].subjects == [f]]
        for ha in only(field_a, groups.get(field_a, [])):
            for hb in only(field_b, groups.get(field_b, [])):
                c = self.citation_chain(ha.work_id, hb.work_id)
                if c and c["verified"] and (chain is None or len(c["hops"]) < len(chain["hops"])):
                    chain = c
        report["citation_chain"] = chain
        if not any(report["fields"].values()):
            report["note"] = "No supporting passage found"
        return report

    def answer(self, q: str, backend: str = "extractive", k: int = 8, model: str = "llama3.1:8b") -> dict:
        hits = self.search(q, k=k)
        if backend == "extractive":
            return extractive_answer(q, hits, self.works, self.corpus.passages, self.rows)
        return generative_answer(q, hits, self.corpus.passages, self.rows, OllamaGenerator(model))

    def trends(self, decay: float | None = None) -> dict:
        topics: dict[str, set] = defaultdict(set)
        for c in self.claims:
            if c["kind"] == "topic":
                topics[c["work_id"]].add(c["label"])
        f, skipped = topic_year_counts(list(self.works.values()), topics)
        years = sorted({y for c in f.values() for y in c})
        d = decay if decay is not None else self.cfg["trend"]["decay"]
        return {"counts": {k: dict(sorted(c.items())) for k, c in sorted(f.items())}, "excluded_without_year": skipped,
                "T_k": trend_scores(f, years[0], years[-1], d) if years else {}, "decay": d, "bin": "year"}

    def provenance(self) -> dict:
        return {"label": self.label, "config_hash": config_hash(self.cfg), "encoder": self.corpus.encoder.card(),
                "run_info": self.corpus.run_info, "n_works": len(self.works), "n_passages": len(self.corpus.passages)}


# ---------------------------------------------------------------- reporting

def render_report(wb: Workbench, question: str, banner: str = "", k: int = 5, backend: str = "extractive",
                  field_a: str | None = None, field_b: str | None = None) -> str:
    lines = [f"# Search report: {question}", ""]
    if banner:
        lines += [f"> **{banner}**", ""]
    prov = wb.provenance()
    lines += [f"Encoder: `{prov['encoder']['name']}` (transformer: {prov['encoder']['is_transformer']}) · alpha={wb.cfg['retrieval']['alpha']}"
              f" · backend `{prov['run_info']['backend']}` · workers {prov['run_info']['workers']} · mode {prov['run_info']['mode']}", ""]
    lines += ["## Ranked papers and why", "", "| # | paper | score | semantic raw | BM25 raw | alpha |", "|---|---|---|---|---|---|"]
    ex = wb.explain(question, k)
    for e in ex:
        b = e["breakdown"]
        lines.append(f"| {e['rank']} | {e['source_id']} ({e['year']}) {e['title']} | {e['score']:.3f} | {b['semantic_raw']:.3f} | {b['lexical_raw']:.3f} | {b['alpha']} |")
    lines += ["", "### Evidence passage behind each rank", ""]
    for e in ex:
        ev = e["evidence"]
        if ev:
            lines.append(f"- **{e['source_id']}** `{ev['passage_id']}` chars {ev['start']}-{ev['end']}: \"{ev['text']}\"")
    ans = wb.answer(question, backend)
    lines += ["", f"## Attributed synthesis ({ans['backend']})", ""]
    if ans["abstained"]:
        lines.append("No supporting passage found.")
    for s in ans["statements"]:
        lines.append(f"- {s['text']} — `{s['passage_id']}` [{s['start']}:{s['end']}] ({wb.tag(s['work_id'])})")
    if field_a and field_b:
        cf = wb.compare_fields(question, field_a, field_b)
        lines += ["", f"## Methods by field: {field_a} vs {field_b}", ""]
        for f, ms in cf["fields"].items():
            lines.append(f"**{f}**: " + (", ".join(m["label"] for m in ms) or "none found"))
            for m in ms[:2]:
                c = m["evidence"]
                lines.append(f"  - {m['label']}: `{c['passage_id']}` [{c['start']}:{c['end']}] {m['work']}")
        lines += ["", "### Verified citation chain", ""]
        ch = cf["citation_chain"]
        if ch:
            for h in ch["hops"]:
                ev = h["evidence"]
                lines.append(f"- {h['citing']} cites {h['cited']} (marker [{h['marker']}], intent **{h['intent']}**): "
                             f"\"{ev['text']}\" — `{ev['passage_id']}` [{ev['start']}:{ev['end']}]")
        else:
            lines.append("No verified citation chain found.")
    tr = wb.trends()
    lines += ["", "## Trend counts by year (description only, not a forecast)", "",
              "| topic | counts by year | T_k (decay %s) |" % tr["decay"], "|---|---|---|"]
    for t, c in tr["counts"].items():
        lines.append(f"| {t} | {json.dumps(c)} | {tr['T_k'][t]:.3f} |")
    return "\n".join(lines) + "\n"


def counts_by_type(claims: list[dict]) -> dict:
    return dict(Counter(c["kind"] for c in claims))
