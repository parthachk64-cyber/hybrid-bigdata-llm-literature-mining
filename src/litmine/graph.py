"""Typed knowledge graph G=(V,E) with directed, evidence-carrying edges.

Edge types
  cites         work -> work     explicit, from a resolved reference; evidence = citing passage. inferred=False
  authored_by   work -> author   explicit metadata. inferred=False
  uses_method   work -> method   lexicon/rule extraction, evidence = passage span. inferred=True
  uses_dataset  work -> dataset  same. inferred=True
  about_topic   work -> topic    same. inferred=True

Model/rule-inferred associations never masquerade as citations: they carry inferred=True and a different type.
"""
from __future__ import annotations

from collections import Counter, deque

from .data import Work, norm_title, resolve_references
from .retrieval import lex_tokens


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.rejected: list[dict] = []
        self._out: dict[tuple[str, str], list[dict]] = {}
        self.aliases: dict[str, str] = {}  # lowercase surface form -> node id

    # ------------------------------------------------------------ building
    def add_node(self, nid: str, ntype: str, label: str, **attrs) -> None:
        self.nodes.setdefault(nid, {"type": ntype, "label": label, **attrs})

    def add_edge(self, src: str, dst: str, etype: str, evidence: dict | None, inferred: bool, **attrs) -> dict:
        e = {"src": src, "dst": dst, "type": etype, "evidence": evidence, "inferred": inferred, **attrs}
        self.edges.append(e)
        self._out.setdefault((src, etype), []).append(e)
        return e

    @classmethod
    def build(cls, works: list[Work], claims: list[dict] | None = None) -> KnowledgeGraph:
        g = cls()
        for w in works:
            g.add_node(w.work_id, "work", w.title, year=w.year, source_id=w.source_id, subjects=w.subjects)
        g._add_authors(works)
        for e in resolve_references(works):
            if not e["resolved"]:
                g.rejected.append({**e, "reason": "unresolved_reference"})
            elif e["evidence"] is None:
                g.rejected.append({**e, "reason": "no_citing_passage"})
            elif e["src_year"] is not None and e["dst_year"] is not None and e["src_year"] < e["dst_year"]:
                g.rejected.append({**e, "reason": "cites_a_later_work"})
            else:
                g.add_edge(e["src"], e["dst"], "cites", e["evidence"], False, marker=e["marker"],
                           src_year=e["src_year"], dst_year=e["dst_year"])
        for c in claims or []:
            nid = f"{c['kind'][0]}:{c['label'].lower()}"
            g.add_node(nid, c["kind"], c["label"])
            g.aliases[c["label"].lower()] = nid
            for s in c.get("surfaces", []):
                g.aliases[s.lower()] = nid
            et = {"method": "uses_method", "dataset": "uses_dataset", "topic": "about_topic"}[c["kind"]]
            g.add_edge(c["work_id"], nid, et, {"passage_id": c["passage_id"], "start": c["start"], "end": c["end"],
                                              "text": c.get("text")}, True, extractor=c.get("extractor", "lexicon"))
        return g

    def _add_authors(self, works: list[Work]) -> None:
        for w in works:
            for a in w.authors:
                orcid, name = a.get("orcid"), a["name"]
                if orcid:
                    nid = f"a:orcid:{orcid}"
                    amb = False
                else:
                    amb = True  # name only: cannot be told apart from any same-name author
                    nid = f"a:name:{name.lower()}"
                self.add_node(nid, "author", name, ambiguous=amb, orcid=orcid)
                self.add_edge(w.work_id, nid, "authored_by", {"field": "authors"}, False)

    # ------------------------------------------------------------- queries
    def out(self, nid: str, etype: str) -> list[dict]:
        return self._out.get((nid, etype), [])

    def cites_path(self, src: str, dst: str, max_hops: int = 4) -> list[dict] | None:
        """Shortest directed citation chain src -> ... -> dst, bounded by max_hops. None if none exists."""
        if src == dst:
            return []
        seen = {src}
        q: deque = deque([(src, [])])
        while q:
            node, path = q.popleft()
            if len(path) >= max_hops:
                continue
            for e in sorted(self.out(node, "cites"), key=lambda x: x["dst"]):
                if e["dst"] in seen:
                    continue
                if e["dst"] == dst:
                    return path + [e]
                seen.add(e["dst"])
                q.append((e["dst"], path + [e]))
        return None

    def verify_path(self, path: list[dict]) -> bool:
        """A chain is verified only if every hop is explicit, evidenced, and temporally possible."""
        return bool(path) and all(
            e["type"] == "cites" and not e["inferred"] and e["evidence"] and e["evidence"].get("text")
            and (e.get("src_year") is None or e.get("dst_year") is None or e["src_year"] >= e["dst_year"])
            for e in path)

    def find_chain_between(self, a: str, b: str, max_hops: int = 4) -> list[dict] | None:
        """Chain in either citation direction (newer cites older)."""
        return self.cites_path(a, b, max_hops) or self.cites_path(b, a, max_hops)

    def work_labels(self, work_id: str, etypes=("uses_method", "about_topic")) -> set[str]:
        return {e["dst"] for et in etypes for e in self.out(work_id, et)}

    def topic_overlap(self, a: str, b: str) -> dict:
        la, lb = self.work_labels(a, ("about_topic",)), self.work_labels(b, ("about_topic",))
        ma, mb = self.work_labels(a, ("uses_method",)), self.work_labels(b, ("uses_method",))
        j = len(la & lb) / len(la | lb) if (la | lb) else 0.0
        return {"shared_topics": sorted(la & lb), "shared_methods": sorted(ma & mb), "topic_jaccard": j}

    def collaborators(self, author_id: str) -> dict:
        works = [e["src"] for e in self.edges if e["type"] == "authored_by" and e["dst"] == author_id]
        co: Counter = Counter()
        for w in works:
            for e in self.out(w, "authored_by"):
                if e["dst"] != author_id:
                    co[e["dst"]] += 1
        return {"author": author_id, "ambiguous": self.nodes[author_id].get("ambiguous", False),
                "works": sorted(works), "collaborators": dict(sorted(co.items()))}

    def expansion_terms(self, query: str, max_terms: int = 4) -> list[str]:
        """Explainable query expansion: labels of graph nodes that co-occur (in works) with nodes the query names."""
        ql = " " + " ".join(lex_tokens(query, stop=False)) + " "
        matched = {nid for surf, nid in self.aliases.items() if f" {surf} " in ql}
        if not matched:
            return []
        works = {e["src"] for e in self.edges if e["dst"] in matched and e["type"] in ("uses_method", "about_topic", "uses_dataset")}
        score: Counter = Counter()
        for w in works:
            for et in ("uses_method", "about_topic"):
                for e in self.out(w, et):
                    if e["dst"] not in matched:
                        score[e["dst"]] += 1
        qtoks = set(lex_tokens(query))
        out = []
        for nid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])):
            label = self.nodes[nid]["label"]
            if not set(lex_tokens(label)) <= qtoks:
                out.append(label)
            if len(out) >= max_terms:
                break
        return out

    def stats(self) -> dict:
        et = Counter(e["type"] for e in self.edges)
        nt = Counter(n["type"] for n in self.nodes.values())
        return {"nodes": dict(nt), "edges": dict(et), "rejected_citation_edges": Counter(r["reason"] for r in self.rejected)}


def title_index(works: list[Work]) -> dict[str, str]:
    return {norm_title(w.title): w.work_id for w in works}
