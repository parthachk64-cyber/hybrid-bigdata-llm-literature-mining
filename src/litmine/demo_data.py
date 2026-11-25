"""Loader for the bundled SYNTHETIC demo corpus. Its outputs must never enter benchmark reports."""
from __future__ import annotations

import json
from pathlib import Path

from .data import Work, dedupe, norm_doi, normalize_record, read_jsonl

DEMO_DIR = Path(__file__).resolve().parent / "demo"
SYNTHETIC_BANNER = "SYNTHETIC DEMO DATA - not a benchmark; excluded from every benchmark report"


def demo_works() -> tuple[list[Work], list[dict]]:
    raw = [normalize_record(r, origin="demo/corpus.jsonl") for r in read_jsonl(DEMO_DIR / "corpus.jsonl")]
    return dedupe(raw)


def demo_queries() -> dict[str, str]:
    return {r["qid"]: r["text"] for r in read_jsonl(DEMO_DIR / "queries.jsonl")}


def demo_qrels(works: list[Work]) -> dict[str, dict]:
    by_doi = {w.doi: w.work_id for w in works}
    out: dict[str, dict] = {}
    for i, line in enumerate(open(DEMO_DIR / "qrels.tsv", encoding="utf-8")):
        if i == 0:
            continue
        qid, did, g = line.rstrip("\n").split("\t")
        out.setdefault(qid, {})[by_doi[norm_doi(did.removeprefix("doi:"))]] = int(g)
    return out


def demo_split() -> dict:
    return json.loads((DEMO_DIR / "split.json").read_text(encoding="utf-8"))


def demo_sessions() -> dict:
    return json.loads((DEMO_DIR / "feedback_sessions.json").read_text(encoding="utf-8"))


def demo_lexicon_path() -> Path:
    return DEMO_DIR / "lexicon.yaml"
