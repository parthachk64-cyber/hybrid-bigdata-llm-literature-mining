"""Ingestion: JSONL records -> normalized works -> version-aware dedup -> passages with spans.

Nothing here invents a DOI, reference, edge or year. Missing fields stay missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
VERSION_RE = re.compile(r"(?:v|_v|\.v)(\d+)$", re.IGNORECASE)
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip()


def norm_title(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", norm_text(s).lower()).strip()


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    m = DOI_RE.search(doi.strip())
    if not m:
        return None
    d = m.group(0).lower().rstrip(".,;)")
    return VERSION_RE.sub("", d)


def parse_year(date: str | None) -> int | None:
    m = re.match(r"(\d{4})", date or "")
    return int(m.group(1)) if m else None


def stable_id(doi: str | None, title: str, first_author: str = "", year: int | None = None) -> str:
    """Version-independent ID: DOI if present, else title+first author+year."""
    key = f"doi:{doi}" if doi else f"t:{norm_title(title)}|{norm_title(first_author)}|{year or ''}"
    return "w_" + hashlib.sha1(key.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    work_id: str
    field: str  # title | abstract | body
    start: int
    end: int
    text: str


@dataclass
class Work:
    work_id: str
    source_id: str
    doi: str | None
    version: int
    title: str
    authors: list
    date: str | None
    year: int | None
    abstract: str
    body: str
    references: list
    license: str | None
    provenance: dict = field(default_factory=dict)
    versions_seen: list = field(default_factory=list)
    subjects: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    def full_text(self) -> str:
        return " ".join(x for x in (self.title, self.abstract, self.body) if x)


def _norm_author(a) -> dict:
    if isinstance(a, str):
        a = {"name": a}
    return {"name": norm_text(a.get("name")), "orcid": a.get("orcid") or None,
            "affiliation": norm_text(a.get("affiliation")) or None}


def normalize_record(rec: dict, origin: str = "") -> Work:
    """Map a raw JSON record to a Work, keeping original field names in provenance."""
    doi = norm_doi(rec.get("doi"))
    m = VERSION_RE.search(str(rec.get("doi") or "")) or VERSION_RE.search(str(rec.get("source_id") or ""))
    version = int(rec.get("version") or (m.group(1) if m else 1))
    title = norm_text(rec.get("title"))
    authors = [_norm_author(a) for a in rec.get("authors", [])]
    date = rec.get("date") or (str(rec["year"]) if rec.get("year") else None)
    year = parse_year(date)
    source_id = str(rec.get("source_id") or rec.get("id") or "")
    wid = stable_id(doi, title, authors[0]["name"] if authors else "", year)
    return Work(
        work_id=wid, source_id=source_id, doi=doi, version=version, title=title, authors=authors,
        date=date, year=year, abstract=norm_text(rec.get("abstract")), body=norm_text(rec.get("body")),
        references=[dict(r) for r in rec.get("references", [])], license=rec.get("license"),
        provenance={"origin": origin, "original_fields": sorted(rec.keys())}, versions_seen=[version],
        subjects=list(rec.get("subjects", [])),
    )


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # fail loudly with location, never skip silently
                raise ValueError(f"{path}:{i + 1}: invalid JSON ({e})") from e


def dedupe(works: Iterable[Work]) -> tuple[list[Work], list[dict]]:
    """Version-aware dedup. Key = DOI (version stripped) else normalized title. Highest version wins."""
    groups: dict[str, list[Work]] = {}
    for w in works:
        key = f"doi:{w.doi}" if w.doi else f"t:{norm_title(w.title)}"
        groups.setdefault(key, []).append(w)
    kept, dropped = [], []
    for key in sorted(groups):
        ws = sorted(groups[key], key=lambda x: (x.version, x.source_id))
        best = ws[-1]
        best.versions_seen = sorted({v for x in ws for v in (x.versions_seen or [x.version])})
        kept.append(best)
        for w in ws[:-1]:
            if w.source_id == best.source_id and w.version == best.version:
                continue  # same record seen again (re-ingestion): not a duplicate to report
            dropped.append({"dropped_source_id": w.source_id, "kept_source_id": best.source_id,
                            "reason": "superseded_version" if w.version != best.version else "exact_duplicate"})
    return sorted(kept, key=lambda w: w.work_id), dropped


def segment(work: Work, max_chars: int = 420, n_sent: int = 3) -> list[Passage]:
    """Sentence-window passages; passage.text == source_field[start:end] exactly (no rewriting)."""
    out: list[Passage] = []
    if work.title:
        out.append(Passage(f"{work.work_id}#title#0", work.work_id, "title", 0, len(work.title), work.title))
    for fname in ("abstract", "body"):
        src = getattr(work, fname)
        if not src:
            continue
        spans, pos = [], 0
        for m in SENT_RE.finditer(src):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(src)))
        i, idx = 0, 0
        while i < len(spans):
            s, e = spans[i]
            j = i + 1
            while j < len(spans) and j - i < n_sent and spans[j][1] - s <= max_chars:
                e = spans[j][1]
                j += 1
            out.append(Passage(f"{work.work_id}#{fname}#{idx}", work.work_id, fname, s, e, src[s:e]))
            idx += 1
            i = j
    return out


def resolve_references(works: list[Work]) -> list[dict]:
    """Resolve reference dicts to corpus works by DOI, then exact title. Unresolved stay unresolved.

    Each edge carries the citing passage (the sentence holding the [n] marker) as evidence.
    """
    by_doi = {w.doi: w for w in works if w.doi}
    by_title = {norm_title(w.title): w for w in works}
    edges = []
    for w in works:
        passages = segment(w)
        for idx, ref in enumerate(w.references, start=1):
            tgt = by_doi.get(norm_doi(ref.get("doi"))) or (by_title.get(norm_title(ref.get("title"))) if ref.get("title") else None)
            edges.append({"src": w.work_id, "dst": tgt.work_id if tgt else None, "marker": idx, "raw": ref,
                          "resolved": tgt is not None, "evidence": _marker_evidence(w, passages, idx),
                          "src_year": w.year, "dst_year": tgt.year if tgt else None})
    return edges


def _marker_evidence(w: Work, passages: list[Passage], idx: int) -> dict | None:
    tag = f"[{idx}]"
    for p in passages:
        if p.field != "title" and tag in p.text:
            k = p.text.index(tag)
            return {"passage_id": p.passage_id, "field": p.field, "start": p.start, "end": p.end,
                    "marker_start": p.start + k, "text": p.text}
    return None


# ---------- snapshots (all-or-nothing writes) ----------

def corpus_hash(works: list[Work]) -> str:
    h = hashlib.sha256()
    for w in sorted(works, key=lambda x: x.work_id):
        h.update(json.dumps(w.to_json(), sort_keys=True).encode())
    return h.hexdigest()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_snapshot(works: list[Work], dropped: list[dict], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "corpus.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(w.to_json(), sort_keys=True, ensure_ascii=False) + "\n" for w in sorted(works, key=lambda x: x.work_id))
    os.replace(tmp, out / "corpus.jsonl")
    manifest = {"n_works": len(works), "hash": corpus_hash(works), "dropped": dropped}
    _atomic_json(out / "snapshot.json", manifest)
    return manifest


def load_snapshot(out_dir: str | Path) -> list[Work]:
    return [Work(**r) for r in read_jsonl(Path(out_dir) / "corpus.jsonl")]


def ingest(paths: list, out_dir: str | Path) -> dict:
    """Idempotent batch/incremental ingest. Merges with existing snapshot, dedups, rewrites only on change."""
    out = Path(out_dir)
    existing = load_snapshot(out) if (out / "corpus.jsonl").exists() else []
    prev_hash = corpus_hash(existing) if existing else None
    raw = list(existing)
    n_in = 0
    for p in paths:
        for rec in read_jsonl(p):
            raw.append(normalize_record(rec, origin=str(p)))
            n_in += 1
    works, dropped = dedupe(raw)
    new_hash = corpus_hash(works)
    changed = new_hash != prev_hash
    if changed:
        write_snapshot(works, dropped, out)
    return {"records_read": n_in, "works_after_dedup": len(works), "dropped": len(dropped),
            "hash": new_hash, "changed": changed}
