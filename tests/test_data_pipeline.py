import json

import numpy as np
import pytest

from litmine.config import load_config
from litmine.data import ingest, load_snapshot, norm_doi, normalize_record, segment
from litmine.demo_data import DEMO_DIR, demo_works
from litmine.distributed import ShardedSearcher, current_generation, publish_generation
from litmine.encoders import HashingEncoder, cosine_matrix, enc_tokens


def test_demo_corpus_shape(works):
    assert len(works) >= 30
    raw = [json.loads(x) for x in open(DEMO_DIR / "corpus.jsonl", encoding="utf-8")]
    assert len(raw) == len(works) + 1  # one duplicate version dropped
    assert any(not w.abstract for w in works)  # item without abstract


def test_version_aware_dedup_keeps_latest():
    ws, dropped = demo_works()
    s12 = [w for w in ws if w.doi == "10.5555/litmine.syn.012"]
    assert len(s12) == 1 and s12[0].version == 2 and s12[0].versions_seen == [1, 2]
    assert dropped == [{"dropped_source_id": "S12v1", "kept_source_id": "S12v2", "reason": "superseded_version"}]


def test_norm_doi_strips_version_and_case():
    assert norm_doi("DOI:10.5555/ABC.1v3") == "10.5555/abc.1"


def test_passage_spans_preserved(works):
    for w in works:
        for p in segment(w):
            src = w.title if p.field == "title" else getattr(w, p.field)
            assert src[p.start:p.end] == p.text


def test_encoder_input_keeps_negation_and_symbols():
    toks = enc_tokens("Effect was NOT significant (p<0.05) at 5%")
    assert "not" in toks and "<" in toks and "%" in toks


def test_ingest_idempotent_and_incremental(tmp_path):
    src = DEMO_DIR / "corpus.jsonl"
    a = ingest([src], tmp_path)
    b = ingest([src], tmp_path)
    assert a["changed"] and not b["changed"] and a["hash"] == b["hash"]
    extra = tmp_path / "extra.jsonl"
    extra.write_text(json.dumps({"source_id": "N1", "doi": "10.5555/new.1", "title": "A new work", "abstract": "Text.", "date": "2025-01-01"}) + "\n")
    c = ingest([extra], tmp_path)
    assert c["changed"] and c["works_after_dedup"] == a["works_after_dedup"] + 1
    assert len(load_snapshot(tmp_path)) == c["works_after_dedup"]


def test_bad_json_line_fails_loudly(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"title": "x"}\n{broken\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        ingest([p], tmp_path / "o")


def test_missing_fields_are_not_invented():
    w = normalize_record({"title": "No metadata"})
    assert w.doi is None and w.year is None and w.references == []


def test_embedding_shape_and_zero_policy():
    e = HashingEncoder(64)
    m = e.encode(["abrupt shift", ""])
    assert m.shape == (2, 64) and m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m[0]), 1.0) and not m[1].any()
    assert cosine_matrix(m[1:], m[:1])[0, 0] == 0.0


def test_partial_ingestion_restart_keeps_old_generation(tmp_path, works):
    cfg = load_config(overrides={"distributed": {"workers": 1, "shards": 3}})
    m1 = publish_generation(tmp_path, works, cfg)
    cur = current_generation(tmp_path)
    assert m1["generation"] == 1 and cur.name == "gen-0001"
    with pytest.raises(RuntimeError, match="injected"):
        publish_generation(tmp_path, works, cfg, fail_shard=1)
    assert current_generation(tmp_path) == cur  # still the old, complete generation
    assert not list(tmp_path.glob("*.tmp"))
    m2 = publish_generation(tmp_path, works, cfg)  # restart succeeds
    assert m2["generation"] == 2 and current_generation(tmp_path).name == "gen-0002"


def test_local_and_partitioned_parity(tmp_path, works, wb):
    cfg = load_config(overrides={"distributed": {"workers": 2, "shards": 4}})
    publish_generation(tmp_path, works, cfg)
    sh = ShardedSearcher(current_generation(tmp_path), cfg, {w.work_id: w for w in works})
    for q in ["abrupt shifts in sensor streams", "isolation forest vibration", "negation in clinical notes"]:
        local = [(h.work_id, h.score) for h in wb.searcher.search(q, k=10, expand=False)]
        part = sh.search(q, k=10, expand=False)
        assert [d for d, _ in local] == [d for d, _ in part]
        assert all(abs(a - b) < 1e-9 for (_, a), (_, b) in zip(local, part))
