"""Versioned configuration loading. Every run records the config it used."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "config_version": 1,
    "ingest": {"segment_max_chars": 420, "segment_sentences": 3},
    "bm25": {"k1": 1.5, "b": 0.75, "remove_stopwords": True},
    "encoder": {"name": "hashing", "dim": 256, "max_len": 256},
    "retrieval": {"alpha": 0.5, "candidate_depth": 50, "normalization": "minmax", "top_k": 10},
    "expansion": {"enabled": False, "max_terms": 4},
    "feedback": {"eta": 0.2, "gamma": 0.9, "epsilon": 0.1, "seed": 7},
    "trend": {"bin": "year", "decay": 0.7},
    "distributed": {"backend": "multiprocessing-parquet", "workers": 2, "shards": 4},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = deep_merge(cfg, loaded)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
