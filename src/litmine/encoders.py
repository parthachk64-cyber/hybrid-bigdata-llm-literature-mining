"""Encoders. All share one interface: encode(texts, kind) -> float32 [n, dim], plus a card() for the run record.

Three implementations, labelled honestly:
  * HashingEncoder  - deterministic signed feature hashing of word uni/bi-grams. NOT a transformer. Offline substitute.
  * OllamaEncoder   - a real local transformer encoder served by Ollama (e.g. nomic-embed-text). Model digest pinned.
  * HFEncoder       - Hugging Face transformers encoder with an explicit pinned revision.

Zero-vector policy: a text with no tokens encodes to the zero vector and has cosine 0.0 with everything
(never NaN, never a random vector).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np

# keeps words, numbers and every symbol/negation token: "not", "p<0.05", "-", "%" survive
ENC_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def enc_tokens(text: str) -> list[str]:
    return ENC_TOKEN_RE.findall(text.lower())


class Encoder(Protocol):
    def encode(self, texts: list[str], kind: str = "document") -> np.ndarray: ...
    def card(self) -> dict: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    out = np.zeros_like(m)
    ok = n[:, 0] > 0
    out[ok] = m[ok] / n[ok]
    return out.astype(np.float32)


def cosine_matrix(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Cosine between rows; zero vectors give 0.0 (explicit policy)."""
    qn, dn = l2_normalize(q.astype(np.float64)), l2_normalize(d.astype(np.float64))
    return qn @ dn.T


class HashingEncoder:
    def __init__(self, dim: int = 256, max_len: int = 256):
        self.dim, self.max_len = dim, max_len

    def _feat(self, tok: str) -> tuple[int, float]:
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def encode(self, texts: list[str], kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = enc_tokens(t)[: self.max_len]
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                j, s = self._feat(f)
                out[i, j] += s
        return l2_normalize(out)

    def card(self) -> dict:
        return {"name": "hashing", "kind": "deterministic-lexical-substitute", "is_transformer": False,
                "revision": "n/a", "tokenizer": "regex words+numbers+symbols (negation and symbols kept)",
                "max_len": self.max_len, "pooling": "bag of hashed uni+bi-grams", "dim": self.dim,
                "normalization": "l2", "passage_to_doc": "max over passages"}


class OllamaEncoder:
    """Real local transformer encoder via an Ollama server. The model digest is read and recorded."""

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434",
                 max_chars: int = 2000, batch: int = 32, doc_prefix: str = "search_document: ",
                 query_prefix: str = "search_query: "):
        self.model, self.host, self.max_chars, self.batch = model, host.rstrip("/"), max_chars, batch
        self.doc_prefix, self.query_prefix = doc_prefix, query_prefix
        self._digest: str | None = None
        self._dim: int | None = None

    def _post(self, path: str, payload: dict, timeout: float = 300) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def digest(self) -> str:
        if self._digest is None:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read())["models"]
            hit = [m for m in tags if m["name"].split(":")[0] == self.model.split(":")[0]]
            self._digest = hit[0]["digest"] if hit else "unknown"
        return self._digest

    def _embed_single(self, text: str) -> list[float]:
        t = text
        while True:
            try:
                return self._post("/api/embed", {"model": self.model, "input": [t]})["embeddings"][0]
            except urllib.error.HTTPError as e:
                if len(t) < 64:
                    raise RuntimeError(f"embedding failed for a {len(t)}-char text: {e.read()[:200]!r}") from e
                t = t[: len(t) // 2]
                self.truncated = getattr(self, "truncated", 0) + 1

    def encode(self, texts: list[str], kind: str = "document") -> np.ndarray:
        pre = self.query_prefix if kind == "query" else self.doc_prefix
        if kind == "query":  # exact-text memo: a query's vector is deterministic, so re-embedding it is pure waste
            cache = self.__dict__.setdefault("_qcache", {})
            miss = [t for t in dict.fromkeys(texts) if t not in cache]
            if miss:
                for t, v in zip(miss, self._encode_raw(miss, pre)):
                    cache[t] = v
            return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)
        return self._encode_raw(texts, pre)

    def _encode_raw(self, texts: list[str], pre: str) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = [pre + t[: self.max_chars] for t in texts[i:i + self.batch]]
            try:
                rows.extend(self._post("/api/embed", {"model": self.model, "input": chunk})["embeddings"])
            except urllib.error.HTTPError:  # one bad item must not hide which one: retry singly, halving the text on failure
                rows.extend(self._embed_single(t) for t in chunk)
        m = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        self._dim = m.shape[1]
        return l2_normalize(m)

    def card(self) -> dict:
        if self._dim is None:
            self._dim = int(self.encode(["dimension probe"], "query").shape[1])
        return {"name": f"ollama:{self.model}", "kind": "transformer-encoder", "is_transformer": True,
                "revision": self.digest(), "tokenizer": "model-native (server side)", "max_len": f"{self.max_chars} chars",
                "pooling": "model-native", "dim": self._dim, "normalization": "l2",
                "prefixes": {"document": self.doc_prefix, "query": self.query_prefix},
                "passage_to_doc": "max over passages"}


class HFEncoder:
    """transformers encoder with mean pooling. A revision is REQUIRED so a run can be reproduced."""

    def __init__(self, model_id: str, revision: str, max_len: int = 256, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.model_id, self.revision, self.max_len = torch, model_id, revision, max_len
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device

    def encode(self, texts: list[str], kind: str = "document") -> np.ndarray:
        torch = self.torch
        outs = []
        for i in range(0, len(texts), 16):
            b = self.tok(texts[i:i + 16], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return l2_normalize(np.concatenate(outs) if outs else np.zeros((0, 1), dtype=np.float32))

    def card(self) -> dict:
        return {"name": f"hf:{self.model_id}", "kind": "transformer-encoder", "is_transformer": True,
                "revision": self.revision, "tokenizer": type(self.tok).__name__, "max_len": self.max_len,
                "pooling": "mean over attention mask", "dim": int(self.model.config.hidden_size),
                "normalization": "l2", "passage_to_doc": "max over passages"}


def make_encoder(cfg: dict) -> Encoder:
    e = cfg["encoder"]
    if e["name"] == "hashing":
        return HashingEncoder(e.get("dim", 256), e.get("max_len", 256))
    if e["name"].startswith("ollama:"):
        return OllamaEncoder(e["name"].split(":", 1)[1], host=e.get("host", "http://127.0.0.1:11434"))
    if e["name"].startswith("hf:"):
        return HFEncoder(e["name"].split(":", 1)[1], revision=e["revision"], max_len=e.get("max_len", 256))
    raise ValueError(f"unknown encoder {e['name']!r}")
