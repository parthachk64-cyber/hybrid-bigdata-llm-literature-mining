"""Adapters: FIXTURE tests against a local stub server (distinct from a verified model run), plus an optional real-model test."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from litmine.encoders import OllamaEncoder
from litmine.synthesis import OllamaGenerator, generative_answer


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"models": [{"name": "stub-embed:latest", "digest": "abc123"}, {"name": "stub-llm", "digest": "def456"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            self._send({"embeddings": [[float(len(t) % 7 + 1), 1.0, 0.0] for t in body["input"]]})
        else:
            self._send({"response": "Both methods work on sensor streams [P1]. The moon is made of cheese [P1]. No citation here."})


@pytest.fixture()
def host():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_encoder_fixture(host):
    e = OllamaEncoder("stub-embed", host=host)
    m = e.encode(["a", "bb", "ccc"], "query")
    assert m.shape == (3, 3) and np.allclose(np.linalg.norm(m, axis=1), 1)
    c = e.card()
    assert c["revision"] == "abc123" and c["is_transformer"] and c["prefixes"]["query"].startswith("search_query")


def test_generator_output_is_verified_and_unsupported_rejected(host):
    from litmine.data import Passage
    from litmine.retrieval import Hit
    p = Passage("w1#abstract#0", "w1", "abstract", 0, 60, "Both methods work on sensor streams in hydrology.")
    hits = [Hit("w1", 1, 1.0, {})]
    out = generative_answer("which methods work on sensor streams", hits, {p.passage_id: p}, {"w1": [p.passage_id]}, OllamaGenerator("stub-llm", host))
    assert [s["text"] for s in out["statements"]] == ["Both methods work on sensor streams ."] or out["statements"][0]["text"].startswith("Both methods")
    reasons = {r["reason"] for r in out["rejected_statements"]}
    assert reasons == {"not_supported_by_cited_passage", "no_valid_citation"}
    assert out["model_digest"] == "def456"


def _ollama_up(model):
    try:
        t = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2).read())["models"]
        return any(m["name"].startswith(model) for m in t)
    except Exception:
        return False


@pytest.mark.model
@pytest.mark.skipif(not _ollama_up("nomic-embed-text"), reason="needs local ollama + nomic-embed-text")
def test_real_transformer_encoder_semantic_ordering():
    e = OllamaEncoder("nomic-embed-text")
    v = e.encode(["Bayesian changepoint detection in river flow", "detecting abrupt shifts in streamflow", "a recipe for sourdough bread"])
    assert v[0] @ v[1] > v[0] @ v[2]
    assert e.card()["dim"] == v.shape[1] == 768
