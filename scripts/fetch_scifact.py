"""Fetch SciFact in BEIR format from the Hugging Face mirror (pinned revision). Writes data/scifact/ and prints SHA-256s."""
import hashlib, urllib.request
from pathlib import Path

REV = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
BASE = "https://huggingface.co/datasets"
FILES = {
    "corpus.parquet": f"{BASE}/BeIR/scifact/resolve/{REV}/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{BASE}/BeIR/scifact/resolve/{REV}/queries/queries-00000-of-00001.parquet",
    "test.tsv": f"{BASE}/BeIR/scifact-qrels/resolve/main/test.tsv",
    "train.tsv": f"{BASE}/BeIR/scifact-qrels/resolve/main/train.tsv",
}
out = Path("data/scifact"); out.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    data = urllib.request.urlopen(url, timeout=120).read()
    (out / name).write_bytes(data)
    print(name, len(data), hashlib.sha256(data).hexdigest())
