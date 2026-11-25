"""Figures and the results ledger. Every figure/table carries a provenance label; nothing is drawn from rounded paper values
unless it is explicitly labelled PAPER-REPORTED."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTED = {  # transcribed from the paper's Tables I-VI (rounded to the paper's precision)
    "table1": {"TF-IDF": (0.71, 0.66, 0.68), "BM25": (0.78, 0.72, 0.75), "BERT retrieval": (0.86, 0.81, 0.83),
               "Knowledge graph search": (0.88, 0.84, 0.86), "Proposed hybrid": (0.94, 0.91, 0.92)},
    "table2_ndcg10": {"BM25": 0.74, "BERT": 0.82, "RAG-based": 0.87, "Proposed": 0.93},
    "table3_latency_ms": {"TF-IDF": 145, "BM25": 172, "BERT retrieval": 310, "RAG pipeline": 355, "Proposed": 198},
    "table4_qps": {"100K": 420, "300K": 395, "600K": 372, "1M": 351},
    "table5_trend_accuracy": {"baseline": 0.81, "proposed": 0.92},
    "table6_p10": {"Without knowledge graph": 0.88, "Without hybrid retrieval": 0.86, "Without reinforcement ranking": 0.90, "Full": 0.94},
}


def _bar(path: Path, title: str, labels, vals, ylabel: str, prov: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, [0 if v is None else v for v in vals], color="#2b6cb0")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.text(0.01, 0.01, prov, fontsize=7, color="#9b2c2c")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_run(run_dir: str | Path) -> list[str]:
    d = Path(run_dir)
    out = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    prov = f"MEASURED run {meta['run_id']} - {meta['label']}"
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    m = out["metrics"]
    names = list(m)
    paths = [_bar(fig_dir / "p_at_10.png", "Precision@10 (measured)", names, [m[n]["p"] for n in names], "P@10", prov),
             _bar(fig_dir / "ndcg_at_10.png", "NDCG@10 (measured)", names, [m[n]["ndcg"] for n in names], "NDCG@10", prov)]
    lat = out["latency"]
    paths.append(_bar(fig_dir / "latency.png", "Mean latency, retrieval only (measured)", list(lat), [v["mean_ms"] for v in lat.values()], "ms", prov))
    if out.get("ablation"):
        ab = out["ablation"]
        paths.append(_bar(fig_dir / "ablation.png", "Ablation P@10 (measured)", list(ab), [v["p"] for v in ab.values()], "P@10", prov))
    return paths


def write_ledger(runs_root: str | Path, out_path: str | Path) -> str:
    """docs/results.md: paper-reported values next to measured values, plus the explicit reproduction gaps."""
    lines = ["# Results ledger", "", "> Provenance legend: **PAPER-REPORTED** = transcribed from the IEEE paper; **MEASURED** = produced by a run in `runs/<id>/`; "
             "**NOT RUN** = no data or resources. Synthetic demo runs are excluded from this ledger.", ""]
    for rd in sorted(Path(runs_root).glob("*/metrics.json")):
        meta = json.loads((rd.parent / "run.json").read_text(encoding="utf-8"))
        if meta.get("dataset", {}).get("synthetic"):
            continue
        lines += [f"## Measured run `{meta['run_id']}` - {meta['label']}", "", f"Encoder `{meta['encoder']['name']}` rev `{str(meta['encoder']['revision'])[:12]}` "
                  f"(transformer: {meta['encoder']['is_transformer']}); {meta['n_docs']} docs; {meta['n_test_queries']} test queries; commit `{meta['env']['git_commit'][:10]}`.", ""]
        lines.append((rd.parent / "tables" / "tables.md").read_text(encoding="utf-8"))
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
