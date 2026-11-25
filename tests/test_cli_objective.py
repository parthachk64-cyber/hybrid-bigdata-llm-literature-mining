import json

import pytest

from litmine.cli import main
from litmine.demo_data import DEMO_DIR


def test_full_cli_flow(tmp_path, capsys):
    snap, idx, runs = tmp_path / "snap", tmp_path / "idx", tmp_path / "runs"
    main(["ingest", "--input", str(DEMO_DIR / "corpus.jsonl"), "--out", str(snap)])
    main(["index", "--snapshot", str(snap), "--index", str(idx), "--workers", "1", "--shards", "3"])
    capsys.readouterr()
    main(["search", "Bayesian online changepoint detection for river gauges", "--snapshot", str(snap), "--k", "3"])
    out = capsys.readouterr().out
    assert "semantic=" in out and "evidence" in out
    main(["path", "S30", "S05", "--snapshot", str(snap)])
    chain = json.loads(capsys.readouterr().out)
    assert chain["verified"] and len(chain["hops"]) == 2
    main(["synthesize", "citation intent classification", "--snapshot", str(snap)])
    assert json.loads(capsys.readouterr().out)["statements"]
    main(["eval", "--dataset", "demo", "--run-id", "t1", "--runs", str(runs)])
    capsys.readouterr()
    assert (runs / "t1" / "tables" / "tables.md").exists() and (runs / "t1" / "rankings.bm25.jsonl").exists()
    run = json.loads((runs / "t1" / "run.json").read_text())
    assert run["env"]["python"] and run["config_hash"] and run["encoder"]["is_transformer"] is False
    main(["plot", "--run", str(runs / "t1")])
    assert (runs / "t1" / "figures" / "p_at_10.png").exists()


def test_synthetic_runs_excluded_from_ledger(tmp_path):
    from litmine.plots import write_ledger
    runs = tmp_path / "runs"
    main(["eval", "--dataset", "demo", "--run-id", "syn", "--runs", str(runs)])
    text = open(write_ledger(runs, tmp_path / "r.md"), encoding="utf-8").read()
    assert "syn" not in text.replace("synthetic", "")


def test_gradient_flows_through_all_loss_branches():
    torch = pytest.importorskip("torch")
    from litmine.objective import Adapter, Objective, l_embed, l_graph, l_rank, train_step
    g = torch.Generator().manual_seed(0)
    mk = lambda n: torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    batch = {"q": mk(6), "d_pos": mk(6), "d_neg": mk(6), "g_src": mk(6), "g_dst": mk(6), "g_neg": mk(6)}
    ad = Adapter(16)
    for fn in (lambda a, b: l_embed(a(b["q"]), a(b["d_pos"])), lambda a, b: l_rank(a(b["q"]), a(b["d_pos"]), a(b["d_neg"])),
               lambda a, b: l_graph(a(b["g_src"]), a(b["g_dst"]), a(b["g_neg"]))):
        ad.W.grad = None
        fn(ad, batch).backward()
        assert float(ad.W.grad.norm()) > 0
    before = Objective()(ad, batch)["total"].item()
    for _ in range(25):
        res = train_step(Objective(), ad, batch)
    assert res["total"] < before and res["grad_norm"] > 0


def test_objective_weights_combine_components():
    torch = pytest.importorskip("torch")
    from litmine.objective import Adapter, Objective
    x = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    out = Objective(2.0, 3.0, 0.0)(Adapter(8), {"q": x, "d_pos": x, "d_neg": x.flip(0)})
    assert out["total"].item() == pytest.approx(2 * out["embed"].item() + 3 * out["rank"].item(), rel=1e-5)
