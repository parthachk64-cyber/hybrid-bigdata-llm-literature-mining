"""The paper's end-to-end objective  L = l1*L_embed + l2*L_rank + l3*L_graph  as a callable.

The paper does not define the three terms, their weights, or an optimiser. These are COMPANION definitions, chosen to be
standard and replaceable:

  L_embed : InfoNCE contrastive relevance loss on (query, relevant doc, in-batch negatives)
  L_rank  : pairwise margin ranking loss on (query, relevant, non-relevant) triples
  L_graph : link prediction - binary cross-entropy of sigmoid(<z_u, z_v>) on cites edges vs sampled non-edges

Trainable part: a linear adapter W (dim x dim, initialised to identity) applied on top of FROZEN encoder vectors, so the
objective can be trained on CPU in seconds and gradient flow into W is demonstrable (tests/test_objective.py).
An inference-only profile simply never calls train_step; it is labelled "inference-only" in run metadata.
"""
from __future__ import annotations

from dataclasses import dataclass


def _t():
    import torch
    return torch


class Adapter:
    def __init__(self, dim: int, seed: int = 7):
        torch = _t()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim, generator=g))

    def __call__(self, x):
        z = x @ self.W
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def parameters(self):
        return [self.W]


def l_embed(q, d_pos, temperature: float = 0.05):
    torch = _t()
    logits = q @ d_pos.T / temperature
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(q)))


def l_rank(q, d_pos, d_neg, margin: float = 0.2):
    torch = _t()
    s_pos, s_neg = (q * d_pos).sum(-1), (q * d_neg).sum(-1)
    return torch.clamp(margin - s_pos + s_neg, min=0).mean()


def l_graph(z_src, z_dst, z_neg):
    torch = _t()
    pos = torch.nn.functional.logsigmoid((z_src * z_dst).sum(-1))
    neg = torch.nn.functional.logsigmoid(-(z_src * z_neg).sum(-1))
    return -(pos.mean() + neg.mean())


@dataclass
class Objective:
    lam_embed: float = 1.0
    lam_rank: float = 1.0
    lam_graph: float = 0.5

    def __call__(self, adapter: Adapter, batch: dict) -> dict:
        q, dp, dn = adapter(batch["q"]), adapter(batch["d_pos"]), adapter(batch["d_neg"])
        parts = {"embed": l_embed(q, dp), "rank": l_rank(q, dp, dn)}
        if "g_src" in batch:
            parts["graph"] = l_graph(adapter(batch["g_src"]), adapter(batch["g_dst"]), adapter(batch["g_neg"]))
        else:
            parts["graph"] = parts["embed"] * 0.0
        total = self.lam_embed * parts["embed"] + self.lam_rank * parts["rank"] + self.lam_graph * parts["graph"]
        return {"total": total, **parts}


def train_step(obj: Objective, adapter: Adapter, batch: dict, lr: float = 0.05) -> dict:
    torch = _t()
    opt = torch.optim.SGD(adapter.parameters(), lr=lr)
    opt.zero_grad()
    out = obj(adapter, batch)
    out["total"].backward()
    grad_norm = float(adapter.W.grad.norm())
    opt.step()
    return {k: float(v.detach()) for k, v in out.items()} | {"grad_norm": grad_norm}
