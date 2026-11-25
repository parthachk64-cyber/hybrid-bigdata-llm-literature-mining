"""Reinforcement-guided adaptive ranking: the paper's tabular Q-learning update on a small, auditable companion MDP.

Paper update:   Q(s,a) <- Q(s,a) + eta * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Companion design (the paper gives no state/action space or logs, so this is OUR choice, documented in docs/architecture.md)
  action : which fusion-weight profile alpha to use for the query  (default {0.2, 0.5, 0.8}); alpha=0.5 is the neutral default
  state  : (query_length_bucket, lexical/dense agreement bucket)   - derived only from the query and the two candidate lists
  reward : supplied by the caller. In this repo it is NDCG@k computed from relevance judgments of TRAINING queries and is
           labelled a SIMULATED feedback sandbox. Real user clicks are never fabricated; logged clicks would need an
           exposure/position-bias correction before use as rewards.
  terminal: a single-query episode has no next state, so the target is r alone (the max term is dropped).
  explore : epsilon-greedy with a seeded RNG during training; greedy and frozen at evaluation.
Per-user isolation: every user has a private PolicyStore entry; nobody inherits another user's table.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIONS = (0.2, 0.5, 0.8)
NEUTRAL_ACTION = 0.5


def q_update(q: float, r: float, next_max: float | None, eta: float, gamma: float) -> float:
    """One temporal-difference step. next_max=None marks a terminal transition."""
    target = r if next_max is None else r + gamma * next_max
    return q + eta * (target - q)


def make_state(query: str, lex_top: list[str], sem_top: list[str], k: int = 5) -> tuple[str, str]:
    n = len(query.split())
    length = "short" if n <= 6 else "long"
    a, b = set(lex_top[:k]), set(sem_top[:k])
    agree = len(a & b) / max(1, min(k, len(a), len(b))) if a and b else 0.0
    return length, "high" if agree >= 0.5 else "low"


@dataclass
class QPolicy:
    eta: float = 0.2
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = 7
    actions: tuple = ACTIONS
    frozen: bool = False
    q: dict = field(default_factory=dict)
    n_updates: int = 0
    version: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def value(self, s, a) -> float:
        return self.q.get((s, a), 0.0)

    def best(self, s) -> float:
        return max(self.value(s, a) for a in self.actions)

    def greedy(self, s) -> float:
        # ties resolve to the neutral action first, then the smaller alpha: deterministic and neutral by default
        return sorted(self.actions, key=lambda a: (-self.value(s, a), abs(a - NEUTRAL_ACTION), a))[0]

    def act(self, s, explore: bool = False) -> float:
        if explore and not self.frozen and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self.greedy(s)

    def update(self, s, a, r: float, s_next=None) -> float:
        if self.frozen:
            raise RuntimeError("policy is frozen: held-out evaluation must not learn")
        nm = None if s_next is None else self.best(s_next)
        self.q[(s, a)] = q_update(self.value(s, a), r, nm, self.eta, self.gamma)
        self.n_updates += 1
        self.version += 1
        return self.q[(s, a)]

    def freeze(self) -> QPolicy:
        self.frozen = True
        return self

    def to_json(self) -> dict:
        return {"eta": self.eta, "gamma": self.gamma, "epsilon": self.epsilon, "seed": self.seed, "version": self.version,
                "n_updates": self.n_updates, "actions": list(self.actions),
                "q": {f"{s[0]}|{s[1]}|{a}": v for (s, a), v in sorted(self.q.items())}}

    @classmethod
    def from_json(cls, d: dict) -> QPolicy:
        p = cls(d["eta"], d["gamma"], d["epsilon"], d["seed"], tuple(d["actions"]))
        for k, v in d["q"].items():
            ln, g, a = k.split("|")
            p.q[((ln, g), float(a))] = v
        p.n_updates, p.version = d["n_updates"], d["version"]
        return p


class PolicyStore:
    """Per-user policies. Unknown users get a fresh neutral policy; nothing is shared across users."""

    def __init__(self, **kw):
        self._kw, self._p = kw, {}

    def get(self, user: str) -> QPolicy:
        if user not in self._p:
            self._p[user] = QPolicy(**self._kw)
        return self._p[user]


def train_policy(searcher, train_queries: dict[str, str], qrels: dict, reward_fn, epochs: int = 3, sessions: list | None = None,
                 policy: QPolicy | None = None, k: int = 10) -> QPolicy:
    """Train on TRAINING queries only. reward_fn(ranked_ids, qrels_for_query) -> r. Held-out queries are never seen."""
    pol = policy or QPolicy()
    seq = sessions or [[q] for q in sorted(train_queries)]
    for _ in range(epochs):
        for sess in seq:
            states, acts, rews = [], [], []
            for qid in sess:
                text = train_queries[qid]
                s = state_for(searcher, text)
                a = pol.act(s, explore=True)
                ranked = [h.work_id for h in searcher.search(text, alpha=a, k=k)]
                states.append(s), acts.append(a), rews.append(reward_fn(ranked, qrels.get(qid, {})))
            for i in range(len(states)):
                nxt = states[i + 1] if i + 1 < len(states) else None
                pol.update(states[i], acts[i], rews[i], nxt)
    return pol


def state_for(searcher, query: str):
    lex = [w for w, _ in searcher.lexical_only(query, 5)]
    sem = [w for w, _ in searcher.dense_only(query, 5)]
    return make_state(query, lex, sem)
