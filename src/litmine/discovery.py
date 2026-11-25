"""Trend significance T_k = sum_t w_t * F_{k,t} and a forecast kept strictly apart from description.

Binning     : calendar year of Work.year (the record's own publication date; works without a year are excluded and counted).
Weights     : w_t = decay ** (t_end - t), t_end = last bin in the window. Higher weight = more recent. Not normalised
              unless normalize=True (then divided by sum_t w_t, so T_k is a recency-weighted mean frequency).
Description : trend_scores(...) describes the window it is given. It makes no claim about the future.
Forecast    : forecast_topics(...) removes everything after `cutoff_year`, predicts the top-n topics for `target_year`,
              and scores them against a held-out period with a target declared in the call.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .data import Work


def topic_year_counts(works: list[Work], topics_by_work: dict[str, set[str]]) -> tuple[dict, int]:
    """F[k][t] = number of works about topic k in year t. Returns (F, n_excluded_without_year)."""
    f: dict[str, Counter] = defaultdict(Counter)
    skipped = 0
    for w in works:
        if w.year is None:
            skipped += 1
            continue
        for k in topics_by_work.get(w.work_id, ()):
            f[k][w.year] += 1
    return f, skipped


def weights(years: list[int], decay: float = 0.7) -> dict[int, float]:
    end = max(years)
    return {t: decay ** (end - t) for t in years}


def trend_scores(f: dict, start: int, end: int, decay: float = 0.7, normalize: bool = False) -> dict[str, float]:
    years = list(range(start, end + 1))
    w = weights(years, decay)
    z = sum(w.values()) if normalize else 1.0
    return {k: sum(w[t] * c.get(t, 0) for t in years) / z for k, c in f.items()}


def ranked(scores: dict[str, float], n: int) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def forecast_topics(works: list[Work], topics_by_work: dict[str, set[str]], cutoff_year: int, target_year: int,
                    top_n: int = 3, decay: float = 0.7, window: int = 5) -> dict:
    """Predict the top-n topics by publication count in target_year using ONLY works with year <= cutoff_year.

    Target (declared up front): the set of n topics with the most works in target_year; ties broken by name.
    Score: precision@n = |predicted ∩ actual| / n. Future publications, citation counts and graph edges are excluded
    from the earlier snapshot by construction (only the filtered list is ever used for prediction).
    """
    past = [w for w in works if w.year is not None and w.year <= cutoff_year]
    held = [w for w in works if w.year == target_year]
    f_past, _ = topic_year_counts(past, topics_by_work)
    pred = ranked(trend_scores(f_past, cutoff_year - window + 1, cutoff_year, decay), top_n)
    f_held, _ = topic_year_counts(held, topics_by_work)
    actual_counts = {k: c.get(target_year, 0) for k, c in f_held.items()}
    actual = ranked(actual_counts, top_n)
    return {"cutoff_year": cutoff_year, "target_year": target_year, "n_past_works": len(past),
            "n_heldout_works": len(held), "predicted": pred, "actual": actual,
            "precision_at_n": len(set(pred) & set(actual)) / top_n if top_n else None,
            "baseline_last_year": ranked({k: c.get(cutoff_year, 0) for k, c in f_past.items()}, top_n)}
