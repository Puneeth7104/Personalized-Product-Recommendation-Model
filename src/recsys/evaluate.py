"""Offline evaluation: Precision@K, Recall@K, NDCG@K (and hit rate) per user and overall."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .models import BaseRecommender

METRIC_FUNCS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}


def per_user_metrics(
    model: BaseRecommender,
    truth: dict[int, set[int]],
    k_list: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    """One row per evaluated user with every metric at every K.

    Each user is ranked once at the largest K and the metrics are read off that list.
    """
    k_list = sorted(k_list)
    max_k = k_list[-1]
    ids = model.ctx.item_ids
    rows = []
    for user_id, relevant in truth.items():
        if not relevant:
            continue
        recs = [int(i) for i in ids[model.recommend_idx(user_id, max_k)]]
        row: dict[str, float] = {"user_id": user_id}
        for k in k_list:
            for name, fn in METRIC_FUNCS.items():
                row[f"{name}@{k}"] = fn(recs, relevant, k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_user: pd.DataFrame) -> pd.Series:
    """Mean of every metric column over users."""
    cols = [c for c in per_user.columns if c != "user_id"]
    out = per_user[cols].mean()
    out["n_users"] = len(per_user)
    return out


def evaluate_models(
    models: dict[str, BaseRecommender],
    truth: dict[int, set[int]],
    k_list: Iterable[int] = (5, 10, 20),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate several models on the same users; returns (summary table, per-user tables)."""
    per_user = {name: per_user_metrics(m, truth, k_list) for name, m in models.items()}
    summary = pd.DataFrame({name: summarize(df) for name, df in per_user.items()}).T
    summary["n_users"] = summary["n_users"].astype(int)
    return summary, per_user


def restrict_truth_to_items(
    truth: dict[int, set[int]], item_ids: set[int]
) -> dict[int, set[int]]:
    """Keep only relevant items in ``item_ids`` (e.g. cold items); drop users left with none."""
    out = {u: items & item_ids for u, items in truth.items()}
    return {u: items for u, items in out.items() if items}
