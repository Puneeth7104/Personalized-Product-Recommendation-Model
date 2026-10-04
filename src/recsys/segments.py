"""User segmentation and per-segment analysis of recommendation quality and behaviour."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .models import BaseRecommender, RecContext

SEGMENT_ORDER = ["new (0 items)", "light (1-4)", "medium (5-19)", "heavy (20+)"]


def assign_activity_segment(n_items: int) -> str:
    if n_items <= 0:
        return SEGMENT_ORDER[0]
    if n_items < 5:
        return SEGMENT_ORDER[1]
    if n_items < 20:
        return SEGMENT_ORDER[2]
    return SEGMENT_ORDER[3]


def build_user_segments(ctx: RecContext, all_user_ids: Iterable[int]) -> pd.DataFrame:
    """Segment every user using training data only.

    * ``activity_segment``: how many distinct items they interacted with
    * ``dominant_category``: category with the most interaction strength
    * ``price_segment``: budget / mid / premium shopper by strength-weighted mean price
    """
    cats = ctx.products["category"].to_numpy()
    cat_names = sorted(set(cats))
    onehot = np.zeros((ctx.n_items, len(cat_names)))
    for j, c in enumerate(cat_names):
        onehot[cats == c, j] = 1.0
    cat_strength = ctx.matrix @ onehot
    totals = np.asarray(ctx.matrix.sum(axis=1)).ravel()
    prices = ctx.products["price"].to_numpy(dtype=float)
    # compare prices within their category so electronics do not always look "premium"
    rel_price = ctx.products.groupby("category")["price"].transform(lambda s: s.rank(pct=True))
    avg_rel_price = np.where(totals > 0, (ctx.matrix @ rel_price.to_numpy()) / np.where(totals == 0, 1, totals), np.nan)
    avg_price = np.where(totals > 0, (ctx.matrix @ prices) / np.where(totals == 0, 1, totals), np.nan)

    lo, hi = np.nanquantile(avg_rel_price, [1 / 3, 2 / 3]) if np.isfinite(avg_rel_price).any() else (0, 0)

    rows = []
    for user_id in all_user_ids:
        u = ctx.user_to_idx.get(int(user_id))
        if u is None:
            rows.append({"user_id": int(user_id), "n_items_train": 0,
                         "activity_segment": SEGMENT_ORDER[0],
                         "dominant_category": "unknown", "price_segment": "unknown",
                         "avg_price": np.nan})
            continue
        n = int(ctx.user_n_items[u])
        price_seg = "unknown" if n == 0 else (
            "budget" if avg_rel_price[u] <= lo else "premium" if avg_rel_price[u] > hi else "mid"
        )
        rows.append({
            "user_id": int(user_id),
            "n_items_train": n,
            "activity_segment": assign_activity_segment(n),
            "dominant_category": cat_names[int(np.argmax(cat_strength[u]))] if n else "unknown",
            "price_segment": price_seg,
            "avg_price": float(avg_price[u]) if n else np.nan,
        })
    return pd.DataFrame(rows).set_index("user_id")


def segment_metric_table(
    per_user_by_model: dict[str, pd.DataFrame],
    segments: pd.DataFrame,
    metric: str = "ndcg@10",
    segment_col: str = "activity_segment",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Mean metric per segment (rows) and model (columns), plus the number of users."""
    out = {}
    counts = None
    for name, df in per_user_by_model.items():
        joined = df.join(segments[[segment_col]], on="user_id")
        grouped = joined.groupby(segment_col)[metric]
        out[name] = grouped.mean()
        counts = grouped.size()
    table = pd.DataFrame(out)
    table.insert(0, "n_users", counts)
    if order is not None:
        table = table.reindex([s for s in order if s in table.index])
    return table


def recommendation_profile(
    models: dict[str, BaseRecommender],
    ctx: RecContext,
    segments: pd.DataFrame,
    k: int = 10,
    cold_item_threshold: int = 5,
    max_users_per_segment: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Describe *what* each model recommends to each segment.

    Columns: mean price, share of the list taken by the user's top category, number of
    distinct categories, novelty (mean -log2 item popularity share), share of cold items,
    and catalog coverage (distinct items recommended to the segment / catalog size).
    """
    rng = np.random.default_rng(seed)
    cats = ctx.products["category"].to_numpy()
    prices = ctx.products["price"].to_numpy(dtype=float)
    pop_share = (ctx.item_user_counts + 1.0) / (ctx.item_user_counts + 1.0).sum()
    novelty = -np.log2(pop_share)
    cold = ctx.item_user_counts < cold_item_threshold

    rows = []
    for segment in SEGMENT_ORDER:
        members = segments.index[segments["activity_segment"] == segment].to_numpy()
        if len(members) == 0:
            continue
        if len(members) > max_users_per_segment:
            members = rng.choice(members, size=max_users_per_segment, replace=False)
        for name, model in models.items():
            top_share, n_cats, mean_price, nov, cold_share = [], [], [], [], []
            seen_items: set[int] = set()
            for user_id in members:
                idx = model.recommend_idx(int(user_id), k)
                seen_items.update(int(i) for i in idx)
                counts = pd.Series(cats[idx]).value_counts()
                top_share.append(counts.iloc[0] / len(idx))
                n_cats.append(len(counts))
                mean_price.append(prices[idx].mean())
                nov.append(novelty[idx].mean())
                cold_share.append(cold[idx].mean())
            rows.append({
                "segment": segment,
                "model": name,
                "users": len(members),
                "mean_price": float(np.mean(mean_price)),
                "top_category_share": float(np.mean(top_share)),
                "distinct_categories": float(np.mean(n_cats)),
                "novelty": float(np.mean(nov)),
                "cold_item_share": float(np.mean(cold_share)),
                "catalog_coverage": len(seen_items) / ctx.n_items,
            })
    return pd.DataFrame(rows)
