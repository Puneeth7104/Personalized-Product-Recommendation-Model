"""Product metadata features: price tiers used by the content-based model."""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_LABELS = ("budget", "mid", "premium")
_GLOBAL_KEY = "__all__"


def fit_price_cutpoints(products: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-category 33rd / 66th price percentiles (plus a global fallback)."""
    cuts: dict[str, tuple[float, float]] = {}
    for category, group in products.groupby("category"):
        lo, hi = group["price"].quantile([1 / 3, 2 / 3]).to_numpy()
        cuts[str(category)] = (float(lo), float(hi))
    lo, hi = products["price"].quantile([1 / 3, 2 / 3]).to_numpy()
    cuts[_GLOBAL_KEY] = (float(lo), float(hi))
    return cuts


def price_tier(price: float, category: str, cutpoints: dict[str, tuple[float, float]]) -> str:
    """Map a price to budget / mid / premium relative to its category.

    Works for brand-new products too, because it only needs the cut points that
    were learned from the existing catalog.
    """
    lo, hi = cutpoints.get(category, cutpoints[_GLOBAL_KEY])
    return TIER_LABELS[int(np.digitize(price, [lo, hi], right=True))]


def add_price_tier(
    products: pd.DataFrame, cutpoints: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    out = products.copy()
    out["price_tier"] = [
        price_tier(p, c, cutpoints) for p, c in zip(out["price"], out["category"])
    ]
    return out
