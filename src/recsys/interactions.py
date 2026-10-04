"""Turn raw events into a user-item interaction table and a time-based split."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_WEIGHTS: dict[str, float] = {"view": 1.0, "click": 2.0, "cart": 4.0, "purchase": 8.0}
REQUIRED_COLUMNS = ("user_id", "item_id", "event_type", "timestamp")


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and clean a raw event log (use this for your own data).

    Required columns: ``user_id``, ``item_id``, ``event_type`` (one of
    view/click/cart/purchase) and ``timestamp``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing required columns: {missing}")

    out = events.loc[:, list(REQUIRED_COLUMNS)].dropna().copy()
    out["event_type"] = out["event_type"].astype(str).str.lower()
    unknown = sorted(set(out["event_type"]) - set(EVENT_WEIGHTS))
    if unknown:
        raise ValueError(f"unknown event types {unknown}; expected {sorted(EVENT_WEIGHTS)}")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.drop_duplicates().sort_values("timestamp", kind="stable").reset_index(drop=True)


def time_based_split(
    events: pd.DataFrame,
    valid_frac: float = 0.2,
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by calendar time: train is strictly before the cutoff, validation after.

    A random split would let the model train on the future and evaluate on the
    past, which overstates quality. If ``cutoff`` is not given it is placed so the
    last ``valid_frac`` of the time span is held out.
    """
    if cutoff is None:
        start, end = events["timestamp"].min(), events["timestamp"].max()
        cutoff = start + (end - start) * (1.0 - valid_frac)
    train = events[events["timestamp"] < cutoff].reset_index(drop=True)
    valid = events[events["timestamp"] >= cutoff].reset_index(drop=True)
    return train, valid, pd.Timestamp(cutoff)


def aggregate_interactions(
    events: pd.DataFrame,
    weights: dict[str, float] = EVENT_WEIGHTS,
    half_life_days: float | None = None,
) -> pd.DataFrame:
    """Collapse events to one row per (user, item) with an implicit-feedback strength.

    ``weight`` is the sum of event weights (optionally time-decayed); ``strength`` is
    ``log1p(weight)`` which stops a user who viewed one item 50 times from
    dominating the model.
    """
    w = events["event_type"].map(weights).astype(float)
    if half_life_days is not None:
        age_days = (events["timestamp"].max() - events["timestamp"]).dt.total_seconds() / 86400.0
        w = w * np.power(0.5, age_days / half_life_days)

    frame = pd.DataFrame(
        {
            "user_id": events["user_id"].to_numpy(),
            "item_id": events["item_id"].to_numpy(),
            "weight": w.to_numpy(),
            "n_events": 1,
            "last_ts": events["timestamp"].to_numpy(),
        }
    )
    agg = (
        frame.groupby(["user_id", "item_id"], as_index=False)
        .agg(weight=("weight", "sum"), n_events=("n_events", "sum"), last_ts=("last_ts", "max"))
    )
    agg["strength"] = np.log1p(agg["weight"])
    return agg


def build_ground_truth(
    valid_events: pd.DataFrame,
    train_events: pd.DataFrame,
    positive_events: tuple[str, ...] = ("cart", "purchase"),
) -> dict[int, set[int]]:
    """Items each user added to cart or bought during the validation window.

    Items the user already purchased in training are removed, because the models
    never recommend those (see ``RecContext.purchased``).
    """
    pos = valid_events[valid_events["event_type"].isin(positive_events)]
    bought = train_events[train_events["event_type"] == "purchase"]
    already = set(zip(bought["user_id"], bought["item_id"]))

    truth: dict[int, set[int]] = {}
    for user, item in zip(pos["user_id"], pos["item_id"]):
        if (user, item) in already:
            continue
        truth.setdefault(int(user), set()).add(int(item))
    return truth
