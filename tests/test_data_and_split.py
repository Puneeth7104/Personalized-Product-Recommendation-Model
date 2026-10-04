import math

import pandas as pd

from recsys.interactions import (
    EVENT_WEIGHTS,
    aggregate_interactions,
    build_ground_truth,
    time_based_split,
    validate_events,
)

from _shared import small_result


def test_split_has_no_time_leakage():
    r = small_result()
    assert r.train_events["timestamp"].max() < r.cutoff
    assert r.valid_events["timestamp"].min() >= r.cutoff
    assert len(r.train_events) + len(r.valid_events) == len(r.events)


def test_events_only_use_known_types_and_items():
    r = small_result()
    assert set(r.events["event_type"]) <= set(EVENT_WEIGHTS)
    assert set(r.events["item_id"]) <= set(r.products["item_id"])


def test_items_are_not_interacted_with_before_launch():
    r = small_result()
    launch = r.products.set_index("item_id")["launch_day"]
    day = (r.events["timestamp"] - pd.Timestamp("2026-01-01")).dt.total_seconds() / 86400
    assert (day >= r.events["item_id"].map(launch) - 1e-9).all()


def test_funnel_is_monotone():
    r = small_result()
    counts = r.events["event_type"].value_counts()
    assert counts["view"] > counts["click"] > counts["cart"] > counts["purchase"] > 0


def test_ground_truth_excludes_items_already_purchased_in_train():
    r = small_result()
    bought = r.train_events[r.train_events["event_type"] == "purchase"]
    already = set(zip(bought["user_id"], bought["item_id"]))
    for user, items in r.truth.items():
        assert all((user, item) not in already for item in items)


def test_aggregate_interactions_weights_and_log_strength():
    ev = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [5, 5, 5, 6],
            "event_type": ["view", "click", "purchase", "view"],
            "timestamp": pd.to_datetime(["2026-01-01"] * 4),
        }
    )
    agg = aggregate_interactions(ev).set_index(["user_id", "item_id"])
    assert agg.loc[(1, 5), "weight"] == 1 + 2 + 8
    assert agg.loc[(1, 5), "n_events"] == 3
    assert abs(agg.loc[(1, 5), "strength"] - math.log1p(11.0)) < 1e-12


def test_time_based_split_respects_explicit_cutoff():
    ev = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [1, 2, 3],
            "event_type": ["view", "cart", "purchase"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"]),
        }
    )
    train, valid, cutoff = time_based_split(ev, cutoff=pd.Timestamp("2026-01-05"))
    assert list(train["item_id"]) == [1]
    assert list(valid["item_id"]) == [2, 3]
    truth = build_ground_truth(valid, train)
    assert truth == {1: {2, 3}}


def test_validate_events_rejects_bad_input():
    bad_cols = pd.DataFrame({"user_id": [1], "item_id": [1]})
    for frame in (bad_cols,):
        try:
            validate_events(frame)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for missing columns")

    bad_type = pd.DataFrame(
        {"user_id": [1], "item_id": [1], "event_type": ["wishlist"], "timestamp": ["2026-01-01"]}
    )
    try:
        validate_events(bad_type)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown event type")
