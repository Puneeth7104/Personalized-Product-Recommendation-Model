import numpy as np

from recsys.models import HybridRecommender

from _shared import small_result


def _model(prefix):
    r = small_result()
    return next(m for name, m in r.models.items() if name.startswith(prefix))


def _heavy_user(r):
    seg = r.segments
    return int(seg.index[seg["activity_segment"] == "heavy (20+)"][0])


def _new_user(r):
    seg = r.segments
    return int(seg.index[seg["activity_segment"] == "new (0 items)"][0])


def test_popularity_orders_items_by_weighted_interaction_count():
    r = small_result()
    pop = _model("Popularity (all-time)")
    top = pop.recommend(_new_user(r), k=5)
    assert top["score"].is_monotonic_decreasing
    assert top["score"].iloc[0] == pop.scores.max()


def test_recommendations_are_unique_and_sized_k():
    r = small_result()
    user = _heavy_user(r)
    for model in r.models.values():
        recs = model.recommend(user, k=15)
        assert len(recs) == 15
        assert recs["item_id"].is_unique


def test_already_purchased_items_are_never_recommended():
    r = small_result()
    checked = 0
    for user_idx, bought in r.ctx.purchased.items():
        user_id = int(r.ctx.user_ids[user_idx])
        bought_ids = set(int(i) for i in r.ctx.item_ids[bought])
        for model in r.models.values():
            recs = set(model.recommend(user_id, k=20)["item_id"])
            assert not (recs & bought_ids)
        checked += 1
        if checked >= 25:
            break
    assert checked > 0


def test_recommendations_are_deterministic():
    r = small_result()
    user = _heavy_user(r)
    for model in r.models.values():
        a = model.recommend(user, k=10)["item_id"].tolist()
        b = model.recommend(user, k=10)["item_id"].tolist()
        assert a == b


def test_cold_start_user_gets_trending_items_from_hybrid():
    r = small_result()
    user = _new_user(r)
    hybrid = _model("Hybrid")
    trending = _model("Popularity (trending")
    assert hybrid.recommend(user, k=10)["item_id"].tolist() == trending.recommend(user, k=10)["item_id"].tolist()


def test_unknown_user_does_not_crash():
    for model in small_result().models.values():
        assert len(model.recommend(10**9, k=5)) == 5


def test_hybrid_weights_shift_toward_cf_as_history_grows():
    w0, w1, w2, w3 = (HybridRecommender.weights_for(n) for n in (0, 2, 10, 50))
    assert w0 == (0.0, 0.0, 1.0)
    assert w1[0] < w2[0] < w3[0]  # MF weight grows
    assert w1[2] > w2[2] > w3[2]  # popularity weight shrinks
    for w in (w0, w1, w2, w3):
        assert abs(sum(w) - 1.0) < 1e-9


def test_content_model_finds_same_subcategory_for_new_product():
    r = small_result()
    content = _model("Content-based")
    products = r.products.set_index("item_id")
    row = products.iloc[0]
    vec = content.encode_item(row["category"], row["subcategory"], row["brand"], row["tags"], float(row["price"]))
    sims = content.similar_to_vector(vec, k=10)
    same = (products.loc[sims["item_id"], "subcategory"] == row["subcategory"]).mean()
    assert same >= 0.8


def test_content_model_can_score_items_with_no_training_history():
    r = small_result()
    content = _model("Content-based")
    zero_hist = np.flatnonzero(r.ctx.item_user_counts == 0)
    assert len(zero_hist) > 0
    user = _heavy_user(r)
    scores = content.score(user)
    assert (scores[zero_hist] > 0).any()
    mf = _model("Matrix factorization")
    assert np.allclose(mf.score(user)[zero_hist], 0.0)


def test_explanation_points_to_item_in_user_history():
    r = small_result()
    content = _model("Content-based")
    user = _heavy_user(r)
    rec_item = int(content.recommend(user, k=1)["item_id"].iloc[0])
    why = content.explain(user, rec_item, n=1)
    history = set(r.train_events.loc[r.train_events["user_id"] == user, "item_id"])
    assert why and set(why) <= history
