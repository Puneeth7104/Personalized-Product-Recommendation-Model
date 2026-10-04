from _shared import small_result


def test_all_metrics_are_valid_probabilities():
    metrics = small_result().metrics
    cols = [c for c in metrics.columns if c != "n_users"]
    assert ((metrics[cols] >= 0) & (metrics[cols] <= 1)).all().all()


def test_recall_and_hit_rate_do_not_decrease_with_k():
    m = small_result().metrics
    assert (m["recall@5"] <= m["recall@10"]).all() and (m["recall@10"] <= m["recall@20"]).all()
    assert (m["hit_rate@5"] <= m["hit_rate@10"]).all() and (m["hit_rate@10"] <= m["hit_rate@20"]).all()


def test_personalised_models_beat_popularity_baseline():
    m = small_result().metrics
    baseline = m.loc[[i for i in m.index if i.startswith("Popularity")], "ndcg@10"].max()
    for prefix in ("Matrix factorization", "Content-based", "Hybrid"):
        name = next(i for i in m.index if i.startswith(prefix))
        assert m.loc[name, "ndcg@10"] > baseline, name


def test_cold_items_only_reachable_through_content():
    cold = small_result().cold_metrics
    for name in cold.index:
        if name.startswith(("Popularity", "Matrix factorization")):
            assert cold.loc[name, "recall@20"] == 0.0
    content = next(i for i in cold.index if i.startswith("Content-based"))
    hybrid = next(i for i in cold.index if i.startswith("Hybrid"))
    assert cold.loc[content, "recall@20"] > 0
    assert cold.loc[hybrid, "recall@20"] > 0


def test_segment_tables_cover_every_activity_segment():
    r = small_result()
    table = r.segment_tables["ndcg@10 by activity segment"]
    assert list(table.index) == ["new (0 items)", "light (1-4)", "medium (5-19)", "heavy (20+)"]
    assert table["n_users"].sum() == len(r.truth)


def test_hybrid_beats_pure_cf_for_new_users():
    table = small_result().segment_tables["ndcg@10 by activity segment"]
    hybrid = next(c for c in table.columns if c.startswith("Hybrid"))
    mf = next(c for c in table.columns if c.startswith("Matrix factorization"))
    assert table.loc["new (0 items)", hybrid] > table.loc["new (0 items)", mf]


def test_recommendation_profile_columns():
    profile = small_result().profile
    assert {"segment", "model", "mean_price", "novelty", "catalog_coverage", "cold_item_share"} <= set(profile.columns)
    assert ((profile["catalog_coverage"] >= 0) & (profile["catalog_coverage"] <= 1)).all()
