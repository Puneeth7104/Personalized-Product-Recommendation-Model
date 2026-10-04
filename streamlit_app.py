"""Streamlit dashboard for the personalised product recommendation system.

Run locally:   streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recsys.data import BRAND_POOL, CATEGORY_TREE, TAG_POOL  # noqa: E402
from recsys.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from recsys.segments import SEGMENT_ORDER  # noqa: E402

st.set_page_config(page_title="Personalised Product Recommender", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Simulating shop data and training all models (about 10 seconds)...")
def load_result():
    return run_pipeline(PipelineConfig())


res = load_result()
cfg = res.config
ctx = res.ctx
products = res.products.set_index("item_id")
model_names = list(res.models)
HYBRID = next(n for n in model_names if n.startswith("Hybrid"))
MF = next(n for n in model_names if n.startswith("Matrix"))
CONTENT = next(n for n in model_names if n.startswith("Content"))


def title_of(item_id: int) -> str:
    return str(products.loc[int(item_id), "title"])


def catalog_table(item_ids, **extra_columns) -> pd.DataFrame:
    """Readable product table for a list of item ids, plus any extra columns."""
    item_ids = [int(i) for i in item_ids]
    table = products.loc[item_ids, ["title", "category", "brand", "price"]].reset_index()
    for name, values in extra_columns.items():
        table[name] = list(values)
    return table


# ----------------------------------------------------------------------------- sidebar

st.sidebar.header("Controls")
model_name = st.sidebar.selectbox(
    "Recommendation model", model_names, index=model_names.index(HYBRID)
)
k = st.sidebar.slider("Number of recommendations (K)", 3, 20, 10)

segment_choice = st.sidebar.selectbox("Shopper segment", ["All"] + SEGMENT_ORDER)
segments = res.segments
member_ids = segments.index if segment_choice == "All" else segments.index[
    segments["activity_segment"] == segment_choice
]
member_ids = sorted(int(u) for u in member_ids)
# default to a user who has validation purchases so the "did it work?" check is interesting
default_user = next((u for u in member_ids if u in res.truth), member_ids[0])
user_id = st.sidebar.selectbox(
    "Shopper",
    member_ids,
    index=member_ids.index(default_user),
    format_func=lambda u: f"User {u} - {segments.loc[u, 'activity_segment']}",
)

st.sidebar.caption(
    "All data is simulated (see the About tab). Models are trained once on events before "
    f"{res.cutoff:%d %b %Y} and judged on what shoppers did afterwards."
)

st.title("🛍️ Personalised Product Recommendations")
st.caption(
    f"{len(res.users):,} shoppers · {len(res.products):,} products · {len(res.events):,} "
    "view/click/cart/purchase events (simulated)"
)

tab_recs, tab_similar, tab_eval, tab_segments, tab_about = st.tabs(
    ["Recommendations", "Similar & new products", "Model evaluation", "Segment analysis", "About"]
)

# ----------------------------------------------------------------------------- recommendations

with tab_recs:
    seg_row = segments.loc[user_id]
    n_hist = int(seg_row["n_items_train"])
    user_idx = ctx.user_to_idx.get(user_id)
    n_buys = 0 if user_idx is None else len(ctx.purchased.get(user_idx, []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", seg_row["activity_segment"])
    c2.metric("Items interacted with", n_hist)
    c3.metric("Past purchases", n_buys)
    c4.metric("Favourite category", seg_row["dominant_category"])

    left, right = st.columns([2, 3])

    with left:
        st.subheader("What this shopper did before")
        if n_hist == 0:
            st.info("No history before the cutoff: a cold-start shopper. Personalised models have "
                    "nothing to go on, so the hybrid falls back to trending items.")
        else:
            history = ctx.train_events[ctx.train_events["user_id"] == user_id]
            strongest = (
                history.sort_values("event_weight")
                .groupby("item_id")
                .tail(1)
                .sort_values("event_weight", ascending=False)
                .head(10)
            )
            st.dataframe(
                catalog_table(strongest["item_id"], action=strongest["event_type"]),
                hide_index=True,
            )

    with right:
        st.subheader(f"Top {k} recommendations")
        model = res.models[model_name]
        recs = model.recommend(user_id, k)

        def reason(item_id: int) -> str:
            if n_hist == 0:
                return "Trending with all shoppers"
            if hasattr(model, "explain"):
                evidence = model.explain(user_id, item_id)
                if evidence:
                    return f"Similar to: {title_of(evidence[0])}"
            if model_name.startswith("Popularity"):
                return "Popular with shoppers overall"
            return "Shoppers with similar taste engaged with it"

        is_new = [
            "new arrival" if ctx.item_user_counts[ctx.item_to_idx[int(i)]] < cfg.cold_item_threshold else ""
            for i in recs["item_id"]
        ]
        table = catalog_table(
            recs["item_id"],
            score=recs["score"].round(3),
            note=is_new,
            why=[reason(i) for i in recs["item_id"]],
        )
        table.insert(0, "rank", range(1, len(table) + 1))
        st.dataframe(table, hide_index=True)

    st.subheader("Did it work? (validation window)")
    relevant = res.truth.get(user_id, set())
    if not relevant:
        st.write("This shopper did not add anything to cart or buy anything after the cutoff, "
                 "so there is nothing to check recommendations against.")
    else:
        top_k = set(int(i) for i in recs["item_id"])
        hits = relevant & top_k
        st.write(
            f"After the cutoff this shopper added to cart or bought **{len(relevant)}** item(s). "
            f"**{len(hits)}** of them appear in the top {k} above."
        )
        st.dataframe(
            catalog_table(
                sorted(relevant),
                in_top_k=["yes" if i in hits else "no" for i in sorted(relevant)],
            ),
            hide_index=True,
        )

# ----------------------------------------------------------------------------- similar & new products

with tab_similar:
    content_model = res.models[CONTENT]
    mf_model = res.models[MF]

    st.subheader("Products similar to one you pick")
    options = list(products.index)
    pick = st.selectbox(
        "Product",
        options,
        format_func=lambda i: f"{title_of(i)}  ({products.loc[i, 'category']}, ${products.loc[i, 'price']:.2f})",
    )
    a, b = st.columns(2)
    with a:
        st.markdown("**By product metadata** (works for brand-new products too)")
        sim = content_model.similar_items(pick, 8)
        st.dataframe(catalog_table(sim["item_id"], similarity=sim["similarity"].round(3)), hide_index=True)
    with b:
        st.markdown("**By shopper behaviour** (customers who engaged with this also engaged with...)")
        if ctx.item_user_counts[ctx.item_to_idx[int(pick)]] == 0:
            st.info("This product has no interaction history yet, so behaviour-based similarity is "
                    "empty. That is the cold-start problem the metadata model solves.")
        else:
            sim = mf_model.similar_items(pick, 8)
            st.dataframe(catalog_table(sim["item_id"], similarity=sim["similarity"].round(3)), hide_index=True)

    st.divider()
    st.subheader("Cold-start demo: a product launched today")
    st.caption("Describe a product that has never been seen. Only its metadata is needed.")
    n1, n2, n3 = st.columns(3)
    category = n1.selectbox("Category", list(CATEGORY_TREE))
    subcategory = n2.selectbox("Sub-category", CATEGORY_TREE[category])
    brand = n3.selectbox("Brand", sorted(BRAND_POOL))
    n4, n5 = st.columns([3, 1])
    tags = n4.multiselect("Tags (up to 3)", TAG_POOL, default=TAG_POOL[:2], max_selections=3)
    price = n5.number_input("Price ($)", min_value=1.0, value=50.0, step=5.0)

    vec = content_model.encode_item(category, subcategory, brand, " ".join(tags), float(price))
    x, y = st.columns(2)
    with x:
        st.markdown("**Most similar existing products**")
        sim = content_model.similar_to_vector(vec, 8)
        st.dataframe(catalog_table(sim["item_id"], similarity=sim["similarity"].round(3)), hide_index=True)
    with y:
        st.markdown("**Shoppers most likely to be interested**")
        aud = content_model.users_for_vector(vec, 8)
        aud["segment"] = [segments.loc[int(u), "activity_segment"] for u in aud["user_id"]]
        aud["affinity"] = aud["affinity"].round(3)
        st.dataframe(aud, hide_index=True)

# ----------------------------------------------------------------------------- evaluation

with tab_eval:
    st.subheader("How good are the recommendations?")
    n_eval = len(res.truth)
    st.markdown(
        f"**Protocol.** Models learn only from events before **{res.cutoff:%d %b %Y}** "
        f"({len(res.train_events):,} events). They are then scored on the {cfg.valid_days} days "
        f"after the cutoff ({len(res.valid_events):,} events). A recommendation counts as correct "
        f"if the shopper **added it to cart or bought it** in that window. {n_eval:,} shoppers "
        "have at least one such item. Items a shopper already bought are never recommended."
    )
    k_eval = st.radio("K", list(cfg.k_list), index=1, horizontal=True)
    cols = [f"precision@{k_eval}", f"recall@{k_eval}", f"ndcg@{k_eval}", f"hit_rate@{k_eval}"]
    overall = res.metrics[cols].round(4)
    st.dataframe(overall)
    st.bar_chart(res.metrics[[f"ndcg@{k_eval}"]].rename(columns={f"ndcg@{k_eval}": f"NDCG@{k_eval}"}))

    st.markdown(
        "- **Precision@K**: of the K items shown, how many were correct.\n"
        "- **Recall@K**: of everything the shopper went on to want, how much did the top K catch.\n"
        "- **NDCG@K**: like precision, but a correct item at rank 1 counts more than at rank 10.\n"
        "- **Hit rate@K**: share of shoppers with at least one correct item in the top K."
    )

    st.subheader("Cold items only")
    st.caption(
        f"{len(res.cold_item_ids)} products had fewer than {cfg.cold_item_threshold} shoppers "
        "before the cutoff. Only correct items from this group count here."
    )
    st.dataframe(res.cold_metrics[cols].round(4))

# ----------------------------------------------------------------------------- segments

with tab_segments:
    st.subheader("Quality by shopper segment")
    st.caption("Segments use training data only. 'New' shoppers have no history before the cutoff.")
    names = [n for n in res.segment_tables if n.endswith("activity segment")]
    table_name = st.radio("Metric", names, horizontal=True)
    seg_table = res.segment_tables[table_name]
    st.dataframe(seg_table.round(4))
    st.bar_chart(seg_table.drop(columns="n_users"))

    st.subheader("Quality by price segment")
    price_table = res.segment_tables[next(n for n in res.segment_tables if "price" in n)]
    st.dataframe(price_table.round(4))

    st.subheader("What does each model recommend to each segment?")
    seg_pick = st.selectbox("Activity segment", SEGMENT_ORDER, index=len(SEGMENT_ORDER) - 1)
    profile = res.profile[res.profile["segment"] == seg_pick].drop(columns="segment").set_index("model")
    st.dataframe(profile.round(3))
    st.caption(
        "mean_price: average price of the list · top_category_share: how much of the list is the "
        "most common category · distinct_categories: variety · novelty: higher means less "
        "mainstream items · cold_item_share: share of new arrivals · catalog_coverage: share of "
        "the catalog recommended to at least one shopper in the segment."
    )

# ----------------------------------------------------------------------------- about

with tab_about:
    st.markdown(
        """
### How it works
1. **Data prep**: view / click / cart / purchase events become one implicit-feedback score per
   shopper-product pair (weights 1 / 2 / 4 / 8, then `log1p`).
2. **Time-based split**: train strictly before a cutoff date, evaluate after it, so the model never
   sees the future.
3. **Popularity baseline**: all-time and trending (14-day half-life).
4. **Collaborative filtering**: truncated SVD (matrix factorisation) of the shopper x product matrix.
5. **Content-based**: TF-IDF over category, sub-category, brand, tags and price tier. Handles new
   products and shoppers with little history.
6. **Hybrid**: blends the three, leaning on content and trending for thin histories and on
   collaborative signal for rich ones.
7. **Evaluation**: Precision@K, Recall@K, NDCG@K, hit rate, overall, for cold items and per segment.

### About the data
The project brief did not include a dataset, so the shop is simulated: hidden shopper personas,
a long-tailed popularity curve, a purchase funnel, new arrivals shortly before the cutoff and
shoppers who sign up after it. Numbers here show the method works on data with this structure,
not how it would perform on a real store.

Source code and write-up are in the project's GitHub repository (see README).
"""
    )
