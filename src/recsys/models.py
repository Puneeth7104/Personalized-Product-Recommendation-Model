"""Recommendation models.

All models share one interface (``fit(ctx)``, ``score(user_id)``, ``recommend``):

* ``PopularityRecommender``            - non-personalised baseline (all-time or trending)
* ``MatrixFactorizationRecommender``   - collaborative filtering (truncated SVD / "PureSVD")
* ``ContentBasedRecommender``          - TF-IDF over product metadata; handles new products
* ``HybridRecommender``                - blends the three, adapting to how much history a user has
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .features import add_price_tier, fit_price_cutpoints, price_tier
from .interactions import EVENT_WEIGHTS, aggregate_interactions

# --------------------------------------------------------------------------- context


@dataclass
class RecContext:
    """Everything the models need, derived from training data only."""

    products: pd.DataFrame            # row order defines the item index
    item_ids: np.ndarray
    item_to_idx: dict[int, int]
    user_ids: np.ndarray
    user_to_idx: dict[int, int]
    train_events: pd.DataFrame        # with user_idx, item_idx, event_weight columns
    interactions: pd.DataFrame        # one row per (user, item)
    matrix: sp.csr_matrix             # users x items, log1p(weight)
    item_user_counts: np.ndarray      # distinct training users per item
    user_n_items: np.ndarray          # distinct training items per user
    purchased: dict[int, np.ndarray]  # user_idx -> item idx already bought (never re-recommended)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_context(train_events: pd.DataFrame, products: pd.DataFrame) -> RecContext:
    """Index users/items and build the sparse interaction matrix from training events."""
    products = products.reset_index(drop=True)
    item_ids = products["item_id"].to_numpy()
    item_to_idx = {int(i): n for n, i in enumerate(item_ids)}

    events = train_events[train_events["item_id"].isin(item_to_idx)].copy()
    user_ids = np.sort(events["user_id"].unique())
    user_to_idx = {int(u): n for n, u in enumerate(user_ids)}

    events["user_idx"] = events["user_id"].map(user_to_idx).astype(int)
    events["item_idx"] = events["item_id"].map(item_to_idx).astype(int)
    events["event_weight"] = events["event_type"].map(EVENT_WEIGHTS).astype(float)

    inter = aggregate_interactions(events)
    rows = inter["user_id"].map(user_to_idx).to_numpy()
    cols = inter["item_id"].map(item_to_idx).to_numpy()
    matrix = sp.csr_matrix(
        (inter["strength"].to_numpy(), (rows, cols)), shape=(len(user_ids), len(item_ids))
    )

    binary = matrix.copy()
    binary.data[:] = 1.0
    item_user_counts = np.asarray(binary.sum(axis=0)).ravel().astype(int)
    user_n_items = np.asarray(binary.sum(axis=1)).ravel().astype(int)

    bought = events[events["event_type"] == "purchase"]
    purchased = {
        int(u): g["item_idx"].unique()
        for u, g in bought.groupby("user_idx")
    }

    return RecContext(
        products=products,
        item_ids=item_ids,
        item_to_idx=item_to_idx,
        user_ids=user_ids,
        user_to_idx=user_to_idx,
        train_events=events,
        interactions=inter,
        matrix=matrix,
        item_user_counts=item_user_counts,
        user_n_items=user_n_items,
        purchased=purchased,
    )


# --------------------------------------------------------------------------- base


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


class BaseRecommender:
    name = "Base"

    def fit(self, ctx: RecContext) -> "BaseRecommender":
        self.ctx = ctx
        self._fit()
        return self

    def _fit(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def score_user_idx(self, user_idx: int | None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def score(self, user_id: int) -> np.ndarray:
        """Score every catalog item for a user (all zeros for unknown users)."""
        return self.score_user_idx(self.ctx.user_to_idx.get(int(user_id)))

    def recommend_idx(self, user_id: int, k: int = 10, exclude_purchased: bool = True) -> np.ndarray:
        """Top-k item indices, best first. Ties break by item index so output is deterministic."""
        ctx = self.ctx
        scores = np.array(self.score(user_id), dtype=float)
        user_idx = ctx.user_to_idx.get(int(user_id))
        if exclude_purchased and user_idx is not None and user_idx in ctx.purchased:
            scores[ctx.purchased[user_idx]] = -np.inf
        order = np.lexsort((np.arange(len(scores)), -scores))
        return order[:k]

    def recommend(self, user_id: int, k: int = 10, exclude_purchased: bool = True) -> pd.DataFrame:
        """Top-k recommendations as a DataFrame with ``item_id`` and ``score``."""
        idx = self.recommend_idx(user_id, k, exclude_purchased)
        scores = np.asarray(self.score(user_id))[idx]
        return pd.DataFrame({"item_id": self.ctx.item_ids[idx], "score": scores})


# --------------------------------------------------------------------------- popularity


class PopularityRecommender(BaseRecommender):
    """Recommend what everyone interacts with most.

    With ``half_life_days`` set, recent events count more ("trending"), which is a
    much stronger baseline for catalogs with new arrivals.
    """

    def __init__(self, half_life_days: float | None = None, name: str | None = None):
        self.half_life_days = half_life_days
        self.name = name or (
            "Popularity (all-time)" if half_life_days is None
            else f"Popularity (trending, {half_life_days:g}d half-life)"
        )

    def _fit(self) -> None:
        ev = self.ctx.train_events
        w = ev["event_weight"].to_numpy(dtype=float)
        if self.half_life_days is not None:
            age = (ev["timestamp"].max() - ev["timestamp"]).dt.total_seconds().to_numpy() / 86400.0
            w = w * np.power(0.5, age / self.half_life_days)
        self.scores = np.bincount(ev["item_idx"].to_numpy(), weights=w, minlength=self.ctx.n_items)

    def score_user_idx(self, user_idx: int | None) -> np.ndarray:
        return self.scores.copy()


# --------------------------------------------------------------------------- matrix factorization


class MatrixFactorizationRecommender(BaseRecommender):
    """Collaborative filtering via truncated SVD of the implicit-feedback matrix.

    Users and items are embedded into ``n_factors`` latent dimensions; a user's score
    for an item is the reconstruction of their row from those factors, so items that
    similar users liked rise to the top. Items with no training interactions have
    all-zero factors, which is exactly the cold-start gap the content model fills.
    """

    def __init__(self, n_factors: int = 32, random_state: int = 42,
                 name: str = "Matrix factorization (SVD)"):
        self.n_factors = n_factors
        self.random_state = random_state
        self.name = name

    def _fit(self) -> None:
        matrix = self.ctx.matrix
        k = max(1, min(self.n_factors, min(matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(matrix)   # users x k
        self.item_factors = self.svd.components_              # k x items
        norms = np.linalg.norm(self.item_factors, axis=0)
        self._item_unit = (self.item_factors / np.where(norms == 0, 1.0, norms)).T  # items x k

    def score_user_idx(self, user_idx: int | None) -> np.ndarray:
        if user_idx is None:
            return np.zeros(self.ctx.n_items)
        return self.user_factors[user_idx] @ self.item_factors

    def similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        """'Customers who interacted with this also interacted with...' (collaborative)."""
        idx = self.ctx.item_to_idx[int(item_id)]
        sims = self._item_unit @ self._item_unit[idx]
        sims[idx] = -np.inf
        top = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return pd.DataFrame({"item_id": self.ctx.item_ids[top], "similarity": sims[top]})


# --------------------------------------------------------------------------- content-based


class ContentBasedRecommender(BaseRecommender):
    """TF-IDF over product metadata (category, sub-category, brand, tags, price tier).

    A user profile is the strength-weighted average of the items they interacted with.
    Because it only needs product metadata, it can score brand-new products and can
    answer "which existing products are like this new one?".
    """

    name = "Content-based (TF-IDF)"

    def _tokens(self, category: str, subcategory: str, brand: str, tags: str, tier: str) -> str:
        parts = [f"cat_{category}", f"sub_{subcategory}", f"sub_{subcategory}",
                 f"brand_{brand}", f"tier_{tier}"]
        parts += [f"tag_{t}" for t in str(tags).split()]
        return " ".join(p.replace(" ", "_") for p in parts)

    def _fit(self) -> None:
        ctx = self.ctx
        self.cutpoints = fit_price_cutpoints(ctx.products)
        prod = add_price_tier(ctx.products, self.cutpoints)
        docs = [
            self._tokens(r.category, r.subcategory, r.brand, r.tags, r.price_tier)
            for r in prod.itertuples()
        ]
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"\S+", lowercase=False, sublinear_tf=True, norm="l2"
        )
        self.item_vecs = self.vectorizer.fit_transform(docs).tocsr()
        self.user_profiles = normalize(ctx.matrix @ self.item_vecs).tocsr()

    def score_user_idx(self, user_idx: int | None) -> np.ndarray:
        if user_idx is None:
            return np.zeros(self.ctx.n_items)
        profile = self.user_profiles[user_idx]
        return np.asarray((self.item_vecs @ profile.T).todense()).ravel()

    # --- cold-start helpers -------------------------------------------------

    def encode_item(self, category: str, subcategory: str, brand: str, tags: str, price: float):
        """Vectorise a product that is not in the catalog yet."""
        tier = price_tier(price, category, self.cutpoints)
        return self.vectorizer.transform([self._tokens(category, subcategory, brand, tags, tier)])

    def similar_to_vector(self, vec, k: int = 10, exclude_idx: int | None = None) -> pd.DataFrame:
        sims = np.asarray((self.item_vecs @ vec.T).todense()).ravel()
        if exclude_idx is not None:
            sims[exclude_idx] = -np.inf
        top = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return pd.DataFrame({"item_id": self.ctx.item_ids[top], "similarity": sims[top]})

    def similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        idx = self.ctx.item_to_idx[int(item_id)]
        return self.similar_to_vector(self.item_vecs[idx], k, exclude_idx=idx)

    def users_for_vector(self, vec, k: int = 10) -> pd.DataFrame:
        """Existing users whose taste profile best matches a (possibly new) product."""
        sims = np.asarray((self.user_profiles @ vec.T).todense()).ravel()
        top = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return pd.DataFrame({"user_id": self.ctx.user_ids[top], "affinity": sims[top]})

    def explain(self, user_id: int, item_id: int, n: int = 1) -> list[int]:
        """Item ids from the user's history that most resemble the recommended item."""
        ctx = self.ctx
        user_idx = ctx.user_to_idx.get(int(user_id))
        if user_idx is None:
            return []
        row = ctx.matrix[user_idx]
        if row.nnz == 0:
            return []
        target = self.item_vecs[ctx.item_to_idx[int(item_id)]]
        sims = np.asarray((self.item_vecs[row.indices] @ target.T).todense()).ravel()
        evidence = sims * row.data
        best = np.argsort(-evidence)[:n]
        return [int(ctx.item_ids[row.indices[b]]) for b in best if evidence[b] > 0]


# --------------------------------------------------------------------------- hybrid


class HybridRecommender(BaseRecommender):
    """Blend collaborative, content and popularity signals, adapting to user history.

    Weights ``(mf, content, popularity)`` depend on how many distinct items the user
    interacted with in training:

    =========  ===========  ==================================================
    history    weights      reasoning
    =========  ===========  ==================================================
    0 items    0 / 0 / 1    nothing to personalise on: show trending items
    1-4        .20/.50/.30  too little for CF; metadata generalises best
    5-19       .45/.40/.15  enough overlap for CF to start helping
    20+        .55/.35/.10  rich history: lean on collaborative signal
    =========  ===========  ==================================================

    Items with almost no training interactions are "cold": the MF score is
    meaningless for them, so their MF slot is filled with the content score instead.
    The weights are fixed, documented heuristics (not tuned on the validation window).
    """

    name = "Hybrid (MF + content + trending)"

    TIERS = (
        (0, (0.00, 0.00, 1.00)),
        (1, (0.20, 0.50, 0.30)),
        (5, (0.45, 0.40, 0.15)),
        (20, (0.55, 0.35, 0.10)),
    )

    def __init__(self, mf: MatrixFactorizationRecommender, content: ContentBasedRecommender,
                 popularity: PopularityRecommender, cold_item_threshold: int = 5):
        self.mf, self.content, self.popularity = mf, content, popularity
        self.cold_item_threshold = cold_item_threshold

    def _fit(self) -> None:
        self.cold_mask = self.ctx.item_user_counts < self.cold_item_threshold

    @classmethod
    def weights_for(cls, n_items_seen: int) -> tuple[float, float, float]:
        chosen = cls.TIERS[0][1]
        for threshold, weights in cls.TIERS:
            if n_items_seen >= threshold:
                chosen = weights
        return chosen

    def score_user_idx(self, user_idx: int | None) -> np.ndarray:
        n_seen = 0 if user_idx is None else int(self.ctx.user_n_items[user_idx])
        w_mf, w_content, w_pop = self.weights_for(n_seen)

        pop = _minmax(self.popularity.score_user_idx(user_idx))
        if w_mf == 0 and w_content == 0:
            return w_pop * pop

        content = _minmax(self.content.score_user_idx(user_idx))
        mf = _minmax(self.mf.score_user_idx(user_idx))
        mf = np.where(self.cold_mask, content, mf)
        return w_mf * mf + w_content * content + w_pop * pop

    def explain(self, user_id: int, item_id: int, n: int = 1) -> list[int]:
        return self.content.explain(user_id, item_id, n)
