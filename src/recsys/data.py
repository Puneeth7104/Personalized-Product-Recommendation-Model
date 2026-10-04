"""Synthetic e-commerce data generator.

The project brief does not ship a dataset, so this module simulates one with
realistic structure:

* a product catalog with category / sub-category / brand / tags / price,
* a long-tailed item popularity prior,
* shoppers whose taste comes from hidden "personas" (so collaborative signals
  exist that metadata alone cannot fully explain),
* a view -> click -> cart -> purchase funnel where stronger actions are more
  likely for items the shopper actually likes,
* new arrivals that launch shortly before the train/validation cutoff (cold-start
  items) and shoppers who sign up only after the cutoff (cold-start users).

Everything is seeded, so results are reproducible. To use real data instead, see
"Bring your own data" in the README (``interactions.validate_events``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2026-01-01")

CATEGORY_TREE: dict[str, list[str]] = {
    "Electronics": ["Headphones", "Smartphones", "Laptops", "Smart Home", "Cameras"],
    "Fashion": ["Sneakers", "Jackets", "Jeans", "Watches", "Bags"],
    "Home & Kitchen": ["Cookware", "Coffee", "Furniture", "Lighting", "Storage"],
    "Beauty": ["Skincare", "Haircare", "Fragrance", "Makeup"],
    "Sports": ["Yoga", "Running", "Cycling", "Fitness Gear"],
    "Books": ["Fiction", "Business", "Science", "Self-help"],
    "Toys & Games": ["Board Games", "Puzzles", "Building Sets", "Outdoor Play"],
    "Grocery": ["Snacks", "Tea", "Organic", "Beverages"],
}

CATEGORY_BASE_PRICE: dict[str, float] = {
    "Electronics": 120.0,
    "Fashion": 55.0,
    "Home & Kitchen": 40.0,
    "Beauty": 22.0,
    "Sports": 35.0,
    "Books": 15.0,
    "Toys & Games": 25.0,
    "Grocery": 8.0,
}

BRAND_POOL = [
    "Aurora", "Nimbus", "Vertex", "Lumen", "Orbit", "Zenith", "Pioneer", "Evergreen",
    "Summit", "Harbor", "Echo", "Cobalt", "Meridian", "Atlas", "Willow", "Ember",
    "Quartz", "Fjord", "Sable", "Nova", "Terra", "Kestrel", "Juniper", "Onyx",
]

TAG_POOL = [
    "eco-friendly", "wireless", "compact", "classic", "pro", "smart", "lightweight",
    "durable", "organic", "portable", "limited-edition", "ergonomic", "minimal",
    "waterproof", "handmade", "everyday",
]

N_PERSONAS = 8


def generate_products(n_items: int, cutoff_day: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create the product catalog.

    About 10% of items launch in the last 4 days before the cutoff (few training
    interactions) and 5% launch exactly at the cutoff (zero training
    interactions). The rest launch earlier.
    """
    subcats = [(cat, sub) for cat, subs in CATEGORY_TREE.items() for sub in subs]
    sub_weights = rng.dirichlet(np.ones(len(subcats)) * 3.0)
    sub_idx = rng.choice(len(subcats), size=n_items, p=sub_weights)

    category_brands = {
        cat: rng.choice(BRAND_POOL, size=6, replace=False) for cat in CATEGORY_TREE
    }

    n_recent = int(round(0.10 * n_items))
    n_zero = int(round(0.05 * n_items))
    launch = np.concatenate(
        [
            rng.integers(0, 46, size=n_items - n_recent - n_zero),
            rng.integers(cutoff_day - 4, cutoff_day, size=n_recent),
            np.full(n_zero, cutoff_day),
        ]
    )
    rng.shuffle(launch)

    rows = []
    for item_id in range(n_items):
        category, subcategory = subcats[sub_idx[item_id]]
        brand = str(rng.choice(category_brands[category]))
        tags = rng.choice(TAG_POOL, size=3, replace=False)
        price = round(
            max(1.0, CATEGORY_BASE_PRICE[category] * float(rng.lognormal(0.0, 0.55))), 2
        )
        rows.append(
            {
                "item_id": item_id,
                "title": f"{brand} {str(tags[0]).title()} {subcategory}",
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "tags": " ".join(str(t) for t in tags),
                "price": price,
                "launch_day": int(launch[item_id]),
            }
        )
    return pd.DataFrame(rows)


def generate_events(
    products: pd.DataFrame,
    n_users: int,
    total_days: int,
    cutoff_day: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate users and their view / click / cart / purchase events."""
    n_items = len(products)
    subcats = sorted(products["subcategory"].unique())
    sub_pos = {s: i for i, s in enumerate(subcats)}
    item_sub = products["subcategory"].map(sub_pos).to_numpy()
    item_brand = products["brand"].to_numpy()
    item_launch = products["launch_day"].to_numpy()

    # price tier (0 cheap .. 2 expensive) relative to the category
    item_tier = (
        products.groupby("category")["price"]
        .transform(lambda s: s.rank(method="first", pct=True))
        .to_numpy()
    )
    item_tier = np.digitize(item_tier, [1 / 3, 2 / 3])

    item_pop = rng.lognormal(0.0, 0.9, size=n_items)

    # day-by-day availability and "new arrival" novelty boost
    days = np.arange(total_days)[:, None]
    available = (item_launch[None, :] <= days).astype(float)
    age = np.clip(days - item_launch[None, :], 0, None)
    novelty = 1.0 + 1.5 * np.exp(-age / 7.0)
    availability_novelty = available * novelty

    # hidden personas: each concentrates on ~6 sub-categories across categories
    persona = np.full((N_PERSONAS, len(subcats)), 0.01)
    for p in range(N_PERSONAS):
        chosen = rng.choice(len(subcats), size=6, replace=False)
        persona[p, chosen] += rng.dirichlet(np.ones(6) * 2.0)
    persona /= persona.sum(axis=1, keepdims=True)

    # users: late sign-ups produce cold-start users in the validation window
    group = rng.random(n_users)
    signup_day = np.where(
        group < 0.65,
        rng.uniform(0, 30, n_users),
        np.where(
            group < 0.90,
            rng.uniform(30, cutoff_day, n_users),
            rng.uniform(cutoff_day, total_days - 3, n_users),
        ),
    )
    rate = rng.lognormal(np.log(0.30), 0.9, size=n_users)  # sessions per active day

    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "signup_date": START_DATE + pd.to_timedelta(signup_day, unit="D"),
        }
    )

    u_list: list[int] = []
    i_list: list[int] = []
    t_list: list[str] = []
    ts_list: list[float] = []

    funnel = (("click", 30), ("cart", 90), ("purchase", 180))

    for u in range(n_users):
        active_days = total_days - signup_day[u]
        n_sessions = min(int(rng.poisson(rate[u] * active_days)), 300)
        if n_sessions == 0:
            continue

        main, other = rng.choice(N_PERSONAS, size=2, replace=False)
        mix = 0.75 * persona[main] + 0.25 * persona[other]
        fav_subs = rng.choice(np.flatnonzero(persona[main] > 0.05), size=1)
        sub_boost = np.ones(len(subcats))
        sub_boost[fav_subs] = 2.0
        affinity = (mix * sub_boost)[item_sub]

        price_pref = rng.choice(3, p=[0.30, 0.45, 0.25])
        price_factor = np.array([1.0, 0.55, 0.25])[np.abs(item_tier - price_pref)]
        fav_brand = rng.choice(BRAND_POOL)
        brand_factor = np.where(item_brand == fav_brand, 2.5, 1.0)

        affinity = affinity * price_factor * brand_factor
        aff_norm = affinity / affinity.max()
        base = (aff_norm + 1e-4) ** 1.3 * item_pop

        session_days = signup_day[u] + rng.random(n_sessions) * active_days
        for day_f in session_days:
            weights = base * availability_novelty[min(int(day_f), total_days - 1)]
            cum = np.cumsum(weights)
            n_visits = 1 + int(rng.poisson(1.2))
            picks = np.searchsorted(cum, rng.random(n_visits) * cum[-1])
            for item in np.minimum(picks, n_items - 1):
                a = aff_norm[item]
                u_list.append(u)
                i_list.append(int(item))
                t_list.append("view")
                ts_list.append(day_f)

                probs = (0.20 + 0.50 * a, 0.15 + 0.40 * a, 0.45)
                for (event, delay_s), p in zip(funnel, probs):
                    if rng.random() >= p:
                        break
                    u_list.append(u)
                    i_list.append(int(item))
                    t_list.append(event)
                    ts_list.append(day_f + delay_s / 86400.0)

    events = pd.DataFrame(
        {
            "user_id": u_list,
            "item_id": i_list,
            "event_type": t_list,
            "timestamp": START_DATE + pd.to_timedelta(np.array(ts_list), unit="D"),
        }
    )
    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return users, events


def generate_dataset(
    n_users: int = 2000,
    n_items: int = 600,
    total_days: int = 90,
    valid_days: int = 18,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return ``(products, users, events, cutoff)`` for the simulated store."""
    rng = np.random.default_rng(seed)
    cutoff_day = total_days - valid_days
    products = generate_products(n_items, cutoff_day, rng)
    users, events = generate_events(products, n_users, total_days, cutoff_day, rng)
    cutoff = START_DATE + pd.Timedelta(days=cutoff_day)
    return products, users, events, cutoff
