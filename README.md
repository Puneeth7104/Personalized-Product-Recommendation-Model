# Personalised Product Recommendation System

A Python/ML project that recommends products to shoppers from their views, clicks, carts and
purchases, with a Streamlit dashboard for exploring recommendations, evaluation results and
per-segment behaviour.

**Live demo:**[ _add your Streamlit URL here after deploying_](https://personalized-appuct-recommendation-model-zepzsyhqmqoywfj3rob3y.streamlit.app/) &nbsp;·&nbsp;
**CI:** ![CI](https://github.com/Puneeth7104/product-recommendation-system/actions/workflows/ci.yml/badge.svg)

![stack](https://img.shields.io/badge/Python-3.11%2B-blue) ![stack](https://img.shields.io/badge/pandas%20%7C%20NumPy%20%7C%20scikit--learn-informational) ![stack](https://img.shields.io/badge/Streamlit-dashboard-red)

## What it does

| Stage | What is built |
|---|---|
| 1. Data prep | Event log (view / click / cart / purchase) → one implicit-feedback score per shopper-product pair (weights 1 / 2 / 4 / 8, then `log1p`) |
| 2. Baseline | Popularity, all-time and trending (14-day half-life) |
| 3. Collaborative filtering | Matrix factorisation via truncated SVD on the shopper × product matrix |
| 4. Content-based + cold start | TF-IDF over category, sub-category, brand, tags and price tier; scores brand-new products and thin histories |
| 5. Hybrid | Blends the three; weights adapt to how much history a shopper has |
| 6. Evaluation | Precision@K, Recall@K, NDCG@K, hit rate with a **time-based** split; separate cold-item evaluation |
| 7. Segments | Quality and recommendation *behaviour* (price, diversity, novelty, coverage) by activity and price segment |
| 8. Interface | Streamlit app with five tabs; explanations ("similar to X you bought") on every recommendation |

## About the data (please read)

The project brief did not include a dataset, so the repo **simulates a shop**
(`src/recsys/data.py`): 2,000 shoppers, 600 products, ~190k events over 90 days, hidden shopper
"personas", a long-tailed popularity curve, a purchase funnel, new arrivals launched just before
the cutoff, and shoppers who sign up only after it. Everything is seeded and reproducible.

This means the numbers below show that **the method works on data with this structure**, not how
it would perform on a real store. To use real data, see [Bring your own data](#bring-your-own-data).

## Results

Train on events before 14 Mar 2026; evaluate on the following 18 days. A recommendation is correct
if the shopper **added it to cart or bought it** in that window (1,378 shoppers qualify). Items
already purchased are never recommended. Default seed (42); full tables in
[`reports/results.md`](reports/results.md).

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity (all-time) | 0.033 | 0.141 | 0.092 |
| Popularity (trending) | 0.033 | 0.143 | 0.093 |
| Matrix factorisation (SVD) | 0.080 | 0.327 | 0.230 |
| Content-based (TF-IDF) | 0.076 | 0.307 | 0.218 |
| **Hybrid** | **0.088** | **0.357** | **0.245** |

**Cold items** (71 products with < 5 shoppers before the cutoff), Recall@20: popularity 0.00,
matrix factorisation 0.00, content-based 0.34, hybrid 0.53. Behaviour-based models cannot recommend
what nobody has interacted with yet; metadata can.

**Segments** (NDCG@10):

| Segment | Shoppers | Trending popularity | Matrix factorisation | Hybrid |
|---|---|---|---|---|
| new (0 items) | 124 | 0.112 | 0.009 | 0.112 |
| light (1–4) | 70 | 0.074 | 0.179 | 0.178 |
| medium (5–19) | 360 | 0.109 | 0.227 | 0.246 |
| heavy (20+) | 824 | 0.085 | 0.269 | 0.270 |

What this shows:

- **Personalisation roughly 2.5× popularity** on NDCG@10 (0.23-0.25 vs 0.09).
- **New shoppers**: collaborative filtering and content models have no signal and give arbitrary
  lists. The hybrid detects this and shows trending items, matching the best option.
- **The hybrid's edge over plain MF is modest.** It is clearest for medium-history shoppers and for
  cold items; for light and heavy shoppers it is about equal. Across five other seeds the hybrid
  beat MF on NDCG@10 in four, by under 0.01 on average, and lost in one. I have not computed
  confidence intervals, so treat this gap as suggestive.
- **Diversity trade-off** (heavy shoppers): the share of a top-10 list taken by one category is
  0.57 for MF, 0.78 for the hybrid and 0.95 for content-based (a filter-bubble risk), versus
  ~0.45 for popularity. Catalog coverage: popularity 2.5%, MF 32%, hybrid 45%, content-based 62%.

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/Puneeth7104/product-recommendation-system.git
cd product-recommendation-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python -m pytest -q                      # 31 tests
python scripts/run_pipeline.py           # trains, evaluates, writes reports/
streamlit run streamlit_app.py           # dashboard at http://localhost:8501
```

The pipeline takes about 6 seconds. On macOS/Linux use `source .venv/bin/activate` instead of the
`Activate.ps1` line.

## Project structure

```
.
├── streamlit_app.py            # dashboard (also the deployment entry point)
├── src/recsys/
│   ├── data.py                 # synthetic catalog, shoppers and event funnel
│   ├── interactions.py         # validation, time-based split, interaction table, ground truth
│   ├── features.py             # price tiers (also used for brand-new products)
│   ├── models.py               # popularity, matrix factorisation, content-based, hybrid
│   ├── metrics.py              # Precision@K, Recall@K, NDCG@K, hit rate
│   ├── evaluate.py             # per-user and overall evaluation
│   ├── segments.py             # shopper segments + recommendation profiles
│   └── pipeline.py             # end-to-end run + report writer
├── scripts/run_pipeline.py     # CLI entry point
├── tests/                      # 31 tests (metrics, split/leakage, models, pipeline)
├── reports/                    # committed results (CSV + results.md)
├── .github/workflows/ci.yml    # runs tests on every push
├── .streamlit/config.toml      # theme
└── requirements*.txt
```

## Design decisions

- **Time-based split, not random.** Random splits let the model train on the future and overstate
  quality. A test asserts that no training event is at or after the cutoff.
- **Relevant = cart or purchase, not views.** Views are cheap and noisy. Views and clicks still
  feed the model as weaker signals.
- **Only purchased items are excluded** from recommendations (and from ground truth). Viewed or
  carted items can come back, which is realistic retargeting.
- **`log1p` on interaction weight** stops one shopper viewing an item 50 times from dominating.
- **Truncated SVD rather than ALS.** Deterministic, dependency-free (scikit-learn only), and a
  strong implicit-feedback baseline. See "Next steps" for the upgrade path.
- **Hybrid weights are fixed heuristics**, documented in `HybridRecommender`, not tuned on the
  validation window. Tuning and reporting on the same window would inflate the results.
- **Cold items** get their MF slot filled with the content score, since MF has nothing to say
  about them.

## Limitations

- Synthetic data, so absolute numbers are not transferable. The simulator also rewards the model
  families used here (taste clusters and metadata that matter), which flatters them somewhat.
- Single train/validation window; no confidence intervals or multi-window backtests.
- No hyperparameter search (SVD factors = 32, hybrid weights fixed).
- Offline metrics only; they do not capture the effect of showing recommendations on behaviour.
- Retrains from scratch on app start; there is no incremental update or model store.

## Bring your own data

Provide two files and pass them through `validate_events`:

- **events**: `user_id`, `item_id`, `event_type` (`view` / `click` / `cart` / `purchase`), `timestamp`
- **products**: `item_id`, `title`, `category`, `subcategory`, `brand`, `tags` (space-separated), `price`

```python
import pandas as pd
from recsys.interactions import validate_events, time_based_split, build_ground_truth
from recsys.models import build_context, PopularityRecommender, MatrixFactorizationRecommender

events = validate_events(pd.read_csv("events.csv"))
products = pd.read_csv("products.csv")
train, valid, cutoff = time_based_split(events, valid_frac=0.2)
ctx = build_context(train, products)
truth = build_ground_truth(valid, train)
```

Then fit models on `ctx` and evaluate with `recsys.evaluate.evaluate_models` as `pipeline.py` does.

## Deployment

The dashboard deploys to Streamlit Community Cloud straight from this repo: main file
`streamlit_app.py`, dependencies from `requirements.txt`. Step-by-step instructions are in
[`docs/GIT_AND_DEPLOY.md`](docs/GIT_AND_DEPLOY.md). The app generates its data and trains in about
10 seconds on first load, so no data files need to be committed.

## Next steps

- Implicit ALS or BPR instead of SVD; tune factors and the hybrid weights on a separate validation
  window and report on a later test window.
- Walk-forward backtesting with confidence intervals.
- Diversity re-ranking (MMR) to counter the content model's narrowness.
- A small FastAPI `/recommend/{user_id}` endpoint wrapping the same models.
- Session-aware signals (recency within the last visit) and price/stock constraints.

See [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) for a walkthrough of the project and
likely questions.
