# Interview guide

## The 60-second pitch

"I built a recommendation system that takes shoppers' views, clicks, carts and purchases and
suggests products they are likely to want next. I compared four approaches: a popularity baseline,
matrix factorisation for collaborative filtering, a content-based model on product metadata, and a
hybrid that adapts to how much history a shopper has. I evaluated with a time-based split so the
model never sees the future, using Precision@K, Recall@K and NDCG@K, and I broke results down by
shopper segment and for cold-start products. Personalised models were about 2.5× better than
popularity; only the metadata-based models could recommend brand-new products at all. It's deployed
as a Streamlit app with tests and CI. The data is simulated because none was provided, so I'm
careful to say the results show the method works, not how it would do in a real store."

## Walk-through in the order the code runs

1. **Interactions.** Four event types become one score per (shopper, product): weights 1/2/4/8,
   summed, then `log1p`. Why: a purchase is stronger evidence than a view; `log1p` stops heavy
   repeaters dominating.
2. **Split.** Train strictly before a cutoff date; evaluate on the 18 days after. Why: a random
   split leaks the future. There is a test for it.
3. **Ground truth.** Relevant = added to cart or bought in the validation window, minus items
   already bought. Why: views are noisy; re-buying is not what we are predicting.
4. **Popularity.** Baseline everyone must beat. I added a *trending* variant because all-time
   popularity is a weak baseline when new products arrive.
5. **Matrix factorisation.** Truncated SVD of the shopper × product matrix, 32 factors. Score =
   reconstruction of the shopper's row. Similar shoppers' items rise.
6. **Content-based.** TF-IDF over category, sub-category, brand, tags and price tier. Shopper
   profile = weighted average of items they interacted with. Works for products with no history.
7. **Hybrid.** Weighted blend with weights that depend on history length (0 items: trending only;
   20+ items: mostly MF). Items with almost no history use the content score in the MF slot.
8. **Evaluation and segments.** Overall, cold items only, by activity and price segment, plus a
   "what does it recommend" profile (price, diversity, novelty, coverage).

## Numbers worth remembering (default seed)

- NDCG@10: popularity 0.09, MF 0.23, content 0.22, hybrid 0.245.
- Cold items, Recall@20: MF and popularity 0.00; content 0.34; hybrid 0.53.
- New shoppers: MF NDCG@10 is 0.009 versus 0.112 for trending; the hybrid matches trending.
- Heavy shoppers' top-10 share from one category: MF 0.57, hybrid 0.78, content 0.95.
- Across five other seeds the hybrid beat MF four times, by under 0.01 on average. Say "modest".

## Questions you should expect

**Why SVD and not ALS or a neural model?** SVD is deterministic, needs only scikit-learn, and is a
strong implicit-feedback baseline. ALS or BPR handle implicit feedback more properly (confidence
weighting, negative sampling) and are my first upgrade. A neural model would be overkill for this
data size and harder to explain.

**Why NDCG as well as precision and recall?** Precision and recall ignore rank. NDCG rewards putting
the right item at position 1 instead of position 10, which matters on a product shelf.

**How did you avoid leakage?** Time-based split; models and segments use only training events;
items bought in training are excluded from both recommendations and ground truth; a test asserts no
training event is at or after the cutoff.

**How do you handle cold start?** New shoppers: trending items. Thin histories: more weight on
metadata. New products: TF-IDF on metadata (the app has a "launch a new product" demo that finds
similar products and likely audiences). Cold-item evaluation shows it works: MF scores zero there.

**Your data is synthetic. Why should I trust the results?** You shouldn't extrapolate the absolute
numbers. The simulator rewards the model families I used, which flatters them. What the project
demonstrates is a correct pipeline: leakage-free evaluation, sensible baselines, segment analysis
and cold-start handling. The code accepts real event logs (`validate_events`).

**Is the hybrid really better?** Modestly. It is clearly better for cold items and shoppers with
medium history, about equal for light and heavy shoppers, and it lost to MF on one of five other
seeds. I did not tune the weights on the validation window (that would inflate results) and I have
not computed confidence intervals; both are next steps.

**What is the downside of the content model?** Narrowness: heavy shoppers get ~95% of a list from
one category. The hybrid reduces that; a diversity re-ranker such as MMR would reduce it further.

**How would you take it to production?** Precompute embeddings and top-N lists offline, serve them
from a cache or key-value store behind an API, retrain on a schedule, monitor click-through and
coverage, and A/B test against the current experience, because offline metrics are not business
impact.

**How would you improve it?** Tune factors and weights on a separate validation window and report
on a later test window; walk-forward backtests with confidence intervals; ALS/BPR; diversity
re-ranking; stock and price constraints; session-aware recency features.

## Things not to claim

- That the numbers reflect real-world performance.
- That the hybrid is dramatically better than MF.
- That offline metrics prove business impact.
