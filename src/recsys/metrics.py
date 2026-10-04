"""Ranking metrics with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the user's relevant items found in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """1 if at least one relevant item appears in the top-k, else 0."""
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain: rewards relevant items ranked higher."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg
