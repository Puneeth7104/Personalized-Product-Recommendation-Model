import math

from recsys.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_precision_and_recall_hand_computed():
    recs = [10, 11, 12, 13, 14]
    relevant = {11, 14, 99}
    assert close(precision_at_k(recs, relevant, 5), 2 / 5)
    assert close(precision_at_k(recs, relevant, 2), 1 / 2)
    assert close(recall_at_k(recs, relevant, 5), 2 / 3)
    assert close(recall_at_k(recs, relevant, 1), 0.0)


def test_hit_rate():
    assert hit_rate_at_k([1, 2, 3], {3}, 3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {3}, 2) == 0.0


def test_ndcg_perfect_and_imperfect():
    relevant = {1, 2}
    assert close(ndcg_at_k([1, 2, 3], relevant, 3), 1.0)
    # one relevant item at rank 2 instead of rank 1
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2)
    assert close(ndcg_at_k([9, 5], {5}, 2), dcg / idcg)


def test_ndcg_rewards_higher_rank():
    relevant = {7}
    assert ndcg_at_k([7, 1, 2], relevant, 3) > ndcg_at_k([1, 7, 2], relevant, 3) > ndcg_at_k([1, 2, 7], relevant, 3)


def test_empty_relevant_set_is_zero():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # 5 relevant items but only k=2 slots: two hits in the top 2 is a perfect score
    assert close(ndcg_at_k([1, 2], {1, 2, 3, 4, 5}, 2), 1.0)
