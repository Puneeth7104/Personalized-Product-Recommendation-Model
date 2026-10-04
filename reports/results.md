# Results

Simulated store: 2000 users, 600 products, 190,853 events over 90 days. Train: 136,442 events before 2026-03-14; validation: 54,411 events after it. Evaluated users: 1378. Relevant = added to cart or purchased in the validation window.

## Overall (all validation users)

|  | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | n_users |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity (all-time) | 0.0401 | 0.0872 | 0.0722 | 0.0326 | 0.1410 | 0.0924 | 0.0246 | 0.2101 | 0.1141 | 1378 |
| Popularity (trending, 14d half-life) | 0.0414 | 0.0901 | 0.0738 | 0.0327 | 0.1425 | 0.0933 | 0.0301 | 0.2560 | 0.1292 | 1378 |
| Matrix factorization (SVD) | 0.1112 | 0.2244 | 0.1932 | 0.0803 | 0.3272 | 0.2304 | 0.0554 | 0.4505 | 0.2699 | 1378 |
| Content-based (TF-IDF) | 0.1016 | 0.2072 | 0.1811 | 0.0761 | 0.3072 | 0.2184 | 0.0534 | 0.4290 | 0.2577 | 1378 |
| Hybrid (MF + content + trending) | 0.1122 | 0.2262 | 0.1963 | 0.0877 | 0.3566 | 0.2445 | 0.0625 | 0.5106 | 0.2934 | 1378 |

## Cold items only (71 items with < 5 training users)

|  | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | n_users |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity (all-time) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 183 |
| Popularity (trending, 14d half-life) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 183 |
| Matrix factorization (SVD) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 183 |
| Content-based (TF-IDF) | 0.0284 | 0.1275 | 0.0988 | 0.0230 | 0.2058 | 0.1252 | 0.0189 | 0.3352 | 0.1586 | 183 |
| Hybrid (MF + content + trending) | 0.0776 | 0.3525 | 0.2479 | 0.0508 | 0.4572 | 0.2823 | 0.0298 | 0.5310 | 0.3021 | 183 |

## ndcg@10 by activity segment

| activity_segment | n_users | Popularity (all-time) | Popularity (trending, 14d half-life) | Matrix factorization (SVD) | Content-based (TF-IDF) | Hybrid (MF + content + trending) |
|---|---|---|---|---|---|---|
| new (0 items) | 124 | 0.1131 | 0.1124 | 0.0093 | 0.0093 | 0.1124 |
| light (1-4) | 70 | 0.0619 | 0.0740 | 0.1788 | 0.1487 | 0.1778 |
| medium (5-19) | 360 | 0.1112 | 0.1091 | 0.2273 | 0.2126 | 0.2460 |
| heavy (20+) | 824 | 0.0836 | 0.0851 | 0.2694 | 0.2584 | 0.2695 |

## recall@10 by activity segment

| activity_segment | n_users | Popularity (all-time) | Popularity (trending, 14d half-life) | Matrix factorization (SVD) | Content-based (TF-IDF) | Hybrid (MF + content + trending) |
|---|---|---|---|---|---|---|
| new (0 items) | 124 | 0.1897 | 0.1741 | 0.0103 | 0.0103 | 0.1741 |
| light (1-4) | 70 | 0.1030 | 0.1334 | 0.2781 | 0.2303 | 0.2738 |
| medium (5-19) | 360 | 0.1675 | 0.1601 | 0.3498 | 0.3275 | 0.3836 |
| heavy (20+) | 824 | 0.1253 | 0.1308 | 0.3691 | 0.3495 | 0.3792 |

## precision@10 by activity segment

| activity_segment | n_users | Popularity (all-time) | Popularity (trending, 14d half-life) | Matrix factorization (SVD) | Content-based (TF-IDF) | Hybrid (MF + content + trending) |
|---|---|---|---|---|---|---|
| new (0 items) | 124 | 0.0363 | 0.0355 | 0.0032 | 0.0032 | 0.0355 |
| light (1-4) | 70 | 0.0257 | 0.0271 | 0.0557 | 0.0486 | 0.0629 |
| medium (5-19) | 360 | 0.0300 | 0.0286 | 0.0661 | 0.0619 | 0.0711 |
| heavy (20+) | 824 | 0.0337 | 0.0345 | 0.1002 | 0.0955 | 0.1049 |

## ndcg@10 by price segment

| price_segment | n_users | Popularity (all-time) | Popularity (trending, 14d half-life) | Matrix factorization (SVD) | Content-based (TF-IDF) | Hybrid (MF + content + trending) |
|---|---|---|---|---|---|---|
| budget | 420 | 0.0670 | 0.0685 | 0.2418 | 0.2293 | 0.2464 |
| mid | 443 | 0.1116 | 0.1141 | 0.2423 | 0.2065 | 0.2486 |
| premium | 391 | 0.0913 | 0.0901 | 0.2749 | 0.2866 | 0.2798 |
| unknown | 124 | 0.1131 | 0.1124 | 0.0093 | 0.0093 | 0.1124 |
