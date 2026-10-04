"""End-to-end pipeline: data -> time split -> models -> evaluation -> segment analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import generate_dataset
from .evaluate import evaluate_models, restrict_truth_to_items
from .interactions import build_ground_truth, time_based_split, validate_events
from .models import (
    BaseRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    RecContext,
    build_context,
)
from .segments import (
    SEGMENT_ORDER,
    build_user_segments,
    recommendation_profile,
    segment_metric_table,
)


@dataclass
class PipelineConfig:
    n_users: int = 2000
    n_items: int = 600
    total_days: int = 90
    valid_days: int = 18
    seed: int = 42
    k_list: tuple[int, ...] = (5, 10, 20)
    n_factors: int = 32
    cold_item_threshold: int = 5


@dataclass
class PipelineResult:
    config: PipelineConfig
    products: pd.DataFrame
    users: pd.DataFrame
    events: pd.DataFrame
    cutoff: pd.Timestamp
    train_events: pd.DataFrame
    valid_events: pd.DataFrame
    ctx: RecContext
    models: dict[str, BaseRecommender]
    truth: dict[int, set[int]]
    segments: pd.DataFrame
    metrics: pd.DataFrame
    cold_metrics: pd.DataFrame
    per_user: dict[str, pd.DataFrame]
    segment_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    profile: pd.DataFrame | None = None
    cold_item_ids: set[int] = field(default_factory=set)
    seconds: float = 0.0


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    t0 = time.time()

    # 1. data -> clean events -> time-based split (train strictly before cutoff)
    products, users, events, cutoff = generate_dataset(
        cfg.n_users, cfg.n_items, cfg.total_days, cfg.valid_days, cfg.seed
    )
    events = validate_events(events)
    train, valid, cutoff = time_based_split(events, cutoff=cutoff)

    # 2. shared context built from training data only
    ctx = build_context(train, products)

    # 3. models
    popularity = PopularityRecommender().fit(ctx)
    trending = PopularityRecommender(half_life_days=14).fit(ctx)
    mf = MatrixFactorizationRecommender(n_factors=cfg.n_factors, random_state=cfg.seed).fit(ctx)
    content = ContentBasedRecommender().fit(ctx)
    hybrid = HybridRecommender(mf, content, trending, cfg.cold_item_threshold).fit(ctx)
    models: dict[str, BaseRecommender] = {
        m.name: m for m in (popularity, trending, mf, content, hybrid)
    }

    # 4. evaluation on the validation window
    truth = build_ground_truth(valid, train)
    metrics, per_user = evaluate_models(models, truth, cfg.k_list)

    cold_ids = {
        int(i) for i, n in zip(ctx.item_ids, ctx.item_user_counts) if n < cfg.cold_item_threshold
    }
    cold_truth = restrict_truth_to_items(truth, cold_ids)
    cold_metrics, _ = evaluate_models(models, cold_truth, cfg.k_list)

    # 5. segment analysis
    segments = build_user_segments(ctx, users["user_id"])
    k_main = 10 if 10 in cfg.k_list else cfg.k_list[0]
    seg_tables = {
        f"{metric}@{k_main} by activity segment": segment_metric_table(
            per_user, segments, f"{metric}@{k_main}", order=SEGMENT_ORDER
        )
        for metric in ("ndcg", "recall", "precision")
    }
    seg_tables[f"ndcg@{k_main} by price segment"] = segment_metric_table(
        per_user, segments, f"ndcg@{k_main}", segment_col="price_segment",
        order=["budget", "mid", "premium", "unknown"],
    )
    profile = recommendation_profile(
        models, ctx, segments, k=k_main, cold_item_threshold=cfg.cold_item_threshold
    )

    return PipelineResult(
        config=cfg, products=products, users=users, events=events, cutoff=cutoff,
        train_events=train, valid_events=valid, ctx=ctx, models=models, truth=truth,
        segments=segments, metrics=metrics, cold_metrics=cold_metrics, per_user=per_user,
        segment_tables=seg_tables, profile=profile, cold_item_ids=cold_ids,
        seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------- reporting


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col, v in row.items():
            if col == "n_users":
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_reports(result: PipelineResult, out_dir: str | Path = "reports") -> Path:
    """Write small CSV/Markdown reports that are safe to commit."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out / "metrics_overall.csv")
    result.cold_metrics.to_csv(out / "metrics_cold_items.csv")
    for name, table in result.segment_tables.items():
        table.to_csv(out / ("segment_" + name.replace("@", "_at_").replace(" ", "_") + ".csv"))
    result.profile.to_csv(out / "recommendation_profile.csv", index=False)

    ks = result.config.k_list
    main = [f"{m}@{k}" for k in ks for m in ("precision", "recall", "ndcg")]
    cfg = result.config
    text = [
        "# Results",
        "",
        f"Simulated store: {cfg.n_users} users, {cfg.n_items} products, "
        f"{len(result.events):,} events over {cfg.total_days} days. "
        f"Train: {len(result.train_events):,} events before {result.cutoff:%Y-%m-%d}; "
        f"validation: {len(result.valid_events):,} events after it. "
        f"Evaluated users: {len(result.truth)}. Relevant = added to cart or purchased in the validation window.",
        "",
        "## Overall (all validation users)",
        "",
        _md_table(result.metrics[main + ["n_users"]]),
        "",
        f"## Cold items only ({len(result.cold_item_ids)} items with < {cfg.cold_item_threshold} training users)",
        "",
        _md_table(result.cold_metrics[main + ["n_users"]]),
    ]
    for name, table in result.segment_tables.items():
        text += ["", f"## {name}", "", _md_table(table)]
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return out
