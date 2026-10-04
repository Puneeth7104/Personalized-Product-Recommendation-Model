"""Train, evaluate and report. Usage:  python scripts/run_pipeline.py [--users 2000] [--items 600]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from recsys.pipeline import PipelineConfig, run_pipeline, save_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports", default=str(ROOT / "reports"))
    parser.add_argument("--save-data", action="store_true",
                        help="also write the simulated CSVs to data/sample/ (git-ignored)")
    args = parser.parse_args()

    result = run_pipeline(PipelineConfig(n_users=args.users, n_items=args.items, seed=args.seed))

    pd.set_option("display.width", 200, "display.max_columns", 30)
    ks = result.config.k_list
    cols = [f"{m}@{k}" for k in ks for m in ("precision", "recall", "ndcg")]
    print(f"\nEvents: {len(result.events):,} | train {len(result.train_events):,} | "
          f"validation {len(result.valid_events):,} | evaluated users {len(result.truth)}")
    print(f"Pipeline finished in {result.seconds:.1f}s\n")
    print("=== Overall ===")
    print(result.metrics[cols + ["n_users"]].round(4).to_string())
    print("\n=== Cold items only ===")
    print(result.cold_metrics[cols + ["n_users"]].round(4).to_string())
    for name, table in result.segment_tables.items():
        print(f"\n=== {name} ===")
        print(table.round(4).to_string())

    out = save_reports(result, args.reports)
    print(f"\nReports written to {out}")

    if args.save_data:
        data_dir = ROOT / "data" / "sample"
        data_dir.mkdir(parents=True, exist_ok=True)
        result.products.to_csv(data_dir / "products.csv", index=False)
        result.users.to_csv(data_dir / "users.csv", index=False)
        result.events.to_csv(data_dir / "events.csv", index=False)
        print(f"Simulated data written to {data_dir}")


if __name__ == "__main__":
    main()
