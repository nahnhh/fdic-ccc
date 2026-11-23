from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL = ROOT / "data" / "processed" / "part2" / "bank_quarter_panel.parquet"
DEFAULT_OUTPUT = ROOT / "results" / "part3_polars_aggregate_plan.txt"
SOURCE_COLUMNS = ["ASSET", "REPDTE", "FAIL_DT", "EXIT_TYPE"]


def aggregate_query(panel_path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(panel_path)
        .filter(
            (pl.col("ASSET") > 0)
            & (pl.col("REPDTE") <= pl.date(2025, 6, 30))
        )
        .with_columns(
            pl.col("REPDTE").dt.year().alias("YEAR"),
            pl.when(
                pl.col("FAIL_DT").is_not_null()
                & (pl.col("FAIL_DT") > pl.col("REPDTE"))
                & (pl.col("FAIL_DT") <= pl.col("REPDTE").dt.offset_by("12mo"))
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.UInt8)
            .alias("target"),
        )
        .group_by(["YEAR", "EXIT_TYPE"])
        .agg(
            pl.len().alias("rows"),
            pl.col("target").sum().alias("failures"),
            pl.col("ASSET").mean().alias("mean_asset"),
        )
        .sort(["YEAR", "EXIT_TYPE"])
    )


def render_plan(panel_path: Path) -> str:
    lazy_query = aggregate_query(panel_path)
    plan = lazy_query.explain(optimized=True)
    return "\n".join([
        "POLARS AGGREGATE JOB — OPTIMIZED LAZY PLAN",
        f"Input: {panel_path}",
        "Output: counts, failure-label sum, and mean ASSET by YEAR and EXIT_TYPE",
        "",
        "PUSHDOWN ANNOTATIONS",
        "1. Predicate/filter pushdown at the Parquet scan:",
        "   ASSET > 0 AND REPDTE <= 2025-06-30",
        "   The optimized plan places this predicate in the scan SELECTION.",
        "2. Projection/column pushdown:",
        f"   Read only {', '.join(SOURCE_COLUMNS)} (4 of 21 source columns).",
        "   YEAR and target are derived after reading; all other source columns are pruned.",
        "3. No bank-quarter deduplication:",
        "   Part 2 found zero duplicate (CERT, REPDTE) keys, so this aggregate",
        "   does not pay for a sort/hash shuffle just to remove duplicates.",
        "",
        "OPTIMIZED PLAN (Polars explain(optimized=True))",
        "-" * 54,
        plan,
        "",
        "Execution note: explain() shows the optimized logical plan. The lazy",
        "query executes when collect() is called; collection and the benchmark",
        "are measured separately from this plan printout.",
        "",
    ])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.panel.exists():
        raise FileNotFoundError(f"Panel not found: {args.panel}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = render_plan(args.panel)
    args.output.write_text(content, encoding="utf-8")
    print(content)
    print(f"Saved plan printout to {args.output}")


if __name__ == "__main__":
    main()
