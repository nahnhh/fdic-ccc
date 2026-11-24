from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl


CORE_FIELDS = ["ASSET", "DEP", "EQ", "NETINC", "ROA", "NCLNLS", "LNATRES", "RBC1AAJ"]
MODEL_END = __import__("datetime").date(2025, 6, 30)


def flatten_quality_report(value: object, path: str = "") -> list[dict[str, str]]:
    if isinstance(value, dict):
        if not value:
            return [{"path": path, "type": "object", "value": "{}"}]
        rows = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            rows.extend(flatten_quality_report(child, child_path))
        return rows
    if isinstance(value, list):
        if not value:
            return [{"path": path, "type": "array", "value": "[]"}]
        rows = []
        for index, child in enumerate(value):
            rows.extend(flatten_quality_report(child, f"{path}[{index}]"))
        return rows
    if value is None:
        value_type, text = "null", ""
    elif isinstance(value, bool):
        value_type, text = "boolean", str(value).lower()
    elif isinstance(value, (int, float)):
        value_type, text = "number", str(value)
    else:
        value_type, text = "string", str(value)
    return [{"path": path, "type": value_type, "value": text}]


def write_quality_report_csv(report: dict, path: Path) -> None:
    rows = flatten_quality_report(report)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["path", "type", "value"])
        writer.writeheader()
        writer.writerows(rows)


def parse_mmddyyyy(series: pl.Series) -> pl.Series:
    return series.cast(pl.String, strict=False).str.strip_chars().str.to_date(
        "%m/%d/%Y", strict=False
    )


def quarter_end_column() -> pl.Expr:
    return pl.col("REPDTE").dt.offset_by("3mo").dt.month_end()


def read_raw_financials(raw_dir: Path) -> pl.DataFrame:
    files = sorted((raw_dir / "financials").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No financial quarter files under {raw_dir / 'financials'}")

    frames = []
    for path in files:
        frame = pl.read_parquet(path)
        frames.append(
            frame.with_columns(
                pl.col("CERT").cast(pl.Int64, strict=False),
                pl.col("REPDTE").cast(pl.String, strict=False).str.strip_chars(),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


def typed_financials(fin: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        pl.col("CERT").cast(pl.Int64, strict=False),
        pl.col("REPDTE").cast(pl.String, strict=False).str.to_date("%Y%m%d", strict=False),
        pl.col("STNAME").cast(pl.String, strict=False),
    ]
    for field in CORE_FIELDS:
        if field in fin.columns:
            expressions.append(pl.col(field).cast(pl.Float64, strict=False))
    if "P3ASSET" in fin.columns:
        expressions.append(pl.col("P3ASSET").cast(pl.Int64, strict=False))
    return fin.select(*expressions)


def summarize_numeric_values(values: pl.Series) -> dict:
    numeric = values.cast(pl.Float64, strict=False)
    finite = numeric.filter(numeric.is_finite().fill_null(False))
    if finite.is_empty():
        return {
            "finite_rows": 0,
            "minimum": None,
            "p01": None,
            "median": None,
            "p99": None,
            "maximum": None,
            "iqr_fence_lower": None,
            "iqr_fence_upper": None,
            "potential_iqr_outlier_rows": 0,
        }

    q1 = float(finite.quantile(0.25, interpolation="linear"))
    q3 = float(finite.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {
        "finite_rows": finite.len(),
        "minimum": float(finite.min()),
        "p01": float(finite.quantile(0.01, interpolation="linear")),
        "median": float(finite.median()),
        "p99": float(finite.quantile(0.99, interpolation="linear")),
        "maximum": float(finite.max()),
        "iqr_fence_lower": lower,
        "iqr_fence_upper": upper,
        "potential_iqr_outlier_rows": int(
            ((finite < lower) | (finite > upper)).sum()
        ),
    }


def build_financial_value_checks(fin_raw: pl.DataFrame) -> dict:
    nonfinite_counts = {}
    parse_failure_counts = {}
    for field in CORE_FIELDS:
        if field not in fin_raw.columns:
            nonfinite_counts[field] = None
            parse_failure_counts[field] = None
            continue
        raw_values = fin_raw[field]
        numeric = raw_values.cast(pl.Float64, strict=False)
        nonfinite_counts[field] = int(
            (numeric.is_not_null() & ~numeric.is_finite()).fill_null(False).sum()
        )
        parse_failure_counts[field] = int(
            (raw_values.is_not_null() & numeric.is_null()).sum()
        )

    def count_relation(left_name: str, right_name: str, relation: str) -> int | None:
        if left_name not in fin_raw.columns or right_name not in fin_raw.columns:
            return None
        left = pl.col(left_name).cast(pl.Float64, strict=False)
        right = pl.col(right_name).cast(pl.Float64, strict=False)
        valid = left.is_finite() & right.is_finite()
        if relation == "greater_than":
            condition = left > right
        elif relation == "less_than":
            condition = left < right
        else:
            raise ValueError(f"Unsupported numeric relation: {relation}")
        return int(fin_raw.select((valid & condition).fill_null(False).sum()).item())

    def count_negative(field: str) -> int | None:
        if field not in fin_raw.columns:
            return None
        values = pl.col(field).cast(pl.Float64, strict=False)
        return int(
            fin_raw.select((values.is_finite() & (values < 0)).fill_null(False).sum()).item()
        )

    return {
        "nonfinite_counts_by_field": nonfinite_counts,
        "nonnumeric_nonnull_counts_by_field": parse_failure_counts,
        "consistency_checks": {
            "equity_exceeds_assets_rows": count_relation("EQ", "ASSET", "greater_than"),
            "deposits_exceed_assets_rows": count_relation("DEP", "ASSET", "greater_than"),
            "negative_deposit_rows": count_negative("DEP"),
        },
        "consistency_check_notes": {
            "equity_exceeds_assets_rows": (
                "Accounting review flag; not automatically corrected or dropped."
            ),
            "deposits_exceed_assets_rows": (
                "Review warning; this can occur when a bank has negative equity."
            ),
            "negative_deposit_rows": (
                "Review warning; investigate source values before treating as invalid."
            ),
        },
    }


def build_quality_report(fin_raw: pl.DataFrame, fin: pl.DataFrame,
                         failures: pl.DataFrame, institutions: pl.DataFrame) -> dict:
    cert_col = failures.select(pl.col("CERT").cast(pl.Int64, strict=False))
    failures_typed = failures.with_columns(
        pl.col("CERT").cast(pl.Int64, strict=False),
        parse_mmddyyyy(pl.col("FAILDATE")).alias("FAIL_DT"),
    )
    institutions_typed = institutions.with_columns(
        pl.col("CERT").cast(pl.Int64, strict=False),
        parse_mmddyyyy(pl.col("ENDEFYMD")).alias("END_DT"),
    )

    null_fin = {
        field: int(fin_raw[field].null_count())
        if field in fin_raw.columns else None
        for field in CORE_FIELDS
    }
    bad_repdte = int(fin["REPDTE"].null_count())
    duplicate_keys = int(fin_raw.select(pl.struct(["CERT", "REPDTE"]).is_duplicated().sum()).item())
    null_certs = int(fin_raw["CERT"].is_null().sum())

    failure_rows = failures_typed.height
    failure_cert_null = int(failures_typed["CERT"].is_null().sum())
    failure_date_invalid = int(failures_typed["FAIL_DT"].is_null().sum())
    duplicate_failure_certs = int(
        cert_col.group_by("CERT").len().filter(pl.col("len") > 1).height
        if cert_col.height else 0
    )

    institution_duplicates = int(
        institutions_typed.group_by("CERT").len().filter(pl.col("len") > 1).height
    )
    active_count = institutions_typed["ACTIVE"].value_counts().sort("ACTIVE").to_dicts()
    exit_counts = (
        institutions_typed.group_by("CHANGEC1")
        .len()
        .sort("CHANGEC1")
        .with_columns(pl.col("CHANGEC1").cast(pl.String, strict=False))
        .to_dicts()
    )

    era_rows = []
    era_distribution_rows = []
    previous_positive_medians = {field: None for field in CORE_FIELDS}
    for era, start, end in (
        ("1984-1993", "1984-01-01", "1993-12-31"),
        ("1994-2007", "1994-01-01", "2007-12-31"),
        ("2008-2014", "2008-01-01", "2014-12-31"),
        ("2015-2025", "2015-01-01", "2025-12-31"),
        ("2026+", "2026-01-01", "9999-12-31"),
    ):
        era_df = fin_raw.with_columns(
            pl.col("REPDTE").cast(pl.String, strict=False).str.to_date("%Y%m%d", strict=False)
        ).filter(
            pl.col("REPDTE").is_between(
                pl.lit(start).str.to_date(),
                pl.lit(end).str.to_date(),
            )
        )
        nulls = {}
        field_distributions = {}
        for field in CORE_FIELDS:
            if field in era_df.columns and era_df.height:
                nulls[field] = round(float(era_df[field].null_count()) / era_df.height, 6)
                distribution = summarize_numeric_values(era_df[field])
            else:
                nulls[field] = None
                distribution = summarize_numeric_values(pl.Series(field, [], dtype=pl.Float64))
            median = distribution["median"]
            previous_median = previous_positive_medians[field]
            distribution["positive_median_ratio_vs_previous_era"] = (
                median / previous_median
                if median is not None and median > 0
                and previous_median is not None and previous_median > 0
                else None
            )
            previous_positive_medians[field] = (
                median if median is not None and median > 0 else None
            )
            field_distributions[field] = distribution
        era_rows.append({"era": era, "rows": era_df.height, "null_share": nulls})
        era_distribution_rows.append({
            "era": era,
            "rows": era_df.height,
            "fields": field_distributions,
        })

    financial_value_checks = build_financial_value_checks(fin_raw)

    fin_certs = fin["CERT"].drop_nulls().unique()
    failures_with_financials = failures_typed.join(
        fin_certs.to_frame("CERT"), on="CERT", how="inner"
    )
    failure_certs_no_financials = (
        failures_typed.filter(pl.col("CERT").is_not_null())
        .select("CERT").unique()
        .join(fin_certs.to_frame("CERT"), on="CERT", how="anti")
        .height
    )

    # Count missing quarterly observations within each bank's observed span.
    gap_counts = (
        fin.sort(["CERT", "REPDTE"])
        .group_by("CERT")
        .agg(
            (pl.col("REPDTE").diff().dt.total_days() / 91.0).cast(pl.Int64, strict=False)
            .fill_null(0).sub(1).clip(0).sum().alias("missing_quarter_slots")
        )
    )

    return {
        "financials": {
            "raw_rows": fin_raw.height,
            "typed_rows": fin.height,
            "core_null_counts": null_fin,
            "core_null_share_by_era": era_rows,
            "core_value_distributions_by_era": era_distribution_rows,
            "value_checks": financial_value_checks,
            "outlier_rule": (
                "Potential outliers are values outside 1.5 IQR fences within each era. "
                "They are review flags only and are not removed from the panel."
            ),
            "temporal_scale_check_note": (
                "Era quantiles and positive-median ratios are descriptive screens. "
                "Changes may reflect inflation, bank composition, data units, or field "
                "definitions; the latest-vintage extract does not include historical "
                "dictionary versions needed to verify definition changes."
            ),
            "invalid_or_null_REPDTE": bad_repdte,
            "duplicate_CERT_REPDTE_rows": duplicate_keys,
            "null_CERT_rows": null_certs,
            "nonpositive_ASSET_rows": int(
                fin_raw.select(pl.col("ASSET").cast(pl.Float64, strict=False) <= 0).sum().item()
                if "ASSET" in fin_raw.columns else 0
            ),
            "per_bank_gaps": {
                "banks_with_gaps": int(gap_counts.filter(pl.col("missing_quarter_slots") > 0).height),
                "missing_quarter_slots": int(gap_counts["missing_quarter_slots"].sum()),
                "maximum_missing_quarters_for_one_bank": int(
                    gap_counts["missing_quarter_slots"].max()
                ),
            },
        },
        "failures": {
            "rows": failure_rows,
            "distinct_CERT": failures_typed["CERT"].drop_nulls().n_unique(),
            "null_CERT_rows": failure_cert_null,
            "invalid_FAILDATE_rows": failure_date_invalid,
            "CERTs_with_multiple_rows": duplicate_failure_certs,
            "failure_CERTs_without_Financials_rows": failure_certs_no_financials,
            "failure_CERTs_with_Financials_rows": failures_with_financials["CERT"].n_unique(),
        },
        "institutions": {
            "rows": institutions_typed.height,
            "duplicate_CERT_rows": institution_duplicates,
            "active_counts": active_count,
            "invalid_ENDEFYMD_rows": int(institutions_typed["END_DT"].is_null().sum()),
            "exit_counts_by_CHANGEC1": exit_counts,
        },
    }


def build_panel(raw_dir: Path) -> tuple[pl.DataFrame, dict]:
    fin_raw = read_raw_financials(raw_dir)
    fin = typed_financials(fin_raw)

    failures = pl.read_parquet(raw_dir / "failures.parquet").with_columns(
        pl.col("CERT").cast(pl.Int64, strict=False),
        parse_mmddyyyy(pl.col("FAILDATE")).alias("FAIL_DT"),
    )
    institutions = pl.read_parquet(raw_dir / "institutions.parquet").with_columns(
        pl.col("CERT").cast(pl.Int64, strict=False),
        parse_mmddyyyy(pl.col("ENDEFYMD")).alias("END_DT"),
    )

    report = build_quality_report(fin_raw, fin, failures, institutions)

    # One failure event per CERT: use the earliest documented failure date.
    failures_one = (
        failures.filter(pl.col("CERT").is_not_null() & pl.col("FAIL_DT").is_not_null())
        .group_by("CERT")
        .agg(pl.col("FAIL_DT").min().alias("FAIL_DT"))
    )

    inst_exit = institutions.select(
        "CERT", "ACTIVE", "CHANGEC1",
        pl.when(pl.col("END_DT").dt.year() == 9999)
        .then(pl.lit(None, dtype=pl.Date))
        .otherwise(pl.col("END_DT"))
        .alias("EXIT_DT"),
    )

    staged = fin.unique(subset=["CERT", "REPDTE"], keep="first", maintain_order=True)
    dropped_duplicate_keys = fin.height - staged.height

    staged = staged.filter(
        pl.col("CERT").is_not_null()
        & pl.col("REPDTE").is_not_null()
    )
    dropped_invalid_keys = fin.height - dropped_duplicate_keys - staged.height
    if "ASSET" in fin.columns:
        after_asset = staged.filter(pl.col("ASSET") > 0)
        dropped_nonpositive_asset = staged.height - after_asset.height
        staged = after_asset
    else:
        dropped_nonpositive_asset = 0

    after_horizon = staged.filter(pl.col("REPDTE") <= pl.lit(MODEL_END))
    dropped_after_horizon = staged.height - after_horizon.height
    staged = after_horizon

    panel = (
        staged.join(inst_exit, on="CERT", how="left")
        .join(failures_one, on="CERT", how="left")
        .with_columns(
            pl.when(pl.col("FAIL_DT").is_not_null())
            .then(pl.lit("failure"))
            .when(pl.col("EXIT_DT").is_not_null())
            .then(pl.lit("other_exit"))
            .otherwise(pl.lit("open"))
            .alias("EXIT_TYPE"),
            pl.coalesce(["FAIL_DT", "EXIT_DT"]).alias("EXIT_DATE"),
        )
    )

    # The authoritative exit date removes reports dated after an exit.
    before_exit_filter = panel.height
    panel = panel.filter(pl.col("EXIT_DATE").is_null() | (pl.col("REPDTE") <= pl.col("EXIT_DATE")))
    dropped_after_exit = before_exit_filter - panel.height

    panel = panel.with_columns(
        pl.col("REPDTE").dt.offset_by("12mo").alias("LABEL_WINDOW_END")
    )
    panel = panel.with_columns(
        pl.when(
            pl.col("FAIL_DT").is_not_null()
            & (pl.col("FAIL_DT") > pl.col("REPDTE"))
            & (pl.col("FAIL_DT") <= pl.col("LABEL_WINDOW_END"))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.UInt8)
        .alias("FAIL_4Q")
    ).with_columns(
        pl.col("FAIL_4Q").cast(pl.UInt8).alias("target")
    ).sort(["REPDTE", "CERT"])

    report["panel"] = {
        "rows": panel.height,
        "distinct_CERT": panel["CERT"].n_unique(),
        "exit_type_counts": panel["EXIT_TYPE"].value_counts().sort("EXIT_TYPE").to_dicts(),
        "failure_label_count": int(panel["FAIL_4Q"].sum()),
        "label_window": "FAIL_DT > REPDTE and FAIL_DT <= REPDTE + 12 months",
        "rows_dropped_per_rule": {
            "duplicate_CERT_REPDTE": dropped_duplicate_keys,
            "invalid_or_null_key": dropped_invalid_keys,
            "nonpositive_ASSET": dropped_nonpositive_asset,
            "dated_after_2025Q2": dropped_after_horizon,
            "dated_after_bank_exit": dropped_after_exit,
        },
        "raw_rows_minus_dropped_equals_clean_rows": (
            fin_raw.height
            - dropped_duplicate_keys
            - dropped_invalid_keys
            - dropped_nonpositive_asset
            - dropped_after_horizon
            - dropped_after_exit
            == panel.height
        ),
        "rows_in_four_quarters_before_failure": int(
            panel.filter(
                pl.col("FAIL_DT").is_not_null()
                & (pl.col("REPDTE") < pl.col("FAIL_DT"))
                & (pl.col("REPDTE").dt.offset_by("12mo") >= pl.col("FAIL_DT"))
            ).height
        ),
    }
    return panel, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data_raw/fdic"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/part2"))
    args = parser.parse_args()

    panel, report = build_panel(args.raw)
    args.output.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(args.output / "bank_quarter_panel.parquet", compression="zstd")
    report["panel"]["panel_rows"] = panel.height
    (args.output / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_quality_report_csv(report, args.output / "data_quality_report.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
