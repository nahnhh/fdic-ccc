from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "processed" / "part2" / "bank_quarter_panel.parquet"
RAW = ROOT / "data_raw" / "fdic"
OUTPUT = ROOT / "data" / "processed" / "part4"
RESULTS = ROOT / "results"

TRAIN_END = pd.Timestamp("1987-12-31")
VALIDATION_START = pd.Timestamp("1989-03-31")
VALIDATION_END = pd.Timestamp("2006-12-31")
TEST_START = pd.Timestamp("2008-03-31")

FEATURE_NAMES = [
    "equity_to_assets",
    "tier1_leverage",
    "npa_ratio",
    "net_chargeoff_rate",
    "reserve_coverage",
    "roa",
    "provision_to_assets",
    "liquid_asset_ratio",
    "brokered_dep_ratio",
    "core_deposit_ratio",
    "loan_to_assets",
    "re_concentration",
    "delta4_npa_ratio",
    "delta4_equity_ratio",
    "management_expense_ratio",
]

RAW_FEATURE_FIELDS = [
    "DEPDOM", "BRO", "CHBAL", "SC", "LNLSGR", "LNLSNET",
    "LNRE", "LNATRES", "NTLNLS", "P9ASSET", "NAASSET", "ORE",
    "ELNATR", "ROA", "RBC1AAJ", "NONIX",
]


def safe_ratio(numerator: str, denominator: str) -> pl.Expr:
    denominator_value = pl.col(denominator)
    return (
        pl.when(denominator_value.is_not_null() & denominator_value.is_finite() & (denominator_value > 0))
        .then(pl.col(numerator) / denominator_value)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def read_raw_camels(raw_dir: Path) -> pl.DataFrame:
    files = sorted((raw_dir / "financials").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No financial quarter files under {raw_dir / 'financials'}")

    fields = ["CERT", "REPDTE", "ASSET", "DEP", "EQ", *RAW_FEATURE_FIELDS]
    frames = []
    for path in files:
        frame = pl.read_parquet(path, columns=[field for field in fields if field in pl.read_parquet_schema(path)])
        frame = frame.with_columns(
            pl.col("CERT").cast(pl.Int64, strict=False),
            pl.col("REPDTE").cast(pl.String, strict=False).str.to_date("%Y%m%d", strict=False),
        )
        frames.append(frame)
    raw = pl.concat(frames, how="diagonal_relaxed")
    for field in ["ASSET", "DEP", "EQ", *RAW_FEATURE_FIELDS]:
        if field in raw.columns:
            raw = raw.with_columns(pl.col(field).cast(pl.Float64, strict=False))
    return raw


def build_camels_features(panel: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    raw_key = raw.select("CERT", "REPDTE", "ASSET", "DEP", "EQ", *RAW_FEATURE_FIELDS).unique(
        subset=["CERT", "REPDTE"], keep="first"
    )
    features = (
        panel.join(raw_key, on=["CERT", "REPDTE"], how="left", suffix="_raw")
        .sort(["CERT", "REPDTE"])
        .with_columns(
            safe_ratio("EQ", "ASSET").alias("equity_to_assets"),
            pl.col("RBC1AAJ").alias("tier1_leverage"),
            safe_ratio(
                "NAASSET", "ASSET"
            ).alias("_naasset_ratio"),
            safe_ratio(
                "P9ASSET", "ASSET"
            ).alias("_p9asset_ratio"),
            safe_ratio(
                "ORE", "ASSET"
            ).alias("_ore_ratio"),
            safe_ratio("NTLNLS", "LNLSGR").alias("net_chargeoff_rate"),
            safe_ratio("LNATRES", "NAASSET").alias("reserve_coverage"),
            pl.col("ROA").alias("roa"),
            safe_ratio("ELNATR", "ASSET").alias("provision_to_assets"),
            safe_ratio("CHBAL", "ASSET").alias("_cash_to_assets"),
            safe_ratio("SC", "ASSET").alias("_securities_to_assets"),
            safe_ratio("BRO", "DEP").alias("brokered_dep_ratio"),
            safe_ratio("DEPDOM", "DEP").alias("core_deposit_ratio"),
            safe_ratio("LNLSNET", "ASSET").alias("loan_to_assets"),
            safe_ratio("LNRE", "LNLSGR").alias("re_concentration"),
            safe_ratio("NONIX", "ASSET").alias("management_expense_ratio"),
        )
        .with_columns(
            (pl.col("_naasset_ratio") + pl.col("_p9asset_ratio") + pl.col("_ore_ratio")).alias("npa_ratio"),
            (pl.col("_cash_to_assets") + pl.col("_securities_to_assets")).alias("liquid_asset_ratio"),
        )
        .with_columns(
            pl.when(
                pl.col("REPDTE").shift(4).over("CERT")
                == pl.col("REPDTE").dt.offset_by("-12mo").dt.month_end()
            )
            .then(pl.col("npa_ratio") - pl.col("npa_ratio").shift(4).over("CERT"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("delta4_npa_ratio"),
            pl.when(
                pl.col("REPDTE").shift(4).over("CERT")
                == pl.col("REPDTE").dt.offset_by("-12mo").dt.month_end()
            )
            .then(pl.col("equity_to_assets") - pl.col("equity_to_assets").shift(4).over("CERT"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("delta4_equity_ratio"),
        )
        .sort(["REPDTE", "CERT"])
    )
    for name in FEATURE_NAMES:
        features = features.with_columns(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(pl.lit(None)).alias(name)
        )
    features = features.with_columns(
        pl.when(
            pl.col("REPDTE").shift(1).over("CERT")
            == pl.col("REPDTE").dt.offset_by("-3mo").dt.month_end()
        )
        .then(pl.col("REPDTE").shift(1).over("CERT"))
        .otherwise(pl.lit(None, dtype=pl.Date))
        .alias("feature_source_date_lag1q")
    )
    for name in FEATURE_NAMES:
        features = features.with_columns(
            pl.when(
                pl.col("feature_source_date_lag1q")
                == pl.col("REPDTE").dt.offset_by("-3mo").dt.month_end()
            )
            .then(pl.col(name).shift(1).over("CERT"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(f"{name}_lag1q")
        )
    return features


def make_temporal_splits(features: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    train = features.filter(pl.col("REPDTE") <= TRAIN_END)
    validation = features.filter(
        pl.col("REPDTE").is_between(VALIDATION_START, VALIDATION_END)
    )
    test = features.filter(
        (pl.col("REPDTE") >= TEST_START)
        & (pl.col("REPDTE") <= pl.lit(pd.Timestamp("2025-06-30")))
    )
    report = {
        "train": {"start": str(train["REPDTE"].min()), "end": str(train["REPDTE"].max()),
                  "rows": train.height, "failures": int(train["target"].sum())},
        "validation": {
            "start": str(validation["REPDTE"].min()),
            "end": str(validation["REPDTE"].max()),
            "rows": validation.height,
            "failures": int(validation["target"].sum()),
        },
        "test": {
            "start": str(test["REPDTE"].min()),
            "end": str(test["REPDTE"].max()),
            "rows": test.height,
            "failures": int(test["target"].sum()),
        },
        "gap_quarters": 4,
        "label_window": "Failure 0-12 months after the reporting quarter",
        "feature_horizon": "Lagged model uses only the exact prior quarter; delta4 requires an exact four-quarter comparison",
        "revision_control": (
            "The FDIC download is a latest-vintage snapshot; historical filing vintages and "
            "publication timestamps are not present, so revision leakage cannot be ruled out."
        ),
        "filing_lag_check": (
            "The primary specification uses values from exactly REPDTE-1 quarter; "
            "same-quarter values are reported as a validation-period sensitivity comparison."
        ),
        "lagged_feature_coverage": {
            "train_rows_with_exact_prior_quarter": int(train["feature_source_date_lag1q"].is_not_null().sum()),
            "validation_rows_with_exact_prior_quarter": int(validation["feature_source_date_lag1q"].is_not_null().sum()),
            "test_rows_with_exact_prior_quarter": int(test["feature_source_date_lag1q"].is_not_null().sum()),
        },
    }
    return train, validation, report


class TrainingQuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> TrainingQuantileClipper:
        values = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanquantile(values, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def fit_and_evaluate(
    train: pl.DataFrame,
    evaluation: pl.DataFrame,
    feature_names: list[str],
) -> tuple[dict, Pipeline]:
    X_train = train.select(feature_names).to_pandas()
    y_train = train["target"].to_numpy()
    X_evaluation = evaluation.select(feature_names).to_pandas()
    y_evaluation = evaluation["target"].to_numpy()

    pipe = Pipeline([
        ("outlier_clip", TrainingQuantileClipper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    probability = pipe.predict_proba(X_evaluation)[:, 1]
    prediction = (probability >= 0.5).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_evaluation, prediction, labels=[0, 1]).ravel()
    metrics = {
        "validation_rows": int(len(y_evaluation)),
        "validation_failures": int(y_evaluation.sum()),
        "roc_auc": float(roc_auc_score(y_evaluation, probability)),
        "average_precision": float(average_precision_score(y_evaluation, probability)),
        "brier_score": float(brier_score_loss(y_evaluation, probability)),
        "log_loss": float(log_loss(y_evaluation, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "coefficients": {},
        "training_preprocessing_fit": {
            "training_rows": int(len(y_train)),
            "training_failures": int(y_train.sum()),
            "fit_scope": "training observations only",
            "winsor_quantiles": [0.005, 0.995],
            "features": {},
        },
    }
    clipper = pipe.named_steps["outlier_clip"]
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    for index, (name, coefficient) in enumerate(zip(feature_names, pipe.named_steps["model"].coef_[0])):
        metrics["coefficients"][name] = float(coefficient)
        metrics["training_preprocessing_fit"]["features"][name] = {
            "lower_clip": float(clipper.lower_bounds_[index]),
            "upper_clip": float(clipper.upper_bounds_[index]),
            "imputation_median": float(imputer.statistics_[index]),
            "standardization_mean": float(scaler.mean_[index]),
            "standardization_scale": float(scaler.scale_[index]),
        }
    return metrics, pipe


def coefficient_direction_check(metrics: dict) -> dict:
    expected = {
        "net_chargeoff_rate": "positive",
        "core_deposit_ratio": "negative",
        "loan_to_assets": "positive",
        "re_concentration": "positive",
    }
    results = {}
    for name, expected_direction in expected.items():
        coefficient = metrics["coefficients"][f"{name}_lag1q"]
        observed_direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
        results[name] = {
            "expected_direction": expected_direction,
            "observed_direction": observed_direction,
            "coefficient": coefficient,
            "matches_expectation": observed_direction == expected_direction,
        }
    return results


def marginal_direction_diagnostics(train: pl.DataFrame) -> dict:
    expected = {
        "net_chargeoff_rate_lag1q": "positive",
        "core_deposit_ratio_lag1q": "negative",
        "loan_to_assets_lag1q": "positive",
        "re_concentration_lag1q": "positive",
    }
    results = {}
    for name, direction in expected.items():
        values = train.select(name, "target").drop_nulls().to_pandas()
        values = values[np.isfinite(values[name].to_numpy())]
        labels = values["target"].to_numpy()
        feature_values = values[name].to_numpy()
        auc = float(roc_auc_score(labels, feature_values))
        results[name.removesuffix("_lag1q")] = {
            "observations": int(len(values)),
            "median_nonfailure": float(np.median(feature_values[labels == 0])),
            "median_failure": float(np.median(feature_values[labels == 1])),
            "univariate_auc_in_expected_direction": auc if direction == "positive" else 1.0 - auc,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    panel = pl.read_parquet(args.panel)
    features = build_camels_features(panel, read_raw_camels(args.raw))
    args.output.mkdir(parents=True, exist_ok=True)
    features.write_parquet(args.output / "feature_matrix.parquet", compression="zstd")
    train, validation, split_report = make_temporal_splits(features)
    # The filing-lag sensitivity is now evaluated only on validation.
    contemporaneous_metrics, _ = fit_and_evaluate(train, validation, FEATURE_NAMES)
    lagged_names = [f"{name}_lag1q" for name in FEATURE_NAMES]
    lagged_metrics, _ = fit_and_evaluate(train, validation, lagged_names)
    report = {
        "features": split_report,
        "models": {
            "same_quarter_sensitivity": contemporaneous_metrics,
            "one_quarter_lag_primary": lagged_metrics,
        },
        "coefficient_prior_sign_check": coefficient_direction_check(lagged_metrics),
        "development_marginal_direction_diagnostics": marginal_direction_diagnostics(train),
        "coefficient_interpretation": (
            "Expected directions are economic hypotheses, not constraints. Logistic coefficients "
            "are conditional on all included features; correlated CAMELS ratios can therefore "
            "have signs that differ from their marginal risk association."
        ),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "part4_split_and_model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "same_quarter_coefficient": [
            contemporaneous_metrics["coefficients"][name] for name in FEATURE_NAMES
        ],
        "one_quarter_lag_coefficient": [
            lagged_metrics["coefficients"][f"{name}_lag1q"] for name in FEATURE_NAMES
        ],
    }).to_csv(args.results / "part4_coefficients.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
