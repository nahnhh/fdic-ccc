from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from part4_features import FEATURE_NAMES, TrainingQuantileClipper


ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "part4" / "feature_matrix.parquet"
FAILURES_PATH = ROOT / "data_raw" / "fdic" / "failures.parquet"
RESULTS_PATH = ROOT / "results"

FEATURE_COLUMNS = [f"{name}_lag1q" for name in FEATURE_NAMES]
FIT_END = date(1987, 12, 31)
VALIDATION_START = date(1989, 3, 31)
VALIDATION_END = date(2006, 12, 31)
TEST_START = date(2008, 3, 31)
TEST_END = date(2025, 6, 30)
DEVELOPMENT_COST_END = date(2007, 12, 31)
VALIDATION_FOLDS = (
    (date(1987, 12, 31), date(1989, 3, 31), date(1992, 12, 31)),
    (date(1992, 12, 31), date(1994, 3, 31), date(1997, 12, 31)),
    (date(1997, 12, 31), date(1999, 3, 31), date(2002, 12, 31)),
    (date(2002, 12, 31), date(2004, 3, 31), date(2006, 12, 31)),
)


def read_failure_events(path: Path) -> pl.DataFrame:
    failures = (
        pl.read_parquet(path)
        .with_columns(
            pl.col("CERT").cast(pl.Int64, strict=False),
            pl.col("FAILDATE").cast(pl.String, strict=False).str.to_date(
                "%m/%d/%Y", strict=False
            ).alias("FAIL_DT"),
            pl.col("COST").cast(pl.Float64, strict=False),
            pl.col("QBFASSET").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("CERT").is_not_null() & pl.col("FAIL_DT").is_not_null())
        .sort(["CERT", "FAIL_DT"])
        .group_by("CERT", maintain_order=True)
        .agg(
            pl.col("FAIL_DT").first(),
            pl.col("COST").first(),
            pl.col("QBFASSET").first(),
        )
    )
    return failures


def estimate_cost_parameters(failures: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    development_events = failures.filter(
        pl.col("FAIL_DT").is_between(date(1984, 1, 1), DEVELOPMENT_COST_END)
    )
    positive_costs = development_events.filter(pl.col("COST") > 0)
    valid_loss_rates = positive_costs.filter(pl.col("QBFASSET") > 0).with_columns(
        (pl.col("COST") / pl.col("QBFASSET")).alias("loss_to_assets")
    )
    if positive_costs.is_empty() or valid_loss_rates.is_empty():
        raise ValueError("Development-period failures lack positive COST/QBFASSET data")

    median_failure_cost = float(positive_costs["COST"].median())
    median_loss_rate = float(valid_loss_rates["loss_to_assets"].median())
    parameters = {
        "currency_units": "thousands of USD",
        "development_failure_events": development_events.height,
        "positive_cost_events": positive_costs.height,
        "events_with_positive_qbfasset": valid_loss_rates.height,
        "median_missed_failure_cost": median_failure_cost,
        "median_fdic_loss_to_qbfasset_ratio": median_loss_rate,
        "base_false_alarm_fraction_of_median_failure_cost": 0.01,
        "false_alarm_cost_proxy": (
            "Each false alert is a review cost equal to a specified fraction "
            "(default 1%) of the development-period median failed-bank COST. "
            "It is not proportional to reported bank assets."
        ),
        "missed_failure_cost_unit": (
            "One FDIC COST per failed institution if no alert occurs in any positive "
            "12-month label row."
        ),
        "missing_or_nonpositive_failure_cost": (
            "Imputed to the development-period median positive COST."
        ),
        "test_data_used_for_cost_estimation": False,
    }
    return parameters, development_events


def temporal_partitions(
    features: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    fit = features.filter(pl.col("REPDTE") <= FIT_END)
    validation = features.filter(
        pl.col("REPDTE").is_between(VALIDATION_START, VALIDATION_END)
    )
    test = features.filter(pl.col("REPDTE").is_between(TEST_START, TEST_END))
    report = {
        "fit": {
            "start": str(fit["REPDTE"].min()),
            "end": str(fit["REPDTE"].max()),
            "rows": fit.height,
            "failures": int(fit["target"].sum()),
        },
        "fit_to_validation_gap_quarters": 4,
        "validation": {
            "start": str(validation["REPDTE"].min()),
            "end": str(validation["REPDTE"].max()),
            "rows": validation.height,
            "failures": int(validation["target"].sum()),
        },
        "validation_to_test_gap_quarters": 4,
        "test": {
            "start": str(test["REPDTE"].min()),
            "end": str(test["REPDTE"].max()),
            "rows": test.height,
            "failures": int(test["target"].sum()),
        },
        "test_used_for_threshold_or_model_selection": False,
        "test_evaluation": "One final evaluation after model and thresholds were fixed on validation.",
        "threshold_validation_method": (
            "Four expanding-window folds; each fold trains on all earlier available "
            "development rows and evaluates after a four-quarter gap."
        ),
        "expanding_validation_folds": [
            {
                "training_end": str(training_end),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
            }
            for training_end, validation_start, validation_end in VALIDATION_FOLDS
        ],
    }
    return fit, validation, test, report


def attach_event_costs(
    frame: pl.DataFrame,
    failures: pl.DataFrame,
    median_failure_cost: float,
) -> pd.DataFrame:
    cost_lookup = failures.select(
        "CERT", pl.col("COST").alias("event_cost")
    )
    return (
        frame.join(cost_lookup, on="CERT", how="left")
        .select("CERT", "target", "ASSET", "event_cost")
        .to_pandas()
    )


def normalized_cost_rows(frame: pd.DataFrame, median_failure_cost: float) -> pd.DataFrame:
    rows = frame.copy()
    rows["target"] = rows["target"].astype(np.uint8)
    rows["CERT"] = rows["CERT"].astype(np.int64)
    rows["ASSET"] = pd.to_numeric(rows["ASSET"], errors="coerce")
    rows["event_cost"] = pd.to_numeric(rows["event_cost"], errors="coerce")
    invalid_cost = rows["event_cost"].isna() | (rows["event_cost"] <= 0)
    rows.loc[invalid_cost, "event_cost"] = median_failure_cost
    rows["false_alarm_cost_unit"] = float(median_failure_cost)
    if rows["ASSET"].isna().any() or not np.isfinite(rows["ASSET"]).all():
        raise ValueError("ASSET must be finite for false-alarm cost calculation")
    return rows


def event_cost_components(
    rows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    loss_rate: float,
    false_alarm_multiplier: float = 1.0,
) -> dict:
    predicted = scores >= threshold
    y = rows["target"].to_numpy(dtype=np.uint8)
    false_alarm_mask = predicted & (y == 0)
    if "false_alarm_cost_unit" in rows:
        unit_cost = rows.loc[false_alarm_mask, "false_alarm_cost_unit"].to_numpy(dtype=np.float64)
        false_alarm_cost = float((unit_cost * loss_rate * false_alarm_multiplier).sum())
    else:
        false_alarm_cost = float(
            (rows.loc[false_alarm_mask, "ASSET"].to_numpy(dtype=np.float64)
             * loss_rate * false_alarm_multiplier).sum()
        )

    positive = rows.loc[y == 1, ["CERT", "event_cost"]].drop_duplicates("CERT")
    detected_ids = set(rows.loc[predicted & (y == 1), "CERT"].tolist())
    missed = positive.loc[~positive["CERT"].isin(detected_ids)]
    missed_failure_cost = float(missed["event_cost"].sum())
    return {
        "false_alarm_rows": int(false_alarm_mask.sum()),
        "false_alarm_cost": false_alarm_cost,
        "missed_failure_banks": int(len(missed)),
        "missed_failure_cost": missed_failure_cost,
        "total_estimated_cost": false_alarm_cost + missed_failure_cost,
    }


def choose_cost_threshold(
    rows: pd.DataFrame,
    scores: np.ndarray,
    loss_rate: float,
    false_alarm_multiplier: float = 1.0,
) -> tuple[float, dict]:
    y = rows["target"].to_numpy(dtype=np.uint8)
    negative = y == 0
    positive = y == 1
    if not negative.any() or not positive.any():
        raise ValueError("Threshold validation requires both classes")

    negative_scores = scores[negative]
    if "false_alarm_cost_unit" in rows:
        negative_costs = (
            rows.loc[negative, "false_alarm_cost_unit"].to_numpy(dtype=np.float64)
            * loss_rate
            * false_alarm_multiplier
        )
    else:
        negative_costs = (
            rows.loc[negative, "ASSET"].to_numpy(dtype=np.float64)
            * loss_rate
            * false_alarm_multiplier
        )
    negative_order = np.argsort(negative_scores)
    negative_scores = negative_scores[negative_order]
    negative_cumulative = np.cumsum(negative_costs[negative_order])

    event_rows = rows.loc[positive, ["CERT", "event_cost"]].copy()
    event_rows["score"] = scores[positive]
    events = event_rows.groupby("CERT", sort=False).agg(
        max_score=("score", "max"),
        event_cost=("event_cost", "first"),
    )
    event_order = np.argsort(events["max_score"].to_numpy())
    event_scores = events["max_score"].to_numpy()[event_order]
    event_costs = events["event_cost"].to_numpy(dtype=np.float64)[event_order]
    event_cumulative = np.cumsum(event_costs)

    candidates = np.unique(np.concatenate((
        [np.nextafter(float(scores.min()), -np.inf)],
        scores,
        [np.nextafter(float(scores.max()), np.inf)],
    )))
    negative_indices = np.searchsorted(negative_scores, candidates, side="left")
    false_alarm_costs = np.full(len(candidates), negative_cumulative[-1])
    has_alerts = negative_indices > 0
    false_alarm_costs[has_alerts] = (
        negative_cumulative[-1] - negative_cumulative[negative_indices[has_alerts] - 1]
    )
    event_indices = np.searchsorted(event_scores, candidates, side="left")
    missed_costs = np.zeros(len(candidates))
    missed_events = event_indices > 0
    missed_costs[missed_events] = event_cumulative[event_indices[missed_events] - 1]
    total_costs = false_alarm_costs + missed_costs

    best_cost = float(total_costs.min())
    best_indices = np.flatnonzero(total_costs == best_cost)
    best_index = int(best_indices[-1])
    threshold = float(candidates[best_index])
    details = event_cost_components(rows, scores, threshold, loss_rate, false_alarm_multiplier)
    details.update({
        "validation_threshold": threshold,
        "validation_cost_candidates": int(len(candidates)),
        "false_alarm_cost_multiplier": false_alarm_multiplier,
    })
    return threshold, details


def evaluate_predictions(
    rows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    loss_rate: float,
) -> dict:
    y = rows["target"].to_numpy(dtype=np.uint8)
    predicted = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    positive = rows.loc[y == 1, ["CERT"]].drop_duplicates()
    detected = set(rows.loc[(predicted == 1) & (y == 1), "CERT"].tolist())
    metrics = {
        "rows": int(len(y)),
        "failures": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "balanced_accuracy": float(
            (tp / (tp + fn) + tn / (tn + fp)) / 2
        ) if tp + fn and tn + fp else 0.0,
        "alert_rate": float(predicted.mean()),
        "failure_banks": int(len(positive)),
        "detected_failure_banks": int(len(detected)),
        "failure_bank_recall": float(len(detected) / len(positive)) if len(positive) else 0.0,
    }
    metrics["estimated_cost"] = event_cost_components(rows, scores, threshold, loss_rate)
    return metrics


def model_cost_sensitivity(
    rows: pd.DataFrame,
    scores: np.ndarray,
    base_false_alarm_fraction: float,
) -> list[dict]:
    sensitivity = []
    for false_alarm_fraction in (0.001, 0.01, 0.1, 1.0):
        threshold, details = choose_cost_threshold(
            rows,
            scores,
            false_alarm_fraction,
            false_alarm_multiplier=1.0,
        )
        sensitivity.append({
            "false_alarm_fraction_of_median_failure_cost": false_alarm_fraction,
            "false_alarm_cost_multiplier": false_alarm_fraction / base_false_alarm_fraction,
            "threshold": threshold,
            "validation_total_estimated_cost": details["total_estimated_cost"],
            "validation_false_alarm_rows": details["false_alarm_rows"],
            "validation_missed_failure_banks": details["missed_failure_banks"],
        })
    return sensitivity


def capacity_threshold(scores: np.ndarray, alert_fraction: float = 0.01) -> float:
    quantile = 1.0 - alert_fraction
    return float(np.quantile(scores, quantile, method="higher"))


def bank_bootstrap_average_precision(
    rows: pd.DataFrame,
    scores: np.ndarray,
    n_bootstrap: int = 300,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    bank_codes, unique_banks = pd.factorize(rows["CERT"], sort=False)
    estimates = []
    y = rows["target"].to_numpy(dtype=np.uint8)
    bank_count = len(unique_banks)
    for _ in range(n_bootstrap):
        counts = np.bincount(
            rng.integers(0, bank_count, size=bank_count),
            minlength=bank_count,
        )
        weights = counts[bank_codes]
        sampled = weights > 0
        sampled_y = y[sampled]
        if sampled_y.min() == 0 and sampled_y.max() == 1:
            estimates.append(float(average_precision_score(
                sampled_y,
                scores[sampled],
                sample_weight=weights[sampled],
            )))
    estimates = np.asarray(estimates, dtype=np.float64)
    if not len(estimates):
        return {"n_bootstrap": n_bootstrap, "valid_replicates": 0}
    return {
        "unit": "cluster bootstrap by CERT",
        "n_bootstrap": n_bootstrap,
        "valid_replicates": int(len(estimates)),
        "average_precision_mean": float(estimates.mean()),
        "average_precision_ci_lower": float(np.quantile(estimates, 0.025)),
        "average_precision_ci_upper": float(np.quantile(estimates, 0.975)),
    }


def per_era_test_metrics(
    rows: pd.DataFrame,
    scores: np.ndarray,
    report_dates: list[date],
    threshold: float,
    loss_rate: float,
) -> list[dict]:
    eras = (
        ("2008-2014", 2008, 2014),
        ("2015-2025", 2015, 2025),
    )
    results = []
    years = np.asarray([report_date.year for report_date in report_dates])
    for era, start_year, end_year in eras:
        mask = (years >= start_year) & (years <= end_year)
        if not mask.any():
            continue
        era_rows = rows.loc[mask].reset_index(drop=True)
        era_scores = scores[mask]
        metrics = evaluate_predictions(era_rows, era_scores, threshold, loss_rate)
        results.append({"era": era, **metrics})
    return results


def fit_sklearn_models(
    fit: pl.DataFrame,
    validation: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict, np.ndarray]:
    development = pl.concat([fit, validation], how="vertical").sort("REPDTE")
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coefficients: dict[str, dict[str, float] | None] = {}
    validation_dates = validation["REPDTE"].to_numpy()
    scored_validation = np.zeros(validation.height, dtype=bool)
    npa_feature = ["npa_ratio_lag1q"]
    models = {
        "npa_logistic_baseline": npa_feature,
        "camels_logistic": FEATURE_COLUMNS,
    }
    oof_scores = {
        name: np.full(validation.height, np.nan, dtype=np.float64)
        for name in (
            "prevalence_baseline",
            *models,
            "camels_hist_gradient_boosting",
        )
    }

    def new_pipeline() -> Pipeline:
        return Pipeline([
            ("outlier_clip", TrainingQuantileClipper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs", class_weight=None)),
        ])

    def new_tree_pipeline() -> Pipeline:
        return Pipeline([
            ("outlier_clip", TrainingQuantileClipper()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.1,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ])

    for training_end, validation_start, validation_end in VALIDATION_FOLDS:
        fold_fit = development.filter(pl.col("REPDTE") <= training_end)
        fold_validation = validation.filter(
            pl.col("REPDTE").is_between(validation_start, validation_end)
        )
        fold_mask = np.isin(validation_dates, fold_validation["REPDTE"].to_numpy())
        if not fold_fit.height or not fold_validation.height:
            raise ValueError(f"Empty expanding validation fold starting {validation_start}")
        scored_validation |= fold_mask

        prior = DummyClassifier(strategy="prior")
        prior.fit(np.zeros((fold_fit.height, 1)), fold_fit["target"].to_numpy())
        oof_scores["prevalence_baseline"][fold_mask] = prior.predict_proba(
            np.zeros((int(fold_mask.sum()), 1))
        )[:, 1]
        for name, columns in models.items():
            pipeline = new_pipeline()
            pipeline.fit(
                fold_fit.select(columns).to_pandas(),
                fold_fit["target"].to_numpy(),
            )
            oof_scores[name][fold_mask] = pipeline.predict_proba(
                fold_validation.select(columns).to_pandas()
            )[:, 1]
        tree_pipeline = new_tree_pipeline()
        tree_pipeline.fit(
            fold_fit.select(FEATURE_COLUMNS).to_pandas(),
            fold_fit["target"].to_numpy(),
        )
        oof_scores["camels_hist_gradient_boosting"][fold_mask] = (
            tree_pipeline.predict_proba(
                fold_validation.select(FEATURE_COLUMNS).to_pandas()
            )[:, 1]
        )

    if not scored_validation.any():
        raise ValueError("Expanding validation folds produced no out-of-fold scores")

    final_fit = development.filter(pl.col("REPDTE") <= VALIDATION_END)
    y_final = final_fit["target"].to_numpy()
    for name, columns in models.items():
        final_pipeline = new_pipeline()
        final_pipeline.fit(final_fit.select(columns).to_pandas(), y_final)
        test_scores = final_pipeline.predict_proba(test.select(columns).to_pandas())[:, 1]
        results[name] = (oof_scores[name][scored_validation], test_scores)
        coefficients[name] = (
            {
                column: float(coefficient)
                for column, coefficient in zip(
                    columns, final_pipeline.named_steps["model"].coef_[0]
                )
            }
            if name == "camels_logistic"
            else None
        )

    final_tree_pipeline = new_tree_pipeline()
    final_tree_pipeline.fit(
        final_fit.select(FEATURE_COLUMNS).to_pandas(),
        y_final,
    )
    tree_test_scores = final_tree_pipeline.predict_proba(
        test.select(FEATURE_COLUMNS).to_pandas()
    )[:, 1]
    results["camels_hist_gradient_boosting"] = (
        oof_scores["camels_hist_gradient_boosting"][scored_validation],
        tree_test_scores,
    )
    coefficients["camels_hist_gradient_boosting"] = None

    final_prior = DummyClassifier(strategy="prior")
    final_prior.fit(np.zeros((final_fit.height, 1)), y_final)
    results["prevalence_baseline"] = (
        oof_scores["prevalence_baseline"][scored_validation],
        final_prior.predict_proba(np.zeros((test.height, 1)))[:, 1],
    )
    coefficients["prevalence_baseline"] = None
    results = {
        name: results[name]
        for name in (
            "prevalence_baseline",
            "npa_logistic_baseline",
            "camels_logistic",
            "camels_hist_gradient_boosting",
        )
    }
    return results, coefficients, scored_validation


def run_spark_mllib(
    features_path: Path,
    failures: pl.DataFrame,
    development_events: pl.DataFrame,
    loss_rate: float,
) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
    from pyspark.ml.feature import Imputer, StandardScaler as SparkStandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession, functions as F

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("FDIC-Part5-MLlib-Validation")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.caseSensitive", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        development_failure_lookup = spark.createDataFrame(
            development_events.select(
                "CERT", pl.col("COST").alias("event_cost")
            ).to_pandas()
        )
        all_failure_lookup = spark.createDataFrame(
            failures.select("CERT", pl.col("COST").alias("event_cost")).to_pandas()
        )
        base = (
            spark.read.parquet(str(features_path))
            .filter(F.col("REPDTE") <= F.lit(TEST_END).cast("date"))
            .select(
                "CERT", "REPDTE", "ASSET",
                F.col("target").cast("double").alias("target"),
                *FEATURE_COLUMNS,
            )
        )
        fit = base.filter(F.col("REPDTE") <= F.lit(FIT_END).cast("date"))
        validation = base.filter(
            F.col("REPDTE").between(
                F.lit(VALIDATION_START).cast("date"),
                F.lit(VALIDATION_END).cast("date"),
            )
        ).join(development_failure_lookup, on="CERT", how="left")
        test = base.filter(
            F.col("REPDTE").between(
                F.lit(TEST_START).cast("date"),
                F.lit(TEST_END).cast("date"),
            )
        ).join(all_failure_lookup, on="CERT", how="left")
        fit_rows = fit.count()
        validation_rows = validation.count()
        test_rows_count = test.count()
        bounds = fit.stat.approxQuantile(FEATURE_COLUMNS, [0.005, 0.995], 0.001)
        fitted_bounds = {
            name: (float(values[0]), float(values[1]))
            for name, values in zip(FEATURE_COLUMNS, bounds)
        }

        def clip_to_bounds(frame, bounds):
            for feature, (lower, upper) in bounds.items():
                value = F.col(feature)
                frame = frame.withColumn(
                    feature,
                    F.when(value.isNull(), value)
                    .when(value < F.lit(lower), F.lit(lower))
                    .when(value > F.lit(upper), F.lit(upper))
                    .otherwise(value),
                )
            return frame

        fit = clip_to_bounds(fit, fitted_bounds)
        validation = clip_to_bounds(validation, fitted_bounds)

        imputed_columns = [f"{feature}_imputed" for feature in FEATURE_COLUMNS]
        stages = [
            Imputer(
                strategy="median",
                inputCols=FEATURE_COLUMNS,
                outputCols=imputed_columns,
            ),
            VectorAssembler(
                inputCols=imputed_columns,
                outputCol="assembled_features",
                handleInvalid="error",
            ),
            SparkStandardScaler(
                inputCol="assembled_features",
                outputCol="features",
                withMean=True,
                withStd=True,
            ),
            SparkLogisticRegression(
                featuresCol="features",
                labelCol="target",
                maxIter=100,
                regParam=0.0,
                elasticNetParam=0.0,
                standardization=False,
            ),
        ]
        pipeline = SparkPipeline(stages=stages)
        validation_model = pipeline.fit(fit)

        def collect_scores(frame, fitted_model) -> tuple[pd.DataFrame, np.ndarray]:
            predictions = (
                fitted_model.transform(frame)
                .select(
                    "CERT",
                    "target",
                    "ASSET",
                    "event_cost",
                    vector_to_array(F.col("probability"))[1].alias("score"),
                )
                .toPandas()
            )
            scores = predictions.pop("score").to_numpy(dtype=np.float64)
            return predictions, scores

        validation_data, validation_scores = collect_scores(validation, validation_model)
        final_fit = base.filter(
            (F.col("REPDTE") <= F.lit(FIT_END).cast("date"))
            | F.col("REPDTE").between(
                F.lit(VALIDATION_START).cast("date"),
                F.lit(VALIDATION_END).cast("date"),
            )
        )
        final_fit_rows = final_fit.count()
        final_bounds_values = final_fit.stat.approxQuantile(
            FEATURE_COLUMNS, [0.005, 0.995], 0.001
        )
        final_bounds = {
            name: (float(values[0]), float(values[1]))
            for name, values in zip(FEATURE_COLUMNS, final_bounds_values)
        }
        final_fit = clip_to_bounds(final_fit, final_bounds)
        final_test = clip_to_bounds(test, final_bounds)
        final_model = pipeline.fit(final_fit)
        test_data, test_scores = collect_scores(final_test, final_model)
        demo = {
            "engine": "Spark MLlib LogisticRegression",
            "spark_version": spark.version,
            "master": "local[2]",
            "fit_rows": fit_rows,
            "final_fit_rows": final_fit_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows_count,
            "training_only_preprocessing": [
                "0.5th/99.5th percentile clipping bounds",
                "median imputation",
                "standardization",
            ],
            "pipeline_stages": [
                stage.__class__.__name__ for stage in pipeline.getStages()
            ],
            "threshold_source": "fixed validation window only",
            "false_alarm_fraction_used": loss_rate,
        }
        return demo, validation_data, validation_scores, test_data, test_scores
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--failures", type=Path, default=FAILURES_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--skip-spark", action="store_true")
    args = parser.parse_args()

    features = pl.read_parquet(args.features)
    failures = read_failure_events(args.failures)
    cost_parameters, development_events = estimate_cost_parameters(failures)
    median_failure_cost = cost_parameters["median_missed_failure_cost"]
    false_alarm_fraction = cost_parameters["base_false_alarm_fraction_of_median_failure_cost"]
    fit, validation, test, split_report = temporal_partitions(features)

    validation_cost_rows = normalized_cost_rows(
        attach_event_costs(validation, development_events, median_failure_cost),
        median_failure_cost,
    )
    test_cost_rows = normalized_cost_rows(
        attach_event_costs(test, failures, median_failure_cost),
        median_failure_cost,
    )

    score_sets, coefficients, validation_score_mask = fit_sklearn_models(
        fit, validation, test
    )
    validation_cost_rows = validation_cost_rows.loc[
        validation_score_mask
    ].reset_index(drop=True)
    split_report["threshold_validation_oof_rows"] = int(validation_score_mask.sum())
    model_results = {}
    sensitivity_rows = []
    era_metric_rows = []
    test_report_dates = test["REPDTE"].to_list()
    for name, (validation_scores, test_scores) in score_sets.items():
        threshold, validation_cost = choose_cost_threshold(
            validation_cost_rows,
            validation_scores,
            false_alarm_fraction,
            false_alarm_multiplier=1.0,
        )
        top_one_percent_threshold = capacity_threshold(validation_scores, 0.01)
        test_metrics = evaluate_predictions(
            test_cost_rows, test_scores, threshold, false_alarm_fraction
        )
        if name == "camels_logistic":
            test_metrics["bank_bootstrap_average_precision"] = (
                bank_bootstrap_average_precision(test_cost_rows, test_scores)
            )
        test_era_metrics = per_era_test_metrics(
            test_cost_rows,
            test_scores,
            test_report_dates,
            threshold,
            false_alarm_fraction,
        )
        if name == "camels_hist_gradient_boosting":
            model_family = "HistGradientBoostingClassifier"
            model_parameters = {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 42,
            }
            training_only_preprocessing = [
                "0.5th/99.5th percentile clipping",
                "median imputation",
            ]
        elif name == "prevalence_baseline":
            model_family = "DummyClassifier"
            model_parameters = None
            training_only_preprocessing = []
        else:
            model_family = "LogisticRegression"
            model_parameters = None
            training_only_preprocessing = [
                "0.5th/99.5th percentile clipping",
                "median imputation",
                "standardization",
            ]
        model_results[name] = {
            "validation": evaluate_predictions(
                validation_cost_rows,
                validation_scores,
                threshold,
                false_alarm_fraction,
            ),
            "test": test_metrics,
            "test_by_era": test_era_metrics,
            "threshold_selection": validation_cost,
            "threshold_cost_sensitivity": model_cost_sensitivity(
                validation_cost_rows,
                validation_scores,
                false_alarm_fraction,
            ),
            "capacity_operating_point": {
                "alert_fraction": 0.01,
                "validation_threshold": top_one_percent_threshold,
                "validation": evaluate_predictions(
                    validation_cost_rows,
                    validation_scores,
                    top_one_percent_threshold,
                    false_alarm_fraction,
                ),
                "test": evaluate_predictions(
                    test_cost_rows,
                    test_scores,
                    top_one_percent_threshold,
                    false_alarm_fraction,
                ),
            },
            "final_model_coefficients": coefficients[name],
            "model_family": model_family,
            "model_parameters": model_parameters,
            "training_only_preprocessing": training_only_preprocessing,
            "final_refit": "Fit on 1984Q1–2006Q4 after validation threshold selection; 2007 is excluded as the gap.",
        }
        for era_metrics in test_era_metrics:
            era_metric_rows.append({
                "model": name,
                "era": era_metrics["era"],
                "rows": era_metrics["rows"],
                "failures": era_metrics["failures"],
                "prevalence": era_metrics["prevalence"],
                "average_precision": era_metrics["average_precision"],
                "roc_auc": era_metrics["roc_auc"],
                "brier_score": era_metrics["brier_score"],
                "log_loss": era_metrics["log_loss"],
                "validation_selected_threshold": threshold,
                "precision": era_metrics["precision"],
                "recall": era_metrics["recall"],
                "failure_bank_recall": era_metrics["failure_bank_recall"],
                "alert_rate": era_metrics["alert_rate"],
            })
        for row in model_results[name]["threshold_cost_sensitivity"]:
            sensitivity_rows.append({"model": name, **row})

    spark_demo = None
    if not args.skip_spark:
        (
            spark_demo,
            spark_validation_data,
            spark_validation_scores,
            spark_test_data,
            spark_test_scores,
        ) = run_spark_mllib(
            args.features,
            failures,
            development_events,
            false_alarm_fraction,
        )
        spark_validation_rows = normalized_cost_rows(
            spark_validation_data, median_failure_cost
        )
        spark_test_rows = normalized_cost_rows(
            spark_test_data, median_failure_cost
        )
        spark_threshold, spark_validation_cost = choose_cost_threshold(
            spark_validation_rows,
            spark_validation_scores,
            false_alarm_fraction,
        )
        spark_test_metrics = evaluate_predictions(
            spark_test_rows,
            spark_test_scores,
            spark_threshold,
            false_alarm_fraction,
        )
        spark_test_era_metrics = per_era_test_metrics(
            spark_test_rows,
            spark_test_scores,
            test_report_dates,
            spark_threshold,
            false_alarm_fraction,
        )
        spark_top_one_percent = capacity_threshold(spark_validation_scores, 0.01)
        model_results["spark_mllib_logistic"] = {
            "validation": evaluate_predictions(
                spark_validation_rows,
                spark_validation_scores,
                spark_threshold,
                false_alarm_fraction,
            ),
            "test": spark_test_metrics,
            "test_by_era": spark_test_era_metrics,
            "threshold_selection": spark_validation_cost,
            "threshold_cost_sensitivity": model_cost_sensitivity(
                spark_validation_rows,
                spark_validation_scores,
                false_alarm_fraction,
            ),
            "capacity_operating_point": {
                "alert_fraction": 0.01,
                "validation_threshold": spark_top_one_percent,
                "validation": evaluate_predictions(
                    spark_validation_rows,
                    spark_validation_scores,
                    spark_top_one_percent,
                    false_alarm_fraction,
                ),
                "test": evaluate_predictions(
                    spark_test_rows,
                    spark_test_scores,
                    spark_top_one_percent,
                    false_alarm_fraction,
                ),
            },
            "final_model_coefficients": None,
            "model_family": "Spark MLlib LogisticRegression",
            "model_parameters": None,
            "training_only_preprocessing": [
                "0.5th/99.5th percentile clipping bounds",
                "median imputation",
                "standardization",
            ],
            "final_refit": (
                "Validation threshold is selected from the initial 1984Q1–1987Q4 fit; "
                "the Spark pipeline is then refit on all eligible 1984Q1–2006Q4 "
                "development rows before test scoring."
            ),
        }
        for era_metrics in spark_test_era_metrics:
            era_metric_rows.append({
                "model": "spark_mllib_logistic",
                "era": era_metrics["era"],
                "rows": era_metrics["rows"],
                "failures": era_metrics["failures"],
                "prevalence": era_metrics["prevalence"],
                "average_precision": era_metrics["average_precision"],
                "roc_auc": era_metrics["roc_auc"],
                "brier_score": era_metrics["brier_score"],
                "log_loss": era_metrics["log_loss"],
                "validation_selected_threshold": spark_threshold,
                "precision": era_metrics["precision"],
                "recall": era_metrics["recall"],
                "failure_bank_recall": era_metrics["failure_bank_recall"],
                "alert_rate": era_metrics["alert_rate"],
            })
        for row in model_results["spark_mllib_logistic"]["threshold_cost_sensitivity"]:
            sensitivity_rows.append({"model": "spark_mllib_logistic", **row})

    report = {
        "validation_scheme": split_report,
        "cost_estimates": cost_parameters,
        "metrics_justification": {
            "primary": "Average precision/PR-AUC, because failures are rare.",
            "validation": "Out-of-fold predictions from four expanding-window folds are used for threshold selection; the final scikit-learn models are refit on all eligible development rows before the single test scoring.",
            "secondary": [
                "ROC-AUC for ranking across thresholds",
                "Brier score and log loss for probability quality",
                "precision, recall, specificity, and bank-level failure recall at the cost-selected threshold",
                "A bank-clustered bootstrap confidence interval for test average precision",
                "A validation-selected top-1% capacity threshold as an operational alternative",
                "Test average precision and threshold metrics by era, using the unchanged validation-selected threshold",
            ],
        },
        "models": model_results,
        "spark_mllib_demo": spark_demo,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "part5_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    metric_rows = []
    for model_name, model_report in model_results.items():
        for period in ("validation", "test"):
            metrics = model_report[period]
            metric_rows.append({
                "model": model_name,
                "period": period,
                "threshold": metrics["threshold"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "failure_bank_recall": metrics["failure_bank_recall"],
                "alert_rate": metrics["alert_rate"],
                "false_alarm_rows": metrics["estimated_cost"]["false_alarm_rows"],
                "missed_failure_banks": metrics["estimated_cost"]["missed_failure_banks"],
                "estimated_cost_thousands_usd": metrics["estimated_cost"]["total_estimated_cost"],
            })
    pd.DataFrame(metric_rows).to_csv(
        args.results / "part5_model_metrics.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        args.results / "part5_threshold_sensitivity.csv", index=False
    )
    pd.DataFrame(era_metric_rows).to_csv(
        args.results / "part5_test_per_era_metrics.csv", index=False
    )
    coefficient_rows = [
        {
            "model": model_name,
            "feature": feature,
            "standardized_coefficient": coefficient,
        }
        for model_name, model_coefficients in coefficients.items()
        if model_coefficients
        for feature, coefficient in model_coefficients.items()
    ]
    pd.DataFrame(coefficient_rows).to_csv(
        args.results / "part5_final_model_coefficients.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

