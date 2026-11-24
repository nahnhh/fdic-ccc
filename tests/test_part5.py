from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from part5_model_validation import (
    bank_bootstrap_average_precision,
    capacity_threshold,
    choose_cost_threshold,
    estimate_cost_parameters,
    event_cost_components,
    fit_sklearn_models,
    per_era_test_metrics,
)
from part4_features import FEATURE_NAMES


class Part5Test(unittest.TestCase):
    def test_cost_estimates_exclude_test_period(self):
        failures = pl.DataFrame({
            "CERT": [1, 2, 3],
            "FAIL_DT": [date(1990, 3, 31), date(2000, 3, 31), date(2010, 3, 31)],
            "COST": [100.0, 400.0, 100000.0],
            "QBFASSET": [1000.0, 1000.0, 1000.0],
        })
        parameters, _ = estimate_cost_parameters(failures)
        self.assertEqual(parameters["development_failure_events"], 2)
        self.assertEqual(parameters["median_missed_failure_cost"], 250.0)
        self.assertAlmostEqual(
            parameters["median_fdic_loss_to_qbfasset_ratio"], 0.25
        )
        self.assertFalse(parameters["test_data_used_for_cost_estimation"])

    def test_cost_threshold_uses_validation_costs_and_failure_events(self):
        validation = pd.DataFrame({
            "CERT": [1, 1, 2, 3, 4],
            "target": [1, 1, 1, 0, 0],
            "ASSET": [100.0, 100.0, 100.0, 100.0, 100.0],
            "event_cost": [100.0, 100.0, 50.0, 50.0, 50.0],
        })
        scores = np.array([0.4, 0.8, 0.1, 0.7, 0.05])

        threshold, details = choose_cost_threshold(
            validation,
            scores,
            loss_rate=10.0,
            false_alarm_multiplier=1.0,
        )

        self.assertEqual(threshold, 0.8)
        self.assertEqual(details["missed_failure_banks"], 1)
        self.assertEqual(details["missed_failure_cost"], 50.0)
        self.assertEqual(details["false_alarm_rows"], 0)
        self.assertEqual(details["total_estimated_cost"], 50.0)

        all_negative = event_cost_components(
            validation, scores, threshold=1.0, loss_rate=10.0
        )
        self.assertEqual(all_negative["missed_failure_banks"], 2)
        self.assertEqual(all_negative["missed_failure_cost"], 150.0)

    def test_review_cost_fraction_and_capacity_operating_point(self):
        validation = pd.DataFrame({
            "CERT": [1, 2, 3, 4, 5],
            "target": [1, 0, 0, 0, 0],
            "ASSET": [100.0] * 5,
            "event_cost": [100.0] * 5,
            "false_alarm_cost_unit": [100.0] * 5,
        })
        scores = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
        threshold = capacity_threshold(scores, alert_fraction=0.60)
        self.assertGreaterEqual(threshold, 0.95)
        metrics = event_cost_components(
            validation,
            scores,
            threshold=threshold,
            loss_rate=0.01,
            false_alarm_multiplier=1.0,
        )
        self.assertGreater(metrics["false_alarm_rows"], 0)

    def test_bank_bootstrap_interval(self):
        rows = pd.DataFrame({
            "CERT": [1, 1, 2, 2, 3, 3, 4, 4],
            "target": [1, 0, 1, 0, 0, 0, 0, 0],
        })
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = bank_bootstrap_average_precision(
            rows, scores, n_bootstrap=20, seed=7
        )
        self.assertGreater(result["valid_replicates"], 0)
        self.assertLessEqual(
            result["average_precision_ci_lower"],
            result["average_precision_ci_upper"],
        )

    def test_hist_gradient_boosting_and_per_era_metrics(self):
        quarter_ends = ((3, 31), (6, 30), (9, 30), (12, 31))
        rows = []
        for year in range(1984, 2007):
            for month, day in quarter_ends:
                for cert in range(1, 21):
                    target = int(cert % 5 == 0)
                    row = {
                        "CERT": cert,
                        "REPDTE": date(year, month, day),
                        "target": target,
                    }
                    for index, feature in enumerate(FEATURE_NAMES):
                        row[f"{feature}_lag1q"] = (
                            cert / 20 + target * 0.2 + index * 0.01
                        )
                    rows.append(row)
        for year in (2008, 2015):
            for cert in range(1, 21):
                target = int(cert % 5 == 0)
                row = {
                    "CERT": cert,
                    "REPDTE": date(year, 3, 31),
                    "target": target,
                }
                for index, feature in enumerate(FEATURE_NAMES):
                    row[f"{feature}_lag1q"] = (
                        cert / 20 + target * 0.2 + index * 0.01
                    )
                rows.append(row)
        frame = pl.DataFrame(rows)
        fit = frame.filter(pl.col("REPDTE") <= date(1987, 12, 31))
        validation = frame.filter(
            pl.col("REPDTE").is_between(date(1989, 3, 31), date(2006, 12, 31))
        )
        test = frame.filter(pl.col("REPDTE").is_in([
            date(2008, 3, 31),
            date(2015, 3, 31),
        ]))

        scores, coefficients, validation_mask = fit_sklearn_models(
            fit, validation, test
        )

        self.assertIn("camels_hist_gradient_boosting", scores)
        self.assertEqual(len(scores["camels_hist_gradient_boosting"][1]), test.height)
        self.assertTrue(np.isfinite(scores["camels_hist_gradient_boosting"][0]).all())
        self.assertIsNone(coefficients["camels_hist_gradient_boosting"])
        self.assertTrue(validation_mask.any())

        rows_for_metrics = pd.DataFrame({
            "CERT": [1, 2, 3, 4],
            "target": [1, 0, 1, 0],
            "ASSET": [100.0] * 4,
            "event_cost": [50.0] * 4,
            "false_alarm_cost_unit": [50.0] * 4,
        })
        era_metrics = per_era_test_metrics(
            rows_for_metrics,
            np.array([0.9, 0.1, 0.8, 0.2]),
            [date(2008, 3, 31), date(2008, 3, 31),
             date(2015, 3, 31), date(2015, 3, 31)],
            threshold=0.5,
            loss_rate=0.01,
        )
        self.assertEqual([row["era"] for row in era_metrics], ["2008-2014", "2015-2025"])
        self.assertEqual([row["rows"] for row in era_metrics], [2, 2])
        self.assertTrue(all(row["threshold"] == 0.5 for row in era_metrics))


if __name__ == "__main__":
    unittest.main()
