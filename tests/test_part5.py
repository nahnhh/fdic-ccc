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
)


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


if __name__ == "__main__":
    unittest.main()
