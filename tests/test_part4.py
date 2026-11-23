from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from part4_features import (
    FEATURE_NAMES,
    TrainingQuantileClipper,
    build_camels_features,
    make_temporal_splits,
    safe_ratio,
)


class Part4Test(unittest.TestCase):
    def test_feature_definitions_and_temporal_split(self):
        panel = pl.DataFrame({
            "CERT": [1, 1, 1, 1, 1, 1],
            "REPDTE": [
                date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30),
                date(2023, 12, 31), date(2024, 3, 31), date(2024, 6, 30),
            ],
            "target": [0, 0, 0, 0, 1, 0],
        })
        raw = pl.DataFrame({
            "CERT": [1] * 6,
            "REPDTE": panel["REPDTE"].to_list(),
            "ASSET": [100.0] * 6,
            "DEP": [80.0] * 6,
            "EQ": [10.0] * 6,
            "DEPDOM": [70.0] * 6,
            "BRO": [10.0] * 6,
            "CHBAL": [20.0] * 6,
            "SC": [5.0] * 6,
            "LNLSGR": [60.0] * 6,
            "LNLSNET": [55.0] * 6,
            "LNRE": [30.0] * 6,
            "LNATRES": [5.0] * 6,
            "NTLNLS": [1.0] * 6,
            "P9ASSET": [2.0] * 6,
            "NAASSET": [3.0] * 6,
            "ORE": [1.0] * 6,
            "ELNATR": [1.0] * 6,
            "ROA": [1.0] * 6,
            "RBC1AAJ": [12.0] * 6,
            "NONIX": [2.0] * 6,
        })

        features = build_camels_features(panel, raw)
        self.assertTrue(set(FEATURE_NAMES).issubset(features.columns))
        self.assertEqual(features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "equity_to_assets"], 0.1)
        self.assertAlmostEqual(
            features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "npa_ratio"],
            0.06,
        )
        self.assertAlmostEqual(
            features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "net_chargeoff_rate"],
            1.0 / 60.0,
        )
        self.assertAlmostEqual(
            features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "management_expense_ratio"],
            0.02,
        )
        self.assertAlmostEqual(
            features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "equity_to_assets_lag1q"],
            0.1,
        )
        self.assertEqual(
            features.filter(pl.col("REPDTE") == date(2024, 6, 30))[0, "feature_source_date_lag1q"],
            date(2024, 3, 31),
        )

        train, test, report = make_temporal_splits(features)
        self.assertEqual(train.height, 0)
        self.assertEqual(test.height, 6)
        self.assertEqual(report["gap_quarters"], 4)
        self.assertIn("latest-vintage snapshot", report["revision_control"])

    def test_denominator_guard_and_training_quantile_bounds(self):
        ratios = pl.DataFrame({"n": [5.0, 5.0], "d": [0.0, 10.0]}).select(
            safe_ratio("n", "d").alias("ratio")
        )
        self.assertIsNone(ratios[0, "ratio"])
        self.assertEqual(ratios[1, "ratio"], 0.5)

        clipper = TrainingQuantileClipper()
        clipper.fit([[0.0], [1.0], [2.0], [1000.0]])
        transformed = clipper.transform([[10000.0]])
        self.assertLess(transformed[0, 0], 10000.0)
        self.assertAlmostEqual(clipper.upper_bounds_[0], 985.03)

    def test_full_outputs(self):
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "part4_features.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        feature_matrix = pl.read_parquet(
            ROOT / "data" / "processed" / "part4" / "feature_matrix.parquet"
        )
        report = json.loads(
            (ROOT / "results" / "part4_split_and_model_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(feature_matrix.height, 1_659_606)
        self.assertTrue(set(FEATURE_NAMES).issubset(feature_matrix.columns))
        self.assertGreater(report["features"]["train"]["rows"], 100_000)
        self.assertGreater(report["features"]["test"]["rows"], 50_000)
        self.assertGreaterEqual(report["features"]["test"]["failures"], 50)
        self.assertEqual(report["features"]["gap_quarters"], 4)
        self.assertEqual(
            report["features"]["filing_lag_check"].startswith("The primary specification"),
            True,
        )
        self.assertIn("revision_control", report["features"])
        self.assertEqual(
            report["features"]["lagged_feature_coverage"]["test_rows_with_exact_prior_quarter"],
            434415,
        )
        self.assertEqual(
            set(report["models"]["one_quarter_lag_primary"]["coefficients"].keys()),
            {f"{name}_lag1q" for name in FEATURE_NAMES},
        )
        for model in report["models"].values():
            self.assertTrue(0.0 <= model["roc_auc"] <= 1.0)
            self.assertTrue(0.0 <= model["average_precision"] <= 1.0)
            self.assertEqual(
                model["training_preprocessing_fit"]["training_rows"],
                report["features"]["train"]["rows"],
            )
            self.assertEqual(
                set(model["training_preprocessing_fit"]["features"].keys()),
                set(model["coefficients"].keys()),
            )
        self.assertEqual(
            set(report["coefficient_prior_sign_check"].keys()),
            {"net_chargeoff_rate", "core_deposit_ratio", "loan_to_assets", "re_concentration"},
        )
        self.assertGreater(
            report["development_marginal_direction_diagnostics"]["net_chargeoff_rate"][
                "univariate_auc_in_expected_direction"
            ],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
