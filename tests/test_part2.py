from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from part2_data_quality_panel import (
    build_financial_value_checks,
    build_panel,
    summarize_numeric_values,
)


ROOT = Path(__file__).resolve().parent.parent


class Part2Test(unittest.TestCase):
    def test_part2_outputs(self):
        output = ROOT / "data" / "processed" / "part2"
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "part2_data_quality_panel.py"),
             "--raw", str(ROOT / "data_raw" / "fdic"),
             "--output", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        panel = pl.read_parquet(output / "bank_quarter_panel.parquet")
        report = json.loads((output / "data_quality_report.json").read_text(encoding="utf-8"))
        with (output / "data_quality_report.csv").open(
            encoding="utf-8", newline=""
        ) as csv_file:
            report_csv_paths = {
                row["path"] for row in csv.DictReader(csv_file)
            }

        self.assertGreater(panel.height, 1_500_000)
        self.assertFalse(
            panel.select(pl.struct(["CERT", "REPDTE"]).is_duplicated().any()).item()
        )
        self.assertEqual(panel["CERT"].null_count(), 0)
        self.assertEqual(panel["REPDTE"].null_count(), 0)
        self.assertTrue(panel["FAIL_4Q"].is_in([0, 1]).all())
        self.assertTrue(panel["target"].is_in([0, 1]).all())
        self.assertTrue((panel["target"] == panel["FAIL_4Q"]).all())
        self.assertEqual(report["financials"]["duplicate_CERT_REPDTE_rows"], 0)
        self.assertEqual(report["panel"]["rows"], panel.height)
        self.assertIn("core_null_share_by_era", report["financials"])
        self.assertIn("core_value_distributions_by_era", report["financials"])
        self.assertIn("value_checks", report["financials"])
        self.assertIn("panel.panel_rows", report_csv_paths)
        self.assertIn(
            "financials.core_value_distributions_by_era[0].fields.ASSET.median",
            report_csv_paths,
        )
        self.assertIn("per_bank_gaps", report["financials"])
        self.assertIn("rows_dropped_per_rule", report["panel"])
        self.assertIn("rows_in_four_quarters_before_failure", report["panel"])
        self.assertIn("exit_counts_by_CHANGEC1", report["institutions"])
        self.assertIn("failure_CERTs_without_Financials_rows", report["failures"])
        self.assertTrue(report["panel"]["raw_rows_minus_dropped_equals_clean_rows"])

    def test_financial_value_audit_reports_outliers_and_inconsistencies(self):
        financials = pl.DataFrame({
            "ASSET": ["100", "100", "100", "10000", "inf", None, "bad"],
            "DEP": ["90", "110", "95", "90", "-2", "90", "90"],
            "EQ": ["10", "10", "5", "11000", "10", "10", "10"],
            "NETINC": [1.0] * 7,
            "ROA": [1.0] * 7,
            "NCLNLS": [1.0] * 7,
            "LNATRES": [1.0] * 7,
            "RBC1AAJ": [12.0] * 7,
        })

        checks = build_financial_value_checks(financials)
        outlier_summary = summarize_numeric_values(
            pl.Series("ASSET", [100.0, 100.0, 100.0, 10_000.0])
        )

        self.assertEqual(outlier_summary["potential_iqr_outlier_rows"], 1)
        self.assertEqual(checks["nonfinite_counts_by_field"]["ASSET"], 1)
        self.assertEqual(checks["nonnumeric_nonnull_counts_by_field"]["ASSET"], 1)
        self.assertEqual(
            checks["consistency_checks"]["equity_exceeds_assets_rows"], 1
        )
        self.assertEqual(
            checks["consistency_checks"]["deposits_exceed_assets_rows"], 1
        )
        self.assertEqual(checks["consistency_checks"]["negative_deposit_rows"], 1)

    def test_synthetic_exit_and_point_in_time_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            financial_dir = raw / "financials"
            financial_dir.mkdir()

            quarters = [
                date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30),
                date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30),
                date(2026, 3, 31),
            ]
            rows = []
            for cert in (10, 20, 30, 31, 40, 50):
                for repdte in quarters:
                    rows.append({
                        "CERT": cert,
                        "REPDTE": repdte.strftime("%Y%m%d"),
                        "STNAME": "Test",
                        "ASSET": 100,
                        "DEP": 90,
                        "EQ": 10,
                        "NETINC": 1,
                        "ROA": 1.0,
                        "NCLNLS": 1.0,
                        "LNATRES": 1.0,
                        "RBC1AAJ": 12.0,
                    })
            # This row must be dropped as a nonpositive asset.
            rows.append({
                "CERT": 60, "REPDTE": "20240331", "STNAME": "Test",
                "ASSET": 0, "DEP": 0, "EQ": 0, "NETINC": 0,
                "ROA": 0.0, "NCLNLS": 0.0, "LNATRES": 0.0, "RBC1AAJ": 0.0,
            })
            pl.DataFrame(rows).write_parquet(financial_dir / "REPDTE=synthetic.parquet")

            pl.DataFrame({
                "CERT": [10, 20, 30, 31, 40],
                "FAILDATE": ["6/30/2024", "12/31/2024", "3/31/2025",
                             "6/30/2025", "3/31/2024"],
            }).write_parquet(raw / "failures.parquet")

            pl.DataFrame({
                "CERT": [10, 20, 30, 31, 40, 50, 60],
                "ACTIVE": [0, 0, 0, 0, 0, 0, 1],
                "CHANGEC1": [223.0, 223.0, 223.0, 223.0, 223.0, 223.0, None],
                "ENDEFYMD": ["12/31/2024", "12/31/2024", "12/31/2025",
                             "12/31/2025", "12/31/2024", "6/30/2024", "12/31/9999"],
            }).write_parquet(raw / "institutions.parquet")

            panel, report = build_panel(raw)

            def row(cert, repdte):
                return panel.filter(
                    (pl.col("CERT") == cert)
                    & (pl.col("REPDTE") == date.fromisoformat(repdte))
                )

            self.assertEqual(row(10, "2024-03-31")["FAIL_4Q"].item(), 1)
            self.assertEqual(row(10, "2024-06-30")["FAIL_4Q"].item(), 0)
            self.assertEqual(row(20, "2024-03-31")["FAIL_4Q"].item(), 1)
            self.assertEqual(row(30, "2024-03-31")["FAIL_4Q"].item(), 1)
            self.assertEqual(row(31, "2024-03-31")["FAIL_4Q"].item(), 0)
            self.assertEqual(row(40, "2024-03-31")["FAIL_4Q"].item(), 0)
            self.assertTrue(
                panel.filter(pl.col("CERT") == 50)["FAIL_4Q"].eq(0).all()
            )
            self.assertFalse(
                panel.filter(pl.col("CERT") == 50)
                .select(pl.col("REPDTE") > pl.col("EXIT_DATE"))
                .to_series()
                .any()
            )

            self.assertFalse(
                (panel["REPDTE"] > date(2025, 6, 30)).any()
            )
            self.assertTrue(
                panel.filter(pl.col("FAIL_DT").is_not_null())
                .select((pl.col("REPDTE") <= pl.col("FAIL_DT")).all())
                .item()
            )
            self.assertTrue((panel["ASSET"] > 0).all())
            reconciliation = report["panel"]["raw_rows_minus_dropped_equals_clean_rows"]
            self.assertTrue(reconciliation)


if __name__ == "__main__":
    unittest.main()
