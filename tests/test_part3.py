from __future__ import annotations

import csv
import json
import sys
import unittest
from datetime import date
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from part3_scalable_processing import (
    benchmark_jobs,
    pandas_four_quarter_label,
    pandas_full,
    pandas_label_join,
    pandas_quarter_lags,
    polars_four_quarter_label,
    polars_full,
    polars_label_join,
    polars_quarter_lags,
)


class Part3Test(unittest.TestCase):
    def test_core_job_benchmark_and_full_output_match(self):
        panel = pl.read_parquet(
            ROOT / "data" / "processed" / "part2" / "bank_quarter_panel.parquet"
        )
        rows = benchmark_jobs(panel, repeats=1)
        self.assertEqual({row["job"] for row in rows}, {
            "label_join",
            "four_quarter_label",
            "four_quarter_lags",
            "dedup_check",
            "full_pipeline",
        })
        self.assertTrue(all(row["output_match"] for row in rows))
        self.assertTrue(all(row["runs"] == 1 for row in rows))

    def test_synthetic_core_transformations_match(self):
        panel = pl.DataFrame({
            "CERT": [1, 1, 2, 2, 3],
            "REPDTE": [
                date(2024, 3, 31),
                date(2024, 6, 30),
                date(2024, 3, 31),
                date(2024, 6, 30),
                date(2025, 3, 31),
            ],
            "ASSET": [100.0, 100.0, 100.0, 100.0, 0.0],
            "FAIL_DT": [
                date(2024, 6, 30),
                None,
                date(2025, 9, 30),
                None,
                None,
            ],
            "CHANGEC1": [223.0, 223.0, 223.0, 223.0, None],
            "EXIT_DT": [None, None, None, None, None],
            "EXIT_TYPE": ["open", "open", "failure", "failure", "other_exit"],
        })
        pandas_panel = panel.to_pandas()

        polars_join = polars_label_join(panel).height
        pandas_join = len(pandas_label_join(pandas_panel))
        self.assertEqual(polars_join, pandas_join)

        polars_label = (
            polars_four_quarter_label(panel)
            .sort(["CERT", "REPDTE"])
            .to_dicts()
        )
        pandas_label = (
            pandas_four_quarter_label(pandas_panel)
            .sort_values(["CERT", "REPDTE"])
            .reset_index(drop=True)
            .to_dict("records")
        )
        expected = [
            {key: int(row[key]) if key in ("CERT", "target") else row[key]
             for key in ("CERT", "REPDTE", "target")}
            for row in polars_label
        ]
        for row in pandas_label:
            row["CERT"] = int(row["CERT"])
            row["REPDTE"] = row["REPDTE"].date()
            row["target"] = int(row["target"])
        self.assertEqual(expected, pandas_label)

        lag_p = polars_quarter_lags(panel).to_pandas().sort_values(
            ["CERT", "REPDTE"]
        ).reset_index(drop=True)
        lag_pd = pandas_quarter_lags(pandas_panel).sort_values(
            ["CERT", "REPDTE"]
        ).reset_index(drop=True)
        for col in ("ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"):
            lag_p[col] = lag_p[col].round(8)
            lag_pd[col] = lag_pd[col].round(8)
        self.assertTrue(lag_p.equals(lag_pd))

    def test_saved_machine_and_artifact_contract(self):
        machine_path = ROOT / "results" / "part3_machine_metadata.json"
        sample_path = ROOT / "results" / "part3_random_sample_benchmark.csv"
        scale_path = ROOT / "results" / "part3_replication_benchmark.csv"
        core_path = ROOT / "results" / "part3_core_job_benchmark.csv"
        spark_path = ROOT / "results" / "part3_spark_validation.json"
        for path in (machine_path, sample_path, scale_path, core_path, spark_path):
            self.assertTrue(path.exists(), path)
        machine = json.loads(machine_path.read_text(encoding="utf-8"))
        self.assertIn("total_memory_gb", machine)
        self.assertIn("logical_cpu_count", machine)
        self.assertIn("seed", machine)
        with core_path.open(encoding="utf-8", newline="") as csv_file:
            self.assertTrue(
                all(int(row["runs"]) >= 5 for row in csv.DictReader(csv_file))
            )
        with sample_path.open(encoding="utf-8", newline="") as csv_file:
            self.assertTrue(
                all(int(row["runs"]) >= 5 for row in csv.DictReader(csv_file))
            )
        with scale_path.open(encoding="utf-8", newline="") as csv_file:
            completed_runs = [
                int(row["runs"])
                for row in csv.DictReader(csv_file)
                if row["status"] == "complete"
            ]
        self.assertTrue(completed_runs)
        self.assertTrue(all(runs >= 5 for runs in completed_runs))
        spark = json.loads(spark_path.read_text(encoding="utf-8"))
        self.assertTrue(spark["success"])
        self.assertTrue(spark["label_join"]["output_match"])
        self.assertGreaterEqual(spark["label_join"]["runs"], 5)
        completed = [scale for scale in spark["scales"] if scale["status"] == "complete"]
        self.assertGreaterEqual(len(completed), 2)
        self.assertTrue(all(scale["polars_match"] for scale in completed))
        self.assertTrue(all(scale["runs"] >= 5 for scale in completed))
        self.assertFalse(spark["deduplicate_used"])


if __name__ == "__main__":
    unittest.main()
