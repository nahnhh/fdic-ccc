from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import polars as pl
import psutil


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "processed" / "part2" / "bank_quarter_panel.parquet"
RESULTS = ROOT / "results"
SEED = 20241123


def machine_metadata() -> dict:
    vm = psutil.virtual_memory()
    cpu_model = ""
    if sys.platform.startswith("win"):
        cpu_model = os.popen(
            'powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"'
        ).read().strip().replace("\n", "; ")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "polars_thread_pool_size": pl.thread_pool_size(),
        "cpu_model": cpu_model or platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_gb": round(vm.total / 1024**3, 2),
        "seed": SEED,
    }


def replicate_polars(panel: pl.DataFrame, factor: int) -> pl.DataFrame:
    if factor == 1:
        return panel
    pieces = []
    for i in range(factor):
        pieces.append(
            panel.with_columns(
                (pl.col("CERT") + i * 100_000_000).cast(pl.Int64).alias("CERT")
            )
        )
    return pl.concat(pieces, rechunk=True)


def polars_full(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.lazy()
        .filter(pl.col("ASSET") > 0, pl.col("REPDTE") <= pl.date(2025, 6, 30))
        .with_columns(
            pl.col("REPDTE").dt.year().alias("YEAR"),
            pl.when(
                pl.col("FAIL_DT").is_not_null()
                & (pl.col("FAIL_DT") > pl.col("REPDTE"))
                & (pl.col("FAIL_DT") <= pl.col("REPDTE").dt.offset_by("12mo"))
            ).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.UInt8).alias("target"),
        )
        .group_by(["YEAR", "EXIT_TYPE"])
        .agg(
            pl.len().alias("rows"),
            pl.col("target").sum().alias("failures"),
            pl.col("ASSET").mean().alias("mean_asset"),
        )
        .sort(["YEAR", "EXIT_TYPE"])
        .collect()
    )


def pandas_full(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.loc[
        (panel["ASSET"] > 0)
        & (panel["REPDTE"] <= pd.Timestamp("2025-06-30"))
    ].copy()
    work["YEAR"] = work["REPDTE"].dt.year
    work["target"] = (
        work["FAIL_DT"].notna()
        & (work["FAIL_DT"] > work["REPDTE"])
        & (work["FAIL_DT"] <= work["REPDTE"] + pd.DateOffset(months=12))
    ).astype("uint8")
    return (
        work.groupby(["YEAR", "EXIT_TYPE"], as_index=False)
        .agg(rows=("CERT", "size"), failures=("target", "sum"), mean_asset=("ASSET", "mean"))
        .sort_values(["YEAR", "EXIT_TYPE"])
        .reset_index(drop=True)
    )


def polars_label_join(panel: pl.DataFrame) -> pl.DataFrame:
    failures = (
        panel.filter(pl.col("FAIL_DT").is_not_null())
        .select(pl.col("CERT"), pl.col("FAIL_DT"))
        .unique()
    )
    institutions = panel.select("CERT", "CHANGEC1", "EXIT_DT").unique()
    return panel.select("CERT", "REPDTE").join(failures, on="CERT", how="left").join(
        institutions, on="CERT", how="left"
    )


def pandas_label_join(panel: pd.DataFrame) -> pd.DataFrame:
    failures = (
        panel.loc[panel["FAIL_DT"].notna(), ["CERT", "FAIL_DT"]]
        .drop_duplicates()
    )
    institutions = panel[["CERT", "CHANGEC1", "EXIT_DT"]].drop_duplicates()
    work = panel[["CERT", "REPDTE"]].merge(failures, on="CERT", how="left")
    return work.merge(institutions, on="CERT", how="left")


def polars_four_quarter_label(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.lazy().filter(
        pl.col("ASSET") > 0,
        pl.col("REPDTE") <= pl.date(2025, 6, 30),
    ).with_columns(
        pl.when(
            pl.col("FAIL_DT").is_not_null()
            & (pl.col("FAIL_DT") > pl.col("REPDTE"))
            & (pl.col("FAIL_DT") <= pl.col("REPDTE").dt.offset_by("12mo"))
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("target")
    ).select("CERT", "REPDTE", "target").collect()


def pandas_four_quarter_label(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.loc[
        (panel["ASSET"] > 0)
        & (panel["REPDTE"] <= pd.Timestamp("2025-06-30"))
    ].copy()
    work["target"] = (
        work["FAIL_DT"].notna()
        & (work["FAIL_DT"] > work["REPDTE"])
        & (work["FAIL_DT"] <= work["REPDTE"] + pd.DateOffset(months=12))
    ).astype("uint8")
    return work[["CERT", "REPDTE", "target"]]


def polars_quarter_lags(panel: pl.DataFrame) -> pl.DataFrame:
    ordered = panel.lazy().sort(["CERT", "REPDTE"])
    expressions = []
    for lag in range(1, 5):
        previous_date = pl.col("REPDTE").shift(lag).over("CERT")
        previous_asset = pl.col("ASSET").shift(lag).over("CERT")
        expected_date = pl.col("REPDTE").dt.offset_by(f"-{lag * 3}mo").dt.month_end()
        expressions.append(
            pl.when(previous_date == expected_date)
            .then(previous_asset)
            .otherwise(None)
            .alias(f"ASSET_LAG{lag}Q")
        )
    return ordered.with_columns(*expressions).select(
        "CERT", "REPDTE", "ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"
    ).collect()


def pandas_quarter_lags(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.sort_values(["CERT", "REPDTE"]).copy()
    grouped = work.groupby("CERT", sort=False)
    for lag in range(1, 5):
        previous_date = grouped["REPDTE"].shift(lag)
        expected_date = work["REPDTE"] - pd.DateOffset(months=3 * lag)
        previous_month_end = expected_date + pd.offsets.MonthEnd(0)
        work[f"ASSET_LAG{lag}Q"] = grouped["ASSET"].shift(lag).where(
            previous_date == previous_month_end
        )
    return work[[
        "CERT", "REPDTE", "ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"
    ]]


def polars_dedup_check(panel: pl.DataFrame) -> int:
    return panel.select(pl.struct(["CERT", "REPDTE"]).is_duplicated().sum()).item()


def pandas_dedup_check(panel: pd.DataFrame) -> int:
    return int(panel.duplicated(["CERT", "REPDTE"]).sum())


def measure(func, repeats: int = 3) -> dict:
    gc.collect()
    func()
    timings = []
    rss_deltas = []
    rss_baselines = []
    rss_peaks = []
    result = None
    process = psutil.Process()
    for _ in range(repeats):
        gc.collect()
        before = process.memory_info().rss
        peak = [before]
        stop = threading.Event()

        def watch_rss() -> None:
            while not stop.wait(0.005):
                peak[0] = max(peak[0], process.memory_info().rss)

        watcher = threading.Thread(target=watch_rss, daemon=True)
        watcher.start()
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        after = process.memory_info().rss
        stop.set()
        watcher.join()
        peak[0] = max(peak[0], after)
        timings.append(elapsed)
        rss_deltas.append(max(0, peak[0] - before) / 1024**2)
        rss_baselines.append(before / 1024**2)
        rss_peaks.append(peak[0] / 1024**2)
    return {
        "median_s": round(statistics.median(timings), 6),
        "min_s": round(min(timings), 6),
        "max_s": round(max(timings), 6),
        "spread_s": round(max(timings) - min(timings), 6),
        "runs": repeats,
        "median_baseline_rss_mb": round(statistics.median(rss_baselines), 3),
        "median_peak_rss_mb": round(statistics.median(rss_peaks), 3),
        "median_peak_rss_increase_mb": round(statistics.median(rss_deltas), 3),
        "result": result,
    }


def random_sample(panel: pl.DataFrame, fraction: float) -> pl.DataFrame:
    size = max(1, int(panel.height * fraction))
    return panel.sample(n=size, seed=SEED, shuffle=True)


def benchmark_jobs(panel: pl.DataFrame, repeats: int) -> list[dict]:
    pandas_panel = panel.to_pandas()
    pandas_panel["EXIT_TYPE"] = pandas_panel["EXIT_TYPE"].astype("category")
    jobs = [
        ("label_join", lambda: polars_label_join(panel), lambda: pandas_label_join(pandas_panel)),
        ("four_quarter_label", lambda: polars_four_quarter_label(panel),
         lambda: pandas_four_quarter_label(pandas_panel)),
        ("four_quarter_lags", lambda: polars_quarter_lags(panel),
         lambda: pandas_quarter_lags(pandas_panel)),
        ("dedup_check", lambda: polars_dedup_check(panel),
         lambda: pandas_dedup_check(pandas_panel)),
        ("full_pipeline", lambda: polars_full(panel), lambda: pandas_full(pandas_panel)),
    ]
    rows = []
    for name, polars_job, pandas_job in jobs:
        polars = measure(polars_job, repeats)
        pandas = measure(pandas_job, repeats)
        polars_result = polars.pop("result")
        pandas_result = pandas.pop("result")
        if name == "dedup_check":
            match = polars_result == pandas_result
        elif name == "label_join":
            p_result = polars_result.sort(["CERT", "REPDTE", "FAIL_DT"]).to_pandas()
            pa_result = pandas_result.sort_values(
                ["CERT", "REPDTE", "FAIL_DT"]
            ).reset_index(drop=True)
            p_result = p_result.reset_index(drop=True)
            match = (
                len(p_result) == len(pa_result)
                and p_result[["CERT", "REPDTE", "FAIL_DT"]].equals(
                    pa_result[["CERT", "REPDTE", "FAIL_DT"]]
                )
            )
        elif name == "four_quarter_label":
            p_height = polars_result.height
            pa_height = len(pandas_result)
            match = p_height == pa_height
        elif name == "four_quarter_lags":
            p_result = polars_result.to_pandas().sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
            pa_result = pandas_result.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
            for col in ("ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"):
                p_result[col] = p_result[col].round(8)
                pa_result[col] = pa_result[col].round(8)
            match = p_result[
                ["CERT", "REPDTE", "ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"]
            ].equals(
                pa_result[
                    ["CERT", "REPDTE", "ASSET_LAG1Q", "ASSET_LAG2Q", "ASSET_LAG3Q", "ASSET_LAG4Q"]
                ]
            )
        else:
            pp = polars_result.to_pandas()
            pa = pandas_result
            pp = pp.sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
            pa = pa.sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
            pp["mean_asset"] = pp["mean_asset"].round(8)
            pa["mean_asset"] = pa["mean_asset"].round(8)
            match = pp.to_dict("records") == pa.to_dict("records")
        rows.append({
            "job": name,
            "input_rows": panel.height,
            "polars_input_size_mb": round(panel.estimated_size() / 1024**2, 3),
            "pandas_input_size_mb": round(
                pandas_panel.memory_usage(index=True, deep=True).sum() / 1024**2, 3
            ),
            "polars_median_s": polars["median_s"],
            "polars_min_s": polars["min_s"],
            "polars_max_s": polars["max_s"],
            "polars_spread_s": polars["spread_s"],
            "polars_median_baseline_rss_mb": polars["median_baseline_rss_mb"],
            "polars_median_peak_rss_mb": polars["median_peak_rss_mb"],
            "polars_median_peak_rss_increase_mb": polars["median_peak_rss_increase_mb"],
            "pandas_median_s": pandas["median_s"],
            "pandas_min_s": pandas["min_s"],
            "pandas_max_s": pandas["max_s"],
            "pandas_spread_s": pandas["spread_s"],
            "pandas_median_baseline_rss_mb": pandas["median_baseline_rss_mb"],
            "pandas_median_peak_rss_mb": pandas["median_peak_rss_mb"],
            "pandas_median_peak_rss_increase_mb": pandas["median_peak_rss_increase_mb"],
            "runs": repeats,
            "speedup_x": round(pandas["median_s"] / polars["median_s"], 3),
            "output_match": match,
        })
    return rows


def benchmark_random_samples(full: pl.DataFrame, repeats: int) -> list[dict]:
    rows = []
    for fraction in (0.25, 0.50, 1.0):
        sample = random_sample(full, fraction)
        pandas_sample = sample.to_pandas()
        pandas_sample["EXIT_TYPE"] = pandas_sample["EXIT_TYPE"].astype("category")
        polars = measure(lambda: polars_full(sample), repeats)
        pandas = measure(lambda: pandas_full(pandas_sample), repeats)
        pp = polars["result"].to_pandas().sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
        pa = pandas["result"].sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
        pp["mean_asset"] = pp["mean_asset"].round(8)
        pa["mean_asset"] = pa["mean_asset"].round(8)
        match = pp.to_dict("records") == pa.to_dict("records")
        rows.append({
            "fraction": fraction,
            "sampling": f"random seed={SEED}",
            "input_rows": sample.height,
            "polars_input_size_mb": round(sample.estimated_size() / 1024**2, 3),
            "pandas_input_size_mb": round(
                pandas_sample.memory_usage(index=True, deep=True).sum() / 1024**2, 3
            ),
            "polars_median_s": polars["median_s"],
            "polars_spread_s": polars["spread_s"],
            "polars_median_baseline_rss_mb": polars["median_baseline_rss_mb"],
            "polars_median_peak_rss_mb": polars["median_peak_rss_mb"],
            "polars_median_peak_rss_increase_mb": polars["median_peak_rss_increase_mb"],
            "pandas_median_s": pandas["median_s"],
            "pandas_spread_s": pandas["spread_s"],
            "pandas_median_baseline_rss_mb": pandas["median_baseline_rss_mb"],
            "pandas_median_peak_rss_mb": pandas["median_peak_rss_mb"],
            "pandas_median_peak_rss_increase_mb": pandas["median_peak_rss_increase_mb"],
            "runs": repeats,
            "speedup_x": round(pandas["median_s"] / polars["median_s"], 3),
            "output_match": match,
        })
    return rows


def benchmark_replication(full: pl.DataFrame, repeats: int) -> list[dict]:
    rows = []
    scale_fields = ["CERT", "REPDTE", "ASSET", "FAIL_DT", "EXIT_TYPE"]
    base = full.select(scale_fields)
    available_memory = psutil.virtual_memory().available
    pandas_base = base.to_pandas()
    pandas_base["EXIT_TYPE"] = pandas_base["EXIT_TYPE"].astype("category")
    pandas_base_bytes = int(pandas_base.memory_usage(index=True, deep=True).sum())
    for factor in (1, 2, 4, 8):
        estimated_bytes = pandas_base_bytes * factor * 2
        if estimated_bytes > available_memory * 0.70:
            rows.append({
                "replication_factor": factor,
                "input_rows": base.height * factor,
                "status": "skipped_memory_guard",
                "estimated_pandas_peak_mb": round(estimated_bytes / 1024**2, 1),
                "available_memory_mb": round(available_memory / 1024**2, 1),
                "runs": 0,
                "output_match": None,
            })
            continue
        scaled = replicate_polars(base, factor)
        pandas_scaled = pd.concat(
            [pandas_base.assign(CERT=pandas_base["CERT"] + i * 100_000_000)
             for i in range(factor)],
            ignore_index=True,
        )
        try:
            polars = measure(lambda: polars_full(scaled), repeats)
            pandas = measure(lambda: pandas_full(pandas_scaled), repeats)
            pp = polars["result"].to_pandas().sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
            pa = pandas["result"].sort_values(["YEAR", "EXIT_TYPE"]).reset_index(drop=True)
            pp["mean_asset"] = pp["mean_asset"].round(8)
            pa["mean_asset"] = pa["mean_asset"].round(8)
            match = pp.to_dict("records") == pa.to_dict("records")
            status = "complete"
            polars_median = polars["median_s"]
            pandas_median = pandas["median_s"]
            polars_spread = polars["spread_s"]
            pandas_spread = pandas["spread_s"]
            polars_rss = polars["median_peak_rss_increase_mb"]
            pandas_rss = pandas["median_peak_rss_increase_mb"]
            polars_base_rss = polars["median_baseline_rss_mb"]
            polars_peak_rss = polars["median_peak_rss_mb"]
            pandas_base_rss = pandas["median_baseline_rss_mb"]
            pandas_peak_rss = pandas["median_peak_rss_mb"]
        except (MemoryError, RuntimeError) as exc:
            status = f"failed: {type(exc).__name__}"
            match = False
            polars_median = pandas_median = polars_spread = pandas_spread = 0.0
            polars_rss = pandas_rss = 0.0
            polars_base_rss = polars_peak_rss = pandas_base_rss = pandas_peak_rss = 0.0
        rows.append({
            "replication_factor": factor,
            "input_rows": scaled.height,
            "status": status,
            "estimated_pandas_peak_mb": round(estimated_bytes / 1024**2, 1),
            "available_memory_mb": round(available_memory / 1024**2, 1),
            "polars_input_size_mb": round(scaled.estimated_size() / 1024**2, 1),
            "pandas_input_size_mb": round(
                pandas_scaled.memory_usage(index=True, deep=True).sum() / 1024**2, 1
            ),
            "polars_median_s": polars_median,
            "polars_spread_s": polars_spread,
            "polars_median_baseline_rss_mb": polars_base_rss,
            "polars_median_peak_rss_mb": polars_peak_rss,
            "polars_median_peak_rss_increase_mb": polars_rss,
            "pandas_median_s": pandas_median,
            "pandas_spread_s": pandas_spread,
            "pandas_median_baseline_rss_mb": pandas_base_rss,
            "pandas_median_peak_rss_mb": pandas_peak_rss,
            "pandas_median_peak_rss_increase_mb": pandas_rss,
            "runs": repeats,
            "speedup_x": round(pandas_median / polars_median, 3) if polars_median and pandas_median else None,
            "output_match": match,
        })
        del pandas_scaled
        del scaled
        gc.collect()
    return rows


def run_spark(panel_path: Path) -> dict:
    java_candidates = []
    if java_home := os.environ.get("JAVA_HOME"):
        java_candidates.append(Path(java_home))
    java_candidates.append(Path(r"C:\Users\ADMIN\AppData\Local\Temp\temurin17\jdk-17.0.21.1+2"))

    java_home = next(
        (p for p in java_candidates if (p / "bin" / "java.exe").exists()),
        None,
    )
    if java_home is None:
        return {"available": False, "reason": "JAVA_HOME is not set and no JDK 17 was found"}

    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    process = subprocess.run(
        [sys.executable, str(ROOT / "src" / "part3_spark_demo.py"), str(panel_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return {"available": True, "success": False, "stderr": process.stderr[-4000:]}
    lines = [line for line in process.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {
            "available": True,
            "success": False,
            "reason": "PySpark completed without JSON",
            "stdout": process.stdout[-4000:],
        }
    spark_result = json.loads(lines[-1])

    full = pl.read_parquet(panel_path)
    for scale in spark_result.get("scales", []):
        if scale.get("status") != "complete":
            continue
        replicated = replicate_polars(full, scale["factor"])
        polars_aggregate = (
            polars_full(replicated)
            .to_pandas()
            .sort_values(["YEAR", "EXIT_TYPE"])
            .reset_index(drop=True)
        )
        polars_aggregate["mean_asset"] = polars_aggregate["mean_asset"].round(8)
        polars_records = polars_aggregate.to_dict("records")
        scale["polars_match"] = polars_records == scale["aggregate"]
        scale["polars_aggregate_rows"] = len(polars_records)
    completed = [
        scale for scale in spark_result.get("scales", [])
        if scale.get("status") == "complete"
    ]
    spark_result["success"] = bool(completed) and all(
        scale.get("polars_match") for scale in completed
    )
    return spark_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-spark", action="store_true")
    args = parser.parse_args()

    if not args.panel.exists():
        raise FileNotFoundError(f"{args.panel} not found")

    full = pl.read_parquet(args.panel)
    job_rows = benchmark_jobs(full, args.repeats)
    sample_rows = benchmark_random_samples(full, args.repeats)
    scale_rows = benchmark_replication(full, args.repeats)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(job_rows).write_csv(RESULTS / "part3_core_job_benchmark.csv")
    pl.DataFrame(sample_rows).write_csv(RESULTS / "part3_random_sample_benchmark.csv")
    pl.DataFrame(scale_rows).write_csv(RESULTS / "part3_replication_benchmark.csv")
    (RESULTS / "part3_machine_metadata.json").write_text(
        json.dumps(machine_metadata(), indent=2), encoding="utf-8"
    )

    spark = {"available": False, "reason": "skipped"} if args.skip_spark else run_spark(args.panel)
    (RESULTS / "part3_spark_validation.json").write_text(
        json.dumps(spark, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "machine": machine_metadata(),
        "core_jobs": job_rows,
        "random_samples": sample_rows,
        "replication": scale_rows,
        "spark": spark,
    }, indent=2))


if __name__ == "__main__":
    main()
