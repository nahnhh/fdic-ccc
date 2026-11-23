from __future__ import annotations

import json
import sys
import time

import psutil
from pyspark.sql import SparkSession, functions as F


def aggregate_panel(panel):
    return (
        panel.withColumn(
            "target",
            F.when(
                F.col("FAIL_DT").isNotNull()
                & (F.col("FAIL_DT") > F.col("REPDTE"))
                & (F.col("FAIL_DT") <= F.add_months(F.col("REPDTE"), 12)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .filter(
            (F.col("ASSET") > 0)
            & (F.col("REPDTE") <= F.lit("2025-06-30").cast("date"))
        )
        .withColumn("YEAR", F.year(F.col("REPDTE")))
        .groupBy("YEAR", "EXIT_TYPE")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum("target").alias("failures"),
            F.avg("ASSET").alias("mean_asset"),
        )
        .orderBy("YEAR", "EXIT_TYPE")
    )


def main(panel_path: str) -> None:
    available_memory = psutil.virtual_memory().available
    startup_start = time.perf_counter()
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("FDIC-Part3-MultiScale")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    startup_elapsed = time.perf_counter() - startup_start

    base = (
        spark.read.parquet(panel_path)
        .filter(
            (F.col("ASSET") > 0)
            & (F.col("REPDTE") <= F.lit("2025-06-30").cast("date"))
        )
        .select("CERT", "REPDTE", "ASSET", "FAIL_DT", "EXIT_TYPE")
    )
    base_count = base.count()
    estimated_row_bytes = 40
    scales = []
    physical_plan = None
    for factor in (1, 2, 4, 8):
        estimated_cache_bytes = base_count * factor * estimated_row_bytes
        if estimated_cache_bytes > available_memory * 0.20:
            scales.append({
                "factor": factor,
                "input_rows": base_count * factor,
                "status": "skipped_memory_guard",
                "estimated_cache_mb": round(estimated_cache_bytes / 1024**2, 1),
                "available_memory_mb": round(available_memory / 1024**2, 1),
                "runs": 0,
            })
            continue
        pieces = [base]
        if factor > 1:
            for i in range(1, factor):
                pieces.append(
                    base.withColumn(
                        "CERT",
                        (F.col("CERT") + i * 100_000_000).cast("long"),
                    )
                )
        replicated = pieces[0] if factor == 1 else pieces[0].unionByName(
            pieces[1]
        )
        if factor > 2:
            for piece in pieces[2:]:
                replicated = replicated.unionByName(piece)
        setup_start = time.perf_counter()
        replicated = replicated.cache()
        input_count = replicated.count()
        setup_elapsed = time.perf_counter() - setup_start
        aggregate = aggregate_panel(replicated)
        warmup_start = time.perf_counter()
        aggregate.collect()
        warmup_elapsed = time.perf_counter() - warmup_start
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            rows = aggregate.collect()
            timings.append(time.perf_counter() - start)
        import statistics

        median_elapsed = round(statistics.median(timings), 6)
        group_results = [
            {
                "YEAR": row["YEAR"],
                "EXIT_TYPE": row["EXIT_TYPE"],
                "rows": row["rows"],
                "failures": row["failures"],
                "mean_asset": round(float(row["mean_asset"]), 8),
            }
            for row in rows
        ]
        scales.append({
            "factor": factor,
            "status": "complete",
            "input_rows": input_count,
            "estimated_cache_mb": round(estimated_cache_bytes / 1024**2, 1),
            "available_memory_mb": round(available_memory / 1024**2, 1),
            "setup_and_cache_s": round(setup_elapsed, 6),
            "warmup_aggregate_s": round(warmup_elapsed, 6),
            "median_collect_s": median_elapsed,
            "warm_total_after_startup_s": round(
                setup_elapsed + warmup_elapsed + median_elapsed, 6
            ),
            "min_collect_s": round(min(timings), 6),
            "max_collect_s": round(max(timings), 6),
            "spread_s": round(max(timings) - min(timings), 6),
            "runs": 3,
            "failure_labels": replicated.where(
                (F.col("ASSET") > 0)
                & (F.col("REPDTE") <= F.lit("2025-06-30").cast("date"))
                & F.col("FAIL_DT").isNotNull()
                & (F.col("FAIL_DT") > F.col("REPDTE"))
                & (F.col("FAIL_DT") <= F.add_months(F.col("REPDTE"), 12))
            ).count(),
            "aggregate": group_results,
        })
        if factor == 1:
            physical_plan = aggregate._jdf.queryExecution().executedPlan().toString()
        replicated.unpersist()

    result = {
        "available": True,
        "success": True,
        "engine": "PySpark local[2]",
        "startup_elapsed_s": round(startup_elapsed, 6),
        "deduplicate_used": False,
        "zero_duplicate_keys_verified_in_part2": True,
        "reason_deduplicate_not_used": "Part 2 verified zero duplicate (CERT, REPDTE) keys",
        "filter_applied_before_cache": True,
        "scales": scales,
        "physical_plan": physical_plan,
    }
    print(json.dumps(result))
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
