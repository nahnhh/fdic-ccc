"""Part 1 storage benchmark: CSV vs Parquet layouts. Fills Table 3 of the report.

Usage: python tests/storage_benchmark.py
Output: results/storage_benchmark.csv (+ printed table)
Each (layout, workload) runs in a fresh subprocess, 1 warm-up + 5 timed runs, median reported.
"""
from __future__ import annotations
import gzip, json, shutil, statistics, subprocess, sys, time
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "financials"
RAW = ROOT / "data_raw" / "fdic" / "financials"
BENCH = ROOT / "data" / "bench"
RESULTS = ROOT / "results"
RATIOS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
AMOUNTS = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNCI",
           "LNCON", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE", "NETINC",
           "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR"]
LAYOUTS = ["csv", "pq_none", "pq_snappy", "pq_zstd", "pq_year_zstd", "pq_qtr_zstd"]
WORKLOADS = ["W1_full", "W2_5cols", "W3_1990_2000", "W4_one_bank"]
REPEATS = 5


def path(layout: str) -> str:
    return {"csv": str(BENCH / "panel.csv"),
            "pq_none": str(BENCH / "panel_none.parquet"),
            "pq_snappy": str(BENCH / "panel_snappy.parquet"),
            "pq_zstd": str(BENCH / "panel_zstd.parquet"),
            "pq_year_zstd": str(BENCH / "year" / "**" / "*.parquet"),
            "pq_qtr_zstd": str(BENCH / "qtr" / "**" / "*.parquet")}[layout]


def load_panel() -> pl.DataFrame:
    processed_files = list(SRC.glob("**/*.parquet")) if SRC.exists() else []
    if processed_files:
        return (pl.read_parquet(str(SRC / "**" / "*.parquet"), hive_partitioning=True)
                .sort(["REPDTE", "CERT"]))

    raw_files = sorted(RAW.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"No processed data at {SRC} and no raw quarter files at {RAW}. "
            "Run process_storage.py or download the FDIC financials first."
        )

    frames = []
    for raw_file in raw_files:
        frame = (pl.read_parquet(raw_file)
                 .with_columns(
                     pl.col("CERT").cast(pl.UInt32),
                     pl.col("REPDTE").cast(pl.Utf8).str.to_date("%Y%m%d"),
                     pl.col("STNAME").cast(pl.String),
                     *[pl.col(c).cast(pl.Int64) for c in AMOUNTS if c in pl.read_parquet_schema(raw_file)],
                     *[pl.col(c).cast(pl.Float32) for c in RATIOS if c in pl.read_parquet_schema(raw_file)],
                 ))
        frames.append(frame)

    return (pl.concat(frames, how="diagonal_relaxed")
            .with_columns(pl.col("REPDTE").dt.year().cast(pl.Int16).alias("YEAR"))
            .sort(["REPDTE", "CERT"]))


def remove_bench() -> None:
    if not BENCH.exists():
        return
    for attempt in range(10):
        try:
            shutil.rmtree(BENCH)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def build() -> None:
    remove_bench()
    BENCH.mkdir(parents=True)
    df = load_panel()
    df.write_csv(path("csv"))
    df.write_parquet(path("pq_none"), compression="uncompressed")
    df.write_parquet(path("pq_snappy"), compression="snappy")
    df.write_parquet(path("pq_zstd"), compression="zstd", compression_level=3)
    for key, name in (("YEAR", "year"), ("REPDTE", "qtr")):
        for (k,), part in df.partition_by(key, as_dict=True, maintain_order=True).items():
            d = BENCH / name / f"{key}={k}"
            d.mkdir(parents=True, exist_ok=True)
            part.write_parquet(d / "part.parquet", compression="zstd", compression_level=3)
    cert = df["CERT"].value_counts().sort("count", descending=True)["CERT"][0]
    (BENCH / "cert.txt").write_text(str(cert))
    with open(path("csv"), "rb") as f, gzip.open(BENCH / "panel.csv.gz", "wb") as g:
        shutil.copyfileobj(f, g)


def size_mb(layout: str) -> float:
    p = Path(path(layout).split("**")[0])
    files = [p] if p.is_file() else [x for x in p.rglob("*.parquet")]
    return sum(x.stat().st_size for x in files) / 1e6


def scan(layout: str) -> pl.LazyFrame:
    if layout == "csv":
        return pl.scan_csv(path(layout), try_parse_dates=True)
    return pl.scan_parquet(path(layout), hive_partitioning=True)


def current_process_rss_mb() -> float:
    try:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                    ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1e6
    except (ImportError, AttributeError):
        pass
    return 0.0


def child(layout: str, workload: str) -> None:
    lf, t0 = scan(layout), time.perf_counter()
    if workload == "W1_full":
        out = lf.collect()
    elif workload == "W2_5cols":
        out = lf.select(["CERT", "REPDTE", *RATIOS]).collect()
    elif workload == "W3_1990_2000":
        pred = (pl.col("YEAR").is_between(1990, 2000) if layout == "pq_year_zstd" else
                pl.col("REPDTE").is_between(pl.date(1990, 3, 31), pl.date(2000, 12, 31)))
        out = lf.filter(pred).select(["CERT", "REPDTE", *RATIOS]).collect()
    else:
        cert = int((BENCH / "cert.txt").read_text())
        out = lf.filter(pl.col("CERT") == cert).select(["CERT", "REPDTE", *RATIOS]).collect()
    sec = time.perf_counter() - t0
    print(json.dumps({"sec": sec, "peak_mb": current_process_rss_mb(), "rows": out.height}))


def run_one(layout: str, workload: str) -> tuple[float, float, int]:
    res = []
    for i in range(REPEATS + 1):
        o = subprocess.run([sys.executable, __file__, "--child", layout, workload],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout
        if i:
            res.append(json.loads(o.strip().splitlines()[-1]))
    return (statistics.median(r["sec"] for r in res),
            statistics.median(r["peak_mb"] for r in res), res[0]["rows"])


def main() -> None:
    build()
    rows = []
    for lay in LAYOUTS:
        r = {"layout": lay, "size_mb": round(size_mb(lay), 1)}
        for w in WORKLOADS:
            s, m, n = run_one(lay, w)
            r[w + "_s"], r[w + "_rows"] = round(s, 3), n
            r["peak_mb_" + w] = round(m, 0)
        rows.append(r)
    out = pl.DataFrame(rows)
    for w in WORKLOADS:
        assert out[w + "_rows"].n_unique() == 1, f"row mismatch across layouts in {w}"
    csv = out.filter(pl.col("layout") == "csv").row(0, named=True)
    out = out.with_columns((csv["size_mb"] / pl.col("size_mb")).round(2).alias("size_ratio_vs_csv"))
    gz = (BENCH / "panel.csv.gz").stat().st_size / 1e6
    RESULTS.mkdir(exist_ok=True)
    out.write_csv(RESULTS / "storage_benchmark.csv")
    print(out)
    print(f"CSV gzip size (MB): {gz:.1f}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
