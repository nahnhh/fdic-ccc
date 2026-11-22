"""Part 1: raw per-quarter Parquet (from the downloader) -> typed, YEAR-partitioned zstd Parquet."""
import json
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

RAW, OUT = Path("data_raw/fdic"), Path("data/processed")
KEYS = ["CERT", "REPDTE", "STNAME"]
AMOUNTS = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "LNLSGR", "LNLSNET", "LNRE", "LNCI",
           "LNCON", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE", "NETINC",
           "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR"]          # thousands of USD -> Int64
RATIOS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]                  # percent -> Float32


def main() -> None:
    raw = pl.scan_parquet(RAW / "financials" / "*.parquet")
    typed = (raw.with_columns(
                pl.col("CERT").cast(pl.UInt32),
                pl.col("REPDTE").cast(pl.Utf8).str.to_date("%Y%m%d"),
                pl.col("STNAME").cast(pl.Categorical),
                *[pl.col(c).cast(pl.Int64) for c in AMOUNTS],
                *[pl.col(c).cast(pl.Float32) for c in RATIOS])
             .with_columns(pl.col("REPDTE").dt.year().cast(pl.Int16).alias("YEAR"))
             .sort(["REPDTE", "CERT"]))
    df, src = typed.collect(), raw.collect()

    # ---- losslessness + completeness checks (raise on failure) ----
    manifest = json.loads((RAW / "manifest_financials.json").read_text())
    assert df.height == src.height == sum(m["api_total"] for m in manifest.values())
    assert df.select(pl.struct("CERT", "REPDTE").is_duplicated().any()).item() is False
    assert df.select(pl.col("CERT", "REPDTE").null_count()).row(0) == (0, 0)
    for c in AMOUNTS:                                  # Int64 cast must not truncate or change totals
        assert src[c].cast(pl.Float64).sum() == df[c].cast(pl.Float64).sum(), c
        assert ((src[c] != src[c].round()).sum() or 0) == 0, f"{c} not integral"
    for c in RATIOS:                                   # Float32 error bound
        assert (src[c].cast(pl.Float64) - df[c].cast(pl.Float64)).abs().max() < 1e-4, c

    ds.write_dataset(
        df.to_arrow(), OUT / "financials", format="parquet",
        partitioning=ds.partitioning(pa.schema([("YEAR", pa.int16())]), flavor="hive"),
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd", compression_level=3),
        existing_data_behavior="overwrite_or_ignore")
    print(f"OK: {df.height:,} rows, {df['YEAR'].n_unique()} year partitions")


if __name__ == "__main__":
    main()
