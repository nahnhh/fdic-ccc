# FDIC Bank failure - Early Warning Model

Logistic regression model predicting U.S. bank failures (1984Q1 to 2025Q2) using ...

Split: the test period comes after the train period, with a gap of at least four quarters, and contains at least 50 failures (failures cluster in 1984–1993 and 2008–2014)

## Part 1 – Acquisition and storage

### Data acquisition

`notebooks/fdic_download_starter.py` downloads the FDIC BankFind API data with
no API key. It collects quarterly financials plus the failures and institutions
reference tables. Financial requests are paginated in 10,000-row pages,
retried with exponential backoff, and saved as one Parquet file per reporting
quarter. The downloader is resumable: quarters marked complete in
`data_raw/fdic/manifest_financials.json` are skipped. The manifest records the
downloaded row count, API row count, and a completeness flag; the final check
also verifies unique `(CERT, REPDTE)` keys.

The downloaded financial data cover 1984Q1–2026Q2: **170 quarters and
1,682,738 bank-quarter rows**. The project’s modeling horizon ends at 2025Q2;
the later quarter(s) remain in the acquisition archive but are excluded from
the Part 2 modeling panel. Raw downloads are under `data_raw/fdic/` and are
intentionally excluded from Git.

Run the downloader from the repository root:

```powershell
python notebooks/fdic_download_starter.py
```

### Typed, partitioned storage

`tests/process_storage.py` converts the raw quarterly Parquet files into
`data/processed/financials/`, partitioned by reporting `YEAR` and compressed
with Zstandard level 3. It casts identifiers and dates to typed columns,
amounts to integer storage where values are integral, ratios to `Float32`, and
state to a categorical type. Quarter files are normalized before concatenation
because the API’s Parquet schemas are not identical across all quarters.
Losslessness checks compare row counts, API totals, keys, numeric totals, and
ratio cast error before writing. This produces the columnar, year-partitioned
layout used for efficient downstream reads.

Build the processed dataset from existing raw downloads:

```powershell
python tests/process_storage.py
```

### Storage benchmark

`tests/storage_benchmark.py` compares CSV and Parquet layouts using four
workloads: full-panel read, five-column projection, 1990–2000 date filter, and
single-bank filter. The checked-in benchmark results are in
`results/storage_benchmark.csv`. On this 1,682,738-row dataset, CSV occupied
319.4 MB, while Zstandard Parquet occupied 132.3 MB (**2.41× smaller**).
Unpartitioned Zstandard Parquet had the fastest full-panel read (0.203 s);
year-partitioned Zstandard took 0.221 s. Quarter partitioning was slower for
the five-column projection and date-filter workloads, so partitioning does not
automatically improve every query. Zstandard provides the best size in these
measurements, while the best layout depends on the access pattern.

Run the benchmark (it rebuilds the benchmark files under `data/bench/`):

```powershell
python tests/storage_benchmark.py
```

## Part 2

The raw download is the single source of truth for the Part 2 quality audit and panel.
The year-partitioned processed files in `data/processed/financials` are derived from the
same raw financial Parquet files and must remain schema/type-compatible with them.

Run order:

1. `python tests/storage_benchmark.py` (Part 1 benchmark, optional for Part 2)
2. `python src/part2_data_quality_panel.py`
3. `python tests/test_part2.py`
4. `python src/part3_scalable_processing.py`
5. `python tests/test_part3.py`

Run the automated data-quality audit and build the bank-quarter panel:

```powershell
python src/part2_data_quality_panel.py
python tests/test_part2.py
```

Outputs are written to `data/processed/part2/` and are ignored by Git:

- `bank_quarter_panel.parquet`: typed bank-quarter observations with `EXIT_TYPE`, `EXIT_DATE`, and `FAIL_4Q`
- `data_quality_report.json`: row counts, key integrity, nulls, failure/exit reconciliation, and panel coverage

The canonical model label column is `target`; `FAIL_4Q` is retained as a
backward-compatible name and must remain equal to `target`.

## Part 3

Run the fair Polars/pandas benchmark and local PySpark demonstration:

```powershell
python src/part3_scalable_processing.py --repeats 3
python src/part3_polars_plan.py
python tests/test_part3.py
```

The Spark step requires Java 17 or newer. Install the project dependencies from
`pyproject.toml`/`uv.lock` first.

The benchmark times label joining, point-in-time four-quarter labels, four
quarterly asset lags, duplicate-key checks, and the complete Polars/pandas
aggregation separately. It also uses seeded random samples, reports run
counts, spreads, process RSS growth, machine details, and scales the narrow
five-column workload to 2x, 4x, and 8x. It stops before a replication if an
estimated pandas working set would exceed the available-memory guard.
Replications are synthetic repeated panels with bank identifiers offset to
preserve unique bank-quarter keys; the five columns are `CERT`, `REPDTE`,
`ASSET`, `FAIL_DT`, and `EXIT_TYPE`. Timings exclude loading and constructing
those repeated inputs. Each engine receives the same already-materialized
input, and output aggregates are compared group by group.

Polars builds a lazy expression plan: filters and projections are optimized
before execution, native columnar operators run across its thread pool, and
group-by aggregates use native parallel execution. The benchmark explicitly
measures collection (not an assumed streaming execution). The Part 2 audit
found zero duplicate `(CERT, REPDTE)` keys, so deduplication is measured as a
separate diagnostic but omitted from the production pipeline and Spark plan
to avoid a needless full-key shuffle.

`results/part3_core_job_benchmark.csv`,
`results/part3_random_sample_benchmark.csv`,
`results/part3_replication_benchmark.csv`, and
`results/part3_machine_metadata.json` contain the measured evidence.
`results/part3_spark_validation.json` contains per-group Spark-versus-Polars
checks and the physical plan. Spark reports JVM startup separately from
post-cache aggregation timings. Its local run uses two cores, whereas Polars
uses the local thread pool; the comparison is an engine demonstration, not a
hardware-matched cluster speed test.
The physical plan shows pushed Parquet filters and the expected aggregation
shuffle; it has no `(CERT, REPDTE)` deduplication shuffle.
The annotated Polars aggregate plan for reports is printed to
`results/part3_polars_aggregate_plan.txt`.

On the current Windows laptop (Intel i5-9300H, 4 physical/8 logical cores,
7.85 GiB RAM), full-panel aggregation took 0.141 s in Polars and 1.256 s in
pandas (8.9x). With 8x replication (13.28M rows), Polars took 0.959 s and
pandas 8.639 s; pandas' measured peak RSS increase was about 900 MiB. Both
completed, so this experiment found no pandas memory failure or cluster
crossover at 8x. Local Spark completed per-group checks at 1x, 2x, and 4x;
the 8x Spark cache was skipped by the memory guard. Spark's JVM startup was
36.1 s, and cache/setup plus one warm aggregation took about 12.9 s at 1x and
22.1 s at 4x. Those setup costs make a cluster unjustified for this workload.
A cluster becomes reasonable when projected data/intermediate working sets
exceed available machine memory, or when distributed input/output, throughput,
and fault-tolerance requirements outweigh startup, shuffle, and data-transfer
costs.

## Part 4

The CAMELS feature matrix contains 15 point-in-time features across capital,
asset quality, earnings, liquidity, sensitivity, management, and four-quarter
dynamics. Net charge-offs use `NTLNLS` (not `NCLNLS`, which is noncurrent loans
and leases). Invalid or nonpositive ratio denominators produce missing values,
not inflated ratios. Dynamic changes require an exact four-quarter comparison.
The management proxy is noninterest expense to assets (`NONIX / ASSET`), an
operating-cost burden measure rather than a complete management-quality rating.
Labels remain based on the Part 2 12-month future-failure window.

The temporal model split uses observations through 2006Q4 for training, excludes
2007, and evaluates on observations from 2008Q1 onward. The primary specification
uses Call Report values from exactly the prior quarter; a same-quarter model is
reported as a sensitivity check. The model pipeline fits 0.5th/99.5th percentile
outlier bounds, median imputation, and standardization on development data only.
Their fitted bounds, medians, means, and scales are included in the Part 4 JSON
report. Expected coefficient directions are economic hypotheses, not constraints;
the report shows both fitted signs and development-period marginal diagnostics.
The downloaded FDIC files are a latest-vintage snapshot without filing vintage
or publication timestamps, so historical revisions cannot be controlled or
ruled out; this limitation is recorded in the report.

Run:

```powershell
uv sync
python src/part4_features.py
python tests/test_part4.py
```