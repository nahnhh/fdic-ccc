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

Requires Python 3.14 or newer. Reproduce Part 1 from the repository root in
PowerShell, following Appendix A3:

```powershell
uv sync --frozen
python notebooks/fdic_download_starter.py
python tests/process_storage.py
python tests/storage_benchmark.py # optional Part 1 benchmark
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

The optional benchmark command rebuilds benchmark files under `data/bench/`.

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
- `data_quality_report.json`: row counts, key integrity, nulls, failure/exit reconciliation, panel coverage, numeric consistency checks, and per-era value distributions
- `data_quality_report.csv`: flattened `path,type,value` rows for spreadsheet inspection; the JSON report remains unchanged and authoritative

The audit flags nonfinite or unparseable core values, equity above assets,
negative deposits, and deposits above assets. It also reports per-era quantiles
and 1.5-IQR outlier counts for core fields, plus positive-median ratios between
eras as a scale-change screen. These are review diagnostics: unusual values are
not automatically dropped; deposits above assets can occur with negative equity.
Distribution shifts alone cannot establish historical unit or field-definition
changes.

The canonical model label column is `target`; `FAIL_4Q` is retained as a
backward-compatible name and must remain equal to `target`.

## Part 3

Run the fair Polars/pandas benchmark and local PySpark demonstration:

```powershell
python src/part3_scalable_processing.py --repeats 5
python src/part3_polars_plan.py
python tests/test_part3.py
```

The Spark step requires Java 17 or newer. Install the project dependencies from
`pyproject.toml`/`uv.lock` first.

The benchmark times label joining, point-in-time four-quarter labels, four
quarterly asset lags, duplicate-key checks, and the complete Polars/pandas
aggregation separately. It also uses seeded random samples, reports run
counts (five measured runs for each completed benchmark), spreads, process RSS
growth, machine details, and scales the narrow five-column workload to 2x, 4x,
and 8x. It stops before a replication if an estimated pandas working set would
exceed the available-memory guard.
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
checks, the Spark reproduction of the CERT label join, and the physical plans.
The join output is compared with Polars as a multiset; Spark reports its
row-count timing over five measured runs. Spark reports JVM startup separately
from post-cache aggregation timings, which also use five measured runs. Its
local run uses two cores, whereas Polars
uses the local thread pool; the comparison is an engine demonstration, not a
hardware-matched cluster speed test.
The aggregate physical plan shows pushed Parquet filters and the expected
aggregation shuffle; it has no `(CERT, REPDTE)` deduplication shuffle.
Running `python src/part3_polars_plan.py` captures Polars'
`LazyFrame.explain(optimized=True)` output and writes it with annotations for
the pushed filter, projected columns, and lack of deduplication shuffle to
`results/part3_polars_aggregate_plan.txt`.

On the current Windows laptop (Intel i5-9300H, 4 physical/8 logical cores,
7.85 GiB RAM), the five-run full-panel aggregation median was 0.036 s in Polars
and 0.375 s in pandas (10.4x). At 4x replication (6.64M rows), medians were
0.132 s and 1.289 s (9.8x); 8x was skipped by the pandas memory guard in this
run. Local Spark reproduced the 1,659,606-row CERT label join in five measured
runs (0.899 s median) with zero multiset differences from Polars. Spark's
aggregate also matched Polars at 1x, 2x, and 4x with five runs each; its 8x
cache was skipped by the memory guard. Spark startup was 10.7 s, separate from
post-cache timings. These results demonstrate local execution, not a cluster
crossover. A cluster becomes reasonable when projected data/intermediate
working sets exceed available machine memory, or when distributed input/output,
throughput, and fault-tolerance requirements outweigh startup, shuffle, and
data-transfer costs.

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

Run Part 4:

```powershell
uv sync
python src/part4_features.py
python tests/test_part4.py
```

## Part 5

Part 5 compares a prior-prevalence baseline, a one-feature NPA logistic
baseline, a 15-feature CAMELS logistic model, and a 15-feature
HistGradientBoosting classifier. The boosted-tree model uses training-only
0.5th/99.5th-percentile clipping and median imputation (scaling is omitted
because tree splits do not require it); its fixed settings are recorded in the
JSON report. For the scikit-learn models, thresholds are selected from
out-of-fold predictions over four expanding-window validation folds, each
separated from its training period by a four-quarter gap. The scikit-learn
models are then refit on all eligible development rows through 2006Q4 before
the final 2008Q1–2025Q2 test is scored; 2007 is the test gap. Thresholds and
model choices are fixed before the test is scored.

The label is highly imbalanced, so average precision is primary, with ROC-AUC,
probability calibration, thresholded precision/recall, and bank-level failure
recall also reported. Failure costs use observed FDIC `COST`. False alarms
have no observed cost, so they are modeled as review costs equal to a fraction
of the development-period median failed-bank `COST`. The base fraction is 1%;
sensitivity covers 0.1%, 1%, 10%, and 100%. The threshold minimizes
validation-period total estimated cost. A missed failure is charged once per
failed bank only if no alert occurs in its positive 12-month window. As an
independent capacity operating point, the top 1% of validation scores is also
evaluated. Test average precision is reported with a bank-clustered bootstrap
interval. A test-era CSV reports scikit-learn model metrics separately for
2008–2014 and 2015–2025 using each model's unchanged validation-selected
threshold; these descriptive test slices are not used to select models or
thresholds. A fresh Spark run also adds its per-era rows. Costs are in thousands
of USD.

The Spark MLlib LogisticRegression demonstration uses a training-only
clipping/imputation/scaling pipeline, selects its illustrative threshold on the
fixed validation window, then refits on the full eligible development period
before test scoring. It requires Java 17 or newer. Run Part 4 first to create
the feature matrix, then run:

```powershell
python src/part5_model_validation.py
python tests/test_part5.py
```

The report and threshold sensitivity table are saved to
`results/part5_validation_report.json` and
`results/part5_threshold_sensitivity.csv`; the concise model comparison is
`results/part5_model_metrics.csv`. Test-period era comparisons are saved to
`results/part5_test_per_era_metrics.csv`. Final standardized logistic
coefficients are saved to `results/part5_final_model_coefficients.csv`.

## Part 6: Conclusion - In the docs

## Reproducibility

Use the locked dependencies and run commands from the repository root in
PowerShell. The project requires Python 3.14 or newer; the Spark demonstrations
in Parts 3 and 5 also require Java 17 or newer.
`requirements.txt` is an exact-version, hash-pinned export of `uv.lock` for
pip-based environments; install it with
`python -m pip install --require-hashes -r requirements.txt`. With uv, use
`uv sync --frozen` as shown below.

```powershell
uv sync --frozen

# Acquire source data if the raw FDIC files are not already present.
python notebooks/fdic_download_starter.py
python tests/process_storage.py

# Rebuild the analytical outputs in dependency order.
python src/part2_data_quality_panel.py
python src/part3_scalable_processing.py --repeats 3
python src/part3_polars_plan.py
python src/part4_features.py
python src/part5_model_validation.py

# Run the corresponding checks.
python tests/test_part2.py
python tests/test_part3.py
python tests/test_part4.py
python tests/test_part5.py
```

Set `JAVA_HOME` to the installed JDK before running the Spark commands if Java
is not already on `PATH`. Part 1's storage benchmark is optional:
`python tests/storage_benchmark.py`.

The raw downloads and processed datasets are generated locally and are not
stored in Git. Re-downloading the FDIC API later may produce different data:
the API is live, and the available financial files are latest-vintage
snapshots without historical filing vintages. Consequently, model outputs
can change if source data change. Fixed date splits, validation folds, and
bootstrap seed (`42`) are recorded in the source and result reports. Benchmark
timings also depend on local hardware and should be treated as measurements,
not expected constants.
