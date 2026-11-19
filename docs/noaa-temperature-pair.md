# Separate forecasts of daily minimum and maximum temperature

Neither the seasonal regressions nor the last-value baselines forecast a
minimum above the maximum on the 365 matched 2025 dates at Central Park. The
separately fitted seasonal regressions also lowered daily-range MAE by 14.8%.
Their minimum and maximum errors had a correlation of 0.60. See the [saved
results and audit](artifacts/noaa-extrema-summary.json).

## Design

Use the [frozen 2025 maximum-temperature run](noaa-later-period.md) as the
reference. Acquire the metric NOAA maximum and minimum series for 2023–2025
and preserve the source bytes. All 1,096 downloaded maximum values matched
the earlier snapshot, and no observed minimum exceeded its paired maximum.
The study fits the same annual sine and cosine pair plus one lag to the
minimum series using a separate scalar backtest. Scoring uses the same
2025 dates, issuance labels and eligible date sets as the maximum backtest.
The point forecasts pair at the analysis stage. See the [NOAA archive](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily).

The archived dates simulate availability and the exploratory comparison reuses
2025. The 2025 maximum evaluation had already informed the selection of this
two-series task. Operational forecasts require real release times and vintages.

## Results

| Matched point forecasts | Minimum MAE °C | Maximum MAE °C | Range MAE °C | Crossings | Error correlation |
|---|---:|---:|---:|---:|---:|
| Last observed values | 2.152 | 3.214 | 2.655 | 0 / 365 | 0.554 |
| Separate annual curves plus lag | 2.050 | 2.945 | 2.263 | 0 / 365 | 0.603 |

Daily range means maximum minus minimum. The tightest predicted ranges were
1.10 °C for persistence and 4.20 °C for the seasonal regressions. Their ordering
held throughout this cohort. The correlated errors show that the two forecasts
often miss in the same direction. Since range error is maximum error minus
minimum error, that dependence matters when estimating uncertainty about the
range. If a task needs a probability for the range, evaluate a joint
predictive distribution on a new cohort. The existing scalar interface serves
the point forecasts used here. See the [next experiment](next-session.md#next-work)
for a proposed comparison that isolates the effect of pairing residuals.

The audit independently recalculated the order check, range MAE and paired
error correlation. The matched case table, summary, minimum forecasts,
benchmark forecasts and scores replayed byte for byte. All 86 targeted tests
passed on Python 3.12.

## Reproduce the comparison

Use the [Python 3.12 profile](python312-public-forecast.md) and complete the
[frozen maximum-temperature run](noaa-later-period.md#reproduce-the-frozen-workflow).
Then explicitly download the separate metric archive with attributes and run
the pairing study in a new directory.

```bash
mkdir -p runs/noaa-extrema-input
curl --fail --show-error --location \
  'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2023-01-01&endDate=2025-12-31&dataTypes=TMIN,TMAX&units=metric&includeAttributes=true&format=csv' \
  --output runs/noaa-extrema-input/noaa-extrema.csv
PYTHONPATH=src runs/python312-forecast-env/bin/python -m scripts.study_noaa_extrema \
  --maximum-study runs/noaa-2025-new \
  --source runs/noaa-extrema-input/noaa-extrema.csv \
  --output-directory runs/noaa-extrema-new
```

The command reads local archives and preserves the new raw and normalized
minimum snapshots. It runs the minimum forecast and persistence benchmark,
checks the origin and eligibility fields across both saved runs, and writes
`matched-extrema.csv`, `summary.json`, and `paired-forecasts.png`.
