# Annual seasonality in the Central Park forecast

On the 366 matched 2024 Central Park forecasts, an annual sine and cosine
pair combined with yesterday's maximum temperature gave the lowest MAE and
RMSE of the five models compared. Its MAE was 8.6% lower than the last-value
benchmark and its RMSE was 9.4% lower. It improved in each season and gave up
accuracy on quiet days. This is an exploratory result from the same
development year used to inspect the original forecast errors.

## What the original forecasts missed

The original Gaussian local-level model had a mean error of −0.020 °C and
the last-value benchmark had a mean error of −0.015 °C, with error defined as
forecast minus actual. Both had their highest MAE in spring and lowest MAE
in summer. The last-value benchmark's 68 days with absolute temperature
changes of at least 5 °C accounted for 65.8% of its squared error. Its MAE on
those days was 6.85 °C. On the 157 days with changes under 2 °C, its MAE was
0.97 °C. These are groups defined by realized outcomes after the forecast.
They can describe the errors without supplying information to the model.

## Matched comparison

Use the saved [NOAA archive example](python312-public-forecast.md), including
its 2023 starting history, 2024 target dates, and simulated availability.
Each regression refits by ordinary least squares at each origin, using the
same eligible input history. The three variants fit the same response dates.
The first eligible observation supplies an initial lag, so the fit uses
response dates from the second observation onward.

The annual terms use one sine and cosine pair with a fixed period of 365.2425
days and phase origin January 1, 2023. The lag-only model fits an intercept
and yesterday's observed maximum. The combined model fits all four terms.
The fixed annual terms capture the recurring seasonal shape. The lag lets the
forecast move with recent weather. See [Hyndman and Athanasopoulos on harmonic
terms and lagged predictors](https://otexts.com/fpp3/dynamic-regression.html).

| Forecast | MAE °C | RMSE °C | Winter MAE | Spring MAE | Summer MAE | Autumn MAE |
|---|---:|---:|---:|---:|---:|---:|
| Last observed value | 2.936 | 3.758 | 3.084 | 3.894 | 2.090 | 2.674 |
| Gaussian local level | 2.940 | 3.772 | 3.148 | 3.724 | 2.125 | 2.765 |
| Annual curve | 3.428 | 4.357 | 4.371 | 3.834 | 2.568 | 2.945 |
| Lag only | 2.930 | 3.691 | 3.035 | 3.716 | 2.365 | 2.602 |
| Annual curve plus lag | 2.682 | 3.407 | 2.926 | 3.327 | 2.041 | 2.434 |

Compared with the lag-only model, adding the annual terms lowered MAE by
0.248 °C. Against last-value persistence, the combination reduced MAE from
6.85 °C to 5.38 °C on days with moves of at least 5 °C. Its MAE on days with
moves under 2 °C rose from 0.97 °C to 1.56 °C. The annual curve alone performed
worse overall. The regressions return point forecasts, so the table uses
point scores across all five models.

The study holds the calendar and histories fixed. Shared comparison reports
verify that each saved forecast uses the same snapshot, training cutoff,
history hash, target, and scoring vintage. The archived daily values still
use simulated availability. The [frozen 2025 evaluation](noaa-later-period.md)
uses later dates that did not guide this comparison. Operational use requires
verified release timing and vintages.

## Run the comparison again

After producing a completed NOAA source run, invoke the study explicitly in
the Python 3.12 profile. Pass a new output directory. The command uses local
saved files and verified snapshots.

```bash
PYTHONPATH=src runs/python312-forecast-env/bin/python scripts/study_noaa_seasonality.py \
  --source-directory runs/noaa-central-park-new \
  --output-directory runs/noaa-seasonality-new
```

The script freezes the three model configurations, saves baseline diagnostics,
fits the three shared-runner backtests, and generates comparison reports. It
also writes `matched-cases.csv` and `summary.json` with overall, seasonal,
monthly, and realized-change scores. See the [curated result](artifacts/noaa-seasonality-summary.json).
