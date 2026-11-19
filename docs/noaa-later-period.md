# Testing the Central Park seasonal model on 2025

The annual curve plus one temperature lag beat both frozen point-forecast
controls on the 365 Central Park dates in 2025. Against the last-observation
forecast, MAE fell 8.4% and RMSE fell 11.4%. The quiet-day tradeoff from 2024
also recurred. [Curated plan, audit and results](artifacts/noaa-2025-summary.json).

## Design and timing

The [2024 comparison](noaa-seasonality.md) informed model choice. Before
downloading the 2025 data, freeze the annual-plus-lag and lag-only model
configurations, last-value benchmark, 2025 dates, and scoring rules. The
annual-plus-lag model must beat both controls on overall MAE and RMSE. Seasonal
and realized-change scores describe where the errors occur.

The checked historical snapshot contributes all 731 dates from 2023 and 2024.
Those bytes remain the exact prefix of the new input. Append only 2025 data
from the [NOAA GHCN-Daily archive](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily).
The run uses January 1, 2025 through December 31, 2025 as its final target
period. It refits coefficients from eligible observations at each forecast
date, so previously observed dates in 2025 may enter later fits. The feature
set, annual period, lag, and scoring rules stay fixed.

Availability is simulated using the archived date labels and values. The
archive leaves actual historical release times and vintages unverified. The
evaluation measures a later period for these frozen models under that
assumption. See the [original public task](python312-public-forecast.md).

## Results

| Model | MAE °C | RMSE °C | Mean forecast error °C |
|---|---:|---:|---:|
| Last observed value | 3.214 | 4.341 | +0.030 |
| Lag only | 3.197 | 4.255 | +0.062 |
| Annual curve plus lag | 2.945 | 3.845 | +0.423 |

The combined model had lower MAE in all four seasons. Its winter advantage
over the last-value benchmark was only 0.012 °C. Across all 365 dates, the
combined forecast averaged 0.42 °C too warm. On the 80 days when the maximum
moved by at least 5 °C, its MAE was 6.02 °C versus 7.66 °C for persistence.
On the 158 days with moves under 2 °C, its MAE was 1.45 °C versus 0.92 °C.
These groups use realized outcomes after scoring and supply no model input.

The audit checked the original input prefix and plan and script hashes, matched
training histories and cutoffs across all 365 cases, and independently
recomputed MAE and RMSE. Reusing the same plan and later archive replayed eight
prediction, result, and score files byte for byte. All 84 targeted tests
passed on Python 3.12.

## Reproduce the frozen workflow

From the repository root with the [Python 3.12 profile](python312-public-forecast.md#set-up-the-verified-profile),
freeze the design using a completed 2024 seasonal study. Then explicitly
download the separate 2025 file. Use a new study directory.

```bash
PYTHONPATH=src runs/python312-forecast-env/bin/python -m scripts.check_noaa_later_period freeze \
  --development-directory runs/noaa-seasonality-new \
  --output-directory runs/noaa-2025-new
curl --fail --show-error --location \
  'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2025-01-01&endDate=2025-12-31&dataTypes=TMAX&units=metric&includeAttributes=true&format=csv' \
  --output runs/noaa-2025-new/download-2025.csv
PYTHONPATH=src runs/python312-forecast-env/bin/python -m scripts.check_noaa_later_period evaluate \
  --study-directory runs/noaa-2025-new \
  --source runs/noaa-2025-new/download-2025.csv
```

The `freeze` command saves the plan, its digest, and the original normalized
history. The `evaluate` command validates those inputs and the metric NOAA
CSV, then saves snapshots, both forecasts, the required benchmark, and a
comparison chart under `evaluation/`. It also writes `summary.json` and
`matched-cases.csv`. Reproduce the original result with the original frozen
plan and retrieved bytes. New downloads may reflect archive revisions.
