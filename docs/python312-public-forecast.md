# Python 3.12 Gaussian forecasts with public NOAA data

Use this profile for the shared CSV workflow on Python 3.12. It covers preserved
input, a fixed-variance Gaussian local-level forecast, a last-observation
benchmark, full predictive scoring, and a saved chart. The source checkout
was verified on Linux x86-64 with CPython 3.12.16. The project lock and package
metadata describe the Python 3.14 research environment.

## Set up the verified profile

The four direct packages are pandas for reading CSV, NumPy and SciPy for shared
distribution calculations, and Matplotlib for the report. The pinned file
includes the nine dependencies those packages resolved in the verified
environment. Use the workplace-approved installation source and package list.
The workflow runs from a checkout using `PYTHONPATH=src`.

```bash
python3.12 -m venv runs/python312-forecast-env
runs/python312-forecast-env/bin/python -m pip install -r requirements-python312.txt
runs/python312-forecast-env/bin/python -m pip check
```

If uv is already available, the verified setup used these equivalent commands.
Choose a new environment directory.

```bash
uv venv --python 3.12 runs/python312-forecast-uv
uv pip sync --python runs/python312-forecast-uv/bin/python requirements-python312.txt
uv pip check --python runs/python312-forecast-uv/bin/python
```

The package pins are in [the Python 3.12 profile](../requirements-python312.txt).
The full research and optional audit workflows have their own environment.
See the [research environment](../README.md#development-environment).

## Prepare the public task

The target is the next day's maximum temperature in degrees Celsius at the
NY City Central Park station `USW00094728`. Use the 731 archived observations
from January 1, 2023 through December 31, 2024. Use 2023 as initial history and
score the 366 next-day targets in 2024, including leap day. Each successful
origin uses an expanding history. The last-value benchmark uses the same
origins and eligible histories. See the [NOAA GHCN-Daily archive](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily).

Acquire the metric CSV with the attribute columns explicitly. If the data
were obtained separately, pass the existing file to the script.

```bash
mkdir -p runs/noaa-input
curl --fail --show-error --location \
  'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2023-01-01&endDate=2024-12-31&dataTypes=TMAX&units=metric&includeAttributes=true&format=csv' \
  --output runs/noaa-input/central-park.csv
PYTHONPATH=src runs/python312-forecast-env/bin/python scripts/run_noaa_temperature.py \
  --source runs/noaa-input/central-park.csv \
  --output-directory runs/noaa-central-park-new
```

The archive contains a date and a value for each day. The script uses midnight
UTC as a calendar label and simulates immediate availability at that label.
This is a development replay with the retrieved archive values. It does not
establish the historical time at which NOAA released an observation or which
vintage was available then. Treat operational forecast timing as a separate
data requirement. See the [NOAA API reference](https://www.ncei.noaa.gov/access/search/documentation/data-service/)
and [GHCN-Daily field and flag definitions](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt).

The command reads a local CSV and makes a new output directory. It validates
the expected station, full daily calendar and quality flags, preserves the
raw and normalized input bytes as separate snapshots, and freezes the input
IDs and settings in `design.json`. It then runs the backtest and benchmark,
scores the frozen forecasts, and draws `forecast/forecast.png` from the saved
results. Missing temperatures, duplicates, quality flags or gaps require the
caller to reconcile the source.

## Verified result

All 366 model and benchmark forecasts completed on Python 3.12.16. The Gaussian
model used fixed initial mean 15, initial variance 100, observation variance 4,
and evolution variance 9 in the corresponding Celsius units. The baseline
was slightly better on both point scores. The Gaussian model also supplied
uncertainty and full predictive distributions.

| Forecast | MAE in °C | RMSE in °C |
|---|---:|---:|
| Fixed Gaussian local level | 2.9402 | 3.7722 |
| Last observed value | 2.9355 | 3.7584 |

Mean CRPS was 2.1188 °C. The 95% prediction region covered 349 of 366 observations
and had a mean width of 15.68 °C. These measurements describe one development
history under simulated availability. The [curated summary](artifacts/python312-noaa-summary.json)
records snapshot IDs, settings, and the verification environment.

The profile also passed 80 targeted tests. Since then, the CI workplace job runs
the whole suite on this profile for every push and pull request, skipping only
the MCMC and XLSX tests that need packages the profile omits; see the
[CI guide](continuous-integration.md#what-runs-automatically). The forecast and benchmark CSVs,
scores, and normalized NOAA input replayed byte for byte in a second clean
Python 3.12 environment. Run the profile on the workplace platform with
its approved packages to establish compatibility there.

The [seasonal follow-up](noaa-seasonality.md) uses this output directory as
its source, keeping the same 366 cases and original benchmark.
