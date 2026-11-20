# Probability of tomorrow’s temperature range at Central Park

Compare two probability forecasts for the event that tomorrow’s daily maximum
minus minimum is at least a specified Celsius threshold. The
[paired 2025 study](noaa-temperature-pair.md) supplies the calibration cohort.
Its observed range reached 10 °C on 82 of 365 dates (22.5%). The later study
uses the same point-model settings and refits separate annual-plus-lag
regressions with the eligible expanding histories.

For each 2025 date, take actual minus forecast for the minimum and maximum.
Add those errors to the two point forecasts and take maximum minus minimum.
The same-day method uses the 365 date-matched error pairs. The independent
method uses all 133,225 combinations of the same marginal errors. Each draw
within a method has equal weight. Counting over these finite sets gives the
probability of a range at least as large as the threshold and the probability
of a strictly negative range. Crossed draws stay in their denominators.

The score is the mean squared difference between forecast probability and
the event indicator (Brier score, lower is better). The summary reports the
mean same-day score minus the independent score on the same dates, so a
negative difference favors same-day pairing for that cohort. It also reports
event counts, mean probabilities, and observed event rates in fixed
probability bins of width 0.1. These are descriptive comparisons. Date-label
availability is simulated, and the results pertain to this station, event,
and evaluation window.

## January–June 2026 results

The study froze January 1 through June 30, 2026, a 10 °C threshold and paired
Brier scoring before requesting the later archive. The
[NOAA Central Park archive](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
supplied all 181 dates with blank quality flags for both temperatures. The
range reached 10 °C on 64 dates (35.4%), compared with 82 of 365 dates (22.5%)
in the 2025 calibration cohort.

| Residual pairing | Mean Brier score | Mean event probability | Mean probability of a negative range |
|---|---:|---:|---:|
| Independent | 0.22415 | 37.4% | 5.01% |
| Same day | 0.23935 | 30.8% | 0.91% |

Same-day pairing raised the mean Brier loss by 0.01520 on the matched dates.
It also reduced the probability of draws with the minimum above the maximum.
All observed and point-forecast ranges were nonnegative. The results compare
one event on one six-month cohort using simulated availability.

The Python 3.12 audit recomputed all 181 probability and Brier rows by direct
enumeration and replayed ten output files byte for byte. The plan and evaluation
are saved locally in `runs/noaa-range-2026-h1-plan/` and
`runs/noaa-range-2026-h1/`. The tracked
[summary and audit](artifacts/noaa-range-probability-summary.json) record the
source and plan digests, calibration bins and replay receipts.

## Freeze and evaluate

Confirm the inclusive evaluation dates, Celsius threshold, Brier scoring and
seasonal settings before acquiring the later NOAA values. The following
commands reproduce the 10 °C design for January–June 2026. Supply
the completed 2025 studies and new output directories. Run with the research
environment or the [Python 3.12 profile](python312-public-forecast.md).

```bash
PYTHONPATH=src .venv/bin/python -m scripts.study_noaa_range freeze \
  --pair-study runs/noaa-extrema-new \
  --maximum-study runs/noaa-2025-new \
  --start 2026-01-01 --end 2026-06-30 --threshold 10 --score brier \
  --output-directory runs/noaa-range-plan
```

`freeze` verifies the original completed forecasts, origins and normalized
snapshots. It copies the matched 2025 table and exact histories, then saves
`plan.json` and its digest. The plan pins those files, the packaged regression,
NOAA parser, saved-forecast checks and study sources. The caller downloads the
metric archive with attributes from the URL in the frozen plan. It starts on
January 1, 2026 so days before a later requested start are available for
training.

```bash
source_url="$(.venv/bin/python -c 'import json; from pathlib import Path; print(json.loads(Path("runs/noaa-range-plan/plan.json").read_text())["source_url"])')"
curl --fail --show-error --location "$source_url" --output runs/noaa-range-later.csv
PYTHONPATH=src .venv/bin/python -m scripts.study_noaa_range evaluate \
  --study runs/noaa-range-plan --source runs/noaa-range-later.csv \
  --output-directory runs/noaa-range-evaluation
```

`evaluate` verifies the plan, sources and frozen inputs before opening the
later CSV. It validates the station, full calendar and both quality flag
columns, preserves the archive and expanded scalar snapshots, and checks
matched forecast origins and cutoffs before scoring. The study reads the
runner’s saved predictions and evaluates both probability methods on the
same cases. Use the same frozen plan, archive bytes, environment and a new
output directory to replay. NOAA can revise the archive, so use the preserved
source file for exact reproduction.

| Output | Contents |
|---|---|
| `inputs.json` | Frozen plan digest and raw archive snapshot ID |
| `tmin/`, `tmax/` | Scalar forecasts, schedules, scores and snapshot references |
| `matched-range.csv` | Date, issuance, cutoff, observed and predicted ranges, event indicator, both probabilities, crossing probabilities and paired Brier losses |
| `summary.json` | Evaluation event count, mean scores and difference, negative-range diagnostics and calibration bins |
