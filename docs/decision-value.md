# Cost-loss decisions from saved forecasts

A probability forecast supports a decision when it changes whether you
take a protective action. This study asks how much that rule would have cost
on the saved [Central Park range forecasts](noaa-range-probability.md). The
event is a next-day temperature range of at least 10 °C.

## Decision and expense

Taking protection costs `C`. An unprotected event costs `L`. Protection
avoids the event loss, and waiting on a day without the event costs zero.
Divide all expenses by `L` and write `alpha = C / L`, with
`0 < alpha < 1`. Acting costs `alpha`. The forecast’s expected expense
of waiting is its event probability `p`. Act when `p > alpha`. At equality,
wait.

| Decision | Event occurs | Event does not occur |
|---|---|---|
| Act | Hit, expense `alpha` | False alarm, expense `alpha` |
| Wait | Miss, expense `1` | Correct negative, expense `0` |

Climatology uses a constant event probability `s` estimated from the 2025
calibration cohort. It acts on every case if `alpha < s` and waits on every
case otherwise. The observed evaluation outcomes determine the resulting
expense. Using their event rate to choose the climatology action would use
information unavailable when the forecasts were issued.

A perfect forecast acts on every event and waits otherwise. Its mean
expense is `alpha` times the observed evaluation event rate.

Relative economic value compares how much of the possible savings over
climatology the forecast achieved.

```text
V = (mean climatology expense − mean forecast expense)
    / (mean climatology expense − mean perfect expense)
```

`V = 1` matches the perfect forecast’s expense. `V = 0` matches
climatology. A negative value costs more than climatology. If climatology
and the perfect forecast have the same mean expense, the JSON records
`null` and a reason. The CSV leaves the value blank and retains the reason.

## Five cases by hand

Take `alpha = 0.20` and a separate calibration event rate of `0.40`.

| Case | Forecast probability | Event | Action | Expense |
|---|---:|---:|---|---:|
| 1 | 0.80 | 1 | Act | 0.20 |
| 2 | 0.45 | 1 | Act | 0.20 |
| 3 | 0.60 | 0 | Act | 0.20 |
| 4 | 0.20 | 0 | Wait | 0 |
| 5 | 0.10 | 0 | Wait | 0 |

The forecast has two hits, one false alarm, zero misses, and two correct
negatives. The mean forecast expense is `0.60 / 5 = 0.12`. Climatology
always acts because `0.20 < 0.40`, so its mean expense is `0.20`. The
perfect forecast acts on the two events and pays `0.40 / 5 = 0.08`.
The relative value is `(0.20 − 0.12) / (0.20 − 0.08) = 2/3`.

## Run the NOAA comparison

Supply a completed range evaluation directory and a new output directory.
The study reads `matched-range.csv` and `summary.json`, verifies the
evaluation dates and counts, and takes the climatology rate from
`summary.json` at `calibration.observed_events / calibration.cases`.
It scores the independent and same-day methods on the same cases at
ratios from 0.05 to 0.95 in steps of 0.05.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.study_decision_value \
  --input-directory runs/noaa-range-2026-h1 \
  --output-directory runs/noaa-range-decision-value
```

The output directory contains `decision-value.csv` with counts, mean
expenses and value for each method and ratio,
`decision-value-summary.json` with those results and the input SHA-256
digests, alpha grid and base-rate source, and `decision-value.png`.
The command requires a new directory for each run.

The calculations are in
[decision_value.py](../src/model_pipelining/decision_value.py).
`evaluate_decisions(cases, *, alpha, base_rate)` takes an iterable of
`(probability, event)` pairs. `value_curve(cases, *, alphas, base_rate)`
evaluates the same pairs at each supplied ratio in order.

## Read the result

The plot shows the 181 frozen January–June 2026 forecasts. The 2025
calibration event rate was 82 of 365 days. The vertical line marks that rate
and the switch in climatology’s action. The horizontal line at zero marks
equal cost to climatology. Gaps indicate undefined value.

![Relative economic value of the independent and same-day NOAA range forecasts across cost-loss ratios](figures/noaa-range-decision-value.png)

Results are descriptive for one station, one event, and one evaluation period.
Value depends on the cost-loss ratio. A forecast can save money for one user
and cost another user more than climatology. The fixed grid describes that
dependence, and no ratio or event threshold is selected or tuned on these
evaluation outcomes.
