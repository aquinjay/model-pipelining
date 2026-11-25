# Evaluation, baselines, and decision tools guide

A plain-English reference to the evaluation metrics, probabilistic baselines,
comparison reports, and decision tools built into `model-pipelining`. Use this
page to see what tools exist, which command to run, and how to read every
diagnostic without digging through JSON files or research notes.

For the full mathematical details and contracts, see
[decision value](decision-value.md), [distribution scoring](distribution-scoring.md),
[experiment review](experiment-review.md), and the
[Python API reference](../src/model_pipelining/README.md).

## 1. Tools at a glance

Everything below is **saved-only** after a backtest finishes: comparisons,
reviews, and decision evaluations never refit a model or modify source runs.
Each workflow is invoked explicitly by the caller; `run_backtest` never runs
decisions, comparisons, or reviews automatically.

| Goal | CLI command or Python function | What it writes automatically |
|---|---|---|
| Run a zero-tuning probabilistic baseline | `python -m scripts.run_probabilistic_baseline --model <name> ...` | Standard backtest run directory + `forecast.png` |
| Compare two saved forecast runs | `compare_runs(first, second, output_directory=...)` | `comparison.md`, `comparison.json`, `comparison.png`, and 3-panel `distribution_scores.png` (when both save full laws) |
| Freeze forecast-time decision recommendations | `python -m scripts.run_decisions issue ...` | `decision_spec.json`, `recommendations.csv`, `decision_manifest.json` (reads forecast-time files only; never reads outcomes) |
| Score frozen decisions against later outcomes | `python -m scripts.run_decisions score ...` | `decision_evaluation.csv`, `decision_summary.json`, `checks.json`, pending `review.json`, and `decision_scores.png` (omit with `--no-plot`) |
| Compare two scored decision runs | `python -m scripts.run_decisions compare ...` | `decision_comparison.md`, `decision_comparison.json`, and 3-panel `decision_comparison.png` |
| Assemble forecast + evidence + calibration + decisions | `python scripts/review_experiment.py ...` | `report.md`, `report.json`, `checks.json`, pending `review.json` (links existing PNGs; does not create charts) |

## 2. Metrics and diagnostics: what each one means

A single average metric like RMSE or mean log score can be dominated by a
handful of extreme days and says nothing about decision value or calibration.
`comparison.md`, `report.md`, and `decision_comparison.md` surface the full
picture together:

### Point error and bias diagnostics (`robust_scoring.py`)
- **Mean bias & Median bias (`mean_error`, `median_error`):** Signed
  `prediction - actual`. Comparing mean vs. median bias immediately reveals
  error skewness (for example, gradual warmups versus sudden cold fronts).
- **MdAE (Median Absolute Error):** Typical-day error, immune to outliers.
- **MAE & p95 / max absolute error:** Average error alongside upper-tail error
  magnitude.
- **RMSE (explicitly labeled outlier-sensitive):** Retained for continuity,
  always paired with **Top-5% squared-error share**.
- **Top-5% concentration (`top_5pct_abs_error_share`, `top_5pct_sq_error_share`, `top_5pct_expense_share`):**
  Fraction of total error or expense coming from the worst 5% of origins. If
  40–50% of a squared-error difference sits in 5% of days, an aggregate average
  is telling you about a handful of shocks rather than everyday performance.

### Full-distribution scores (`distribution_scoring.py`)
- **CRPS (Continuous Ranked Probability Score, lower is better):** Evaluates
  the entire predictive distribution in the target's original units. Far more
  robust to single tail outliers than log score, and exact for all three
  supported families (`student_t`, `gaussian_mixture`, `empirical_samples`).
- **Negative log density (`log_score`, lower is better, outlier-sensitive):**
  `-log p(y)` for continuous laws (`student_t`, `gaussian_mixture`). Reported
  with median and top-5% mean so single tail surprises are visible. Explicitly
  `None` for discrete `empirical_samples` laws (no density floor or hidden KDE).
- **HPD coverage/width and pinball loss (`uncertainty.py`):** Surfaced in
  `report.md` and `comparison.md` alongside CRPS so interval sharpness and
  quantile accuracy stay visible.

### Paired comparison stability (`robust_scoring.py`, `reporting.py`, `decisions.py`)
- **Win / loss / tie rates & median difference:** Shows whether Model 1 beat
  Model 2 consistently across origins or only won on average due to a few days.
- **Bartlett (Newey-West) HAC SE:** Time-series forecast errors are often
  serially correlated; naive standard errors understate uncertainty. The
  deterministic Bartlett kernel (lag `floor(n^(1/3))`) adjusts for serial
  correlation while replaying identically every time.
- **Cumulative CRPS / expense difference plots:** In `distribution_scores.png`
  and `decision_comparison.png`, a steady downward slope means consistent skill;
  a sudden vertical step reveals a single regime shift or shock day.

### Event-probability calibration (`decision_value.py`)
When scoring binary event decisions without fallbacks, `brier_decomposition`
splits the Brier score into:

```text
Brier = Reliability (down) - Resolution (up) + Uncertainty + within_bin_term
```

- **Reliability (lower is better) & ECE (Expected Calibration Error):** Measures
  whether predicted probabilities match observed frequencies (when a model says
  30%, does the event occur ~30% of the time?). Plotted as a **reliability
  diagram** in `decision_scores.png`.
- **Resolution (higher is better):** Measures how sharply the model separates
  high-risk days from low-risk days. A model with high resolution but poor
  reliability has genuine signal that recalibration can unlock; low resolution
  means weak underlying signal.
- **Brier Skill Score (`brier_skill_score`):** Improvement over the predeclared
  calibration `base_rate` (never using in-sample evaluation prevalence).

### Decision policies and Murphy dominance diagrams (`decision_value.py`, `decisions.py`)
- **Binary cost-loss (`binary_cost_loss`):** Act when event probability
  `p > alpha`, where `alpha = C / L` is predeclared in `DecisionSpec`. Reports
  hits, false alarms, misses, correct negatives, realized expense vs.
  calibration climatology and perfect foresight, and **Relative Economic Value**
  (`1` = perfect, `0` = climatology, `< 0` = worse than climatology).
- **Continuous asymmetric / newsvendor (`continuous_asymmetric`):** Uses
  positive `under_cost` ($c_u$) and `over_cost` ($c_o$) and recommends the exact
  predictive quantile at $\tau = c_u / (c_u + c_o)$, directly turning predictive
  quantiles and pinball loss into realized expense.
- **Murphy diagrams (Ehm et al., 2016):** Evaluate elementary scoring functions
  across a whole grid of thresholds—cost-loss ratios $\alpha \in (0, 1)$ for
  binary decisions, target thresholds $\theta$ for continuous asymmetric
  decisions (`compare_decision_scores`) and point/mean forecasts (`compare_runs`).
  - `first_dominates` / `second_dominates`: one model has lower or equal
    elementary loss at *every* threshold (robust across all consistent scoring
    rules, not just one metric).
  - `crossing`: neither model dominates everywhere; inspect the Murphy plot in
    `decision_comparison.png` to see which regimes favor which model.

## 3. Probabilistic baseline adapters

Use [probabilistic_baselines.py](../src/model_pipelining/probabilistic_baselines.py)
to benchmark Bayesian or custom models against zero-tuning probabilistic
references on the exact same histories and targets:

| Adapter (`--model`) | Predictive family | Best used as baseline for | Notes |
|---|---|---|---|
| `gaussian-climatology` | `student_t` ($df = n - 1$) | Stationary series without strong seasonality | Exact Jeffreys-prior predictive law; supports HPD and log score; optional rolling `--window` |
| `residual-persistence` | `student_t` ($df = m$) | Random-walk / persistent series at any horizon | Centers at last value with scale from matched historical lead differences; supports HPD and log score |
| `empirical-climatology` | `empirical_samples` | Multimodal, bounded, or seasonal series | Preserves exact historical shape; rolling `--window 30` tracks local seasonality; `log_score=None`, no HPD |
| `empirical-residual-persistence` | `empirical_samples` | Skewed or heavy-tailed error series at any horizon | Forms $\{y_{\text{last}} + e_i\}$ from historical lead changes; preserves asymmetric shocks; `log_score=None`, no HPD |

All four adapters implement `from_configuration`, are registered in
[adapters.py](../src/model_pipelining/adapters.py), and work identically on
both Python 3.14 and the Python 3.12 workplace profile.

## 4. End-to-end quickstart

```bash
# 1. Run a nonparametric probabilistic baseline on a preserved snapshot
PYTHONPATH=src .venv/bin/python -m scripts.run_probabilistic_baseline \
  --model empirical-residual-persistence \
  --dataset my-dataset \
  --snapshot-id <64-hex-snapshot-id> \
  --output-directory runs/baseline-erp

# 2. Freeze forecast-time recommendations (never reads outcomes)
PYTHONPATH=src .venv/bin/python -m scripts.run_decisions issue \
  --run-directory runs/baseline-erp \
  --spec path/to/decision_spec.json \
  --output-directory runs/baseline-erp-issued

# 3. Score against realized outcomes (creates decision_scores.png + reliability diagram)
PYTHONPATH=src .venv/bin/python -m scripts.run_decisions score \
  --issued-directory runs/baseline-erp-issued \
  --outcomes runs/baseline-erp \
  --output-directory runs/baseline-erp-scored

# 4. Compare two scored decision runs (writes decision_comparison.md + Murphy plot)
PYTHONPATH=src .venv/bin/python -m scripts.run_decisions compare \
  --first runs/model-a-scored \
  --second runs/baseline-erp-scored \
  --output-directory runs/decision-comparison

# 5. Assemble a unified review linking the forecast and its decision evaluation
PYTHONPATH=src .venv/bin/python scripts/review_experiment.py \
  --forecast-directory runs/baseline-erp \
  --decision-directories runs/baseline-erp-scored \
  --require-decisions \
  --output-directory runs/baseline-erp-review
```

To compare two saved forecast runs directly in Python and generate
`comparison.md`, `comparison.json`, `comparison.png`, and `distribution_scores.png`:

```python
from model_pipelining.reporting import compare_runs

compare_runs(
    "runs/model-a",
    "runs/baseline-erp",
    output_directory="runs/model-a-vs-erp",
)
```
