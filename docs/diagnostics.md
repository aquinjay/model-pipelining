# Local-level influence and model criticism

Sensitivity analysis makes assumption influence visible. A prior is supposed
to influence updating. Strong, persistent influence can be appropriate; weak
influence is not an objective or a condition for computational success.
The question is how the stated assumptions shape the update path, uncertainty,
and interpretation of observations.

The diagnostic workflow compares a caller-chosen reference with named
alternatives on the same verified snapshot. It does not search for settings,
rank configurations, or require their paths to agree. All settings remain fixed
within each path. Parameter uncertainty is not integrated out.

## Distinguish the assumptions

| Setting | Meaning | What changing it reveals |
|---|---|---|
| `initial_mean` | Location of the initial Gaussian level prior | How a different initial belief shifts later estimates |
| `initial_variance` | Uncertainty in that initial belief | How initial precision changes update gains, estimates, and uncertainty |
| `observation_variance` | Assumed measurement noise variance | How observations are weighted relative to the predicted level |
| `evolution_variance` | Assumed daily level movement variance | How much changing levels are allowed and how uncertainty evolves |

The initial prior precedes the first daily transition, as in the
[model equations](local-level.md#calculations). Noise assumptions affect the
model throughout the path, not just at initialization. Comparing one changed
setting at a time helps attribute an effect. Joint changes are also supported;
they describe a joint counterfactual, not an isolated effect of one setting.

Every selected observation is processed from the beginning. There is no
30-observation warmup exclusion in this diagnostic: it would hide part of the
initial influence. The updated level at observation `t` conditions only on
observations through `t`; no smoothing with later data is used.

## Read the influence measures

`update_paths.csv` saves one row per observation per successful configuration:
observed value, pre-observation mean/variance, innovation, standardized
innovation, conditional normal PIT, update gain, updated level mean/SD, next
observation SD, and signed differences from the reference in the last three
moment summaries. Differences use the same observation index and units.

The **update gain** `K` is the weight on the latest observation in the model's
update. `initial_mean_weight` has a narrower meaning: it is the derivative of
the updated level mean with respect to the initial mean, **holding all variance
settings fixed**. From the model's update equation:

```text
Q[t] = C[t-1] + W + V
K[t] = (C[t-1] + W) / Q[t]
a[0] = 1
a[t] = a[t-1] * (1 - K[t]) = a[t-1] * V / Q[t]
```

Thus changing only the initial mean by `d` changes the updated mean after `t`
observations by `a[t] * d`, in exact arithmetic. For the worked model with
`initial_variance=V=W=1`, the weights after one and two observations are
`1/3` and `1/8`. An initial-mean difference of 8 therefore gives updated-mean
differences of `8/3` and 1. With a known fixed level (`initial_variance=W=0`),
the weight remains 1: observations do not change that stipulated level.
This strong influence does not fail the diagnostic.

This weight is **not total prior influence**, a fraction of the current
estimate, or a universal robustness score. Changing initial variance or noise
changes the gains, uncertainties, and weights on the observations themselves.
Inspect their complete paths and reference differences. Small initial-mean
weight does not imply those other assumptions have little influence. Very
small weights can underflow to zero in floating-point arithmetic.

`scenario_summaries.json` names the settings changed from the reference and
saves maximum absolute moment differences and the final signed mean difference.
Maxima summarize effect size; they are not scores to minimize. The chart shows
updated means, signed mean differences, updated level SD, and initial-mean
weights. The weight axis uses a logarithmic scale above `1e-12` and a linear
scale near zero. Next-observation SD is saved separately because state and
observation uncertainty answer different questions.

For this fixed-variance model, gains and variance/weight paths depend on the
settings and number of updates, not the observed values. The updated means
and their counterfactual differences also depend on the observations.

## Read the sequential criticism

An innovation is `actual[t] - predicted_mean_before_seeing_actual[t]`.
Standardization divides it by the corresponding pre-observation predictive SD.
The conditional normal probability integral transform (PIT) is the predictive
CDF evaluated at that observation. These calculations use the stated model's
pre-observation distribution, including during initialization.

The report saves mean standardized innovation, mean squared standardized
innovation, and Pearson correlation between consecutive standardized innovations.
The last quantity is `null`, with a reason, when there are fewer than three
observations or either lagged sequence is constant. The charts show innovations
over time, a PIT histogram, and consecutive-innovation pairs.

Under the stated generative model the reference expectations are zero mean,
unit mean square, uniform PIT, and no serial dependence. These are interpretive
reference patterns, not finite-sample acceptance thresholds. A pattern can
motivate revisiting assumptions without identifying one uniquely correct
revision. Initialization and a level change can have different implications;
inspect where discrepancies occur, not just aggregate summaries.

This is **sequential predictive criticism**, not full-data posterior predictive
replication. It does not replace the backtest's frozen forecasts and scores or
an untouched final evaluation. Changing settings after reading it is a
development revision.

## Run it explicitly

First prepare a snapshot as in [the walkthrough](forecasting.md#prepare-the-input-explicitly).
From the repository root, choose an unused output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_local_level.py \
  --dataset synthetic-mean-shift --tag baseline-input \
  --config configs/local_level_diagnostics.json \
  --output-directory runs/tutorial-diagnostics-001
```

`--snapshot-id ID` selects a permanent input instead of a tag; `--store` selects
another input store. A supplied tag resolves once and its ID is saved. The
[example JSON](../configs/local_level_diagnostics.json) contains `reference`
settings and named `scenarios` that override those settings. Unspecified
reference fields use `LocalLevel` defaults; each scenario inherits all unchanged
reference settings. The illustrative changes are explicit examples, not an
automatically generated grid or recommended application settings.

The command verifies/parses the input, calculates the paths/summaries, saves
review questions, and draws both charts automatically. It returns status 1 if
any configuration's computation failed. Large influence does not change the
exit status. Invalid options/configuration/input raises before output creation;
filesystem/reporting errors propagate and may leave partial artifacts.

Library callers make the calculation and plotting separate calls:

```python
from dataclasses import replace
from model_pipelining.diagnostics import (
    run_local_level_diagnostics, plot_local_level_diagnostics,
)
from model_pipelining.local_level import LocalLevel

reference = LocalLevel()
diagnostics = run_local_level_diagnostics(
    dataset="synthetic-mean-shift", tag="baseline-input",
    reference=reference,
    scenarios={
        "different-initial-mean": replace(reference, initial_mean=10),
        "tighter-initial-prior": replace(reference, initial_variance=1),
        "more-observation-noise": replace(reference, observation_variance=4),
    },
    output_directory="runs/tutorial-library-diagnostics-001",
)
plot_local_level_diagnostics(diagnostics)
```

The existing backtest and its run scripts do not invoke this workflow. No input
acquisition, snapshot registration, tag assignment, tuning, or human decision
occurs automatically. Only concrete `LocalLevel` models are supported here;
the shared forecaster protocol and backtest output format are unchanged.

This diagnostic keeps the original immediate/unrevised `timestamp`/`value`
input contract. It does not apply an `ExperimentSpec`'s release rules or final
boundaries and processes its entire input. Prepare an appropriate development-only
snapshot explicitly when preserving a genuinely untouched final period. Backtest
release-aware selection belongs to [the experiment workflow](experiments.md).

## Saved evidence and failures

| File | Meaning |
|---|---|
| `diagnostics.json` | Diagnostic format version 1, permanent input, scope/semantics, Python version, source fingerprints, computation status and configuration counts |
| `update_paths.csv` | Full successful forward paths and reference differences, including initialization |
| `scenario_summaries.json` | Every requested name/configuration, changed setting groups, completion/error, criticism, and influence summaries |
| `review.json` | Pending human decision, interpretation questions, and limitations |
| `sensitivity.png` | Saved influence paths; explicit plotting or automatic CLI action |
| `innovations.png` | Saved sequential criticism; explicit plotting or automatic CLI action |

Numerical failures are retained per configuration and later configurations
continue. Failed configurations have no invented path rows and are named in
chart titles. If the reference fails, valid alternatives still retain their
own paths/criticism, but reference differences and influence summaries are
unavailable. Any failed configuration makes overall computation `failed`.
There is no magnitude-based failure rule or automatic release policy.
The manifest certifies calculation completion; an optional charting failure
does not revise that status and propagates separately.

Existing output directories/charts are rejected. Plotting reads saved files
only and changes no calculation evidence. For another rendering, copy saved
diagnostics into a new directory without its PNG files. Files remain externally
writable, and there is no saved-artifact integrity audit or crash recovery.

To replay, use the saved permanent input and each fully expanded configuration
from `scenario_summaries.json`, with matching code/environment and a new output
directory. Numerical CSV/summaries repeat; execution timestamps differ. Retain
the complete input store and diagnostic folder with matching source and locked
dependencies. Local data/results are ignored by Git.
