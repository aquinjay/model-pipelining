# Model evidence and experiment requirements

An experiment can now declare computational evidence requirements independently
of the model. Models supply calculations through a small evidence provider;
shared workflows select histories, save results, retain failures and draw charts.
No discrepancy, prior influence or model-approval threshold is introduced.

## Choose requirements explicitly

`ExperimentSpec` includes a frozen `EvidencePlan`:

```python
from model_pipelining.experiments import EvidencePlan, ExperimentSpec

experiment = ExperimentSpec(evidence=EvidencePlan(
    required_workflows=("predictive_checks", "prior_influence"),
    required_fit_checks=(),
    replications=1000,
    seed=0,
    interval_level=0.95,
))
```

| Setting | Default | Meaning |
|---|---|---|
| `required_workflows` | `()` | Explicitly requested separate workflows; supported names are `predictive_checks`, `prior_influence` and `sequential_checks`, with no duplicates |
| `required_fit_checks` | `()` | Named checks that must pass at every primary-model forecast origin, in addition to model-required checks |
| `replications` | `1000` | Positive number of independent prior and posterior datasets for predictive checks |
| `seed` | `0` | Integer in `[0, 2**32)` for predictive simulations |
| `interval_level` | `0.95` | Probability in `(0, 1)` for prior-comparison credible/predictive regions |

Empty requirements preserve the earlier behavior. Models' own required numerical
checks still apply. For example, a learned local-level experiment can require
`variance_quadrature_precision_and_tails` and
`marginal_predictive_quantiles_and_hpd`. A constant-mean/variance experiment can
require `exact_posterior_and_predictive_valid`. These are calculation checks,
not judgments about plausible assumptions.

New specifications serialize as version 2 with `task`, `evaluation` and
`evidence`. Version-1 specifications remain readable and receive the default
empty requirements. `to_dict()` expands defaults and writes tuple-valued names
as JSON lists; `from_dict()` restores them. Existing version-1 configuration
files do not need editing. Reporting normalizes explicit version-1/2 specs
before comparison; evidence requirements must match, and missing experiment
metadata is not inferred.

## Run the separate evidence explicitly

Prepare/register an input and tag as in [the forecasting walkthrough](forecasting.md).
Then run the supplied development example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_evidence.py \
  --model local-level-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59+00:00 \
  --experiment configs/daily_evidence_experiment.json \
  --config configs/local_level_variance_priors.json \
  --output-directory runs/experiment-evidence-001
```

The caller chooses the model, prior scenarios, as-of time, specification and
new output directory. The command automatically executes only the declared
separate workflows, saves their computational outcomes and draws their saved
results. `--model normal-mean-variance` with
`configs/normal_mean_variance_priors.json` exercises the same boundary with
exact conjugate calculations. No model backend or configuration file is
discovered automatically.

Library callers invoke the runner and plotting separately:

```python
from dataclasses import replace
from datetime import datetime
from model_pipelining.evidence_workflows import (
    run_experiment_evidence, plot_experiment_evidence,
)
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.local_level_variance import LocalLevelVariance

model = LocalLevelVariance()
output = run_experiment_evidence(
    dataset="synthetic-mean-shift", tag="baseline-input",
    model=model, scenarios={"initial-level-ten": replace(model, initial_mean=10)},
    issued_at=datetime.fromisoformat("2020-06-08T23:59:59+00:00"),
    experiment=ExperimentSpec.load("configs/daily_evidence_experiment.json"),
    output_directory="runs/library-evidence-001",
)
charts = plot_experiment_evidence(output)
```

Selection requires a development experiment and sufficient history before any
final boundary. A tag is resolved once, and every workflow receives that pinned
snapshot. Final/future values stay out of selected histories, simulations and
prior learning paths. The two child workflows retain their established
statistical meanings: [whole-dataset predictive criticism](predictive-checks.md)
and [descriptive prior comparisons](prior-influence.md).

The standalone evidence functions/commands remain available for exploration,
including data-free prior checks. Their explicit arguments control those calls;
they do not certify completion of an experiment's entire evidence plan. Use
`run_experiment_evidence` to execute and record its required separate workflows.

## Read status and failures

The new directory contains `evidence.json`, `checks.json`, `review.json`, and
one subdirectory per started workflow. `evidence.json` records the expanded
plan, pinned context/model, source hashes and each required outcome. Finished
child manifests/check files have SHA-256 receipts. The runner verifies their
experiment, model, history, settings, workflow kind and mandatory check names
before recording success. Receipts identify files at completion; they are not
an immutable artifact store or an automatic later integrity audit.

Missing provider capabilities or missing explicit scenarios for required prior
comparisons produce failed requirements. Numerical failures retain child
evidence and do not prevent later requested workflows from running. A missing
mandatory check cannot pass just because a child labels itself completed.
Invalid plans/selections raise before output creation; I/O errors propagate
and can leave a `running` directory. Existing destinations are never reused.
The CLI returns 1 for a completed evidence run with failed required computation.
Human review remains `pending` in either case.

Per-origin fit checks belong to `run_backtest`, which independently enforces
`required_fit_checks`. If a model omits a requested name, marks it optional and
fails it, or supplies no fit evidence, affected forecasts fail. Constructed fits
retain their original checks plus separate `experiment_requirements`; missing
fits remain visible in `fit_index.csv`. Point benchmarks do not inherit
primary-model fit requirements.

Forecasting does not automatically start expensive separate evidence workflows.
A forecast manifest records their status as `pending` when requested. Its
`completed` status refers to forecasting computation, not completion of the
whole evidence plan. The explicit evidence runner does not modify that saved
forecast run, execute fits, score final forecasts, or approve a model. There is
no aggregate experiment registry or release system.

## Provider boundary and compatibility

`model.evidence_provider()` returns an `EvidenceProvider` with `EvidenceMetadata`
and three operations:

- `simulate(observation_count=..., replications=..., seed=..., request=None)`:
  joint prior datasets without a request, whole-history posterior replication
  with one; every requested replication must appear once as a dataset or failure.
- `summarize(case, level)`: prior/learning parameter and predictive summaries;
  `case=None` means the empty-history prior.
- `validate_scenarios(models)`: enforce which settings may vary in prior
  comparisons, such as holding evolution variance and quadrature fixed.

Providers own conjugate distributions or numerical integration, state smoothing,
fresh-path simulation, undefined moments and parameter-specific summaries.
Shared workflows own release-aware selection, statistics, differences, artifact
schemas, structural validation and computational requirements. Metadata records
inference semantics, helper sources, fields and plot interpretation. Plotting
reads that saved metadata and numerical artifacts without instantiating a model.
The two earlier artifact kinds retain explicit legacy presentation support.

The existing historical-evidence providers are `NormalMeanVariance` with
`mcmc=None`, `LocalLevelVariance`, and [the categorical-W joint model](joint-evidence.md). Unsupported models continue to forecast through their
existing capabilities; requiring unavailable separate evidence fails visibly.
This bounded interface preserves the existing mean/variance/state records. It
is not a universal parameter schema, configurable arbitrary-stage dispatcher,
sampler registry, exploratory override policy or release framework. Further
extensions should follow a concrete model/evidence need.

[Sequential checks](sequential-checks.md) also implement a required development workflow; their predictive laws are frozen before scoring and use full predictive CDFs.

## Review completed evidence together

[Saved experiment review](experiment-review.md) links an explicitly supplied
forecast and bundle, verifies child receipts and required checks, and optionally
includes separate calibration exercises. Missing/failed work blocks readiness;
computation never supplies the human decision. Existing sources are unchanged.
