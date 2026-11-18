# Predictive checks and prior comparisons for joint variance learning

The [joint local-level model](joint-variance.md) now supplies whole-dataset
predictive replication and prior-influence capabilities as well as sequential
checks. The shared workflow and experiment requirement boundaries are unchanged.

Each prior dataset draws W from its categorical prior, V from its inverse-gamma
prior, and the initial state from its Gaussian prior. Each posterior dataset
first draws W from its posterior probabilities, then V and the initial state
from their conditional posterior. This preserves W/V/state dependence. It then
resimulates all process innovations and measurement noise. A fitted latent path
is never copied into a replication. Historical time gaps scale the fresh process
innovations. A W=0 draw has an exactly constant state path.

Predictive artifacts add `evolution_variance` to parameter draws and retain
`latent_levels.csv`. The new field is optional for older model records. Joint
providers must supply it for every completed replication. Distribution metadata
records the full joint integration evidence. Numerical failures retain prior
results and explicit failed posterior precision; descriptive discrepancies never
approve a model.

## Explicit prior comparisons

The caller can change categorical W prior probabilities, inverse-gamma V priors
and initial-state priors. W support and quadrature settings must stay identical:
changing support is a different assumption comparison, not the same prior path.

Learning paths add W mean/median/discrete credible bounds, full support
probabilities, signed W-summary differences and posterior V/W covariance. Existing
state/V/predictive quantities keep their meaning. Scalar prior-location
coefficients and Student-t degrees of freedom remain null for this model.
The new optional columns are null for earlier model families; old artifacts are
not rewritten.

`evolution_priors.png` shows support probabilities for each prior configuration.
The learning chart also shows expected W and V/W covariance. W equal-tail bounds
are support points and can include more than the nominal probability. State
summaries use a Gaussian/point mixture: a known initial state with W=0 contributes
an actual probability atom. Generalized quantiles preserve that atom instead of
inventing a narrow continuous variance or a density-based state HPD interval.
Observation predictions remain continuous because V>0 and retain mixture HPD.

## Run and read evidence

Use [the evidence runner](evidence.md) with the joint model:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_evidence.py \
  --model local-level-joint-variance \
  --dataset synthetic-mean-shift --tag baseline-input \
  --as-of 2020-06-08T23:59:59+00:00 \
  --experiment configs/daily_evidence_experiment.json \
  --config configs/local_level_joint_priors.json \
  --output-directory runs/joint-evidence-example-001
```

The config now includes illustrative prior-weight, observation-variance and
initial-level alternatives. All unchanged settings are restored from the
reference, including immutable support and numerical settings. The caller must
prepare the input, choose the config/experiment and invoke evidence explicitly.
The CLI saves the requested evidence and draws charts automatically. Library
callers invoke plotting separately. `compare_priors.py` and
`check_sequential.py` also accept `--model local-level-joint-variance`.

No shared workflow dispatches on the model's concrete type. Saved presentation
metadata controls optional evolution-prior charts, and older artifact readers
retain their existing behavior. Unsupported calibration is still rejected for
this joint model until a workflow handles its discrete/mixed probability masses.

Ten new tests check joint prior frequencies/covariance, posterior W/V dependence,
fresh innovations, omission failures, support restrictions, known-state atoms,
prior probabilities, future/final isolation, saved-only charts, precision
failures and an end-to-end CLI requiring all three historical workflows. The
suite contains 333 tests.

The initial joint exercise completed 1000 prior and 1000 posterior replications
on 60 selected development observations and three priors over 62 learning points.
All required computations completed; human review remains pending. Artifacts and
five charts are local under `runs/joint-evidence-validation-uvamnxxk/evidence/`.
All 333 tests passed.
