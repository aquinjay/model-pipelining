# Behavioral verification

No executable tests exist yet. Add them with the behavior they verify.

Priorities for the first slice:

- Forecasts use only information available at issuance.
- Prediction keys and target times align with the requested horizon.
- Changing future actuals cannot alter an earlier forecast.
- Invalid adapter output and failed folds remain visible.
- Missing or errored required checks cannot appear as successful validation.
- Saved artifacts reconstruct a run within stated numerical tolerance.
- Reports read saved evidence without model fitting.

Model-specific verification, such as Bayesian recursion checks or synthetic state recovery, belongs with the relevant implementation.
