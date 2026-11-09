# Behavioral verification

Run the ingestion tests from the repository root after installing dependencies:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`test_ingest.py` uses standard-library unittest and temporary CSV/XLSX fixtures. It verifies exact copies, overwrite protection, failure cleanup, acquisition receipts, CSV/XLSX table equivalence, worksheet selection, headers, and invalid requests. Test fixtures do not depend on local generated data.

Priorities for the future forecasting slice:

- Forecasts use only information available at issuance.
- Prediction keys and target times align with the requested horizon.
- Changing future actuals cannot alter an earlier forecast.
- Invalid adapter output and failed folds remain visible.
- Missing or errored required checks cannot appear as successful validation.
- Saved artifacts reconstruct a run within stated numerical tolerance.
- Reports read saved evidence without model fitting.

Model-specific verification, such as Bayesian recursion checks or synthetic state recovery, belongs with the relevant implementation.
