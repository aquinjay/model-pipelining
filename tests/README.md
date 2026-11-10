# Behavioral verification

Run all tests from the repository root after installing dependencies:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`test_ingest.py` uses standard-library unittest and temporary CSV/XLSX fixtures. It verifies exact copies, overwrite protection, failure cleanup, acquisition receipts, CSV/XLSX table equivalence, worksheet selection, headers, and invalid requests. Test fixtures do not depend on local generated data.

`test_snapshots.py` verifies single-file registration, separate records for matching contents, revised historical values, verified retrieval, tampering detection, failed-copy cleanup, dataset boundaries, store relocation, and XLSX reading. It also checks listing and ordering histories, notes that do not change identity, old records without notes, and read-only inspection of missing or malformed stores. It uses temporary fixtures and the same unittest harness.

`test_tags.py` verifies explicit assignments and movement, retained permanent references, dataset-scoped names, history and caller context, invalid or corrupt targets, failed-publication safety, store relocation, and assignment order despite clock changes. It uses temporary fixtures and injects a failure at the final file replacement to check preservation of the previous record.

Priorities for the future forecasting slice:

- Forecasts use only information available at issuance.
- Prediction keys and target times align with the requested horizon.
- Changing future actuals cannot alter an earlier forecast.
- Invalid adapter output and failed folds remain visible.
- Missing or errored required checks cannot appear as successful validation.
- Saved artifacts reconstruct a run within stated numerical tolerance.
- Reports read saved evidence without model fitting.

Model-specific verification, such as Bayesian recursion checks or synthetic state recovery, belongs with the relevant implementation.
