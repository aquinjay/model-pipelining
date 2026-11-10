# Behavioral verification

Run all tests from the repository root after installing dependencies:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The current suite has 57 tests: 17 ingestion, 27 snapshot, and 13 tag tests. The receipt workflow is covered end to end: copying returns a receipt, and passing `ingestion_receipt=receipt` to registration automatically saves it with the registration event. Omitting it records `None`; older records also return `None` without being rewritten. See the [documented workflow](../src/model_pipelining/README.md#one-complete-workflow).

`test_ingest.py` uses standard-library unittest and temporary CSV/XLSX fixtures. It verifies exact copies, overwrite protection, failure cleanup, acquisition receipts, CSV/XLSX table equivalence, worksheet selection, headers, and invalid requests. Test fixtures do not depend on local generated data.

`test_snapshots.py` verifies single-file registration, separate records for matching contents, revised historical values, verified retrieval, tampering detection, failed-copy cleanup, dataset boundaries, store relocation, and XLSX reading. It also checks listing and ordering histories, notes and ingestion receipts that do not change identity, old records without optional fields, and read-only inspection of missing or malformed stores. Receipt checks cover original context after source removal or store relocation, distinct acquisition/registration times, UTC normalization, mismatched destinations, changed file sizes, invalid types, and naive timestamps. It uses temporary fixtures and the same unittest harness.

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
