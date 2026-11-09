# Python implementation

Start reading `ingest.py` in this order:

1. `IngestResult` describes a completed file copy: source, destination, UTC copy time, and byte count.
2. `ingest_local_file` copies one file, rejects existing destinations, and cleans up a partial copy after a handled failure.
3. `read_local_table` reads a CSV or one XLSX worksheet into a pandas DataFrame. It offers worksheet selection and a header-row option.

Use imports from `model_pipelining.ingest`. The project does not yet have installable packaging; run from the repository root with `PYTHONPATH=src` after installing the dependencies into `.venv`.

See [the ingestion notes](../../docs/ingestion.md) for examples, limits, and later enhancements, and [the tests](../../tests/test_ingest.py) for executable examples. No model runner or snapshot registration has been implemented.
