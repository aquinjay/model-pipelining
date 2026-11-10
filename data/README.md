# Local data

Keep input snapshots here while developing. Git ignores other contents of this directory by default; this orientation file is tracked.

Reproduction will require an explicitly preserved snapshot or reliable versioned reference, including source and retrieval/availability assumptions. A hash alone cannot recreate data. Small synthetic fixtures may later be committed deliberately in a separate fixture location.

The [local snapshot module](../docs/snapshots.md) defaults to `data/versioned/`. Keep its entire directory tree together: it contains preserved files, manifests, registration records with any supplied ingestion receipts, and tag histories. These local contents are ignored by Git and are not uploaded when code is pushed to GitHub.

## Synthetic starting data

Generate the local datasets from the repository root with:

```bash
.venv/bin/python scripts/generate_synthetic.py
```

The generator uses only the Python standard library and writes to `data/synthetic/`, which is ignored by Git. It refuses to overwrite existing dataset files; use `--output-dir` to generate a separate copy.

Both CSV files contain 200 daily observations starting on 2020-01-01, with columns `timestamp` (UTC) and `value` (arbitrary units):

- `constant_mean.csv`: mean 10 with independent Gaussian noise of standard deviation 1.
- `mean_shift.csv`: the same noise, with the mean changing from 10 to 15 at observation 101 (2020-04-10).

The shared noise intentionally isolates the effect of the mean shift. Seed 42 and generation settings, Python version, timing assumptions, and file hashes are saved in `manifest.json`. Use the pinned Python interpreter when reproducing the files.

Each observation is assumed available at its timestamp. Forecasts should be issued after observing the current value and predict the next day's value. Generation settings describe simulation truth; they should not be supplied as model inputs. These datasets contain no missing values, seasonality, or publication delays.

## Copy and preserve an input

After generating the source, launch Python from the repository root with `PYTHONPATH=src .venv/bin/python`:

```python
from model_pipelining.ingest import ingest_local_file
from model_pipelining.snapshots import register_snapshot

receipt = ingest_local_file(
    "data/synthetic/mean_shift.csv",
    "data/ingested/data-readme-example/mean_shift.csv",
)
snapshot = register_snapshot(
    receipt.destination,
    dataset="synthetic-mean-shift",
    ingestion_receipt=receipt,
)
```

Choose an unused destination when repeating the example. Copying creates the receipt; registration saves it automatically when you pass `ingestion_receipt=receipt`. You do not save a separate receipt file. Omitting that argument leaves the original ingestion context unknown, and old registrations are not filled in automatically.

| Location | Contents |
|---|---|
| `data/synthetic/` | Generated source files and their generation manifest |
| `data/ingested/` | Completed copies at caller-selected destinations |
| `data/versioned/<dataset>/snapshots/<id>/` | Preserved bytes and snapshot manifest |
| `data/versioned/<dataset>/acquisitions/` | Registration records, including any supplied original ingestion receipt |
| `data/versioned/<dataset>/tags/` | Explicit tag assignments and their history |

Each saved receipt retains the original source, copy destination, UTC copy completion time, and byte count. Registration time is separate. Receipt details do not change the snapshot ID, which identifies the preserved bytes. See the [complete function guide](../src/model_pipelining/README.md#one-complete-workflow) for tagging, retrieval, and inspection.
