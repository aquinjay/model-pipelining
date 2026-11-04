# Local data

Keep input snapshots here while developing. Git ignores other contents of this directory by default; this orientation file is tracked.

Reproduction will require an explicitly preserved snapshot or reliable versioned reference, including source and retrieval/availability assumptions. A hash alone cannot recreate data. Small synthetic fixtures may later be committed deliberately in a separate fixture location.

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
