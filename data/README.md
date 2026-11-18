# Local data

The [explicit experiment evidence runner](../docs/evidence.md) uses this same
verified store and pins a selected tag once across its required workflows.
It does not acquire or register new input. Generated evidence stays under the
caller-selected run directory, separately from preserved snapshots.

Keep input snapshots here while developing. Git ignores other contents of this directory by default; this orientation file is tracked.

Reproduction requires the preserved snapshot and its dataset/ID reference, together with the relevant source and availability assumptions. A hash alone cannot recreate data. Small synthetic fixtures may later be committed deliberately in a separate fixture location.

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

The shared noise isolates the effect of the mean shift. Seed 42, generation settings, Python version, timing assumptions, and file hashes are saved in `data/synthetic/manifest.json`. This generator manifest describes simulation truth; it is separate from a snapshot manifest and from a run manifest. Use the pinned interpreter to reproduce source files.

Each observation is assumed available at its timestamp. Forecasts should be issued after observing the current value and predict the next day's value. Generation settings describe simulation truth; they should not be supplied as model inputs. These datasets contain no missing values, seasonality, or publication delays.

The [first forecasting workflow](../docs/forecasting.md) now implements this convention with a training-mean model. After explicitly registering/tagging a CSV, the backtest pins its ID, validates the daily calendar and values, and predicts each remaining day after 30 initial observations by default. A 200-row input therefore produces 170 forecasts. No input acquisition, registration, or tag assignment happens automatically when starting a run.

The [Bayesian local-level adapter](../docs/local-level.md) uses the same inputs and forecast dates. Its prior/variance settings are explicit model configuration, independent of generator truth. Its [shared quantiles and highest-density regions](../docs/uncertainty.md) describe future observed values, not hidden generator states, and remain conditional on fixed settings. The saved-result comparison requires matching permanent inputs and cases. The local-level script also saves a separate simulated prior trajectory under its run directory, not as an acquired input snapshot.

The separate [influence diagnostics](../docs/diagnostics.md) use a verified CSV
snapshot and process all observations, including initialization. They save
model evidence under an explicitly chosen output folder, not a new acquired
input. Reference/scenario settings are caller choices; the diagnostic does not
read the generator's hidden settings or automatically register/tag inputs.

[Experiment specifications](../docs/experiments.md) can instead name target/time
columns and a release-time column, select historical revisions as of issuance,
and choose first-release or latest-in-snapshot actuals for scoring. A versioned
snapshot can preserve an ordered long release table; registration itself does
not interpret or invent release timestamps. Calendar gaps remain rejected.
Acquisition/registration times do not substitute for release histories. The
default sample files remain immediate, unrevised daily fixtures.

The [unknown-mean adapter](../docs/parameter-uncertainty.md) uses these same
preserved inputs while learning a constant mean and integrating its uncertainty.
Its optional MCMC audit saves paired parameter/future-observation draws under
the run, not as new acquired data. The mean-shift series violates its constant-
mean assumption; sampler agreement with exact inference does not approve that
assumption. Generator settings are never read by the adapter.

The [mean-and-variance adapter](../docs/normal-mean-variance.md) also learns the
observation variance from eligible input values, using an explicit
normal/inverse-gamma prior. It returns exact Student-t observation forecasts.
Its joint sampler saves parameter/future draws in the run directory. Neither
adapter reads the generator truth or changes input acquisition/preservation.

[Predictive checks](../docs/predictive-checks.md) generate whole simulated
datasets under a separate check folder in `runs/`, not as acquired snapshots.
Prior-only checks need no data. Observed checks verify a selected snapshot and
choose only the releases available by an explicit development as-of time;
declared final periods are excluded. They retain selected records/history hash
for replay and never read generator truth. The source store remains unchanged.

[Prior comparisons](../docs/prior-influence.md) use identical histories for
every chosen prior. Each daily issuance selects only then-available releases;
later revisions never rewrite earlier updates. The explicit final cutoff uses
the same development-history boundary as predictive checks. Selected release
records/hashes are saved under `runs/`; no new acquired input is created.

[The learned-variance local-level model](../docs/local-level-variance.md) uses
the same preserved sources and as-of histories, with fixed evolution variance
and learned observation variance. Its predictive checks save fresh latent
trajectories as well as observations. Posterior replication draws initial state
and variance jointly, then resimulates process innovations; it does not copy a
fitted state path. Smoothing is confined to the selected development history
for these checks. Forecast filtering never receives future actuals.

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

[Sequential checks](../docs/sequential-checks.md) use historical as-of training releases and cutoff-eligible scoring releases from the preserved snapshot. Input preparation and registration remain caller actions.

[Simulation calibration](../docs/calibration.md) generates local prior/data trials without source snapshots. It saves simulated truth and observations in the selected run directory before fitting; no external/final data is used.

The [controlled dynamics study](../docs/joint-variance.md) deliberately generates and registers stable/gradual/abrupt fixtures with shared noise under its selected run directory. Final targets remain excluded from its development comparisons; these known fixtures do not establish an untouched application evaluation.
