# Model Pipelining

A Python toolkit for repeatable forecasting runs. Preserve an input file, choose
a model, make historical predictions using only the allowed past data, and save
the predictions, checks, errors, and uncertainty together.

The shared infrastructure lets models use the same data selection, execution,
evaluation, and reporting. The current models are small examples for developing
and verifying that infrastructure. Better scores alone do not establish that
an implementation is correct or that a model is suitable for an application.

## Start here

1. Follow [the forecasting walkthrough](docs/forecasting.md). It explains the
   workflow and runs both models against the same preserved input.
2. Read [the local-level model guide](docs/local-level.md) to understand the
   Bayesian model, its four fixed settings, and its calculations.
3. Read [the uncertainty guide](docs/uncertainty.md) for highest posterior density
   (HPD) regions, quantiles, their meaning, and their evaluation.

For a review of the recent modeling work, follow that same order. Then inspect
`models.py`, `local_level.py`, `uncertainty.py`, `backtest.py`, and `reporting.py`
in [the source directory](src/model_pipelining/). The [test guide](tests/README.md)
explains which checks support each implementation claim.

## What works today

| Part | Current behavior |
|---|---|
| Input preservation | One local file per snapshot; verified retrieval, registration history, optional copy receipts, and explicit dataset tags |
| Table reading | CSV or one selected XLSX worksheet; forecasting currently accepts CSV only |
| Forecasting task | One daily numerical series, next-day predictions, expanding training history, 30 initial observations by default |
| Training mean | Average the eligible history; supplies a point prediction only |
| Bayesian local level | Estimate a changing hidden level using a Gaussian prior and fixed noise variances; supplies a mean, quantiles, and HPD prediction regions |
| Shared execution | Validate inputs/outputs, freeze predictions before scoring, retain failed cases, and save configuration and checks |
| Reporting | Read saved results to draw forecasts or compare complete runs on matching cases; prefer HPD regions when available |

The forecast uncertainty describes the **next observed value**, conditional on
the model's assumptions. It is separate from uncertainty about the hidden level
or a model parameter. The sample task assumes immediate observation availability
and no revisions. All example evaluations are development evidence; there is no
untouched final test period or automatic model approval.

## Workflow and caller actions

```text
Choose a forecasting question and its timing assumptions
  → obtain a completed input file
  → preserve it as a snapshot; optionally assign a readable tag
  → run historical forecasts with one model
  → inspect saved checks, predictions, scores, and assumptions
  → report or compare saved results
```

You obtain/register the input and choose model settings explicitly. Starting a
run does not fetch data, register a file, or move a tag. The runner verifies the
selected snapshot, selects eligible histories, executes the model, and saves
results automatically. Library callers invoke reporting separately; the two
run scripts also draw their charts automatically. The local-level script adds
a seeded prior simulation for human review.

A completed run means the required computational checks and scoring succeeded.
It does not decide whether the assumptions are useful. `review.json` starts as
a pending human review record. The [walkthrough](docs/forecasting.md#read-the-results)
explains how to read both execution and review status.

## Documentation map

| Need | Read |
|---|---|
| Run and understand the complete example | [Forecasting walkthrough](docs/forecasting.md) |
| Understand the Bayesian model and its equations | [Local-level model](docs/local-level.md) |
| Understand HPD regions, quantiles, and output rules | [Shared uncertainty](docs/uncertainty.md) |
| Look up a function, argument, or return type | [Python API reference](src/model_pipelining/README.md) |
| Copy/read a local CSV or workbook | [Ingestion](docs/ingestion.md) |
| Preserve, retrieve, or tag a file | [Snapshots and tags](docs/snapshots.md) |
| Find saved inputs or run files | [Data directory](data/README.md), [run directory](runs/README.md) |
| Understand module responsibilities and future design | [Architecture](docs/architecture.md) |
| See current status and review priorities | [Roadmap](docs/roadmap.md) |
| Understand previous choices | [Decision history](docs/decisions.md) |
| Read the earlier, broader storage proposal | [Historical data-versioning proposal](docs/data-versioning.md) |

The walkthrough describes current behavior. Architecture sections marked as
planned describe future responsibilities. Decision history and the historical
storage proposal explain earlier choices; they are not setup instructions.

## Development environment

Use an isolated environment named `.venv`. The setup supports standard Python tooling and an optional uv workflow; the pipeline itself will not depend on uv.

| Setting | Starting choice |
|---|---|
| Supported interpreter | Standard CPython 3.14 (`>=3.14,<3.15`) |
| Recommended development patch | 3.14.10, recorded in `.python-version` |
| Project version | 0.1.0, recorded in `pyproject.toml`; the API is still being designed |
| Runtime dependencies | pandas, openpyxl, and matplotlib; exact versions in `uv.lock` and `requirements.txt` |
| Test harness | Standard-library unittest |
| Local environment | `.venv/`, excluded from Git |

Create an isolated environment using either setup path below. The pinned environment has been exercised locally; verify it on the machine where you will run experiments.

`pyproject.toml` declares dependencies. `uv.lock` records the resolved versions. `requirements.txt` is a generated, pip-compatible export of that resolution, including development groups when present. Commit all three files and `.python-version`; keep the environment, caches, credentials, local data, and generated run outputs out of Git.

Python, project, and dependency versions are separate choices. Update the interpreter pin deliberately when adopting a newer patch. Adding a dependency requires updating the declaration, lockfile, and pip export together. pandas and matplotlib bring their numerical dependencies; additional model-fitting, notebook, and formatting tools remain unselected.

### Standard Python setup: venv and pip

Install an approved CPython 3.14 interpreter first; use 3.14.10 to match the development pin. The following commands assume a Linux/macOS shell and that `python3.14` is on your path. Unlike uv, `venv` and pip do not automatically select an interpreter from `.python-version` or enforce `requires-python` when installing a requirements file.

```bash
cd /path/to/model-pipelining
python3.14 --version
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys; assert sys.version_info[:2] == (3, 14); assert sys.prefix != sys.base_prefix; print(sys.version); print(sys.executable)"
```

The version check enforces the supported minor version; compare the printed patch with `.python-version` for an exact match. If your Python installation lacks `venv` or pip, obtain those components through your environment's normal provisioning process.

On Windows PowerShell, replace environment creation and activation with:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then use the same `python -m pip` and verification commands. Workflow examples
use Linux/macOS shell notation. In an activated PowerShell environment, set
`$env:PYTHONPATH="src"`, use `python` instead of `.venv/bin/python`, and place
multi-line shell commands on one line. To open the Python session used by the
walkthrough:

```powershell
$env:PYTHONPATH="src"
python
```

Select `.venv` as your editor/notebook interpreter. Run `deactivate` to leave an
activated environment.

At work, use the approved interpreter and package source with this pip route. A different Python minor version requires a separate compatibility decision and dependency resolution; a requirements export alone does not establish compatibility with that runtime.

### Optional setup with uv

If uv is installed, it can provision the pinned interpreter and create the environment from the lockfile:

```bash
cd /path/to/model-pipelining
uv python install 3.14.10
uv sync --locked
uv run --locked python -c "import sys; assert sys.version_info[:3] == (3, 14, 10); assert sys.prefix != sys.base_prefix; print(sys.version); print(sys.executable)"
```

`uv run` uses the project environment without requiring shell activation. Alternatively, activate `.venv` using the standard commands above. The first sync may need network access to provision the interpreter or dependencies.

### Maintaining dependency versions

Use `pyproject.toml` as the editable dependency declaration. With uv available, add an agreed runtime package with `uv add PACKAGE`, or a development tool with `uv add --dev PACKAGE`. Review the resulting changes and regenerate the pip export:

```bash
uv lock
uv export --locked --all-groups --no-emit-project --format requirements.txt --output-file requirements.txt
```

The export is tracked so a machine without uv can install the selected versions using pip. It includes the table-reading and plotting dependencies and their transitive requirements. Do not hand-edit the generated export or use an unreviewed `pip freeze` of an unrelated environment as the project's dependency definition.

A pip-only work machine can consume the committed export without uv. If it needs a different dependency set or approved versions, make an explicit configuration change and regenerate the lock/export in an environment with the resolution tools available. pip installs from the export; it does not update `uv.lock`.

Use a fresh `.venv` when checking reproduction with pip: `pip install -r` does not remove unrelated packages already installed. uv's `sync --locked` reconciles the environment with the lockfile. Dependencies may also require platform support, so validate the selected stack on the actual work runtime when it becomes known.

Setup references: [Python's packaging guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), [uv project guide](https://docs.astral.sh/uv/guides/projects/), and [exporting a uv lockfile for pip](https://docs.astral.sh/uv/concepts/projects/export/).

## Development checks and contributions

Run the behavioral suite from the repository root after setup:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests check calculations, future-data isolation, reproducible outputs, visible
failures, and reporting without model calls. See [the test map](tests/README.md)
for details. There is no CI workflow yet.

Keep changes small enough to review. Follow [AGENTS.md](AGENTS.md), update affected
documentation in the same task, and record consequential design decisions in
[the decision log](docs/decisions.md). Details of a future real-data task,
probabilistic extensions, and model-specific checks belong in
[the roadmap](docs/roadmap.md), not as implied features in usage examples.

## Repository and license

Repository: [aquinjay/model-pipelining](https://github.com/aquinjay/model-pipelining).
Licensed under [Apache License 2.0](LICENSE). The Bayesian workflow background
and mathematical references are collected in [references](docs/references.md).
