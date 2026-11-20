# Agent handoff: changes since `c3fe4ff`

Written 2026-11-19 at `main` = `9dbc5ad`. Read this before starting work if
you last saw the repository at or before `c3fe4ff`, the paired NOAA
minimum/maximum study. Check `git log --first-parent c3fe4ff..main` for
anything newer than this note. The [checkpoint note](next-session.md) remains
the full project status. This page covers only what changed, the conventions
those changes introduced, and the open decision.

## Starting point

At `c3fe4ff` the NOAA work (2024 seasonality, frozen 2025 evaluation and the
minimum/maximum pair) sat unmerged on `feature/noaa-temperature-pair`. A
codebase review then found one latent bug and several maintenance issues. All
of them were fixed without changing forecasts or saved file contents.

## What changed

All changes merged into `main` through pull requests with merge commits. CI
passed on each.

| PR | Merge | Change |
|---|---|---|
| [#9](https://github.com/aquinjay/model-pipelining/pull/9) | `65035a1` | Merged the NOAA seasonality, frozen 2025 evaluation and min/max pair studies unchanged |
| [#10](https://github.com/aquinjay/model-pipelining/pull/10) | `6dd958b` | Fixed saved MCMC settings not reloading; `from_configuration` on every adapter; script model catalog; calibration provider check |
| [#11](https://github.com/aquinjay/model-pipelining/pull/11) | `e54def4` | Benchmark and workflow names defined once; shared helpers made public; one JSON/CSV writer module |
| [#12](https://github.com/aquinjay/model-pipelining/pull/12) | `5162d72` | Weekly harmonic adapter moved into the package; `scripts/__init__.py` added |
| [#13](https://github.com/aquinjay/model-pipelining/pull/13) | `fc59e0b` | Second CI job running the suite on the Python 3.12 workplace profile |
| [#14](https://github.com/aquinjay/model-pipelining/pull/14) | `9dbc5ad` | Refreshed the roadmap scope table and the decisions "Current choices at a glance" |

Each code change has a dated entry at the end of the [decision log](decisions.md).

### Bug fixed

`NormalMean` and `NormalMeanVariance` saved `mcmc` as a dictionary in
`configuration`, but their constructors only accept `MCMCSettings`. As a
result, `NormalMeanVariance(**saved_configuration)` raised `ValueError` for any
run with MCMC enabled. Both now have `from_configuration`, which rebuilds
`MCMCSettings`.

### New and moved code

| Location | What it is |
|---|---|
| `src/model_pipelining/adapters.py` | `MODEL_TYPES` (name to class), `EVIDENCE_MODEL_NAMES`, `build_model(name, configuration)`, `build_scenarios(reference, overrides)`. Scripts use it; the runner and evidence workflows never do |
| `from_configuration(configuration)` | Now on all six repository adapters: `NormalMean`, `NormalMeanVariance`, `LocalLevel`, `LocalLevelVariance`, `LocalLevelJointVariance`, `WeeklyHarmonicRegression` |
| `src/model_pipelining/artifacts.py` | `write_json(path, value, *, overwrite=False)` and `write_csv(path, fields, rows)`. Files are created exclusively by default; only a workflow's final manifest rewrite passes `overwrite=True` |
| `src/model_pipelining/harmonic_regression.py` | `WeeklyHarmonicRegression`, moved from `scripts/run_harmonic_example.py` with its name and configuration unchanged |
| `models.BENCHMARK_MODELS` | `{"training-mean": TrainingMean, "last-value": LastValue}` |
| `experiments.EVIDENCE_WORKFLOWS` | `("predictive_checks", "prior_influence", "sequential_checks")` |
| `evidence.PROVIDER_WORKFLOWS` | `EVIDENCE_WORKFLOWS + ("calibration",)` |
| `scripts/__init__.py` | Makes `python -m scripts.…` use a regular package |
| `tests/optional_dependencies.py` | `requires(*modules)`, `optional_module(name)`, `SAMPLER_PACKAGES` |

### Renamed or removed names

Code or notes that use the old names will fail:

| Old | New |
|---|---|
| `backtest._write_json`, `backtest._write_csv`, `evidence._write_json`, `evidence._write_csv`, `diagnostics._write_json` | `artifacts.write_json`, `artifacts.write_csv` (CSV takes explicit `fields`) |
| `backtest._read_observations` | `experiments.read_observations` |
| `predictive_checks._tail_summary` | `predictive_checks.tail_summary` |
| `evidence_workflows._MANIFESTS` | `evidence_workflows.WORKFLOW_MANIFESTS` |
| `evidence_workflows._REQUIRED_CHECKS` | `evidence_workflows.REQUIRED_WORKFLOW_CHECKS` |
| `NormalMeanVariance(**config)` in scripts | `adapters.build_model(name, config)` or `<Adapter>.from_configuration(config)` |

### Behavior changes

Forecasts, scores and saved file bytes are unchanged; tests check the writers
byte for byte. The behavior differences are:
- Runner output files other than the final `manifest.json` are now created
  exclusively, matching the evidence workflows.
- A JSON value that cannot be serialized no longer leaves a partial file.
- Calibration provider selection raises `ValueError` when
  `calibration_parameters` is missing, empty, duplicated or blank.
- Source-hash inventories in new manifests include `artifacts.py`.
  Diagnostics list `experiments.py` in place of `backtest.py`. Harmonic runs
  record `harmonic_regression.py` and the class path
  `model_pipelining.harmonic_regression.WeeklyHarmonicRegression`. Run
  directories saved before these changes report the edited modules as
  changed sources, and `scripts/continue_model_review.py` will not reuse
  completed parts of earlier studies.

## Conventions to follow now

- **New adapter:** give it `from_configuration`. Register it in
  `adapters.MODEL_TYPES` only if scripts must build it by name.
- **Names:** add benchmark or workflow names to `BENCHMARK_MODELS` or
  `EVIDENCE_WORKFLOWS`, not as new literals.
- **Saved files:** write JSON and CSV through `artifacts.py`.
- **Tests needing `emcee`, `arviz` or `openpyxl`:** decorate them with
  `requires(...)` from `tests/optional_dependencies.py`. Import those packages
  lazily or through `optional_module`.
- **Python 3.12:** shared code must keep running on CPython 3.12. A dependency
  change updates both `uv.lock` (with `requirements.txt`) and
  `requirements-python312.txt`.
- **Frozen NOAA scripts:** do not edit `scripts/check_noaa_later_period.py`,
  `scripts/run_noaa_temperature.py` or `scripts/study_noaa_seasonality.py`. The
  frozen 2025 plan stores their SHA-256 hashes, and `evaluate` refuses to run
  if any changes. Those scripts retain their original implementations. Use
  the [packaged temperature adapter and helpers](../src/model_pipelining/README.md#temperature-regression-and-noaa-inputs)
  for new work.
- **Branches:** merged feature branches are kept on GitHub, so a stacked PR is
  not retargeted automatically when its base merges. Retarget it with
  `gh pr edit <n> --base main`.

## Verification state at `9dbc5ad`

| Environment | Tests | Skipped | Result |
|---|---|---|---|
| CPython 3.14.10 research lock, `REQUIRE_OPTIONAL_TEST_DEPENDENCIES=1` | 475 | 0 | pass |
| CPython 3.12.16 workplace profile | 475 | 31 (MCMC and XLSX) | pass |

`scripts/check_documentation.py` reports 733 local links and anchors valid at
that commit. CI runs both jobs automatically on every push and pull request;
see the [CI guide](continuous-integration.md). Local commands:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/check_documentation.py
```

## Open decision for the user: innovation integration gate

Three documents still say forecast or provider integration of the rare-jump
innovation model "remains gated":
[innovation-particles.md](innovation-particles.md),
[innovation-probability-bounds.md](innovation-probability-bounds.md) and the
roadmap section on probability bounds. Later work changed the picture:
- the [state-grid method](innovation-lattice.md) meets the probability
  criteria on all 17 cases;
- an independent Gibbs check agreed on `abrupt-42`;
- a later decision chose the existing Gaussian local-level adapters for the
  first BDLM workflow.

No entry says whether the gate still applies. The proposed wording is: the state
grid meets the probability criteria, but integrating the rare-jump model stays
deferred until a concrete task needs it. **The user has not decided this.
Do not change the gate wording without their approval.**

## Next work

The [range-probability study](noaa-range-probability.md) is complete. The user
delegated the design choices, and the plan froze January–June 2026, a 10 °C
threshold and paired Brier scoring before retrieval. The 181-case run favors
independent pairing on Brier score, while same-day pairing assigns less
probability to negative ranges. The [checkpoint](next-session.md#next-work)
records the outcome and the archived inputs. Freeze a new design before
retrieving observations for a follow-up evaluation.

The [seasonal follow-up](noaa-range-probability.md#seasonal-calibration-follow-up)
diagnosed the April–May loss gap on saved dates and ran a frozen ten-day July
pilot. Season matching reduced negative-range probability and increased Brier
loss for both pairing methods in the pilot. A separate July 11–December 31
plan was frozen before retrieval and awaits its complete archive.
