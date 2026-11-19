# Repository checks in GitHub Actions

[The workflow](../.github/workflows/ci.yml) defines one Ubuntu 24.04 job named
`Python tests and documentation`. Once pushed, branch pushes and pull requests
start it automatically; `workflow_dispatch` also permits an explicit manual run.
All branches are included so feature branches receive the same checks as main.
A newer run for the same branch/pull request cancels an obsolete in-progress run.

## What runs automatically

1. Check out the requested revision with read-only repository permission and
   without persisting Git credentials. Third-party actions use full commit pins.
2. Install uv 0.13.7, install the Python version in `.python-version`, and run
   `uv sync --locked --all-groups`. Verify the exact interpreter patch and isolated
   environment, then check dependency compatibility.
3. Regenerate the pip requirements export in temporary storage and compare it
   with `requirements.txt`, ignoring the generated command header. A declaration,
   lock or export mismatch fails the job instead of changing tracked files.
4. Check repository Markdown local links and section anchors, parse Python/Bash
   fenced examples, and execute the forecasting walkthrough in temporary storage.
5. Run the complete `unittest` suite, including independent numerical references,
   information boundaries, failure cases, saved-result replay and CLI tests.
6. Require that checks leave tracked files unchanged.

Writable Matplotlib, ArviZ and uv caches live under the runner's temporary
folder. The uv download cache is reused by lock/interpreter identity; the virtual
environment is synchronized on each run. Tests and the walkthrough create their
own local fixtures/stores. GitHub retains the ordinary job logs. The timeout is
15 minutes. The defined runtime is Linux x86-64/CPython 3.14; this is not evidence
of compatibility on an untested work-machine runtime.

## Run the same checks locally

After installing the pinned interpreter and uv:

```bash
uv sync --locked --all-groups
uv pip check --python .venv/bin/python
PYTHONPATH=src uv run --locked --no-sync python scripts/check_documentation.py --walkthrough
PYTHONPATH=src uv run --locked --no-sync python -m unittest discover -s tests -v
```

Use the [documented writable cache settings](parameter-uncertainty.md#run-exact-forecasts-or-request-an-audit)
if the home cache directory is restricted. The workflow contains the exact
runtime/export comparisons. With standard pip, the existing committed requirements
setup and `.venv/bin/python` commands remain available; uv is a CI/setup tool,
not part of the forecasting library API.

`scripts/check_documentation.py` uses the standard library and Bash. It discovers
tracked and new Markdown through Git while excluding ignored generated data/runs
and dependency folders. It checks local inline/reference/image links, ATX heading
anchors (including repeated headings), and plain `python`, `bash` or `sh` fenced
example syntax. Inline and fenced code are excluded from link extraction. This
is a bounded repository convention checker, not a complete Markdown renderer.

Without `--walkthrough`, examples are parsed without execution. With that flag,
the trusted Python blocks in `docs/forecasting.md` run verbatim in a temporary
working directory after the sample generator creates fresh inputs. It checks both
170-case runs and saved charts/comparison, then removes that temporary tree.
Other examples receive syntax checks; behavioral tests cover the supported CLI
and library workflows. Failed links, anchors, syntax or execution return a
nonzero process status.

## Scientific evidence and review

The independent Gibbs checks compare the Gaussian state block against dense
calculations and exercise known state atoms. The [full cross-check and timing
study](innovation-moment-check.md) runs on explicit invocation.

The state-grid regression tests compare lattice likelihoods with dense Gaussian
references and check atoms, unseen-event intervals, replay and resource errors.
The full [state-grid study](innovation-lattice.md) runs on explicit invocation.

The conservative innovation-probability tests automatically check the independent
dense-likelihood envelope identity, unseen-event bounds, near-dominant labels,
atoms/Gaussian limits, invalid budgets, RNG isolation and bounded CLI/replay
fixtures. The [full saved-study follow-up](innovation-probability-bounds.md)
requires explicit caller execution and is not rerun or relabelled by CI.

The suite includes small seeded calibration/reference checks. Large original and
follow-up calibration studies remain explicit caller actions with saved evidence,
not an automatic CI deliverable. CI completion means repository computations and
examples passed in the stated environment. Model adequacy, interpretation of
predictive discrepancies/prior influence, scientific approval and merging remain
separate decisions. Repository branch protection is configured by a repository
administrator if a mandatory status check is wanted.

The [three-model development review](model-review-study.md), its large calibration
and alternative-prior simulations remain explicit research commands. CI runs the
small cohort/continuation guards and seeded quantile regression in the ordinary
suite; it does not rerun the large study or update its curated results.
The bounded innovation-reference tests also run both numerical methods, exact/
Gaussian limits and small CLI fixtures automatically. They require no saved local
study; the seven-case reference command and any larger scientific validation
remain explicit exercises.
The particle tests add small independent conditioning/proposal/resampling and
oracle checks, a deliberately unresolved 160-observation fit with tiny resource
limits, and temporary CLI persistence/replay exercises. The full 17-case
[particle study](innovation-particles.md) remains an explicit caller action; CI
does not regenerate its curated results or imply that its empirical uncertainty
checks have passed.

## Verification at introduction

A fresh temporary environment installed all 24 locked dependencies on Python
3.14.10; dependency checks and the requirements comparison passed. The exact guide
examples completed both 170-case runs and the saved comparison. Nine checker tests
cover stale links/anchors, encoded paths, reference links, ignored artifacts,
syntax errors, non-execution and CLI failure codes. The complete suite has 391
tests. The workflow passed local actionlint 1.7.7 validation, with its official
release checksum verified. Hosted push and PR runs subsequently passed before
[PR #6](https://github.com/aquinjay/model-pipelining/pull/6) merged at `0ff5def`.
See the [checkpoint note](next-session.md) for the run links and next tasks.

References: [GitHub Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python),
[uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/),
[checkout action](https://github.com/actions/checkout),
[setup-uv action](https://github.com/astral-sh/setup-uv).
