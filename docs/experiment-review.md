# Saved experiment review

Collect a forecast, its required historical evidence and any model calibration
exercises into one report. This is an explicit saved-only operation: it does not
fit models, simulate data, retrieve snapshots, run missing workflows or approve a
model. Source folders remain unchanged; retain them alongside the report.

```bash
PYTHONPATH=src .venv/bin/python scripts/review_experiment.py \
  --forecast-directory runs/my-forecast \
  --evidence-directory runs/my-evidence \
  --calibration-directories runs/calibration-original runs/calibration-followup \
  --require-calibration \
  --output-directory runs/my-review
```

Replace the example paths with existing artifact folders. The output must be new
and separate from every source tree. Calibration and evidence folders are supplied
explicitly, without searching local runs. Omit `--require-calibration` to make
omitted calibration optional. Historical workflows declared in the forecast's
experiment remain required even if no evidence folder is supplied. Supplied failed
or incompatible sources always block readiness.

## Identity and scope

The anchor is a finished version-3 forecast with an explicit experiment. The report
rechecks its experiment hash, required checks and prediction statuses. For models
with fit evidence it rechecks every indexed fit's model, input, origin, history,
training count, status and caller-required check names.

Historical evidence must match the dataset/snapshot, expanded model settings,
task, evidence plan, benchmarks and development/final boundaries. Its own role is
development: a final-role forecast may link earlier development evidence. The
selected history hash, availability/order, cutoff and declared minimum history
are rechecked. This does not claim that one evidence cutoff covers every forecast
origin or prove that final data were untouched during research.

Every required child must have the expected kind/settings/context and named
checks. Existing bundle receipts verify child manifest/check bytes against their
completion receipts. Calibration matches the exact model configuration and must
retain all declared trial/parameter rows. It is a prior/data model-level exercise,
not external-data validation or verification at every historical sample size.
Original and follow-up exercises stay separate, with their own seeds and summaries.

Shared declared implementation hashes and environment versions must agree.
Existing bundle/sequential formats omit environment versions; these unknowns are
listed in `version_omissions`, rather than assumed equal. Historical artifacts are
compared to each other, not to today's working tree. Current SHA256 inventories
snapshot all source bytes; they do not authenticate old unreceipted numerical
files. Symlinks/path escapes are rejected and inventories are checked again before
writing, to detect concurrent source changes.

## Output and interpretation

| File | Purpose |
|---|---|
| `report.md` | Readable case/score, workflow and calibration summaries, source links and existing PNG links |
| `report.json` | Complete identities, checks, file hashes, source reviews, conflicts, omissions and scope limitations |
| `checks.json` | Report requirements, separate from each source's checks |
| `review.json` | New pending human decision; never inferred from computations or copied approval |

`ready_for_review` means supplied computations and requested evidence are
compatible and complete. `incomplete` means a requested source/workflow is missing.
`failed` means a supplied artifact is incompatible or its computation failed. The
CLI returns zero only for `ready_for_review`; missing/failed reports return one.
Malformed anchors, unsafe paths and concurrent modification raise errors.

The Markdown links existing charts; it does not create them. Run the relevant
plotting function first if charts are wanted. Reviewers still interpret predictive
discrepancies, prior influence, finite-trial uncertainty and modeling assumptions.
No score threshold, automatic ranking or release decision is introduced.

## Verification

Eighteen tests cover saved-only construction, untouched sources, final/development
linking, missing requirements, child receipts, model/input/history/version
mismatches, failed computations, indexed fit requirements, incomplete calibration,
nonfinite data, unsafe paths, concurrent changes and CLI statuses. A local exercise
combined 130 development forecasts, all three historical workflows, 200 predictive
replications per phase and two 50-trial calibration runs with saved charts. These
small calibration exercises validate assembly mechanics, not calibration claims.

## Full-law scores

For forecasts declaring full distributions, review also checks required scoring
checks, frozen-law and score case identities, actual vintages, finite losses and
mean aggregates. Markdown displays mean CRPS/log loss when complete. This reads
saved artifacts and does not recalculate the model or authenticate files. See
[distribution scoring](distribution-scoring.md).
