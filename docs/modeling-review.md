# Modeling and project direction review — 2026-11-17

This review follows completion of the learned local-level observation-variance
model, marginal prediction, predictive checks, prior comparisons, documentation
and verification. It assesses that completed work against the project's goal:
repeatable forecasting infrastructure with interchangeable models and
inspectable evidence. It does not select an application or approve a model.

## Assessment

The work is on track. The new adapter exercises a computation that the earlier
conjugate examples could not: numerical integration over an unknown parameter
while an evolving latent state is integrated analytically. It uses the existing
request, uncertainty, fit-evidence, failure, evaluation and saved-reporting
boundaries. Shared historical forecasting still does not branch on model names.
This adds evidence about the infrastructure as well as another model.

The progression from fixed settings to analytic parameter learning and then
controlled numerical marginalization is useful at this stage. A new MCMC backend
would not answer an additional question for this scalar integral. Existing
optional sampler audits remain available for the conjugate models; they are
not required for the new model's forecasts.

The separation of computational validity, model criticism and forecasting
evaluation is consistent with the iterative modeling and checking described in
Gelman et al.'s [Bayesian Workflow](https://arxiv.org/abs/2011.01808). The API and
next-step recommendation here are project-specific judgments, not requirements
from that paper.

| Project responsibility | Evidence from this slice | Remaining limit |
|---|---|---|
| Keep model calculations behind an adapter | Existing forecast requests/results carry integrated means, quantiles and HPD sets | Numerical integration and HPD resolution have explicit tolerances; this is not an exact closed-form predictive law |
| Preserve the information available at issuance | Release-aware histories, future/final isolation tests, and identical comparison histories | Supplied historical release claims are not independently verified |
| Make computation inspectable | Saved quadrature/refinement/tail evidence, explicit failed cases, independent numerical references | Finite test coverage and grid agreement do not prove correctness for every pathological input |
| Understand model implications | Data-free prior simulation, posterior fresh-path replication, prior learning paths | Discrepancy fractions and strong prior influence need human interpretation |
| Repeat and report completed work | Numerical/fit replay and saved-only charts | Artifacts remain local; no deployment, release or retention service is implied |

## What the completed exercises say

All 269 tests passed, including 29 new model/evidence tests. Independent batch
Gaussian conditioning, adaptive scalar integration and a known-level conjugate
limit support the new numerical calculations. Replication tests check fresh
state innovations as well as initial-state posterior conditioning.

Each of the two 200-observation synthetic fixtures produced 130 development
forecasts using the first 160 observations; the final 40 were excluded. All 143
numerical/fit files per series replayed byte for byte with matching inputs,
configuration and environment, excluding execution timestamps and PNGs. Each
series also generated 1000 prior and 1000 posterior datasets and compared five
priors across 162 learning points. Artifacts are local under
`runs/local-level-variance-20261117-bs3pk_xh/`; the verification and replay
summaries are saved there. They are ignored by Git and are not shipped with a
fresh checkout.

The numerical checks passed, but the replicated datasets expose structural
questions:

| Fixture | Posterior fraction at least as large as observed: lag correlation | Half-mean difference | Maximum CUSUM |
|---|---|---|---|
| Stable mean | 1.000 | 0.480 | 1.000 |
| Abrupt mean shift | 0.104 | 0.047 | 0.002 |

These fractions are descriptive posterior predictive comparisons, not calibrated
tests or approval thresholds. For the stable fixture, the chosen random walk
produces more serial structure and cumulative movement than the observed series.
For the shift fixture, the observed sustained change is unusual under fresh
paths even though sequential forecasts adapt after the shift. This is a reason
to examine fixed evolution variance and the random-walk assumption. It does not
by itself establish which richer model would be appropriate.

The prior comparisons also separate uncertainty from location. On the shift
fixture at the cutoff, changing the observation-variance prior scale from 2 to
18 moves the predictive mean from 15.380 to 15.363 but widens the 95% predictive
HPD region from 4.648 to 5.098 units. Initial-state prior differences have much
smaller endpoint effects for these settings. Those observations describe this
history; they do not establish that prior influence should vanish or that the
narrowest interval is preferable.

## Next bounded infrastructure step

Consolidate how experiments request and record evidence from the existing
workflows before adding another family or inference backend. The forecast
runner already has a model-independent capability boundary, but predictive
checks and prior comparisons now contain concrete-model branches and some
fields that apply only to conjugate models. A small evidence capability and
metadata boundary would let each model own its simulation and parameter
summaries while shared code owns selection, persistence and reporting.

The first scope should cover the models and evidence already implemented:
declare which evidence is requested or required, retain numerical failures,
keep human criticism separate from execution status, and preserve replay and
legacy artifacts. The caller must still invoke separate evidence workflows
explicitly today. An experiment-owned check plan is a design priority, not a
feature delivered by this slice. A large registry or orchestration service is
not necessary to start that work.

For the next modeling decision, use the collected criticism to state the
question first: for example, whether learning evolution variance or changing
the transition law would explain both stable periods and abrupt changes.
That would need its own priors, independent numerical reference and predictive
checks. More MCMC, learned evolution variance and formal simulation-based
calibration are not prerequisites for preserving this verified checkpoint.

The current result is a tested development checkpoint for the infrastructure.
Application choice, model approval and any release decision remain explicit
future actions. See the [roadmap](roadmap.md), [architecture](architecture.md),
and [new model guide](local-level-variance.md) for current interfaces and limits.
