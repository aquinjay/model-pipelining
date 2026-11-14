# Next modeling detour: parameter uncertainty and MCMC

Status: planned, not implemented. The user prioritized reusable infrastructure
and requested another modeling detour soon, including parameter uncertainty and
MCMC considerations. A real application is not a prerequisite. The current
`LocalLevel` adapter continues to condition on fixed variance settings.

## Statistical target

For an eligible history `D`, a parameter-learning model should integrate over
the parameter posterior when predicting a future observation:

```text
p(y_future | D) = integral p(y_future | theta, D) p(theta | D) d theta
```

Posterior predictive simulation pairs parameter draws with conditional future
observation draws, retaining both parameter and observation uncertainty. This
differs from substituting a single posterior mean parameter into a conditional
forecast. See the primary [Stan posterior prediction guide](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html).

The model's latent-state uncertainty, parameter uncertainty, future process/
observation noise, and numerical Monte Carlo error answer different questions.
The latter concerns precision of computed summaries, not the scientific spread
of the predictive distribution; effective sample size informs its assessment.
See [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html).

Parameter uncertainty does not itself require MCMC: an analytic posterior or
other justified computation can supply the same statistical target. For the
first detour, choose a small model with independently calculable reference
results before expanding the local-level model to unknown noise variances.
The exact model and inference backend remain to be selected.

## Proposed implementation sequence

1. Define an unknown-parameter model and priors with an analytic or independently
   validated posterior/predictive reference. Test marginal prediction calculations
   and prior influence before making comparative forecast scores a focus.
2. Integrate parameter uncertainty into the existing `Forecast` boundary. Means,
   quantiles, and HPD regions must describe the marginalized future observation.
   Quantiles or HPD regions cannot in general be obtained by averaging
   conditional interval endpoints. Preserve the current HPD priority and
   adapter responsibility for density semantics.
3. Choose a maintained inference library after checking the project's CPython
   3.14 runtime and dependency constraints. Add saved fit evidence with a chain
   and draw dimension, including warmup separation, parameter names/transforms,
   initialization, per-chain seeds, backend versions, settings, and diagnostics.
4. Add explicit computation criteria and failure/applicability handling. Compare
   posterior/predictive summaries against the reference within justified Monte
   Carlo tolerances. Then consider richer unknown-variance local-level inference.

This is a proposal for the next slice, not an installed sampler, a new API, or
an instruction to implement a home-grown general inference engine now.

## MCMC evidence to retain

The planned evidence includes multiple-chain rank-normalized split R-hat,
bulk/tail effective sample sizes, and Monte Carlo standard errors for relevant
summaries. These assess computational behavior and precision; passing them does
not prove that the model or priors are suitable. Diagnostic scope should include
the quantities used for prediction, not just a convenient subset of parameters.
The definitions and finite-chain limitations are described in
[Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html).

Backend-specific problems must remain explicit and applicable: for an HMC/NUTS
backend, retain divergences and trajectory/energy diagnostics; other algorithms
need their own relevant evidence. Missing required diagnostics must not silently
become success. Exact diagnostic criteria and the distinction between failed
computation and continued exploratory evidence collection belong to the future
check-plan design. They are separate from a rule about how strongly a prior
may influence updating.

Primary background for these backend-specific warnings is the
[Stan diagnostic guide](https://mc-stan.org/learn-stan/diagnostics-warnings.html).

Each fit must be associated with its eligible history, issuance, and target.
Seeds should be assigned explicitly per fit/chain, with enough saved evidence
to audit execution independent of scheduling order. Reusing a fit trained on
later observations or later revisions would violate the experiment contract.
No automatic fit caching or warm starts are implemented; any later reuse must
be keyed to the training information and inference settings.

## Infrastructure that is ready, and what remains open

[Experiment specifications](experiments.md) already select as-of histories,
issuance/targets, benchmark cohorts, and development/final periods independently
of the adapter. The current forecast outputs can express marginalized means,
quantiles, and HPD regions. The runner still refits from each immutable request.

Posterior chain storage, fit-result interfaces, sampler diagnostic dispatch,
predictive draws/densities, computational precision criteria, and backend
selection are not implemented. The next detour should add those responsibilities
around a concrete, verifiable model rather than hide them in forecast scores.
Prior/noise sensitivity remains descriptive: strong influence should be visible
and interpretable, not automatically minimized or rejected.
