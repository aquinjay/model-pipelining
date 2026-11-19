# Repository instructions

## Keep forecasting model independent

Bayesian dynamic linear models are the first main use case. The shared pipeline
also serves future state-space models, harmonic regression, multivariate
forecasting, and other model families. Keep model-specific fitting and
diagnostics in their adapters or evidence providers. Extend shared interfaces
when a concrete model needs a new capability.

## Keep documentation current

After making implementation or workflow changes, update the affected documentation as part of the same task, before reporting the work complete.

- Review all repository `README.md` files and update those affected by the change, including examples, function signatures, behavior, and usage instructions.
- Update relevant files under `docs/`, including the roadmap and decision log when the change affects project status, next steps, or design decisions.
- Make clear which steps happen automatically and which require a caller action.
- Check that updated examples and local links remain accurate. Do not leave documentation updates as a later task.
