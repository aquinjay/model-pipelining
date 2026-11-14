# References

These are optional background and mathematical sources. The workflow/model/uncertainty guides explain the current tooling without requiring readers to work through these publications first. The sources do not prescribe this project's adapter API or release policy.

- Gelman et al., [Bayesian workflow](https://arxiv.org/abs/2011.01808), 2020 v1: model construction, simulation, computation, predictive checking, comparison, and iteration. The workflow also recognizes the importance of design and measurement outside the paper's main data-analysis scope.
- Hyndman and Athanasopoulos, [Forecasting: Principles and Practice — The forecaster's toolbox](https://otexts.com/fpp3/toolbox.html): benchmarks, forecast evaluation, uncertainty, and rolling-origin cross-validation.
- Särkkä, [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf), 2013, Section 4.3: exact Gaussian filtering and the scalar Gaussian-random-walk example. The [local-level implementation](local-level.md) specifies its own initial-prior timing explicitly.
- [statsmodels unobserved-components documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html): the local-level observation and state specification. statsmodels is a reference, not a project dependency; the current filter uses fixed variances and the standard library.
- Hyndman, [Computing and graphing highest density regions](https://robjhyndman.com/papers/hdrgraph.pdf), 1996: density-based regions, including disconnected sets for multimodal laws. The [shared uncertainty contract](uncertainty.md) preserves segment structure and prioritizes supplied highest-density regions in reporting.
- Hyndman and Athanasopoulos, [Evaluating distributional forecast accuracy](https://otexts.com/fpp3/toolbox.html#evaluating-distributional-forecast-accuracy): quantile and Winkler interval scores. This implementation explicitly uses unscaled pinball loss and applies the interval score to equal-tail bounds only.
- [Python 3.14 NormalDist](https://docs.python.org/3.14/library/statistics.html#statistics.NormalDist): the adapter's standard-library normal quantile calculation. The shared runner does not assume a normal predictive law.

Additional readings on experimental design and causal inference can be added when those questions become active. The current example's intended claim and evaluation limits are stated in [the walkthrough](forecasting.md#understand-the-example).
