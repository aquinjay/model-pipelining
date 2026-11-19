# Reading guide for forecasting models and code

Start with one forecast from a scalar local-level model. Follow the hidden
state, the next observation, and the saved result through one forecast origin.
That example gives you a reference for reading both the statistics and the code.
Then move to seasonal regression and models that forecast several series together.

## Start with the current project

1. **Gaussian conditioning and uncertainty.** Read conditional normals, means,
   variances, covariance, and Bayes' rule. Learn to distinguish uncertainty about
   the current hidden level from uncertainty about the next observed value. Then
   read conjugate priors and marginal prediction using the constant-mean model as
   a small example. [Local-level equations](local-level.md),
   [normal and inverse-gamma example](normal-mean-variance.md).
2. **State-space models and the Kalman filter.** Learn the observation and
   transition equations, process noise versus measurement noise, prediction,
   the measurement update, and the innovation. Then compare the filtered state,
   which uses observations through that time, with the smoothed state, which also
   uses later observations. Start with chapters 5 and 6 and the Rauch–Tung–Striebel
   smoother in the authors' book. [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf),
   [scalar Python implementation](../src/model_pipelining/local_level.py).
3. **Forecast timing and model evaluation.** Read rolling-origin evaluation,
   release dates and data revisions, forecast leakage, and simple baselines.
   Then read prediction interval coverage and width, probability integral
   transforms (PIT), CRPS, and log loss. Trace which observations a model can use
   at an origin and when the runner joins the actual outcome for scoring.
   [Time series cross-validation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation),
   [experiment contract](experiments.md), [distribution scoring](distribution-scoring.md).
4. **Python for reading the pipeline.** Read dataclasses and type hints first,
   then NumPy axes, array shapes, and broadcasting. Learn to read a `unittest`
   test, especially one that compares a hand calculation or an independently
   computed answer with the model output. Trace a `ForecastRequest` from the
   walkthrough into the model and the saved run. [Python dataclasses](https://docs.python.org/3/library/dataclasses.html),
   [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
   [forecasting walkthrough](forecasting.md), [local-level tests](../tests/test_local_level.py).

## Read the numerical methods when you reach them

5. **Likelihoods, numerical integration, and stable arithmetic.** Read Gaussian
   log likelihoods, matrix solves and Cholesky factorization, quadrature, and
   `logsumexp`. Follow how the short-history reference compares a dense Gaussian
   calculation with a sequential calculation. Review underflow and roundoff when
   an observation makes some paths extremely unlikely. [Short-history reference](innovation-reference.md),
   [SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html).
6. **Mixtures and Monte Carlo.** Read latent labels, importance proposals and
   weights, and then particle filtering, resampling, and ancestry. Keep track of
   which variables are sampled and which are integrated analytically. Learn how
   effective sample size and Monte Carlo standard error answer different
   questions. Use a seeded NumPy `Generator` to replay a small calculation.
   [Particle study](innovation-particles.md),
   [NumPy random generators](https://numpy.org/doc/stable/reference/random/generator.html).
7. **Rare paths and numerical evidence.** Read why sampling can miss positive
   probability, why zero visits still allow a positive upper bound, and how
   empirical Bernstein bounds behave for bounded weights. The optional deeper
   topic is Gaussian grid integration. Read the state-grid guide first for the
   reason we used an evenly spaced grid, the tail allowance, and the numerical
   check that exposed FFT roundoff. Poisson summation is the advanced reading
   behind the grid error bound. [Probability bounds](innovation-probability-bounds.md),
   [state-grid study](innovation-lattice.md),
   [Maurer and Pontil, Theorem 4](https://arxiv.org/abs/0907.3740),
   [Micciancio and Regev, Lemma 2.8](https://cims.nyu.edu/~regev/papers/average.pdf).
8. **Simulation and model criticism.** Read prior and posterior predictive
   checks, parameter recovery, simulation-based calibration, and prior
   sensitivity. Ask two questions as you read each test. Does the implementation
   recover what the model says it should? Does the model reproduce features of
   the time series that matter for the task? [Predictive checks](predictive-checks.md),
   [calibration](calibration.md), [development study](model-review-study.md).

## Extend beyond Bayesian dynamic linear models

9. **Harmonic regression and seasonality.** Fit sine and cosine pairs for a
   known period. See how the number of pairs controls the seasonal curve.
   Then read regression with autocorrelated errors, multiple seasonal periods,
   and time-varying seasonal coefficients. Compare a fixed seasonal curve with
   a model that lets it evolve. [Dynamic harmonic regression and time series
   regression](https://otexts.com/fpp3/dynamic-regression.html),
   [complex seasonality](https://otexts.com/fpp3/advanced.html).
10. **Other state-space models.** Move from a scalar level to a state vector
    containing level, slope, and seasonal terms. Follow this with structural
    time series, autoregressive models in state-space form, and maximum
    likelihood estimation via the filter. Later topics include nonlinear
    transitions, non-Gaussian observations, and regime switching.
    [State-space model examples and notation](https://www.statsmodels.org/stable/statespace.html),
    [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
11. **Multivariate forecasting.** Start with covariance and joint versus marginal
    forecast distributions, then read vector autoregressions and dynamic factor
    models. Compare forecasting two series separately with a joint forecast that
    preserves dependence across them. Ask what information is available when
    each series is released on a different date. [Vector autoregressions](https://otexts.com/fpp3/advanced.html),
    [multivariate state-space models](https://www.statsmodels.org/stable/statespace.html),
    [Stock and Watson on forecasting with factors](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JASA_2002.pdf).

For a first practice session, work out one Kalman prediction and update by hand,
find that calculation in the scalar code and its tests, and follow one saved
forecast through the walkthrough. A sinusoid plus noise makes a small second
example that can use the same forecast origins and baseline.

## Run the weekly harmonic example

The example fits an intercept plus a weekly sine and cosine to each eligible
history and returns the next-day point forecast. It uses the same preserved
series, 72 forecast origins, scoring, and comparison report as the training-mean
baseline. The simulated observations have a weekly cycle and Gaussian noise.
This illustrates the contract for another model family.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_harmonic_example.py \
  --output-directory runs/weekly-harmonic-new
```

The command generates and registers the input, runs both models, and writes
the report under `comparison/`. The observed MAE was 0.247 for harmonic
regression and 1.844 for the training mean on this synthetic fixture.
See the [example code](../scripts/run_harmonic_example.py) and the
[forecasting walkthrough](forecasting.md).
