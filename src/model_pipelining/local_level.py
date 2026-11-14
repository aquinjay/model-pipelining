"""A Gaussian local-level model with fixed variances and exact filtering.

level[0] ~ Normal(initial_mean, initial_variance)
level[t] = level[t-1] + Normal(0, evolution_variance)
value[t] = level[t] + Normal(0, observation_variance)

The initial prior is BEFORE the first daily transition and observation. All
noise terms are independent. The adapter returns the next predictive mean and
requested observation quantiles; no MCMC or parameter fitting
is used. See docs/local-level.md for equations, assumptions, and references.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import math
from numbers import Real
import random
from statistics import NormalDist
from typing import ClassVar

from model_pipelining.models import (
    Forecast, ForecastRequest, HighestDensityRegion, Interval, Observation, Quantile,
)
from model_pipelining.uncertainty import interval_probabilities


@dataclass(frozen=True)
class FilterStep:
    """One update, conditioned on observations through timestamp.

    observation_mean/variance describe the prediction BEFORE seeing this value.
    level_mean/variance describe the latent level AFTER seeing this value.
    These training-history diagnostics are not the runner's held-out forecasts.
    """

    timestamp: datetime
    observation_mean: float
    observation_variance: float
    innovation: float
    level_mean: float
    level_variance: float


@dataclass(frozen=True)
class PriorDraw:
    """A simulated latent level and its noisy observation, with no fitted data."""

    level: float
    value: float


@dataclass(frozen=True, kw_only=True)
class LocalLevel:
    """A daily Bayesian local-level adapter with caller-selected fixed settings.

    Variances are squared units, not standard deviations. observation_variance
    must be positive; evolution_variance and initial_variance may be zero.
    Defaults are illustrative development settings, not estimates or tuning.
    Refit from the initial prior for every request; retain no state between calls.
    """

    initial_mean: float = 0.0
    initial_variance: float = 100.0
    observation_variance: float = 1.0
    evolution_variance: float = 0.1
    name: ClassVar[str] = "local-level"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name == "observation_variance" and value <= 0:
                raise ValueError("observation_variance must be positive")
            if name in ("initial_variance", "evolution_variance") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            # Keep configuration JSON-serializable even for numerical scalar types.
            object.__setattr__(self, name, float(value))

    @property
    def configuration(self) -> dict:
        return asdict(self)

    def filter_history(self, history: tuple[Observation, ...]) -> tuple[FilterStep, ...]:
        """Return a fresh forward-filter trace for nonempty, daily eligible history.

        Only forward updates are performed. Smoothing with later observations
        would violate the forecast boundary and is not part of this adapter.
        """
        if not history:
            raise ValueError("LocalLevel needs at least one observation")
        level_mean = self.initial_mean
        level_variance = self.initial_variance
        steps = []
        for observation in history:
            timestamp = observation.timestamp
            if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
                raise ValueError("Observation timestamps must be timezone-aware")
            timestamp = timestamp.astimezone(UTC)
            if steps and timestamp - steps[-1].timestamp != timedelta(days=1):
                raise ValueError("LocalLevel requires ordered observations one UTC day apart")
            value = observation.value
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Observation values must be finite numbers")

            # Predict the next level, then condition it on the noisy observation.
            state_variance = level_variance + self.evolution_variance
            observation_variance = state_variance + self.observation_variance
            if not math.isfinite(observation_variance) or observation_variance <= 0:
                raise ValueError("Filtering produced an invalid predictive variance")
            gain = state_variance / observation_variance
            observation_mean = level_mean
            innovation = value - observation_mean
            level_mean = (1 - gain) * level_mean + gain * value
            # V * gain is algebraically equal to (1 - gain) * state_variance,
            # and avoids subtracting two nearly equal numbers at gain ~ 1.
            level_variance = self.observation_variance * gain
            if not all(math.isfinite(number) for number in (innovation, level_mean, level_variance)):
                raise ValueError("Filtering produced nonfinite posterior moments")
            steps.append(FilterStep(timestamp, observation_mean, observation_variance,
                                    innovation, level_mean, level_variance))
        return tuple(steps)

    def predictive_distribution(self, request: ForecastRequest) -> NormalDist:
        """Return tomorrow's observation law, conditional on fixed settings.

        Include posterior level uncertainty, another day's level movement, and
        observation noise. A latent-level variance alone is not an observation
        forecast variance. No variance-parameter uncertainty is integrated out.
        """
        if (
            request.issued_at.utcoffset() is None
            or request.target_time.utcoffset() is None
            or not request.history
            or request.history[-1].timestamp != request.issued_at
            or request.target_time.astimezone(UTC) - request.issued_at.astimezone(UTC) != timedelta(days=1)
        ):
            raise ValueError("LocalLevel requires issuance at the final observation and a one-day target")
        posterior = self.filter_history(request.history)[-1]
        variance = posterior.level_variance + self.evolution_variance + self.observation_variance
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("Invalid next-observation predictive variance")
        return NormalDist(posterior.level_mean, math.sqrt(variance))

    def forecast(self, request: ForecastRequest) -> Forecast:
        """Return mean, quantiles, and requested highest-density prediction regions.

        For this unimodal symmetric normal law, highest-density regions equal
        the central equal-tail intervals. That is a model-specific fact, not a
        rule imposed on other adapters by the shared runner.
        """
        distribution = self.predictive_distribution(request)
        regions = []
        for level in request.highest_density_levels:
            lower, upper = interval_probabilities(level)
            regions.append(HighestDensityRegion(level, (Interval(distribution.inv_cdf(lower),
                                                                  distribution.inv_cdf(upper)),)))
        return Forecast(
            distribution.mean,
            tuple(Quantile(probability, distribution.inv_cdf(probability)) for probability in request.quantile_levels),
            tuple(regions),
        )

    def simulate_prior(self, *, count: int, seed: int = 0) -> tuple[PriorDraw, ...]:
        """Simulate one seeded trajectory before fitting any observations.

        Use a local generator without changing global random state. These draws
        expose prior assumptions for caller review; finite draws do not establish
        that assumptions are plausible. Conditional on the fixed variances only.
        """
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        generator = random.Random(seed)
        level = generator.gauss(self.initial_mean, math.sqrt(self.initial_variance))
        draws = []
        for _ in range(count):
            level += generator.gauss(0, math.sqrt(self.evolution_variance))
            value = level + generator.gauss(0, math.sqrt(self.observation_variance))
            if not math.isfinite(level) or not math.isfinite(value):
                raise ValueError("Prior simulation produced nonfinite values")
            draws.append(PriorDraw(level, value))
        return tuple(draws)
