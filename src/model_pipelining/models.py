"""Shared mean, quantile, and highest-density forecast outputs, plus a baseline.

The runner owns historical selection and scoring. Models receive a fresh,
immutable request containing eligible observations and the target time only.
The Bayesian LocalLevel adapter in local_level.py uses the same boundary.
"""

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class ForecastRequest:
    history: tuple[Observation, ...]
    issued_at: datetime
    target_time: datetime
    quantile_levels: tuple[float, ...] = ()
    highest_density_levels: tuple[float, ...] = ()


@dataclass(frozen=True)
class Quantile:
    """One quantile of the future observed value, in the target's original units."""

    probability: float
    value: float


@dataclass(frozen=True)
class Interval:
    """One finite, closed segment of a prediction region."""

    lower: float
    upper: float


@dataclass(frozen=True)
class HighestDensityRegion:
    """An adapter-supplied highest-density region of the future observation.

    Multiple segments preserve disconnected regions for multimodal laws. The
    adapter is responsible for their density interpretation and probability mass;
    shared validation checks structure, not highest-density optimality.
    """

    level: float
    segments: tuple[Interval, ...]


@dataclass(frozen=True)
class Forecast:
    """A mean and optional uncertainty summaries of the same future observation.

    Quantiles/regions describe observation uncertainty, not latent-state uncertainty or
    a confidence interval for the mean. The runner validates their values and
    requires exactly the probabilities in its request when supported.
    """

    mean: float
    quantiles: tuple[Quantile, ...] = ()
    highest_density_regions: tuple[HighestDensityRegion, ...] = ()


class Forecaster(Protocol):
    """A model returning a mean and declared observation-uncertainty outputs.

    Refit from the supplied history on every call. Do not retain training data
    between calls. configuration must describe all settings as JSON values.
    supports_quantiles declares whether every requested quantile is required.
    supports_highest_density declares whether requested density regions are required.
    Older float-returning adapters without the flag remain point-only.
    This is a trusted Python interface, not a security sandbox.
    """

    name: str
    supports_quantiles: bool
    supports_highest_density: bool

    @property
    def configuration(self) -> dict: ...

    def forecast(self, request: ForecastRequest) -> Forecast | float: ...


# Retain the earlier type name for callers importing it; the boundary now also
# supports declared quantiles. Plain numeric outputs still mean point-only.
PointForecaster = Forecaster


class TrainingMean:
    """Predict the next value with the arithmetic mean of eligible history."""

    name = "training-mean"
    supports_quantiles = False
    supports_highest_density = False

    @property
    def configuration(self) -> dict:
        return {}

    def forecast(self, request: ForecastRequest) -> float:
        return fmean(observation.value for observation in request.history)
