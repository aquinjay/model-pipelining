"""Zero-tuning probabilistic baseline adapters for full-law and decision benchmarking.

Parametric baselines (`GaussianClimatology`, `ResidualPersistence`) supply exact
reference Student-t predictive laws, quantiles, and highest-density regions.
Nonparametric baselines (`EmpiricalClimatology`, `EmpiricalResidualPersistence`)
supply exact weighted empirical sample predictive laws and quantiles without
imposing symmetry or unimodality (and without inventing a continuous density or
highest-density region).
"""

from dataclasses import asdict, dataclass
from datetime import UTC
import math
from typing import ClassVar

from scipy.stats import t

from model_pipelining.distribution_scoring import empirical_quantile
from model_pipelining.models import (
    Forecast,
    ForecastRequest,
    HighestDensityRegion,
    Interval,
    Quantile,
)
from model_pipelining.uncertainty import interval_probabilities


def _validate_window(window: int | None, *, minimum: int = 2) -> None:
    if window is not None and (type(window) is not int or window < minimum):
        raise ValueError(f"window must be None or an integer >= {minimum}")


def _student_t_forecast(
    *,
    location: float,
    scale: float,
    df: float,
    quantile_levels: tuple[float, ...],
    highest_density_levels: tuple[float, ...],
) -> Forecast:
    if not all(math.isfinite(x) for x in (location, scale, df)) or scale <= 0.0 or df <= 1.0:
        raise ValueError("Predictive Student-t requires finite location, positive scale, and df > 1")
    dist = t(df=df, loc=location, scale=scale)
    quantiles = tuple(Quantile(prob, float(dist.ppf(prob))) for prob in quantile_levels)
    regions = []
    for level in highest_density_levels:
        lower, upper = interval_probabilities(level)
        # Student-t is unimodal and symmetric, so its exact HPD region equals the central interval.
        regions.append(HighestDensityRegion(level, (Interval(float(dist.ppf(lower)), float(dist.ppf(upper))),)))
    return Forecast(
        mean=float(location),
        quantiles=quantiles,
        highest_density_regions=tuple(regions),
        distribution={"family": "student_t", "location": float(location), "scale": float(scale), "df": float(df)},
    )


def _empirical_forecast(values: list[float], *, quantile_levels: tuple[float, ...]) -> Forecast:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Empirical baseline requires nonempty finite values")
    n = len(values)
    weight = 1.0 / n
    record = {"family": "empirical_samples", "values": [float(x) for x in values], "weights": [weight] * n}
    mean_val = math.fsum(record["values"]) / n
    quantiles = tuple(Quantile(prob, empirical_quantile(record, prob)) for prob in quantile_levels)
    return Forecast(
        mean=float(mean_val),
        quantiles=quantiles,
        highest_density_regions=(),
        distribution=record,
    )


def _matched_lead_differences(request: ForecastRequest) -> tuple[float, list[float]]:
    if not request.history:
        raise ValueError("Persistence baseline requires nonempty history")
    if request.target_time.utcoffset() is None or request.history[-1].timestamp.utcoffset() is None:
        raise ValueError("Persistence baseline requires timezone-aware timestamps")
    last_obs = request.history[-1]
    lead = request.target_time.astimezone(UTC) - last_obs.timestamp.astimezone(UTC)
    if lead.total_seconds() <= 0:
        raise ValueError("target_time must follow the latest eligible observation")
    by_time = {obs.timestamp.astimezone(UTC): obs.value for obs in request.history}
    differences = []
    for obs in request.history:
        current_ts = obs.timestamp.astimezone(UTC)
        previous = by_time.get(current_ts - lead)
        if previous is not None:
            differences.append(obs.value - previous)
    return float(last_obs.value), differences


@dataclass(frozen=True, kw_only=True)
class GaussianClimatology:
    """Expanding or rolling reference Student-t climatology baseline.

    Under a normal model with unknown mean and variance and reference Jeffreys
    prior p(mu, sigma^2) proportional to 1 / sigma^2, the exact predictive law
    for a future observation given n values is:

        Student-t(df = n - 1, location = sample_mean, scale = s * sqrt(1 + 1 / n))

    Requires at least 3 observations (df >= 2 for finite first moment and CRPS)
    and strictly positive sample variance. Works at any positive horizon.
    """

    window: int | None = None
    name: ClassVar[str] = "gaussian-climatology"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_window(self.window, minimum=3)

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @classmethod
    def from_configuration(cls, configuration: dict) -> "GaussianClimatology":
        return cls(**configuration)

    def forecast(self, request: ForecastRequest) -> Forecast:
        values = [item.value for item in request.history]
        if self.window is not None:
            values = values[-self.window:]
        n = len(values)
        if n < 3:
            raise ValueError("GaussianClimatology requires at least 3 observations (df >= 2)")
        mean_val = math.fsum(values) / n
        sse = math.fsum((value - mean_val) ** 2 for value in values)
        variance = sse / (n - 1)
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("GaussianClimatology requires positive historical variance")
        scale = math.sqrt(variance * (1.0 + 1.0 / n))
        return _student_t_forecast(
            location=mean_val,
            scale=scale,
            df=float(n - 1),
            quantile_levels=request.quantile_levels,
            highest_density_levels=request.highest_density_levels,
        )


@dataclass(frozen=True, kw_only=True)
class ResidualPersistence:
    """Probabilistic random-walk persistence baseline calibrated from historical lead differences.

    Centers the predictive distribution at the latest eligible observation and
    extracts all historical differences y(t) - y(t - lead) matching the exact
    time delta between the latest eligible observation and target_time.

    Under zero-drift Gaussian increments e_i ~ N(0, sigma^2) with Jeffreys prior
    p(sigma^2) proportional to 1 / sigma^2, the exact predictive law given m
    matched historical differences is:

        Student-t(df = m, location = last_value, scale = sqrt(mean(e_i^2)))

    Requires at least 2 matched historical differences (df >= 2 for finite first
    moment and CRPS) and strictly positive mean squared difference.
    """

    window: int | None = None
    name: ClassVar[str] = "residual-persistence"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_window(self.window, minimum=2)

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @classmethod
    def from_configuration(cls, configuration: dict) -> "ResidualPersistence":
        return cls(**configuration)

    def forecast(self, request: ForecastRequest) -> Forecast:
        last_value, differences = _matched_lead_differences(request)
        if self.window is not None:
            differences = differences[-self.window:]
        m = len(differences)
        if m < 2:
            raise ValueError("ResidualPersistence requires at least 2 matched historical lead differences (df >= 2)")
        mean_sq = math.fsum(diff * diff for diff in differences) / m
        if not math.isfinite(mean_sq) or mean_sq <= 0.0:
            raise ValueError("ResidualPersistence requires positive historical lead-difference variance")
        return _student_t_forecast(
            location=last_value,
            scale=math.sqrt(mean_sq),
            df=float(m),
            quantile_levels=request.quantile_levels,
            highest_density_levels=request.highest_density_levels,
        )


@dataclass(frozen=True, kw_only=True)
class EmpiricalClimatology:
    """Nonparametric expanding or rolling empirical climatology baseline.

    Uses the equal-weight empirical distribution of eligible historical values,
    preserving skewness, bounds, and multimodality (such as seasonal cycles)
    without imposing a Gaussian or unimodal shape.
    """

    window: int | None = None
    name: ClassVar[str] = "empirical-climatology"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = False
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_window(self.window, minimum=1)

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @classmethod
    def from_configuration(cls, configuration: dict) -> "EmpiricalClimatology":
        return cls(**configuration)

    def forecast(self, request: ForecastRequest) -> Forecast:
        values = [item.value for item in request.history]
        if self.window is not None:
            values = values[-self.window:]
        if not values:
            raise ValueError("EmpiricalClimatology requires at least 1 observation")
        return _empirical_forecast(values, quantile_levels=request.quantile_levels)


@dataclass(frozen=True, kw_only=True)
class EmpiricalResidualPersistence:
    """Nonparametric persistence baseline using empirical matched-lead differences.

    Forms the equal-weight empirical predictive distribution {y_last + e_i}
    from historical lead differences e_i = y(t_i) - y(t_i - lead), preserving
    asymmetric error distributions (e.g., gradual warmups vs. sharp drops) and
    heavy tails without assuming Gaussian or Student-t innovations.
    """

    window: int | None = None
    name: ClassVar[str] = "empirical-residual-persistence"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = False
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_window(self.window, minimum=1)

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @classmethod
    def from_configuration(cls, configuration: dict) -> "EmpiricalResidualPersistence":
        return cls(**configuration)

    def forecast(self, request: ForecastRequest) -> Forecast:
        last_value, differences = _matched_lead_differences(request)
        if self.window is not None:
            differences = differences[-self.window:]
        if not differences:
            raise ValueError("EmpiricalResidualPersistence requires at least 1 matched historical lead difference")
        shifted = [last_value + diff for diff in differences]
        return _empirical_forecast(shifted, quantile_levels=request.quantile_levels)


PROBABILISTIC_BENCHMARKS = {
    kind.name: kind
    for kind in (GaussianClimatology, ResidualPersistence, EmpiricalClimatology, EmpiricalResidualPersistence)
}
