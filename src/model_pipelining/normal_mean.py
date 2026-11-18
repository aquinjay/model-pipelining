"""Unknown constant Gaussian mean, with exact parameter-integrated prediction.

mu ~ Normal(prior_mean, prior_variance)
y[t] | mu ~ independent Normal(mu, observation_variance)

Optional MCMC audits this tractable target; forecasts remain explicitly exact.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from numbers import Real
from pathlib import Path
from statistics import NormalDist, fmean
from typing import ClassVar

from model_pipelining.inference import FitEvidence, FittedForecast
from model_pipelining.models import Forecast, ForecastRequest, HighestDensityRegion, Interval, Quantile
from model_pipelining.uncertainty import interval_probabilities


@dataclass(frozen=True)
class MeanPosterior:
    mean: float
    variance: float
    prior_mean_weight: float
    observation_variance: float

    @property
    def predictive_variance(self) -> float:
        return self.observation_variance + self.variance


@dataclass(frozen=True, kw_only=True)
class NormalMean:
    """Learn a constant mean; integrate its uncertainty into future observations.

Positive variances are squared target units. No daily adjacency is required:
the exchangeable model accepts any ordered past observations and future target.
MCMCSettings is optional and deliberately audits, rather than approximates,
the exact forecasts. fit() exposes evidence; forecast() rejects failed audits.
"""

    prior_mean: float = 0.0
    prior_variance: float = 100.0
    observation_variance: float = 1.0
    mcmc: "MCMCSettings | None" = None
    name: ClassVar[str] = "normal-mean"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_fit_evidence: ClassVar[bool] = True

    def __post_init__(self) -> None:
        for name in ("prior_mean", "prior_variance", "observation_variance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "prior_mean" and value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(value))
        if self.mcmc is not None:
            from model_pipelining.sampling import MCMCSettings
            if not isinstance(self.mcmc, MCMCSettings):
                raise ValueError("mcmc must be MCMCSettings or None")

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @property
    def implementation_files(self) -> tuple[Path, ...]:
        return (Path(__file__).with_name("sampling.py"),) if self.mcmc is not None else ()

    def posterior(self, request: ForecastRequest) -> MeanPosterior:
        if (not isinstance(request.issued_at, datetime) or request.issued_at.utcoffset() is None
                or not isinstance(request.target_time, datetime) or request.target_time.utcoffset() is None
                or request.target_time <= request.issued_at or not request.history):
            raise ValueError("NormalMean requires nonempty past history and an aware future target")
        previous = None
        for item in request.history:
            if (not isinstance(item.timestamp, datetime) or item.timestamp.utcoffset() is None
                    or item.timestamp > request.issued_at
                    or (previous is not None and item.timestamp <= previous)):
                raise ValueError("History must be strictly ordered, aware, and available by issuance")
            if isinstance(item.value, bool) or not isinstance(item.value, Real) or not math.isfinite(item.value):
                raise ValueError("History values must be finite numbers")
            previous = item.timestamp
        count = len(request.history)
        # Scale variances before forming weights: no overflowing precisions or
        # n * prior_variance, even for large but finite settings.
        variance_scale = max(self.prior_variance, self.observation_variance)
        prior_scaled = self.prior_variance / variance_scale
        noise_scaled = self.observation_variance / variance_scale
        denominator = noise_scaled + count * prior_scaled
        weight = noise_scaled / denominator
        variance = (prior_scaled / denominator) * self.observation_variance
        data_weight = count * prior_scaled / denominator
        mean = weight * self.prior_mean + data_weight * fmean(item.value for item in request.history)
        predictive_variance = self.observation_variance + variance
        if not all(math.isfinite(x) for x in (mean, variance, predictive_variance)) or variance <= 0:
            raise ValueError("NormalMean produced invalid or unrepresentable posterior moments")
        return MeanPosterior(mean, variance, weight, self.observation_variance)

    def predictive_distribution(self, request: ForecastRequest) -> NormalDist:
        posterior = self.posterior(request)
        return NormalDist(posterior.mean, math.sqrt(posterior.predictive_variance))

    def fit(self, request: ForecastRequest) -> FittedForecast:
        posterior = self.posterior(request)
        distribution = NormalDist(posterior.mean, math.sqrt(posterior.predictive_variance))
        regions = []
        for level in request.highest_density_levels:
            lower, upper = interval_probabilities(level)
            regions.append(HighestDensityRegion(level, (Interval(distribution.inv_cdf(lower),
                                                                  distribution.inv_cdf(upper)),)))
        forecast = Forecast(distribution.mean,
                            tuple(Quantile(p, distribution.inv_cdf(p)) for p in request.quantile_levels),
                            tuple(regions))
        metadata = {
            "inference": "analytic" if self.mcmc is None else "analytic_with_mcmc_audit",
            "forecast_method": "exact_parameter_marginalized_normal",
            "parameter_names": ["mu"], "parameter_units": "original target units",
            "posterior": asdict(posterior), "predictive_variance": posterior.predictive_variance,
            "configuration": self.configuration,
            "prior_influence": "prior_mean_weight is conditional on both variance settings; no magnitude threshold",
        }
        checks = ({"name": "exact_posterior_and_predictive_moments_finite", "required": True, "status": "passed"},)
        if self.mcmc is None:
            checks += ({"name": "mcmc_diagnostics", "required": False, "status": "not_applicable"},)
            return FittedForecast(forecast, FitEvidence(metadata, checks))
        from model_pipelining.sampling import audit_normal_mean
        evidence = audit_normal_mean(self, request, posterior, metadata, checks)
        return FittedForecast(forecast, evidence)

    def forecast(self, request: ForecastRequest) -> Forecast:
        result = self.fit(request)
        if any(item["required"] and item["status"] != "passed" for item in result.evidence.checks):
            raise ValueError("MCMC audit failed; use fit() to inspect retained evidence")
        return result.forecast


# Imported only for static annotations; analytic use does not import samplers.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model_pipelining.sampling import MCMCSettings
