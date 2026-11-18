"""Normal/inverse-gamma learning of a constant mean and observation variance."""

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import ClassVar, TYPE_CHECKING

from model_pipelining.inference import FitEvidence, FittedForecast
from model_pipelining.models import Forecast, ForecastRequest, HighestDensityRegion, Interval, Quantile
from model_pipelining.uncertainty import interval_probabilities, validate_forecast

if TYPE_CHECKING:
    from model_pipelining.sampling import MCMCSettings


@dataclass(frozen=True)
class NormalInverseGammaPosterior:
    mean: float
    mean_precision: float
    shape: float
    scale: float
    prior_mean_weight: float

    @property
    def degrees_of_freedom(self) -> float:
        return 2 * self.shape

    @property
    def mean_scale_squared(self) -> float:
        return self.scale / self.shape / self.mean_precision

    @property
    def predictive_scale_squared(self) -> float:
        return (self.scale / self.shape) * (1 + 1 / self.mean_precision)

    @property
    def observation_variance_mean(self) -> float | None:
        return self.scale / (self.shape - 1) if self.shape > 1 else None

    @property
    def mean_variance(self) -> float | None:
        noise = self.observation_variance_mean
        return noise / self.mean_precision if noise is not None else None

    @property
    def predictive_variance(self) -> float | None:
        noise = self.observation_variance_mean
        return noise * (1 + 1 / self.mean_precision) if noise is not None else None


@dataclass(frozen=True, kw_only=True)
class NormalMeanVariance:
    """Learn mu and v: v ~ InvGamma(shape, scale), mu|v ~ N(m0, v/k0).

    InvGamma density is proportional to v**(-shape-1) * exp(-scale/v).
    Positive shape/scale/mean precision define a proper prior. Undefined
    posterior variances are None, not invented finite moments. Forecasts are
    exact Student-t summaries in both exact and optional MCMC-audit modes.
    """

    prior_mean: float = 0.0
    prior_mean_precision: float = 0.01
    prior_shape: float = 3.0
    prior_scale: float = 2.0
    mcmc: "MCMCSettings | None" = None
    name: ClassVar[str] = "normal-mean-variance"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_fit_evidence: ClassVar[bool] = True

    def __post_init__(self) -> None:
        for name in ("prior_mean", "prior_mean_precision", "prior_shape", "prior_scale"):
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

    def evidence_provider(self):
        """Supply model calculations for shared predictive/prior evidence workflows."""
        from model_pipelining.evidence_normal import NormalMeanVarianceEvidence
        return NormalMeanVarianceEvidence(self)

    @property
    def configuration(self) -> dict:
        return asdict(self)

    @property
    def implementation_files(self) -> tuple[Path, ...]:
        return tuple(Path(__file__).with_name(name) for name in
                     ("sampling.py", "sampling_variance.py")) if self.mcmc is not None else ()

    def posterior(self, request: ForecastRequest) -> NormalInverseGammaPosterior:
        if (not isinstance(request.issued_at, datetime) or request.issued_at.utcoffset() is None
                or not isinstance(request.target_time, datetime) or request.target_time.utcoffset() is None
                or request.target_time <= request.issued_at or not request.history):
            raise ValueError("NormalMeanVariance requires nonempty past history and an aware future target")
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
        kappa = self.prior_mean_precision + count
        weight = self.prior_mean_precision / kappa
        average = fmean(item.value for item in request.history)
        mean = weight * self.prior_mean + (count / kappa) * average
        shape = self.prior_shape + count / 2
        try:
            scatter = math.fsum((item.value - average)**2 for item in request.history)
            adjustment = (math.sqrt(weight * count) * (average - self.prior_mean))**2
            scale = math.fsum((self.prior_scale, scatter / 2, adjustment / 2))
            posterior = NormalInverseGammaPosterior(mean, kappa, shape, scale, weight)
            positive = (kappa, shape, scale, weight, posterior.degrees_of_freedom,
                        posterior.mean_scale_squared, posterior.predictive_scale_squared)
            moments = (posterior.mean_variance, posterior.observation_variance_mean, posterior.predictive_variance)
            if (not math.isfinite(mean) or any(not math.isfinite(x) or x <= 0 for x in positive)
                    or any(x is not None and (not math.isfinite(x) or x <= 0) for x in moments)
                    or posterior.degrees_of_freedom <= 1):
                raise ValueError("Unrepresentable posterior or predictive moments")
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("Unrepresentable posterior or predictive moments") from error
        return posterior

    def predictive_distribution(self, request: ForecastRequest):
        """Return a frozen scipy.stats.t law; scale is not its SD."""
        from scipy.stats import t
        posterior = self.posterior(request)
        return t(df=posterior.degrees_of_freedom, loc=posterior.mean,
                 scale=math.sqrt(posterior.predictive_scale_squared))

    def fit(self, request: ForecastRequest) -> FittedForecast:
        from scipy.stats import t
        posterior = self.posterior(request)
        predictive_scale = math.sqrt(posterior.predictive_scale_squared)
        distribution = t(df=posterior.degrees_of_freedom, loc=posterior.mean, scale=predictive_scale)
        regions = []
        for level in request.highest_density_levels:
            lower, upper = interval_probabilities(level)
            regions.append(HighestDensityRegion(level, (Interval(float(distribution.ppf(lower)),
                                                                 float(distribution.ppf(upper))),)))
        forecast = validate_forecast(Forecast(posterior.mean,
            tuple(Quantile(p, float(distribution.ppf(p))) for p in request.quantile_levels), tuple(regions)),
            request.quantile_levels, request.highest_density_levels)
        references = [
            {"column": "mu", "family": "student_t", "label": "mu (parameter)",
             "df": posterior.degrees_of_freedom, "loc": posterior.mean,
             "scale": math.sqrt(posterior.mean_scale_squared)},
            {"column": "observation_variance", "family": "inverse_gamma", "label": "observation variance",
             "shape": posterior.shape, "scale": posterior.scale},
            {"column": "future_value", "family": "student_t", "label": "future observation",
             "df": posterior.degrees_of_freedom, "loc": posterior.mean, "scale": predictive_scale},
        ]
        metadata = {
            "inference": "analytic" if self.mcmc is None else "analytic_with_mcmc_audit",
            "forecast_method": "exact_normal_inverse_gamma_student_t",
            "parameter_names": ["mu", "observation_variance"],
            "parameter_units": {"mu": "target units", "observation_variance": "squared target units"},
            "posterior": asdict(posterior), "configuration": self.configuration,
            "predictive": {"family": "student_t", "df": posterior.degrees_of_freedom,
                           "loc": posterior.mean, "scale": predictive_scale},
            "predictive_variance": posterior.predictive_variance,
            "mean_variance": posterior.mean_variance,
            "observation_variance_mean": posterior.observation_variance_mean,
            "moment_status": "finite_variance" if posterior.shape > 1 else "variance_undefined_shape_le_one",
            "reference_distributions": references,
            "prior_influence": "prior_mean_weight = prior_mean_precision / posterior_mean_precision; no magnitude threshold",
        }
        checks = ({"name": "exact_posterior_and_predictive_valid", "required": True, "status": "passed"},)
        if self.mcmc is None:
            checks += ({"name": "mcmc_diagnostics", "required": False, "status": "not_applicable"},)
            return FittedForecast(forecast, FitEvidence(metadata, checks))
        from model_pipelining.sampling_variance import audit_normal_mean_variance
        return FittedForecast(forecast, audit_normal_mean_variance(self, request, posterior, metadata, checks))

    def forecast(self, request: ForecastRequest) -> Forecast:
        result = self.fit(request)
        if any(item["required"] and item["status"] != "passed" for item in result.evidence.checks):
            raise ValueError("MCMC audit failed; use fit() to inspect retained evidence")
        return result.forecast
