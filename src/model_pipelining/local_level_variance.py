"""Local-level inference with fixed evolution variance and learned observation variance.

The initial Gaussian state prior and evolution variance are independent of V.
V has an inverse-gamma prior. Integration over log(V) produces a weighted
Gaussian forecast mixture, generally not a Student-t distribution.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import version
import math
from numbers import Real
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp, roots_legendre
from scipy.stats import invgamma

from model_pipelining.inference import FitEvidence, FittedForecast
from model_pipelining.mixtures import NormalMixture
from model_pipelining.models import Forecast, ForecastRequest, Quantile
from model_pipelining.uncertainty import validate_forecast


class NumericalIntegrationError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True, kw_only=True)
class QuadratureSettings:
    minimum_nodes: int = 32
    maximum_nodes: int = 1024
    relative_tolerance: float = 1e-6
    tail_tolerance: float = 1e-8
    maximum_expansions: int = 9

    def __post_init__(self):
        for key in ("minimum_nodes", "maximum_nodes", "maximum_expansions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (8 if key != "maximum_expansions" else 1):
                raise ValueError(f"Invalid {key}")
        if self.maximum_nodes < 2*self.minimum_nodes:
            raise ValueError("maximum_nodes must allow at least one node doubling")
        for key in ("relative_tolerance", "tail_tolerance"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 1e-13 <= value <= 1e-3:
                raise ValueError(f"{key} must be finite and between 1e-13 and 1e-3")


@lru_cache(maxsize=16)
def _rule(count):
    return roots_legendre(count)


def _history(history):
    previous = None
    for item in history:
        if not isinstance(item.timestamp, datetime) or item.timestamp.utcoffset() is None:
            raise ValueError("History timestamps must be aware")
        time = item.timestamp.astimezone(UTC)
        if previous is not None and time <= previous:
            raise ValueError("History must be strictly ordered")
        if isinstance(item.value, bool) or not isinstance(item.value, Real) or not math.isfinite(item.value):
            raise ValueError("History values must be finite numbers")
        previous = time


def conditional_filter(model, history, variances, *, paths=False):
    """Parallel Kalman filters, one per fixed V; first transition is one day.

    Later state increments use actual elapsed UTC days, including gaps. Paths
    are filtered states, never smoothed states or future observations.
    """
    _history(history)
    v = np.asarray(variances, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("Observation variances must be finite and positive")
    mean = np.full(len(v), model.initial_mean)
    variance = np.full(len(v), model.initial_variance)
    log_likelihood = np.zeros(len(v))
    means, covariances, gaps = [], [], []
    previous = None
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for item in history:
            time = item.timestamp.astimezone(UTC)
            gap = (time-previous).total_seconds()/86400 if previous is not None else 1.0
            predicted = variance+model.evolution_variance*gap
            q = predicted+v
            innovation = item.value-mean
            log_likelihood -= 0.5*(math.log(2*math.pi)+np.log(q)+innovation*innovation/q)
            gain = predicted/q
            mean = (v/q)*mean+gain*item.value
            variance = v*gain
            if paths:
                means.append(mean.copy())
                covariances.append(variance.copy())
                gaps.append(gap)
            previous = time
    if not all(np.isfinite(x).all() for x in (mean, variance, log_likelihood)):
        raise ValueError("Nonfinite conditional filter")
    if paths:
        return mean, variance, log_likelihood, np.asarray(means), np.asarray(covariances), tuple(gaps)
    return mean, variance, log_likelihood


def initial_posterior_components(posterior):
    """Conditional smoothed initial-state moments for full-data replication only.

    Forward forecasting never calls this function. Smoothing conditions on the
    selected history, with no observations beyond that history.
    """
    model = posterior.model
    _, _, _, means, covariances, gaps = conditional_filter(model, posterior.history, posterior.variances, paths=True)
    mean, covariance = means[-1].copy(), covariances[-1].copy()
    for index in range(len(means)-2, -1, -1):
        increment = model.evolution_variance*gaps[index+1]
        predicted = covariances[index]+increment
        gain = np.divide(covariances[index], predicted, out=np.zeros_like(predicted), where=predicted > 0)
        conditional = covariances[index]*np.divide(increment, predicted, out=np.zeros_like(predicted), where=predicted > 0)
        mean = means[index]+gain*(mean-means[index])
        covariance = conditional+gain*gain*covariance
    predicted = model.initial_variance+model.evolution_variance
    gain = model.initial_variance/predicted if predicted > 0 else 0.0
    return (model.initial_mean+gain*(mean-model.initial_mean),
            model.initial_variance*(model.evolution_variance/predicted if predicted > 0 else 0)+gain*gain*covariance)


@dataclass(frozen=True)
class VariancePosterior:
    model: object
    history: tuple
    variances: tuple[float, ...]
    weights: tuple[float, ...]
    level_means: tuple[float, ...]
    level_variances: tuple[float, ...]
    log_bounds: tuple[float, float]
    log_evidence: float
    evidence: dict
    passed: bool

    @property
    def observation_variance_mean(self):
        if self.model.prior_shape+len(self.history)/2 <= 1:
            return None
        return math.fsum(w*v for w, v in zip(self.weights, self.variances))

    @property
    def level_mean(self):
        return math.fsum(w*m for w, m in zip(self.weights, self.level_means))

    @property
    def level_variance(self):
        mean = self.level_mean
        return math.fsum(w*(c+(m-mean)**2) for w, m, c in zip(self.weights, self.level_means, self.level_variances))

    def cdf(self, value):
        if value <= 0:
            return 0.0
        boundary = math.log(value)
        if boundary <= self.log_bounds[0]:
            return 0.0
        if boundary >= self.log_bounds[1]:
            return 1.0
        grid = self.model._grid(self.history, (self.log_bounds[0], boundary), self.evidence["nodes"])
        return min(1.0, max(0.0, math.exp(grid["log_evidence"]-self.log_evidence)))

    def ppf(self, probability):
        if isinstance(probability, bool) or not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Variance quantile probability must be in (0,1)")
        root = brentq(lambda z: self.cdf(math.exp(z))-probability, *self.log_bounds, xtol=1e-10, rtol=1e-12)
        result = math.exp(root)
        if abs(self.cdf(result)-probability) > 10*self.model.quadrature.relative_tolerance:
            raise ValueError("Variance quantile precision failed")
        return result

    def predictive(self, elapsed_days=1.0):
        if not math.isfinite(elapsed_days) or elapsed_days < 0:
            raise ValueError("Projection time must be finite and nonnegative")
        return NormalMixture(self.weights, self.level_means,
            tuple(c+self.model.evolution_variance*elapsed_days+v for c, v in zip(self.level_variances, self.variances)))


@dataclass(frozen=True, kw_only=True)
class LocalLevelVariance:
    initial_mean: float = 0.0
    initial_variance: float = 100.0
    evolution_variance: float = 0.1
    prior_shape: float = 3.0
    prior_scale: float = 2.0
    quadrature: QuadratureSettings = field(default_factory=QuadratureSettings)
    name: ClassVar[str] = "local-level-variance"
    supports_quantiles: ClassVar[bool] = True
    supports_highest_density: ClassVar[bool] = True
    supports_fit_evidence: ClassVar[bool] = True
    supports_distribution: ClassVar[bool] = True

    def __post_init__(self):
        for key in ("initial_mean", "initial_variance", "evolution_variance", "prior_shape", "prior_scale"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if key in ("initial_variance", "evolution_variance") and value < 0:
                raise ValueError(f"{key} must be nonnegative")
            if key in ("prior_shape", "prior_scale") and value <= 0:
                raise ValueError(f"{key} must be positive")
            object.__setattr__(self, key, float(value))
        if not isinstance(self.quadrature, QuadratureSettings):
            raise ValueError("quadrature must be QuadratureSettings")

    @classmethod
    def from_configuration(cls, configuration):
        parameters = dict(configuration)
        if "quadrature" in parameters:
            parameters["quadrature"] = QuadratureSettings(**parameters["quadrature"])
        return cls(**parameters)

    def evidence_provider(self):
        """Supply model calculations for shared predictive/prior evidence workflows."""
        from model_pipelining.evidence_local_level import LocalLevelVarianceEvidence
        return LocalLevelVarianceEvidence(self)

    @property
    def configuration(self):
        return asdict(self)

    @property
    def implementation_files(self):
        return (Path(__file__).with_name("mixtures.py"),)

    def _request(self, request):
        _history(request.history)
        if (not request.history or not isinstance(request.issued_at, datetime) or request.issued_at.utcoffset() is None
                or not isinstance(request.target_time, datetime) or request.target_time.utcoffset() is None
                or request.history[-1].timestamp > request.issued_at or request.target_time <= request.issued_at):
            raise ValueError("Need nonempty eligible history and aware future coordinates")
        return (request.target_time.astimezone(UTC)-request.history[-1].timestamp.astimezone(UTC)).total_seconds()/86400

    def _grid(self, history, bounds, count):
        nodes, weights = _rule(count)
        half = (bounds[1]-bounds[0])/2
        log_v = (bounds[1]+bounds[0])/2+half*nodes
        v = np.exp(log_v)
        mean, covariance, likelihood = conditional_filter(self, history, v)
        log_prior = self.prior_shape*math.log(self.prior_scale)-float(gammaln(self.prior_shape))
        log_weights = (np.log(weights*half)+likelihood+log_prior-self.prior_shape*log_v-self.prior_scale/v)
        evidence = float(logsumexp(log_weights))
        if not math.isfinite(evidence):
            raise ValueError("Unrepresentable quadrature evidence")
        posterior_weights = np.exp(log_weights-evidence)
        return {"v": v, "weights": posterior_weights, "means": mean, "covariances": covariance,
                "log_evidence": evidence, "log_v": log_v}

    def _tail_bounds(self, history, bounds, log_evidence):
        shape = self.prior_shape+len(history)/2
        log_constant = (float(gammaln(shape)-gammaln(self.prior_shape))
                        -len(history)/2*(math.log(2*math.pi)+math.log(self.prior_scale)))
        low, high = map(math.exp, bounds)
        # L(V) <= (2*pi*V)^(-n/2), so IG(shape=a0+n/2,scale=b0)
        # bounds the unnormalized posterior in both omitted tails.
        tail = float(invgamma.cdf(low, shape, scale=self.prior_scale)+invgamma.sf(high, shape, scale=self.prior_scale))
        log_bound = log_constant+math.log(max(tail, 1e-300))-log_evidence
        probability = math.exp(min(700, log_bound))
        moment = None
        if shape > 1:
            tail_moment = float(invgamma.cdf(low, shape-1, scale=self.prior_scale)+invgamma.sf(high, shape-1, scale=self.prior_scale))
            log_moment = log_constant+math.log(self.prior_scale/(shape-1))+math.log(max(tail_moment, 1e-300))-log_evidence
            moment = math.exp(min(700, log_moment))
        return probability, moment

    def _integrate(self, history):
        _history(history)
        settings = self.quadrature
        center = math.log(self.prior_scale)-math.log(self.prior_shape+1)
        if not -690 < center < 690:
            raise ValueError("Prior variance scale is outside numeric integration range")
        accepted = False
        comparisons = []
        final = None
        for expansion in range(settings.maximum_expansions):
            width = 4*(1.5**expansion)
            bounds = (max(-700.0, center-width), min(700.0, center+width))
            previous, streak = None, 0
            count = settings.minimum_nodes
            while count <= settings.maximum_nodes:
                current = self._grid(history, bounds, count)
                weights, v = current["weights"], current["v"]
                mean = float(weights@current["means"])
                spread = float(weights@(current["covariances"]+(current["means"]-mean)**2))
                v_mean = float(weights@v) if self.prior_shape+len(history)/2 > 1 else None
                moments = [mean, spread, float(weights@current["log_v"])]
                if v_mean is not None:
                    moments.append(v_mean)
                error = None
                if previous is not None:
                    error = max(abs(current["log_evidence"]-previous["grid"]["log_evidence"]),
                                max(abs(a-b)/max(1, abs(a), abs(b)) for a, b in zip(moments, previous["moments"])))
                    mixture = NormalMixture(tuple(weights), tuple(current["means"]), tuple(current["covariances"]+v+self.evolution_variance))
                    older = NormalMixture(tuple(previous["grid"]["weights"]), tuple(previous["grid"]["means"]),
                                          tuple(previous["grid"]["covariances"]+previous["grid"]["v"]+self.evolution_variance))
                    error = max(error, max(abs(older.cdf(mixture.ppf(p))-p) for p in (0.025, 0.1, 0.5, 0.9, 0.975)))
                    streak = streak+1 if error <= settings.relative_tolerance else 0
                tail, moment_tail = self._tail_bounds(history, bounds, current["log_evidence"])
                tail_passed = tail <= settings.tail_tolerance and (moment_tail is None or
                    moment_tail <= settings.tail_tolerance*max(1, v_mean))
                comparisons.append({"expansion": expansion, "nodes": count, "relative_change": error,
                                    "tail_probability_bound": tail, "variance_moment_tail_bound": moment_tail})
                final = current
                if streak >= 2:
                    if tail_passed:
                        accepted = True
                    break
                previous = {"moments": moments, "grid": current}
                count *= 2
            if accepted or not tail_passed and bounds == (-700.0, 700.0):
                break
            # Refinement failure on this support cannot be repaired by widening it.
            if not tail_passed:
                continue
            break
        evidence = {"method": "Gauss-Legendre integration in log variance", "nodes": len(final["v"]),
                    "versions": {name: version(name) for name in ("numpy", "scipy")},
                    "log_bounds": list(bounds), "log_evidence": final["log_evidence"], "converged": accepted,
                    "settings": asdict(settings), "refinement_history": comparisons,
                    "tail_probability_bound": tail, "variance_moment_tail_bound": moment_tail,
                    "tail_bound_derivation": "L(V) <= (2*pi*V)^(-n/2); integrate prior times this bound outside support",
                    "moment_status": "finite" if self.prior_shape+len(history)/2 > 1 else "predictive_variance_undefined"}
        return VariancePosterior(self, tuple(history), tuple(map(float, final["v"])), tuple(map(float, final["weights"])),
            tuple(map(float, final["means"])), tuple(map(float, final["covariances"])), bounds, final["log_evidence"], evidence, accepted)

    def posterior(self, request):
        self._request(request)
        posterior = self._integrate(request.history)
        if not posterior.passed:
            raise NumericalIntegrationError("Required observation-variance integration checks failed", posterior.evidence)
        return posterior

    def prior(self):
        posterior = self._integrate(())
        if not posterior.passed:
            raise NumericalIntegrationError("Required prior integration checks failed", posterior.evidence)
        return posterior

    def predictive_distribution(self, request):
        gap = self._request(request)
        return self.posterior(request).predictive(gap)

    def fit(self, request):
        gap = self._request(request)
        posterior = self._integrate(request.history)
        mixture = posterior.predictive(gap)
        checks = ({"name": "variance_quadrature_precision_and_tails", "required": True,
                   "status": "passed" if posterior.passed else "failed"},)
        hpds, hpd_evidence, output_error = [], [], None
        quantiles = ()
        try:
            if any(min(p, 1-p) < 100*self.quadrature.tail_tolerance for p in request.quantile_levels):
                raise ValueError("Requested quantile exceeds configured integration tail precision")
            if any(1-level < 100*self.quadrature.tail_tolerance for level in request.highest_density_levels):
                raise ValueError("Requested HPD level exceeds configured integration tail precision")
            quantiles = tuple(Quantile(p, mixture.ppf(p)) for p in request.quantile_levels)
            for level in request.highest_density_levels:
                region, evidence = mixture.highest_density(level)
                hpds.append(region)
                hpd_evidence.append(evidence)
            forecast = validate_forecast(Forecast(mixture.mean, quantiles, tuple(hpds),
                {"family": "gaussian_mixture", "weights": list(mixture.weights),
                 "means": list(mixture.means), "variances": list(mixture.variances)}),
                                         request.quantile_levels, request.highest_density_levels)
        except (ValueError, OverflowError, FloatingPointError) as error:
            output_error = f"{type(error).__name__}: {error}"
            forecast = Forecast(mixture.mean)
        checks += ({"name": "marginal_predictive_quantiles_and_hpd", "required": True,
                    "status": "passed" if output_error is None else "failed"},)
        metadata = {"inference": "deterministic_log_variance_quadrature", "forecast_method": "marginal_gaussian_mixture",
                    "configuration": self.configuration, "quadrature": posterior.evidence,
                    "posterior_level_mean": posterior.level_mean, "posterior_level_variance": posterior.level_variance,
                    "observation_variance_mean": posterior.observation_variance_mean,
                    "predictive_mean": mixture.mean,
                    "predictive_variance": mixture.variance if posterior.observation_variance_mean is not None else None,
                    "elapsed_days_to_target": gap, "hpd_checks": hpd_evidence, "output_error": output_error,
                    "quadrature_nodes": [{"variance": v, "weight": w, "level_mean": m, "level_variance": c}
                        for v, w, m, c in zip(posterior.variances, posterior.weights, posterior.level_means, posterior.level_variances)]}
        return FittedForecast(forecast, FitEvidence(metadata, checks))

    def forecast(self, request):
        result = self.fit(request)
        if any(check["status"] != "passed" for check in result.evidence.checks if check["required"]):
            raise ValueError("Required numerical inference checks failed; use fit() to inspect evidence")
        return result.forecast
