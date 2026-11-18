"""Exact normal/inverse-gamma evidence calculations."""

import math
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.stats import invgamma, t

from model_pipelining.evidence import EvidenceMetadata, PredictiveReplications, ReplicatedDataset
from model_pipelining.models import ForecastRequest
from model_pipelining.normal_mean_variance import NormalInverseGammaPosterior
from model_pipelining.uncertainty import interval_probabilities

def _simulate(*, kind, distribution, observation_count, replications, seed):
    datasets, failures = [], []
    for index in range(replications):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [seed, 0 if kind == "prior" else 1, index])))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Gamma is unit-scale precision: v=b/G, G~Gamma(a,1).
                gamma = float(generator.gamma(distribution["shape"]))
                if not math.isfinite(gamma) or gamma <= 0:
                    raise ValueError("Unrepresentable precision draw")
                variance = distribution["scale"] / gamma
                conditional_variance = variance / distribution["mean_precision"]
                if not all(math.isfinite(x) and x > 0 for x in (variance, conditional_variance)):
                    raise ValueError("Unrepresentable variance draw")
                mu = float(generator.normal(distribution["mean"], math.sqrt(conditional_variance)))
                values = generator.normal(mu, math.sqrt(variance), size=observation_count)
                if not math.isfinite(mu) or not np.isfinite(values).all():
                    raise ValueError("Nonfinite parameter or replicated dataset")
            datasets.append(ReplicatedDataset(index, mu, variance, tuple(map(float, values))))
        except (ValueError, OverflowError, FloatingPointError) as error:
            failures.append({"replication": index, "error": f"{type(error).__name__}: {error}"})
    return PredictiveReplications(kind, seed, replications, observation_count, distribution,
                                  tuple(datasets), tuple(failures))

def _summary(model, case, level):
    history = case.history if case is not None else ()
    if history:
        posterior = model.posterior(ForecastRequest(history, case.issued_at, case.target_time, (), ()))
        average = fmean(item.value for item in history)
        scatter = math.fsum((item.value-average)**2 for item in history) / 2
        disagreement = (math.sqrt(posterior.prior_mean_weight * len(history)) * (average-model.prior_mean))**2 / 2
    else:
        posterior = NormalInverseGammaPosterior(model.prior_mean, model.prior_mean_precision,
                                                model.prior_shape, model.prior_scale, 1.0)
        scatter, disagreement = 0.0, 0.0
    lower, upper = interval_probabilities(level)
    mean_scale = math.sqrt(posterior.mean_scale_squared)
    predictive_scale = math.sqrt(posterior.predictive_scale_squared)
    mean_law = t(df=posterior.degrees_of_freedom, loc=posterior.mean, scale=mean_scale)
    predictive = t(df=posterior.degrees_of_freedom, loc=posterior.mean, scale=predictive_scale)
    variance_law = invgamma(a=posterior.shape, scale=posterior.scale)
    result = {
        "mean_location": posterior.mean, "mean_precision": posterior.mean_precision,
        "shape": posterior.shape, "scale": posterior.scale, "degrees_of_freedom": posterior.degrees_of_freedom,
        "prior_mean_weight": posterior.prior_mean_weight,
        "mean_lower": float(mean_law.ppf(lower)), "mean_upper": float(mean_law.ppf(upper)),
        "mean_variance": posterior.mean_variance,
        "mean_moment_status": "mean_undefined" if posterior.shape <= 0.5 else
                              "variance_undefined" if posterior.shape <= 1 else "finite_mean_and_variance",
        "observation_variance_mean": posterior.observation_variance_mean,
        "observation_variance_median": float(variance_law.ppf(0.5)),
        "variance_lower": float(variance_law.ppf(lower)), "variance_upper": float(variance_law.ppf(upper)),
        "variance_moment_status": "mean_undefined" if posterior.shape <= 1 else
                                  "variance_undefined" if posterior.shape <= 2 else "finite_mean_and_variance",
        "predictive_mean": posterior.mean if posterior.shape > 0.5 else None,
        "predictive_variance": posterior.predictive_variance, "predictive_scale": predictive_scale,
        "predictive_lower": float(predictive.ppf(lower)), "predictive_upper": float(predictive.ppf(upper)),
        "predictive_moment_status": "mean_undefined" if posterior.shape <= 0.5 else
                                    "variance_undefined" if posterior.shape <= 1 else "finite_mean_and_variance",
        "prior_scale_component": model.prior_scale, "data_scatter_component": scatter,
        "prior_data_disagreement_component": disagreement,
    }
    result["predictive_width"] = result["predictive_upper"]-result["predictive_lower"]
    positive = (mean_scale, predictive_scale, result["variance_lower"], result["variance_upper"],
                result["observation_variance_median"], result["predictive_width"])
    if (any(not math.isfinite(x) or x <= 0 for x in positive)
            or any(not math.isfinite(x) for x in result.values() if isinstance(x, (int, float)))):
        raise ValueError("Unrepresentable prior/posterior summary or interval")
    return result


class NormalMeanVarianceEvidence:
    def __init__(self, model):
        if model.mcmc is not None:
            raise ValueError("Evidence requires NormalMeanVariance with mcmc=None")
        self.model = model
        self.metadata = EvidenceMetadata(
            family="normal_mean_variance",
            implementation_files=(Path(__file__), Path(__file__).with_name("normal_mean_variance.py")),
            predictive_scope={
                "generation": "One independent joint parameter pair per entire replicated dataset",
                "posterior": "Exact conjugate posterior draws; no MCMC, refitting per position, or held-out scoring"},
            influence_scope={
                "purpose": "Descriptive prior influence on exact learning and prediction; human interpretation pending",
                "intervals": "Mean and observation-predictive Student-t HPD; variance parameter inverse-gamma equal-tail credible interval",
                "prior_mean_weight": "k0/(k0+n) is the conditional coefficient of prior location; not total prior influence",
                "scale_components": "b = prior_scale + centered data scatter/2 + prior/data mean disagreement term",
                "inference": "Exact distributions only; no simulation, MCMC or held-out scoring"},
            presentation={"generation": "one mean/variance pair per dataset",
                "intervals": "mean/predictive HPD; variance equal-tail credible intervals",
                "predictive_segments": False, "prior_location_coefficient": True, "degrees_of_freedom": True})

    def simulate(self, *, observation_count, replications, seed, request=None):
        if request is None:
            distribution = {"mean": self.model.prior_mean, "mean_precision": self.model.prior_mean_precision,
                            "shape": self.model.prior_shape, "scale": self.model.prior_scale}
        else:
            posterior = self.model.posterior(request)
            distribution = {key: getattr(posterior, key) for key in ("mean", "mean_precision", "shape", "scale")}
        return _simulate(kind="prior" if request is None else "posterior", distribution=distribution,
            observation_count=observation_count, replications=replications, seed=seed)

    def summarize(self, case, level):
        return _summary(self.model, case, level)

    def validate_scenarios(self, models):
        if any(type(model) is not type(self.model) or model.mcmc is not None for model in models):
            raise ValueError("Use matching NormalMeanVariance models with mcmc=None")
