"""Learned local-level variance evidence calculations."""

import json
import math
from pathlib import Path

import numpy as np

from model_pipelining.evidence import EvidenceMetadata, PredictiveReplications, ReplicatedDataset
from model_pipelining.local_level_variance import initial_posterior_components
from model_pipelining.mixtures import NormalMixture
from model_pipelining.state_mixtures import StateMixture
from model_pipelining.models import ForecastRequest
from model_pipelining.uncertainty import interval_probabilities

def _simulate_local(model, *, observation_count, replications, seed, request=None):
    kind = "prior" if request is None else "posterior"
    posterior = None if request is None else model.posterior(request)
    initial_means, initial_variances = (None, None) if posterior is None else initial_posterior_components(posterior)
    gaps = [1.0]*observation_count
    if request is not None:
        gaps[1:] = [(second.timestamp-first.timestamp).total_seconds()/86400
                    for first, second in zip(request.history, request.history[1:])]
    distribution = {"family": "local_level_with_unknown_observation_variance", "configuration": model.configuration,
                    "replication": "Fresh state trajectory from joint initial-state/variance posterior; state innovations resimulated",
                    "elapsed_days": gaps, "quadrature": posterior.evidence if posterior else None}
    datasets, failures = [], []
    for index in range(replications):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0 if kind == "prior" else 1, index])))
        try:
            if posterior is None:
                precision = float(generator.gamma(model.prior_shape))
                if precision <= 0:
                    raise ValueError("Unrepresentable precision draw")
                variance = model.prior_scale/precision
                initial = float(generator.normal(model.initial_mean, math.sqrt(model.initial_variance)))
            else:
                weights = np.asarray(posterior.weights)
                component = generator.choice(len(weights), p=weights/weights.sum())
                variance = posterior.variances[component]
                initial = float(generator.normal(initial_means[component], math.sqrt(initial_variances[component])))
            if not math.isfinite(variance) or variance <= 0 or not math.isfinite(initial):
                raise ValueError("Invalid initial state or variance draw")
            state, levels, values = initial, [], []
            for gap in gaps:
                state = float(generator.normal(state, math.sqrt(model.evolution_variance*gap)))
                levels.append(state)
                values.append(float(generator.normal(state, math.sqrt(variance))))
            if not all(math.isfinite(x) for x in (*levels, *values)):
                raise ValueError("Nonfinite replicated trajectory")
            datasets.append(ReplicatedDataset(index, None, variance, tuple(values), initial, tuple(levels)))
        except (ValueError, OverflowError, FloatingPointError) as error:
            failures.append({"replication": index, "error": f"{type(error).__name__}: {error}"})
    return PredictiveReplications(kind, seed, replications, observation_count, distribution, tuple(datasets), tuple(failures))

def _local_summary(model, case, level, *, posterior=None):
    if posterior is None:
        posterior = model.prior() if case is None else model.posterior(
            ForecastRequest(case.history, case.issued_at, case.target_time, (), ()))
    gap = 1.0 if case is None else (case.target_time-case.history[-1].timestamp).total_seconds()/86400
    predictive = posterior.predictive(gap)
    region, hpd_check = predictive.highest_density(level)
    lower, upper = interval_probabilities(level)
    state = StateMixture(posterior.weights, posterior.level_means, posterior.level_variances)
    state_lower, state_upper = state.ppf(lower), state.ppf(upper)
    result = {}
    result.update(mean_location=posterior.level_mean, mean_variance=posterior.level_variance,
        mean_lower=state_lower, mean_upper=state_upper, mean_moment_status="finite_mean_and_variance",
        observation_variance_mean=posterior.observation_variance_mean,
        observation_variance_median=posterior.ppf(0.5), variance_lower=posterior.ppf(lower), variance_upper=posterior.ppf(upper),
        variance_moment_status="mean_undefined" if posterior.observation_variance_mean is None else "finite_mean",
        predictive_mean=predictive.mean if case is not None or model.prior_shape > 0.5 else None,
        predictive_variance=predictive.variance if posterior.observation_variance_mean is not None else None,
        predictive_lower=region.segments[0].lower, predictive_upper=region.segments[-1].upper,
        predictive_width=math.fsum(item.upper-item.lower for item in region.segments),
        predictive_moment_status="mean_undefined" if case is None and model.prior_shape <= 0.5 else
                                "variance_undefined" if posterior.observation_variance_mean is None else "finite_mean_and_variance",
        state_interval_method="equal_tail", predictive_intervals_json=json.dumps([{"lower": item.lower, "upper": item.upper} for item in region.segments]),
        predictive_segment_count=len(region.segments), quadrature_nodes=posterior.evidence["nodes"],
        quadrature_tail_bound=posterior.evidence["tail_probability_bound"])
    result["_integration_evidence"] = {"quadrature": posterior.evidence, "predictive_hpd": hpd_check}
    return result


class LocalLevelVarianceEvidence:
    def __init__(self, model):
        self.model = model
        self.metadata = EvidenceMetadata(
            family="local_level_variance",
            implementation_files=tuple(Path(__file__).with_name(name) for name in
                ("evidence_local_level.py", "local_level_variance.py", "mixtures.py", "state_mixtures.py")),
            parameter_fields=("mean", "observation_variance", "initial_level"), latent_paths=True,
            integration_checks=True, mean_prior_settings=("initial_mean", "initial_variance"),
            predictive_scope={
                "generation": "One observation variance and initial state per fresh trajectory; state innovations resimulated",
                "posterior": "Joint variance/initial-state posterior via checked quadrature and conditional Gaussian smoothing; fresh process path, not a reused fitted path"},
            influence_scope={
                "purpose": "Descriptive prior influence on numerically integrated learning and prediction; human interpretation pending",
                "inference": "Controlled log-variance integration; no MCMC or held-out scoring",
                "intervals": "State and observation-variance equal-tail credible intervals; observation-predictive mixture HPD with disconnected segments retained",
                "prior_mean_weight": "No scalar prior-location coefficient is claimed for integrated local-level inference",
                "scale_components": "Variance posterior is nonconjugate; no normal/inverse-gamma scale decomposition is claimed"},
            presentation={"generation": "one variance and fresh latent path per dataset",
                "intervals": "state/variance equal-tail credible; predictive mixture HPD",
                "predictive_segments": True, "prior_location_coefficient": False, "degrees_of_freedom": False})

    def simulate(self, *, observation_count, replications, seed, request=None):
        return _simulate_local(self.model, observation_count=observation_count,
            replications=replications, seed=seed, request=request)

    def summarize(self, case, level):
        return _local_summary(self.model, case, level)

    def predictive(self, request):
        posterior = self.model.posterior(request)
        gap = (request.target_time-request.history[-1].timestamp).total_seconds()/86400
        law = posterior.predictive(gap)
        return {"family": "gaussian_mixture", "weights": law.weights, "means": law.means,
                "variances": law.variances, "mean": law.mean,
                "variance": law.variance if posterior.observation_variance_mean is not None else None,
                "inference": posterior.evidence}

    @property
    def calibration_parameters(self):
        return ("observation_variance", *(("initial_level",) if self.model.initial_variance > 0 else ()),
                *(("last_level",) if self.model.initial_variance+self.model.evolution_variance > 0 else ()))

    def calibration_draw(self, count, generator):
        variance = self.model.prior_scale/float(generator.gamma(self.model.prior_shape))
        initial = float(generator.normal(self.model.initial_mean, math.sqrt(self.model.initial_variance)))
        state, values = initial, []
        for _ in range(count):
            state = float(generator.normal(state, math.sqrt(self.model.evolution_variance)))
            values.append(float(generator.normal(state, math.sqrt(variance))))
        truth = {"observation_variance": variance, "initial_level": initial, "last_level": state}
        return {key: truth[key] for key in self.calibration_parameters}, tuple(values)

    def calibration_fit(self, request, truth, level):
        posterior = self.model.posterior(request)
        laws = {"observation_variance": posterior}
        if "initial_level" in truth:
            means, variances = initial_posterior_components(posterior)
            laws["initial_level"] = NormalMixture(posterior.weights, tuple(means), tuple(variances))
        if "last_level" in truth:
            laws["last_level"] = NormalMixture(posterior.weights, posterior.level_means, posterior.level_variances)
        lower, upper = (1-level)/2, (1+level)/2
        return {name: {"cdf": float(law.cdf(truth[name])), "lower": float(law.ppf(lower)),
                       "median": float(law.ppf(.5)), "upper": float(law.ppf(upper))}
                for name, law in laws.items()}, posterior.evidence

    def validate_scenarios(self, models):
        if any(type(model) is not type(self.model) for model in models):
            raise ValueError("Use matching local-level models")
        if any(model.evolution_variance != self.model.evolution_variance or
               model.quadrature != self.model.quadrature for model in models):
            raise ValueError("Local-level prior comparisons must share evolution variance and quadrature settings")
