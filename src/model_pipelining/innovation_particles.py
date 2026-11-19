"""Replicated Rao-Blackwellized particle approximation of the innovation model.

Only V and past innovation labels are sampled. Conditional Gaussian states are
integrated analytically. Monte Carlo diagnostics are empirical, never bounds on
missing posterior mass. This module deliberately supplies no Forecaster adapter.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import math
from numbers import Real

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import t

from model_pipelining.innovation_reference import InnovationMixtureReference, _points, _scale
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.models import ForecastRequest, Observation


MAX_PARTICLE_HISTORY = 512


class ParticlePrecisionError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True, kw_only=True)
class ParticleSettings:
    minimum_particles: int = 512
    maximum_particles: int = 4096
    replicates: int = 16
    seed: int = 0
    resample_fraction: float = .5
    tolerance: float = .05
    confidence: float = .95
    minimum_evidence_fraction: float = .5

    def __post_init__(self):
        for key in ('minimum_particles', 'maximum_particles', 'replicates', 'seed'):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'seed' else 1):
                raise ValueError(f'Invalid {key}')
        ratio = self.maximum_particles // self.minimum_particles
        if (self.minimum_particles < 32 or self.maximum_particles > 65536
                or self.maximum_particles != self.minimum_particles * ratio
                or ratio < 2 or ratio & (ratio - 1) or not 8 <= self.replicates <= 128):
            raise ValueError('Need 32 <= particles <= 65536, a power-of-two refinement ratio >= 2, and 8..128 replicates')
        for key in ('resample_fraction', 'tolerance', 'confidence', 'minimum_evidence_fraction'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'Invalid {key}')
            object.__setattr__(self, key, float(value))
        if not .5 <= self.confidence < 1 or self.tolerance > .1:
            raise ValueError('Need confidence >= .5 and tolerance <= .1')


def _systematic(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative, positions, side='right')


def _condition(model, values, labels, variances):
    """One fixed V/path per particle; independent of the oracle's batch solver."""
    size = len(variances)
    mean = np.full(size, model.initial_mean)
    covariance = np.full(size, model.initial_variance)
    means, covariances = [], []
    w = np.where(labels, model.large_variance, model.small_variance)
    for index, value in enumerate(values):
        predicted = covariance + w[:, index]
        q = predicted + variances
        gain = predicted / q
        mean = (variances / q) * mean + gain * value
        covariance = variances * gain
        means.append(mean)
        covariances.append(covariance)
    last_mean, last_variance = mean.copy(), covariance.copy()
    cross = covariance.copy()
    for index in range(len(values) - 2, -1, -1):
        predicted = covariances[index] + w[:, index + 1]
        gain = np.divide(covariances[index], predicted, out=np.zeros(size), where=predicted > 0)
        conditional = covariances[index] * np.divide(w[:, index + 1], predicted,
            out=np.zeros(size), where=predicted > 0)
        mean = means[index] + gain * (mean - means[index])
        covariance = conditional + gain**2 * covariance
        cross *= gain
    if values:
        predicted = model.initial_variance + w[:, 0]
        gain = np.divide(model.initial_variance, predicted, out=np.zeros(size), where=predicted > 0)
        initial_mean = model.initial_mean + gain * (mean - model.initial_mean)
        initial_variance = model.initial_variance * np.divide(w[:, 0], predicted,
            out=np.zeros(size), where=predicted > 0) + gain**2 * covariance
        cross *= gain
    else:
        initial_mean, initial_variance = mean, covariance
    return dict(initial_mean=initial_mean, initial_variance=initial_variance,
                last_mean=last_mean, last_variance=last_variance, state_covariance=cross)


def _filter(model, values, count, rng, resample_fraction):
    variance = model.prior_scale / rng.gamma(model.prior_shape, size=count)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Unrepresentable inverse-gamma particles')
    labels = np.zeros((count, len(values)), dtype=np.uint8)
    ancestors = np.arange(count)
    mean = np.full(count, model.initial_mean)
    covariance = np.full(count, model.initial_variance)
    log_weights = np.full(count, -math.log(count))
    log_evidence = 0.
    steps = []
    p = model.jump_probability
    log_priors = np.log([1 - p, p], where=np.array([1 - p, p]) > 0,
                        out=np.full(2, -np.inf))
    branch_underflows = weight_underflows = 0
    for index, value in enumerate(values):
        predicted = covariance[:, None] + [model.small_variance, model.large_variance]
        q = predicted + variance[:, None]
        log_branch = log_priors - .5 * (math.log(2 * math.pi) + np.log(q) + (value - mean[:, None])**2 / q)
        log_increment = logsumexp(log_branch, axis=1)
        if not np.isfinite(log_increment).all():
            raise ValueError('Nonfinite particle likelihood; no replacement weights were used')
        jump = np.exp(log_branch[:, 1] - log_increment)
        branch_underflows += int(np.count_nonzero(np.isfinite(log_branch[:, 1]) & (jump == 0)))
        selected = rng.random(count) < jump
        labels[:, index] = selected
        chosen_prediction = covariance + np.where(selected, model.large_variance, model.small_variance)
        chosen_q = chosen_prediction + variance
        gain = chosen_prediction / chosen_q
        mean = (variance / chosen_q) * mean + gain * value
        covariance = variance * gain
        unnormalized = log_weights + log_increment
        normalizer = float(logsumexp(unnormalized))
        log_evidence += normalizer
        log_weights = unnormalized - normalizer
        weights = np.exp(log_weights)
        weight_underflows += int(np.count_nonzero(weights == 0))
        ess = float(1 / (weights @ weights))
        resample = ess < resample_fraction * count and index < len(values) - 1
        if resample:
            parents = _systematic(weights, rng)
            variance, mean, covariance = variance[parents], mean[parents], covariance[parents]
            ancestors, labels = ancestors[parents], labels[parents].copy()
            log_weights.fill(-math.log(count))
        steps.append(dict(index=index, effective_particles=ess, resampled=resample,
                          distinct_variance_ancestors=int(len(np.unique(ancestors)))))
    condition = _condition(model, values, labels, variance)
    if not (np.allclose(condition['last_mean'], mean, atol=1e-12, rtol=1e-12)
            and np.allclose(condition['last_variance'], covariance, atol=1e-12, rtol=1e-12)):
        raise ValueError('Particle ancestry and conditional-state replay disagree')
    return variance, np.exp(log_weights), labels, condition, log_evidence, dict(
        steps=steps, distinct_variance_ancestors=int(len(np.unique(ancestors))),
        distinct_label_paths=int(len(np.unique(labels, axis=0))),
        branch_probability_underflows=branch_underflows, weight_underflows=weight_underflows)


def _state_probability(point, means, variances, inclusive):
    positive = variances > 0
    result = (means <= point if inclusive else means < point).astype(float)
    result[positive] = ndtr((point - means[positive]) / np.sqrt(variances[positive]))
    return result


def _raw(model, values, variance, weights, labels, condition, scale, predictive, state, variance_points):
    """Centered moments retain joint dependence before replicate aggregation."""
    initial = (condition['initial_mean'] - model.initial_mean) / scale
    last = (condition['last_mean'] - model.initial_mean) / scale
    count = labels.sum(axis=1)
    raw = dict(observation_variance_mean=float(weights @ variance / scale**2),
        initial_mean=float(weights @ initial), last_mean=float(weights @ last),
        initial_variance=float(weights @ (initial**2 + condition['initial_variance'] / scale**2)),
        last_variance=float(weights @ (last**2 + condition['last_variance'] / scale**2)),
        initial_last_covariance=float(weights @ (initial * last + condition['state_covariance'] / scale**2)),
        variance_jump_count_covariance=float(weights @ (variance * count) / scale**2),
        jump_count=float(weights @ count), jump_probabilities=(weights @ labels).tolist())
    # Coincident scales make labels analytically independent of all observations/V.
    if model.small_variance == model.large_variance:
        raw['jump_probabilities'] = [model.jump_probability] * len(values)
        raw['jump_count'] = len(values) * model.jump_probability
        raw['variance_jump_count_covariance'] = raw['observation_variance_mean'] * raw['jump_count']
    raw['predictive_cdf'], raw['predictive_density'] = [], []
    for point in predictive:
        cdf = density = 0.
        for probability, w in ((1 - model.jump_probability, model.small_variance),
                               (model.jump_probability, model.large_variance)):
            std = np.sqrt(condition['last_variance'] + variance + w)
            z = (point - condition['last_mean']) / std
            cdf += probability * float(weights @ ndtr(z))
            density += probability * float(weights @ (np.exp(-.5 * z**2) / (math.sqrt(2 * math.pi) * std)))
        raw['predictive_cdf'].append(cdf)
        raw['predictive_density'].append(density)
    for name in ('initial', 'last'):
        for bound, inclusive in (('lower', False), ('upper', True)):
            raw[f'{name}_cdf_{bound}'] = [float(weights @ _state_probability(
                point, condition[f'{name}_mean'], condition[f'{name}_variance'], inclusive)) for point in state]
    raw['variance_cdf'] = [float(weights @ (variance <= point)) for point in variance_points]
    if len(values) <= 8:
        if model.jump_probability in (0, 1):
            raw['sequence_probabilities'] = [1.]
        elif model.small_variance == model.large_variance:
            raw['sequence_probabilities'] = [model.jump_probability**i.bit_count() *
                (1 - model.jump_probability)**(len(values) - i.bit_count()) for i in range(2**len(values))]
        else:
            indices = labels @ (2**np.arange(len(values) - 1, -1, -1))
            raw['sequence_probabilities'] = np.bincount(indices, weights=weights, minlength=2**len(values)).tolist()
    return raw


def _flatten(record):
    return np.concatenate([np.atleast_1d(value) for value in record.values()])


def _unflatten(vector, template):
    result, offset = {}, 0
    for key, value in template.items():
        size = len(value) if isinstance(value, list) else 1
        part = vector[offset:offset + size]
        result[key] = part.tolist() if isinstance(value, list) else float(part[0])
        offset += size
    return result


def _decode(raw, model, scale):
    result = dict(raw)
    im, lm = raw['initial_mean'], raw['last_mean']
    result['initial_variance'] = (raw['initial_variance'] - im**2) * scale**2
    result['last_variance'] = (raw['last_variance'] - lm**2) * scale**2
    result['initial_last_covariance'] = (raw['initial_last_covariance'] - im * lm) * scale**2
    result['variance_jump_count_covariance'] = (raw['variance_jump_count_covariance'] -
        raw['observation_variance_mean'] * raw['jump_count']) * scale**2
    result['observation_variance_mean'] *= scale**2
    result['initial_mean'] = model.initial_mean + im * scale
    result['last_mean'] = model.initial_mean + lm * scale
    result['predictive_mean'] = result['last_mean']
    result['predictive_variance'] = (result['last_variance'] + result['observation_variance_mean'] +
        (1 - model.jump_probability) * model.small_variance + model.jump_probability * model.large_variance)
    result['next_jump_probability'] = model.jump_probability
    return result


def _multipliers(template, scale):
    moments = {'observation_variance_mean', 'initial_variance', 'last_variance',
               'initial_last_covariance', 'variance_jump_count_covariance', 'predictive_variance'}
    means = {'initial_mean', 'last_mean', 'predictive_mean'}
    return _flatten({key: ([scale] * len(value) if key == 'predictive_density' else
        [1.] * len(value)) if isinstance(value, list) else
        (1 / scale**2 if key in moments else 1 / scale if key in means else 1.)
        for key, value in template.items()})


def _aggregate(replicates, model, scale):
    """Ratio of averaged unnormalized estimates, with delete-one-run MCSE."""
    logs = np.array([row['log_evidence'] for row in replicates])
    evidence_weights = np.exp(logs - logs.max())
    raw = np.array([_flatten(row['raw']) for row in replicates])
    numerator = evidence_weights @ raw
    denominator = evidence_weights.sum()
    template = replicates[0]['raw']
    summary = _decode(_unflatten(numerator / denominator, template), model, scale)
    summary['log_evidence'] = float(logs.max() + math.log(denominator / len(logs)))
    deleted = []
    for index in range(len(logs)):
        keep = np.arange(len(logs)) != index
        remaining = evidence_weights[keep].sum()
        if remaining == 0:
            raise ValueError('Replicate evidence collapsed to one representable run')
        item = _decode(_unflatten(evidence_weights[keep] @ raw[keep] / remaining, template), model, scale)
        item['log_evidence'] = float(logs.max() + math.log(remaining / (len(logs) - 1)))
        deleted.append(_flatten(item))
    deleted = np.asarray(deleted)
    mcse = np.sqrt((len(logs) - 1) / len(logs) * np.sum((deleted - deleted.mean(axis=0))**2, axis=0))
    return summary, _unflatten(mcse, summary), float(denominator**2 / (evidence_weights @ evidence_weights))


def _gaussian(model, values, scale, predictive, state, variance_points):
    w = (model.large_variance if model.jump_probability == 1 else model.small_variance)
    scalar = LocalLevelVariance(initial_mean=model.initial_mean, initial_variance=model.initial_variance,
        evolution_variance=w, prior_shape=model.prior_shape, prior_scale=model.prior_scale)
    if values:
        start = datetime(2000, 1, 1, tzinfo=UTC)
        history = tuple(Observation(start + timedelta(days=i), y) for i, y in enumerate(values))
        posterior = scalar.posterior(ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=1)))
    else:
        posterior = scalar.prior()
    variance, weights = np.asarray(posterior.variances), np.asarray(posterior.weights)
    labels = np.full((len(variance), len(values)), int(model.jump_probability == 1), dtype=np.uint8)
    raw = _raw(model, values, variance, weights, labels, _condition(model, values, labels, variance),
               scale, predictive, state, variance_points)
    # Integrate to each boundary; a quadrature rule's discrete node CDF does not
    # inherit its smooth-integrand precision guarantee.
    raw['variance_cdf'] = [posterior.cdf(point) for point in variance_points]
    summary = _decode(raw, model, scale)
    summary['log_evidence'] = posterior.log_evidence
    return dict(mode='gaussian_quadrature', status='completed', posterior=summary, mcse=None,
                numerics=posterior.evidence)


def run_innovation_particles(values, *, model=None, settings=None,
        predictive_points=(-2., 0., 2., 4., 6.), state_points=(0.,), variance_points=(.5, 1., 2.)):
    """Approximate 0..512 daily observations, or raise with every attempted level.

    Resource limits and all requested probes enter the empirical precision gate.
    Independent seed streams depend on particle count and replicate index. The
    same settings/history replay exactly without touching NumPy's global RNG.
    """
    model = InnovationMixtureReference() if model is None else model
    settings = ParticleSettings() if settings is None else settings
    if not isinstance(model, InnovationMixtureReference) or not isinstance(settings, ParticleSettings):
        raise ValueError('Use InnovationMixtureReference and ParticleSettings records')
    if (not isinstance(values, (tuple, list)) or len(values) > MAX_PARTICLE_HISTORY or
            any(isinstance(y, bool) or not isinstance(y, Real) or not math.isfinite(y) for y in values)):
        raise ValueError(f'Use at most {MAX_PARTICLE_HISTORY} finite daily observations as a tuple/list')
    values = tuple(map(float, values))
    predictive, state, variance_points = (_points(predictive_points), _points(state_points),
                                         _points(variance_points, positive=True))
    scale = _scale(model, values)
    result = dict(status='running', model=model.configuration, settings=asdict(settings), values=list(values),
        predictive_points=list(predictive), state_points=list(state), variance_points=list(variance_points),
        comparison_scale=scale, levels=[], uncertainty_kind='empirical independent-run Monte Carlo diagnostics; no certified mass bound')
    if not values or model.jump_probability in (0, 1) or model.small_variance == model.large_variance:
        try:
            result.update(_gaussian(model, values, scale, predictive, state, variance_points))
        except ValueError as error:
            result.update(status='failed', error=str(error), numerical_failure=getattr(error, 'evidence', None))
            raise ParticlePrecisionError('Gaussian-limit quadrature failed', result) from error
        return result
    if model.prior_shape <= 2:
        raise ValueError('Stochastic approximation requires prior_shape > 2 for finite prior variance of V')
    result['mode'] = 'replicated_fully_adapted_particles'
    planned_levels = 1 + int(math.log2(settings.maximum_particles // settings.minimum_particles))
    previous = None
    for count in (settings.minimum_particles * 2**i for i in range(planned_levels)):
        level = dict(particles=count, replicates=[], status='running')
        result['levels'].append(level)
        for replicate in range(settings.replicates):
            rng = np.random.default_rng(np.random.SeedSequence([settings.seed, count, replicate, 167]))
            try:
                variance, weights, labels, condition, log_evidence, diagnostics = _filter(
                    model, values, count, rng, settings.resample_fraction)
                raw = _raw(model, values, variance, weights, labels, condition, scale, predictive, state, variance_points)
                if not np.isfinite(_flatten(raw)).all() or not math.isfinite(log_evidence):
                    raise ValueError('Nonfinite particle summary')
                level['replicates'].append(dict(index=replicate, status='completed', raw=raw,
                                                log_evidence=log_evidence, diagnostics=diagnostics))
            except (ValueError, FloatingPointError, OverflowError) as error:
                level['replicates'].append(dict(index=replicate, status='failed', error=str(error)))
                level['status'], result['status'] = 'failed', 'failed'
                raise ParticlePrecisionError('Particle replicate failed; no replacement replicate was used', result) from error
        try:
            summary, mcse, evidence_ess = _aggregate(level['replicates'], model, scale)
        except (ValueError, FloatingPointError, OverflowError) as error:
            level.update(status='failed', error=str(error))
            result['status'] = 'failed'
            raise ParticlePrecisionError('Particle aggregation failed', result) from error
        if not np.isfinite(_flatten(summary)).all() or not np.isfinite(_flatten(mcse)).all():
            level['status'], result['status'] = 'failed', 'failed'
            raise ParticlePrecisionError('Nonfinite replicate aggregation', result)
        multipliers = _multipliers(summary, scale)
        # Bonferroni over every reported scalar and every planned look. Student-t
        # and jackknife assumptions remain approximate for finite independent runs.
        critical = float(t.isf((1 - settings.confidence) / (2 * len(multipliers) * planned_levels), settings.replicates - 1))
        width = critical * _flatten(mcse) * multipliers
        change = None if previous is None else float(np.max(np.abs(_flatten(summary) - _flatten(previous['posterior'])) * multipliers))
        previous_width = None if previous is None else previous['maximum_scaled_halfwidth']
        refinement = None if change is None else change + float(width.max()) + previous_width
        diversity = min(row['diagnostics']['distinct_variance_ancestors'] for row in level['replicates'])
        checks = dict(halfwidth=float(width.max()) <= settings.tolerance,
            evidence_diversity=evidence_ess >= settings.minimum_evidence_fraction * settings.replicates,
            variance_ancestry=diversity >= 2,
            refinement=refinement is not None and refinement <= 2 * settings.tolerance)
        level.update(status='passed' if all(checks.values()) else 'unresolved', posterior=summary, mcse=mcse,
            critical_value=critical, maximum_scaled_halfwidth=float(width.max()),
            maximum_scaled_change=change, refinement_envelope=refinement,
            replicate_evidence_ess=evidence_ess, minimum_variance_ancestors=diversity, checks=checks)
        if all(checks.values()):
            result.update(status='completed', posterior=summary, mcse=mcse, accepted_particles=count)
            return result
        previous = level
    result['status'] = 'unresolved'
    raise ParticlePrecisionError('Particle precision unresolved at the configured resource limit', result)
