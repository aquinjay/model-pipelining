"""Independent finite-sample probability bounds for the innovation prototype.

These fixed-budget importance draws have a known weight envelope. They do not
resample or reinterpret correlated particles as independent observations.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np
from scipy.special import gammaln, ndtr

from model_pipelining.innovation_particles import MAX_PARTICLE_HISTORY, _condition, _state_probability
from model_pipelining.innovation_reference import InnovationMixtureReference, _points


@dataclass(frozen=True, kw_only=True)
class ProbabilityBoundSettings:
    samples: int = 262144
    seed: int = 0
    confidence: float = .95
    tolerance: float = .05

    def __post_init__(self):
        if type(self.samples) is not int or not 32 <= self.samples <= 4194304:
            raise ValueError('Use 32..4194304 independent samples')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Use a nonnegative integer seed')
        for key in ('confidence', 'tolerance'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'Invalid {key}')
            object.__setattr__(self, key, float(value))
        if not .5 <= self.confidence < 1 or not 0 < self.tolerance <= .1:
            raise ValueError('Need .5 <= confidence < 1 and 0 < tolerance <= .1')


def _draw(model, values, count, rng):
    """Return independent (V, labels, L prior(V)/(C proposal(V))) draws."""
    variance = model.prior_scale / rng.gamma(model.prior_shape + len(values) / 2, size=count)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Unrepresentable inverse-gamma proposal')
    labels = (rng.random((count, len(values))) < model.jump_probability).astype(np.uint8)
    mean = np.full(count, model.initial_mean)
    covariance = np.full(count, model.initial_variance)
    log_weight = np.zeros(count)
    for index, value in enumerate(values):
        w = np.where(labels[:, index], model.large_variance, model.small_variance)
        predicted = covariance + w
        q = predicted + variance
        # Each contribution is <= 0 analytically. Avoid a large normalization
        # constant or cancellation between prior, proposal and likelihood logs.
        log_weight += .5 * np.log(variance / q) - .5 * (value - mean)**2 / q
        gain = predicted / q
        mean = (variance / q) * mean + gain * value
        covariance = variance * gain
    weight = np.exp(log_weight)
    if not np.isfinite(weight).all() or np.any(weight > 1):
        raise ValueError('Invalid bounded importance weights')
    return variance, labels, weight


def _probes(model, values, variance, labels, predictive, state, variance_points):
    condition = _condition(model, values, labels, variance)
    probes = dict(jump_probabilities=labels.astype(float))
    probes['predictive_cdf'] = np.column_stack([
        sum(probability * ndtr((point - condition['last_mean']) /
            np.sqrt(condition['last_variance'] + variance + w))
            for probability, w in ((1 - model.jump_probability, model.small_variance),
                                   (model.jump_probability, model.large_variance)))
        for point in predictive]) if predictive else np.empty((len(variance), 0))
    for name in ('initial', 'last'):
        for bound, inclusive in (('lower', False), ('upper', True)):
            probes[f'{name}_cdf_{bound}'] = np.column_stack([
                _state_probability(point, condition[f'{name}_mean'],
                    condition[f'{name}_variance'], inclusive) for point in state
            ]) if state else np.empty((len(variance), 0))
    probes['variance_cdf'] = np.column_stack([variance <= point for point in variance_points
        ]) if variance_points else np.empty((len(variance), 0))
    return probes


def _ratio_bounds(means, sample_variances, count, confidence):
    """Two-sided Theorem 4, unioned over numerator/denominator expectations."""
    log_term = math.log(4 * len(means) / (1 - confidence))
    radius = np.sqrt(2 * sample_variances * log_term / count) + 7 * log_term / (3 * (count - 1))
    lower, upper = np.maximum(0., means - radius), np.minimum(1., means + radius)
    denominator_lower, denominator_upper = lower[0], upper[0]
    probability_lower = lower[1:] / denominator_upper
    probability_upper = (np.minimum(1., upper[1:] / denominator_lower)
        if denominator_lower > 0 else np.ones(len(means) - 1))
    return probability_lower, probability_upper, (float(denominator_lower), float(denominator_upper)), radius


def bound_innovation_probabilities(values, *, model=None, settings=None,
        predictive_points=(-2., 0., 2., 4., 6.), state_points=(0.,), variance_points=(.5, 1., 2.)):
    """Bound all requested probabilities simultaneously in one fixed-budget call.

    Unresolved calls return their evidence, including [0, 1] intervals when the
    normalizer has no positive lower bound. Confidence covers neither adaptive
    repeated calls nor unrequested events, densities, moments or floating error.
    """
    model = InnovationMixtureReference() if model is None else model
    settings = ProbabilityBoundSettings() if settings is None else settings
    if not isinstance(model, InnovationMixtureReference) or not isinstance(settings, ProbabilityBoundSettings):
        raise ValueError('Use InnovationMixtureReference and ProbabilityBoundSettings records')
    if (not isinstance(values, (tuple, list)) or len(values) > MAX_PARTICLE_HISTORY or
            any(isinstance(y, bool) or not isinstance(y, Real) or not math.isfinite(y) for y in values)):
        raise ValueError(f'Use at most {MAX_PARTICLE_HISTORY} finite daily observations as a tuple/list')
    values = tuple(map(float, values))
    predictive, state, variance_points = (_points(predictive_points), _points(state_points),
                                         _points(variance_points, positive=True))
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 307]))
    totals = squares = visits = None
    template = None
    underflows = 0
    for start in range(0, settings.samples, 4096):
        variance, labels, weight = _draw(model, values, min(4096, settings.samples - start), rng)
        underflows += int(np.count_nonzero(weight == 0))
        probes = _probes(model, values, variance, labels, predictive, state, variance_points)
        template = {key: array.shape[1] for key, array in probes.items()}
        array = np.column_stack([weight, *(weight[:, None] * array for array in probes.values())])
        chunk_totals, chunk_squares = array.sum(axis=0), np.square(array).sum(axis=0)
        if len(values) <= 8:
            if model.jump_probability in (0, 1):
                indices, size = np.zeros(len(weight), dtype=int), 1
            else:
                indices, size = labels @ (2**np.arange(len(values) - 1, -1, -1)), 2**len(values)
            template['sequence_probabilities'] = size
            chunk_totals = np.r_[chunk_totals, np.bincount(indices, weights=weight, minlength=size)]
            chunk_squares = np.r_[chunk_squares, np.bincount(indices, weights=weight**2, minlength=size)]
            if visits is None:
                visits = np.zeros(size, dtype=np.int64)
            visits += np.bincount(indices, minlength=size)
        if totals is None:
            totals, squares = chunk_totals, chunk_squares
        else:
            totals += chunk_totals
            squares += chunk_squares
    means = totals / settings.samples
    sample_variances = np.maximum(0., squares - totals * means) / (settings.samples - 1)
    lower, upper, denominator, _ = _ratio_bounds(means, sample_variances, settings.samples, settings.confidence)
    estimates = means[1:] / means[0] if means[0] > 0 else None
    if not np.isfinite(np.r_[means, sample_variances, lower, upper]).all():
        raise ValueError('Nonfinite probability-bound calculation')
    intervals, offset = {}, 0
    for key, size in template.items():
        part = slice(offset, offset + size)
        intervals[key] = dict(lower=lower[part].tolist(), upper=upper[part].tolist(),
            estimate=None if estimates is None else estimates[part].tolist())
        offset += size
    maximum_width = float(np.max(upper - lower)) if len(lower) else 0.
    log_constant = (float(gammaln(model.prior_shape + len(values) / 2) - gammaln(model.prior_shape)
        - len(values) / 2 * (math.log(2 * math.pi) + math.log(model.prior_scale))) if values else 0.)
    if not math.isfinite(log_constant):
        raise ValueError('Unrepresentable log evidence envelope')
    checks = dict(positive_normalizer=denominator[0] > 0, probability_width=maximum_width <= 2 * settings.tolerance)
    return dict(status='completed' if all(checks.values()) else 'unresolved', model=model.configuration,
        settings=asdict(settings), values=list(values), predictive_points=list(predictive),
        state_points=list(state), variance_points=list(variance_points), intervals=intervals,
        maximum_probability_width=maximum_width, checks=checks,
        normalizer=dict(log_envelope=log_constant, scaled_estimate=float(means[0]),
            scaled_lower=denominator[0], scaled_upper=denominator[1]),
        sufficient_statistics=dict(template=template, means=means.tolist(), sample_variances=sample_variances.tolist()),
        zero_weight_draws=underflows, sequence_proposal_visits=None if visits is None else visits.tolist(),
        uncertainty_kind='Simultaneous finite-sample probability bounds under iid/exact-arithmetic assumptions; one fixed budget; no moment/density or floating-error bound',
        zero_variance_numerator_radius=float(7 * math.log(4 * len(means) / (1 - settings.confidence)) /
                                             (3 * (settings.samples - 1))))
