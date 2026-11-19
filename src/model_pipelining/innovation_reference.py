"""Bounded enumeration reference for independent small/large state innovations.

This is a short-history numerical reference, not a Forecaster or a long-history
approximation. Batch conditioning/adaptive integration and sequential smoothing/
log-variance quadrature independently integrate every positive-prior sequence.
"""

from dataclasses import asdict, dataclass
from itertools import product
import math
from numbers import Real

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln, logsumexp, ndtr, roots_legendre
from scipy.stats import invgamma


MAX_REFERENCE_HISTORY = 8
_LOG_2PI = math.log(2 * math.pi)


class ReferencePrecisionError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True, kw_only=True)
class InnovationMixtureReference:
    initial_mean: float = 0.
    initial_variance: float = 1.
    small_variance: float = .01
    large_variance: float = 4.
    jump_probability: float = .05
    prior_shape: float = 3.
    prior_scale: float = 2.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if (self.initial_variance < 0 or not 0 <= self.small_variance <= self.large_variance
                or not 0 <= self.jump_probability <= 1 or self.prior_shape <= 1 or self.prior_scale <= 0):
            raise ValueError("Need nonnegative ordered variances, probability in [0,1], shape > 1 and scale > 0")

    @property
    def configuration(self):
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ReferenceSettings:
    adaptive_relative_tolerance: float = 1e-9
    adaptive_absolute_tolerance: float = 1e-10
    adaptive_limit: int = 500
    minimum_nodes: int = 32
    maximum_nodes: int = 512
    maximum_expansions: int = 6
    refinement_tolerance: float = 1e-7
    tail_tolerance: float = 1e-9
    agreement_tolerance: float = 5e-6

    def __post_init__(self):
        for name in ("adaptive_limit", "minimum_nodes", "maximum_nodes", "maximum_expansions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 8 <= self.minimum_nodes <= self.maximum_nodes <= 2048:
            raise ValueError("Need 8 <= minimum_nodes <= maximum_nodes <= 2048")
        for name in ("adaptive_relative_tolerance", "adaptive_absolute_tolerance", "refinement_tolerance",
                     "tail_tolerance", "agreement_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1e-3:
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, float(value))


def _values(values):
    if not isinstance(values, (tuple, list)) or len(values) > MAX_REFERENCE_HISTORY:
        raise ValueError(f"Use at most {MAX_REFERENCE_HISTORY} daily observations as a tuple/list")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("Observations must be finite numbers")
    return tuple(map(float, values))


def _points(points, *, positive=False):
    if not isinstance(points, (tuple, list)) or len(points) > 64:
        raise ValueError("Query points must be a tuple/list of at most 64 values")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or positive and x <= 0 for x in points):
        raise ValueError("Query points must be finite (positive for variance)")
    if tuple(sorted(set(points))) != tuple(points):
        raise ValueError("Query points must be strictly increasing")
    return tuple(map(float, points))


def enumerate_sequences(model, observation_count):
    """All positive-prior past labels, including the first pre-observation step."""
    if type(observation_count) is not int or not 0 <= observation_count <= MAX_REFERENCE_HISTORY:
        raise ValueError(f"Observation count must be in [0,{MAX_REFERENCE_HISTORY}]")
    p = model.jump_probability
    if p in (0, 1):
        labels = ((int(p),) * observation_count,)
        logs = np.zeros(1)
    else:
        labels = tuple(product((0, 1), repeat=observation_count))
        logs = np.array([sum(label) * math.log(p) + (observation_count - sum(label)) * math.log1p(-p)
                         for label in labels])
        if np.any(np.exp(logs) == 0):
            raise ValueError("A positive sequence prior is unrepresentable")
    return np.array(labels, dtype=int).reshape(len(labels), observation_count), logs


class _BatchGaussian:
    """Dense covariance conditioning; no Kalman recursion or quadrature grid."""
    def __init__(self, model, values, labels):
        self.model, self.values, self.labels = model, values, labels
        self.n = len(values)
        self.w = np.where(labels, model.large_variance, model.small_variance)
        self.d = np.column_stack((np.full(len(labels), model.initial_variance), self.w))
        self.a = np.column_stack((np.ones(self.n), np.tril(np.ones((self.n, self.n)))))
        self.b = (self.a[None, :, :] * self.d[:, None, :]) @ self.a.T

    def condition(self, variance):
        model, n, m = self.model, self.n, len(self.labels)
        if n == 0:
            return _empty_condition(model, (m, 1))
        covariance = self.b + variance * np.eye(n)
        factor = np.linalg.cholesky(covariance)
        rhs = np.empty((m, n, 3))
        rhs[:, :, 0] = np.asarray(self.values) - model.initial_mean
        rhs[:, :, 1] = model.initial_variance
        rhs[:, :, 2] = self.b[:, :, -1]
        white = np.linalg.solve(factor, rhs)
        solved = np.linalg.solve(np.swapaxes(factor, 1, 2), white)
        initial_mean = model.initial_mean + model.initial_variance * solved[:, :, 0].sum(axis=1)
        last_mean = model.initial_mean + np.sum(self.b[:, :, -1] * solved[:, :, 0], axis=1)
        # Joseph/error-loading form avoids subtracting almost equal variances.
        initial_loading = np.zeros((m, n + 1))
        initial_loading[:, 0] = 1
        initial_loading -= solved[:, :, 1] @ self.a
        last_loading = 1 - solved[:, :, 2] @ self.a
        initial_variance = np.sum(self.d * initial_loading**2, axis=1) + variance * np.sum(solved[:, :, 1]**2, axis=1)
        last_variance = np.sum(self.d * last_loading**2, axis=1) + variance * np.sum(solved[:, :, 2]**2, axis=1)
        cross = np.sum(self.d * initial_loading * last_loading, axis=1) + variance * np.sum(solved[:, :, 1] * solved[:, :, 2], axis=1)
        likelihood = -.5 * (n * _LOG_2PI + 2 * np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(axis=1)
                            + np.sum(white[:, :, 0]**2, axis=1))
        return {key: value[:, None] for key, value in dict(log_likelihood=likelihood,
            initial_mean=initial_mean, last_mean=last_mean, initial_variance=initial_variance,
            last_variance=last_variance, state_covariance=cross).items()}


def _empty_condition(model, shape):
    return {key: np.full(shape, value) for key, value in dict(log_likelihood=0.,
        initial_mean=model.initial_mean, last_mean=model.initial_mean,
        initial_variance=model.initial_variance, last_variance=model.initial_variance,
        state_covariance=model.initial_variance).items()}


def _sequential_condition(model, values, labels, variances):
    """Independent scalar Kalman filters plus backward initial-state smoothing."""
    shape = (len(labels), len(variances))
    if not values:
        return _empty_condition(model, shape)
    w = np.where(labels, model.large_variance, model.small_variance)
    mean = np.full(shape, model.initial_mean, dtype=float)
    covariance = np.full(shape, model.initial_variance, dtype=float)
    likelihood = np.zeros(shape)
    means, covariances = [], []
    for index, value in enumerate(values):
        predicted = covariance + w[:, index, None]
        q = predicted + variances
        innovation = value - mean
        likelihood -= .5 * (_LOG_2PI + np.log(q) + innovation**2 / q)
        gain = predicted / q
        mean = (variances / q) * mean + gain * value
        covariance = variances * gain
        means.append(mean)
        covariances.append(covariance)
    last_mean, last_variance = mean, covariance
    cross = covariance.copy()
    for index in range(len(values) - 2, -1, -1):
        predicted = covariances[index] + w[:, index + 1, None]
        gain = np.divide(covariances[index], predicted, out=np.zeros(shape), where=predicted > 0)
        conditional = covariances[index] * np.divide(w[:, index + 1, None], predicted,
                                                    out=np.zeros(shape), where=predicted > 0)
        mean = means[index] + gain * (mean - means[index])
        covariance = conditional + gain**2 * covariance
        cross = gain * cross
    predicted = model.initial_variance + w[:, 0, None]
    gain = np.divide(model.initial_variance, predicted, out=np.zeros_like(predicted), where=predicted > 0)
    initial_mean = model.initial_mean + gain * (mean - model.initial_mean)
    initial_variance = model.initial_variance * np.divide(w[:, 0, None], predicted,
        out=np.zeros_like(predicted), where=predicted > 0) + gain**2 * covariance
    return dict(log_likelihood=likelihood, initial_mean=initial_mean, last_mean=last_mean,
                initial_variance=initial_variance, last_variance=last_variance, state_covariance=gain * cross)


def _state_cdf(point, mean, variance, inclusive):
    positive = variance > 0
    z = np.divide(point - mean, np.sqrt(variance), out=np.zeros_like(mean), where=positive)
    return np.where(positive, ndtr(z), mean <= point if inclusive else mean < point)


def _scale(model, values):
    scale = max(1., math.sqrt(model.initial_variance + len(values) * model.large_variance),
                math.sqrt(model.prior_scale / (model.prior_shape - 1)),
                max((abs(y - model.initial_mean) for y in values), default=0.))
    if not math.isfinite(scale * scale):
        raise ValueError("Reference moment scale is unrepresentable")
    return scale


def _pack(model, condition, variances, weights, scale, predictive_points, state_points):
    if any(not np.isfinite(value).all() for value in condition.values()):
        raise ValueError("Nonfinite conditional Gaussian calculation")
    if np.any(condition['initial_variance'] < 0) or np.any(condition['last_variance'] < 0):
        raise ValueError("Negative conditional state variance")
    initial = (condition['initial_mean'] - model.initial_mean) / scale
    last = (condition['last_mean'] - model.initial_mean) / scale
    values = [*weights.sum(axis=1), *(weights * variances / scale**2).sum(axis=1),
        np.sum(weights * initial), np.sum(weights * last),
        np.sum(weights * (condition['initial_variance'] / scale**2 + initial**2)),
        np.sum(weights * (condition['last_variance'] / scale**2 + last**2)),
        np.sum(weights * (condition['state_covariance'] / scale**2 + initial * last))]
    for point in predictive_points:
        cdf = density = 0.
        for probability, w in ((1 - model.jump_probability, model.small_variance),
                               (model.jump_probability, model.large_variance)):
            if probability == 0:
                continue
            sd = np.sqrt(condition['last_variance'] + variances + w)
            z = (point - condition['last_mean']) / sd
            cdf += probability * float(np.sum(weights * ndtr(z)))
            density += probability * float(np.sum(weights * np.exp(-.5 * z**2) / sd)) / math.sqrt(2 * math.pi)
        values.extend((cdf, density * scale))
    for point in state_points:
        for which in ('initial', 'last'):
            for inclusive in (False, True):
                values.append(np.sum(weights * _state_cdf(point, condition[f'{which}_mean'],
                                                         condition[f'{which}_variance'], inclusive)))
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Unrepresentable reference integrand")
    return result


def _decode(vector, log_scale, model, labels, scale, predictive_points, state_points):
    count = len(labels)
    normalizer = float(vector[:count].sum())
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Unrepresentable reference evidence")
    probability = vector[:count] / normalizer
    if np.any(probability <= 0):
        raise ValueError("A positive posterior sequence mass is unrepresentable; no paths were pruned")
    v_by_path = vector[count:2 * count] * scale**2 / normalizer
    first, last, second_first, second_last, cross = vector[2 * count:2 * count + 5] / normalizer
    initial_variance = float((second_first - first**2) * scale**2)
    last_variance = float((second_last - last**2) * scale**2)
    if initial_variance < 0 or last_variance < 0 or model.initial_variance > 0 and initial_variance <= 0:
        raise ValueError("Unrepresentable marginal state variance")
    if last_variance == 0 and (model.initial_variance > 0 or
            len(labels[0]) and ((model.small_variance > 0 and model.jump_probability < 1)
                              or (model.large_variance > 0 and model.jump_probability > 0))):
        raise ValueError("Unknown final state variance is unrepresentable")
    variance_mean = float(v_by_path.sum())
    jump_counts = labels.sum(axis=1)
    result = dict(log_evidence=log_scale + math.log(normalizer),
        sequence_probabilities=probability.tolist(), jump_probabilities=(probability @ labels).tolist(),
        observation_variance_mean=variance_mean,
        initial_mean=float(model.initial_mean + first * scale), initial_variance=initial_variance,
        last_mean=float(model.initial_mean + last * scale), last_variance=last_variance,
        initial_last_covariance=float((cross - first * last) * scale**2),
        variance_jump_count_covariance=float(v_by_path @ jump_counts - variance_mean * (probability @ jump_counts)),
        predictive_mean=float(model.initial_mean + last * scale),
        predictive_variance=last_variance + variance_mean + (1 - model.jump_probability) * model.small_variance
                            + model.jump_probability * model.large_variance,
        next_jump_probability=model.jump_probability,
        predictive_points=list(predictive_points), predictive_cdf=[], predictive_density=[],
        state_points=list(state_points), initial_cdf_lower=[], initial_cdf_upper=[], last_cdf_lower=[], last_cdf_upper=[])
    position = 2 * count + 5
    for _ in predictive_points:
        result['predictive_cdf'].append(float(vector[position] / normalizer))
        result['predictive_density'].append(float(vector[position + 1] / normalizer / scale))
        position += 2
    for _ in state_points:
        for which in ('initial', 'last'):
            for bound in ('lower', 'upper'):
                result[f'{which}_cdf_{bound}'].append(float(vector[position] / normalizer))
                position += 1
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in result.values()):
        raise ValueError("Nonfinite marginal reference summary")
    if result['observation_variance_mean'] <= 0 or result['predictive_variance'] <= 0:
        raise ValueError("Unrepresentable observation/predictive variance")
    return result


def _comparison_vector(result, scale):
    scalar = [result[name] / scale for name in ('initial_mean', 'last_mean', 'predictive_mean')]
    scalar += [result[name] / scale**2 for name in ('initial_variance', 'last_variance', 'initial_last_covariance',
                'predictive_variance', 'observation_variance_mean', 'variance_jump_count_covariance')]
    arrays = [result[name] for name in ('sequence_probabilities', 'jump_probabilities', 'predictive_cdf',
              'initial_cdf_lower', 'initial_cdf_upper', 'last_cdf_lower', 'last_cdf_upper', 'variance_cdf')]
    return np.array([*scalar, *(x for array in arrays for x in array),
                     *(x * scale for x in result['predictive_density'])])


def _difference(first, second, scale):
    return max(abs(first['log_evidence'] - second['log_evidence']),
               float(np.max(np.abs(_comparison_vector(first, scale) - _comparison_vector(second, scale)))))


def _batch_adaptive(model, values, labels, log_priors, settings, predictive_points, state_points, variance_points):
    kernel = _BatchGaussian(model, values, labels)
    scale = _scale(model, values)
    v_scale = model.prior_scale / model.prior_shape
    a, b = model.prior_shape, model.prior_scale
    log_constant = float(gammaln(a + len(values) / 2) - gammaln(a)) - len(values) / 2 * (_LOG_2PI + math.log(b))
    dimension = 2 * len(labels) + 5 + 2 * len(predictive_points) + 4 * len(state_points)

    def integrand(t, log_scale, full=True):
        if not 0 < t < 1:
            return np.zeros(dimension if full else 1)
        variance = v_scale * t / (1 - t)
        conditional = kernel.condition(variance)
        log_prior = a * math.log(b) - float(gammaln(a)) - (a + 1) * math.log(variance) - b / variance
        logs = (log_priors[:, None] + conditional['log_likelihood'] + log_prior
                + math.log(v_scale) - 2 * math.log1p(-t) - log_scale)
        with np.errstate(over='raise', under='ignore', invalid='raise'):
            weights = np.exp(logs)
        return (_pack(model, conditional, variance, weights, scale, predictive_points, state_points)
                if full else np.array([float(weights.sum())]))

    def integrate(high, log_scale, full=True):
        vector, error, info = quad_vec(lambda t: integrand(t, log_scale, full), 0., high,
            epsabs=settings.adaptive_absolute_tolerance, epsrel=settings.adaptive_relative_tolerance,
            norm='max', limit=settings.adaptive_limit, full_output=True,
            points=[p for p in (.01, .1, .5, .9, .99) if p < high])
        if not info.success or not np.isfinite(vector).all() or not math.isfinite(error):
            raise ReferencePrecisionError('Adaptive batch integration did not converge',
                {'method': 'batch_adaptive', 'error_estimate': float(error), 'message': str(info.message)})
        return vector, float(error), int(info.neval)

    pilot, pilot_error, pilot_evaluations = integrate(1., log_constant)
    mass = float(pilot[:len(labels)].sum())
    if mass <= 0 or pilot_error > .1 * mass:
        raise ReferencePrecisionError('Adaptive evidence scaling unresolved', {'pilot_mass': mass, 'error': pilot_error})
    log_scale = log_constant + math.log(mass)
    vector, error, evaluations = integrate(1., log_scale)
    result = _decode(vector, log_scale, model, labels, scale, predictive_points, state_points)
    normalizer = float(vector[:len(labels)].sum())
    result['variance_points'], result['variance_cdf'] = list(variance_points), []
    errors = [error / normalizer]
    for value in variance_points:
        partial, partial_error, count = integrate(value / (v_scale + value), log_scale, False)
        result['variance_cdf'].append(float(partial[0] / normalizer))
        errors.append(partial_error / normalizer)
        evaluations += count
    estimated_error = max(errors)
    if estimated_error > settings.agreement_tolerance / 10:
        raise ReferencePrecisionError('Adaptive error estimate exceeds reference budget', {'estimated_error': estimated_error})
    result['numerics'] = {'method': 'batch Gaussian / adaptive rational-V integration over (0,infinity)',
        'estimated_scaled_error': estimated_error, 'evaluations': evaluations + pilot_evaluations,
        'support': 'complete positive variance axis; V=(b/a)*t/(1-t)', 'pilot_log_scale': log_scale}
    return result


def _sequential_grid(model, values, labels, log_priors, bounds, count, scale, predictive_points, state_points):
    nodes, weights = roots_legendre(count)
    half = (bounds[1] - bounds[0]) / 2
    log_v = (bounds[1] + bounds[0]) / 2 + half * nodes
    variances = np.exp(log_v)
    conditional = _sequential_condition(model, values, labels, variances)
    a, b = model.prior_shape, model.prior_scale
    logs = (log_priors[:, None] + conditional['log_likelihood'] + a * math.log(b) - float(gammaln(a))
            - a * log_v - b / variances + np.log(weights * half))
    log_evidence = float(logsumexp(logs))
    normalized = np.exp(logs - log_evidence)
    vector = _pack(model, conditional, variances, normalized, scale, predictive_points, state_points)
    result = _decode(vector, log_evidence, model, labels, scale, predictive_points, state_points)
    return result


def _tail_bounds(model, n, bounds, log_evidence):
    a, b = model.prior_shape, model.prior_scale
    shape = a + n / 2
    log_constant = float(gammaln(shape) - gammaln(a)) - n / 2 * (_LOG_2PI + math.log(b))
    lo, hi = map(math.exp, bounds)
    def bound(power):
        tail = float(np.logaddexp(invgamma.logcdf(lo, shape - power, scale=b),
                                 invgamma.logsf(hi, shape - power, scale=b)))
        log_value = log_constant + tail - log_evidence + (math.log(b / (shape - 1)) if power else 0.)
        return math.exp(log_value) if log_value < 700 else math.inf
    return bound(0), bound(1)


def _sequential_quadrature(model, values, labels, log_priors, settings, predictive_points, state_points, variance_points):
    scale = _scale(model, values)
    center = math.log(model.prior_scale) - math.log(model.prior_shape + 1)
    if not -650 < center < 650:
        raise ValueError('Variance prior scale is outside reference range')
    history, result = [], None
    for expansion in range(settings.maximum_expansions):
        width = 4 * 1.5**expansion
        bounds = (max(-700., center - width), min(700., center + width))
        previous, streak, count = None, 0, settings.minimum_nodes
        while count <= settings.maximum_nodes:
            result = _sequential_grid(model, values, labels, log_priors, bounds, count, scale, predictive_points, state_points)
            result['variance_points'], result['variance_cdf'] = list(variance_points), []
            for value in variance_points:
                boundary = math.log(value)
                if boundary <= bounds[0]:
                    cdf = 0.
                elif boundary >= bounds[1]:
                    cdf = 1.
                else:
                    partial = _sequential_grid(model, values, labels, log_priors, (bounds[0], boundary), count,
                                               scale, (), ())
                    cdf = math.exp(partial['log_evidence'] - result['log_evidence'])
                result['variance_cdf'].append(cdf)
            change = _difference(previous, result, scale) if previous is not None else None
            streak = streak + 1 if change is not None and change <= settings.refinement_tolerance else 0
            probability_tail, moment_tail = _tail_bounds(model, len(values), bounds, result['log_evidence'])
            history.append({'expansion': expansion, 'nodes_per_sequence': count, 'scaled_change': change,
                'tail_probability_bound': probability_tail, 'variance_moment_tail_bound': moment_tail})
            if (streak >= 2 and probability_tail <= settings.tail_tolerance
                    and moment_tail <= settings.tail_tolerance * max(1., result['observation_variance_mean'])):
                result['numerics'] = {'method': 'Kalman filtering/smoothing / log-V Gauss-Legendre',
                    'nodes_per_sequence': count, 'joint_nodes': count * len(labels), 'log_bounds': list(bounds),
                    'tail_probability_bound': probability_tail, 'variance_moment_tail_bound': moment_tail,
                    'refinement_history': history, 'discarded_sequences': 0}
                return result
            previous, count = result, count * 2
    raise ReferencePrecisionError('Sequential quadrature did not establish precision and tails',
                                  {'refinement_history': history, 'candidate': result})


def run_innovation_reference(values, *, model=None, settings=None, predictive_points=(-2., 0., 2., 4., 6.),
                             state_points=(0.,), variance_points=(.5, 1., 2.)):
    """Compare all-sequence batch/adaptive and sequential/grid calculations.

    Inputs are immediately available daily observations only, with one future
    daily transition. Eight observations maximum, including degenerate models.
    Invalid inputs raise; unresolved precision raises ReferencePrecisionError
    retaining diagnostics. Success verifies this finite exercise, not adequacy.
    """
    model = InnovationMixtureReference() if model is None else model
    settings = ReferenceSettings() if settings is None else settings
    if not isinstance(model, InnovationMixtureReference) or not isinstance(settings, ReferenceSettings):
        raise ValueError('Use reference model/settings records')
    values = _values(values)
    predictive_points, state_points = _points(predictive_points), _points(state_points)
    variance_points = _points(variance_points, positive=True)
    labels, log_priors = enumerate_sequences(model, len(values))
    batch = _batch_adaptive(model, values, labels, log_priors, settings, predictive_points, state_points, variance_points)
    sequential = _sequential_quadrature(model, values, labels, log_priors, settings, predictive_points, state_points, variance_points)
    error = _difference(batch, sequential, _scale(model, values))
    result = {'format_version': 1, 'kind': 'innovation_mixture_reference', 'status': 'completed',
        'model': model.configuration, 'settings': asdict(settings), 'values': list(values),
        'sequences': labels.tolist(), 'sequence_prior_probabilities': np.exp(log_priors).tolist(),
        'batch_adaptive': batch, 'sequential_quadrature': sequential, 'maximum_scaled_difference': error,
        'checks': [{'name': 'independent_enumeration_reference_agreement', 'required': True,
                    'status': 'passed' if error <= settings.agreement_tolerance else 'failed'}],
        'scope': 'Finite all-sequence reference with numerical V integration; no adapter, pruning, learned transition or long-history inference'}
    if error > settings.agreement_tolerance:
        raise ReferencePrecisionError('Independent references disagree', result)
    return result
