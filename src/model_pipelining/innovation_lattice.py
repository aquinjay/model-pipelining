"""Bound innovation probabilities with an observation-conditioned lattice proposal."""

from dataclasses import asdict
import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from model_pipelining.innovation_particles import (
    MAX_PARTICLE_HISTORY, _condition, _decode, _flatten, _raw, _unflatten,
)
from model_pipelining.innovation_probability_bounds import (
    ProbabilityBoundSettings, _probes, _ratio_bounds, bound_innovation_probabilities,
)
from model_pipelining.innovation_reference import InnovationMixtureReference, _points, _scale


_LATTICE_ERROR = 1e-4
_TRUNCATION_ERROR = 1e-4
_LOG_PRECISION_STEP = .035


def _precision_limits(model, values, shape, log_constant):
    """Choose a range using a single plausible path as a scale hint.

    The hint affects cost and precision only. Each omitted tail is bounded by
    the Gamma Chernoff inequality, regardless of the quality of the hint.
    """
    centered = np.subtract(values, model.initial_mean)
    prior_variance = model.prior_scale / (model.prior_shape - 1)
    rough_variance = np.mean(np.diff(centered, prepend=0.)**2) / 2
    candidates = prior_variance * np.geomspace(.125, 8., 9)
    if rough_variance > 0:
        candidates = np.r_[candidates, rough_variance * np.geomspace(.125, 8., 9)]
    mean = np.zeros(len(candidates))
    covariance = np.full(len(candidates), model.initial_variance)
    jump_prior = np.array([1 - model.jump_probability, model.jump_probability])
    log_prior = np.log(jump_prior, where=jump_prior > 0, out=np.full(2, -np.inf))
    increments = np.array([model.small_variance, model.large_variance])
    # Evaluate the joint density in log V units at each candidate.
    log_joint = (model.prior_shape * np.log(model.prior_scale / candidates)
                 - gammaln(model.prior_shape) - model.prior_scale / candidates)
    indices = np.arange(len(candidates))
    for y in centered:
        predicted = covariance[:, None] + increments
        q = predicted + candidates[:, None]
        scores = log_prior - .5 * (np.log(2 * np.pi * q) + (y - mean[:, None])**2 / q)
        best = np.argmax(scores, axis=1)
        log_joint += scores[indices, best]
        gain = predicted[indices, best] / q[indices, best]
        mean += gain * (y - mean)
        covariance = candidates * gain
    log_tail = min(-25., float(log_joint.max()) - log_constant - 14.)
    if not math.isfinite(log_tail):
        raise ValueError('Unrepresentable precision tail budget')

    def objective(log_ratio):
        return -shape * (math.expm1(log_ratio) - log_ratio) - log_tail

    low = brentq(objective, -700., 0.)
    high = brentq(objective, 0., 700.)
    # Move both limits outwards, then report the actual combined tail bound.
    limits = shape / model.prior_scale * np.exp([low - 1e-10, high + 1e-10])
    return limits, math.log(2.) + log_tail + log_constant


def _grid_geometry(model, values, precision_limits):
    """Return a finite state lattice and pathwise bounds D(s,path) / L(s,path)."""
    n = len(values)
    dimension = n + 1
    variances = [w for p, w in ((1 - model.jump_probability, model.small_variance),
                               (model.jump_probability, model.large_variance)) if p > 0]
    positive = [w for w in variances if w > 0]
    # The precision adds edge Laplacians, the initial prior, and observation
    # precision. A run of zero-variance edges collapses to a single coordinate.
    precision_bound = precision_limits[1] * (n if 0 in variances else 1)
    if positive:
        precision_bound += 4 / min(positive)
    if model.initial_variance > 0:
        precision_bound += 1 / model.initial_variance
    a = math.log(8 * dimension / _LATTICE_ERROR)
    step = min(.15, math.sqrt(2 * math.pi**2 / (precision_bound * a)))
    exponent = math.exp(-2 * math.pi**2 / (step**2 * precision_bound))
    lattice_error = math.expm1(dimension * math.log1p(2 * exponent / (1 - exponent)))

    # Conditional means lie in the hull of m0 and y. Each observed free state
    # has conditional variance <= V, and an unobserved X0 has variance <= C0.
    max_variance = max(model.initial_variance, 1 / precision_limits[0])
    margin = math.sqrt(2 * max_variance * math.log(2 * dimension * (1 + lattice_error) / _TRUNCATION_ERROR))
    lower = math.floor((min(0., *values) - margin) / step)
    upper = math.ceil((max(0., *values) + margin) / step)
    if upper - lower + 1 > 12000:
        raise ValueError('State lattice exceeds the 12000-point research budget')
    return np.arange(lower, upper + 1) * step, -lower, step, (
        1 - lattice_error - _TRUNCATION_ERROR, 1 + lattice_error)


class _StateLattice:
    """Finite-grid Gaussian mixture filter with independent backward draws."""

    def __init__(self, model, values, precision_limits):
        grid, origin, step, self.mass_ratio = _grid_geometry(model, values, precision_limits)
        self.values = grid
        self.squared_residuals = np.square(np.subtract.outer(values, grid))
        size = len(grid)
        offsets = np.arange(1 - size, size) * step

        def kernel(variance):
            if variance == 0:
                return (offsets == 0).astype(float)
            return (step / math.sqrt(2 * math.pi * variance)) * np.exp(-.5 * offsets**2 / variance)

        self.initial = kernel(model.initial_variance)[size - 1 - origin:2 * size - 1 - origin]
        self.large = model.jump_probability * kernel(model.large_variance)
        self.kernel = (1 - model.jump_probability) * kernel(model.small_variance) + self.large
        self.state_indices = np.arange(size)
        self.identity = not (self.kernel[:size - 1].any() or self.kernel[size:].any())

    def filter(self, precision, *, store=False):
        prior_mass = float(self.initial.sum())
        filtered = self.initial / prior_mass
        log_mass = math.log(prior_mass)
        history = [filtered] if store else None
        for residual in self.squared_residuals:
            if not self.identity:
                # Work with nonnegative products. FFT roundoff can invent
                # substantial relative mass when an observed path is unlikely.
                filtered = np.convolve(filtered, self.kernel, mode='valid')
            with np.errstate(divide='ignore'):
                log_updated = np.log(filtered) - .5 * precision * residual
            offset = float(log_updated.max())
            if not math.isfinite(offset):
                return -math.inf, None
            unnormalized = np.exp(log_updated - offset)
            mass = float(unnormalized.sum())
            if mass <= 0 or not math.isfinite(mass):
                # Remote precision knots can have total floating mass zero.
                # A zero is represented in the saved range of knot masses.
                return -math.inf, None
            filtered = unnormalized / mass
            log_mass += offset + math.log(mass)
            if store:
                history.append(filtered)
        return log_mass, history

    def draw(self, precision, count, rng):
        _, history = self.filter(precision, store=True)
        if history is None:
            raise ValueError('Selected lattice knot has no representable state mass')
        current = rng.choice(len(self.values), p=history[-1], size=count)
        residual = np.zeros(count)
        labels = np.empty((count, len(history) - 1), dtype=np.uint8)
        center = len(self.values) - 1
        for index in range(len(history) - 2, -1, -1):
            residual += self.squared_residuals[index, current]
            if self.identity:
                previous = current
            else:
                # Reuse one backward CDF for all draws at the same next state.
                order = np.argsort(current, kind='stable')
                starts = np.r_[0, np.flatnonzero(np.diff(current[order])) + 1, count]
                previous = np.empty_like(current)
                uniforms = rng.random(count)
                for left, right in zip(starts[:-1], starts[1:], strict=True):
                    group = order[left:right]
                    offsets = current[group[0]] - self.state_indices + center
                    cdf = np.cumsum(history[index] * self.kernel[offsets])
                    if cdf[-1] <= 0:
                        raise ValueError('Selected lattice state has no backward mass')
                    previous[group] = np.searchsorted(cdf, uniforms[group] * cdf[-1], side='right')
            transition = current - previous + center
            labels[:, index] = rng.random(count) < self.large[transition] / self.kernel[transition]
            current = previous
        return residual, labels


def _probability_intervals(means, sample_variances, settings, lattice_ratio, scaled_tail):
    """Expand bounds for grid error and the analytically bounded precision tails."""
    _, _, (den_lower, den_upper), radius = _ratio_bounds(
        means, sample_variances, settings.samples, settings.confidence)
    lattice_low, lattice_high = lattice_ratio
    normalizer_lower = den_lower / lattice_high
    normalizer_upper = den_upper / lattice_low + scaled_tail
    numerator_lower = np.maximum(0., means[1:] - radius[1:]) / lattice_high
    numerator_upper = np.minimum(1., means[1:] + radius[1:]) / lattice_low + scaled_tail
    lower = numerator_lower / normalizer_upper
    upper = (np.minimum(1., numerator_upper / (normalizer_lower + scaled_tail))
             if normalizer_lower > 0 else np.ones(len(lower)))
    return lower, upper, (normalizer_lower, normalizer_upper), float(
        7 * math.log(4 * len(means) / (1 - settings.confidence)) / (3 * (settings.samples - 1)))


def bound_innovation_probabilities_lattice(values, *, model=None, settings=None,
        predictive_points=(-2., 0., 2., 4., 6.), state_points=(0.,), variance_points=(.5, 1., 2.)):
    """Bound posterior probabilities with iid draws from a finite-lattice envelope.

    Results include lattice and tail allowances, finite-sample intervals, and
    descriptive moment/density estimates. The mathematical envelope assumes exact
    arithmetic. Unsupported geometry or arithmetic raises ValueError.
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
    if not values:
        result = bound_innovation_probabilities(values, model=model, settings=settings,
            predictive_points=predictive, state_points=state, variance_points=variance_points)
        return dict(result, proposal={'kind': 'prior'})
    centered = tuple(y - model.initial_mean for y in values)
    shape = model.prior_shape + len(values) / 2
    rate = model.prior_scale
    log_constant = float(gammaln(shape) - gammaln(model.prior_shape)
                         - len(values) / 2 * (math.log(2 * math.pi) + math.log(rate)))
    limits, log_tail = _precision_limits(model, values, shape, log_constant)
    if not np.isfinite(limits).all() or limits[0] <= 0 or limits[1] <= limits[0]:
        raise ValueError('Unrepresentable precision limits')
    lattice = _StateLattice(model, centered, limits)
    cells = math.ceil(math.log(limits[1] / limits[0]) / _LOG_PRECISION_STEP)
    if cells > 1000:
        raise ValueError('Precision lattice exceeds the 1000-cell research budget')
    if not lattice.identity and (cells + 1) * len(values) * len(lattice.values)**2 > 80_000_000_000:
        raise ValueError('Direct grid convolution exceeds the research work budget')
    knots = np.geomspace(*limits, cells + 1)

    def log_gamma(s):
        return shape * math.log(rate) - gammaln(shape) + (shape - 1) * np.log(s) - rate * s

    log_caps = log_gamma(np.clip((shape - 1) / rate, knots[:-1], knots[1:]))
    log_cell_mass = log_caps + np.log(np.diff(knots) / 2)
    knot_mass = np.array([lattice.filter(precision)[0] for precision in knots])
    component_logs = (log_cell_mass[:, None] +
                      np.column_stack((knot_mass[:-1], knot_mass[1:])))
    log_proposal_mass = float(logsumexp(component_logs))
    if not math.isfinite(log_proposal_mass) or not math.isfinite(log_tail):
        raise ValueError('Unrepresentable lattice envelope mass')
    log_envelope = log_constant + log_proposal_mass
    log_scaled_tail = log_tail - log_envelope
    if log_scaled_tail > 700.:
        raise ValueError('Variance tail allowance is too large relative to the proposal mass')
    scaled_tail = math.exp(log_scaled_tail)
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 521]))
    probabilities = np.exp(component_logs.ravel() - log_proposal_mass)
    components = rng.multinomial(settings.samples, probabilities / probabilities.sum()).reshape(-1, 2)

    sums = squares = visits = raw_total = raw_template = None
    zero_weights = 0
    scale = _scale(model, values)
    template = {}
    for knot_index, precision in enumerate(knots):
        left_count = components[knot_index - 1, 1] if knot_index else 0
        right_count = components[knot_index, 0] if knot_index < cells else 0
        for cell, count, from_left in ((knot_index - 1, left_count, False),
                                       (knot_index, right_count, True)):
            for start in range(0, count, 4096):
                count_chunk = min(4096, count - start)
                residual, labels = lattice.draw(precision, count_chunk, rng)
                triangle = np.sqrt(rng.random(count_chunk))
                t = 1 - triangle if from_left else triangle
                left, right = knots[cell:cell + 2]
                s = left + t * (right - left)
                delta = .5 * (right - left) * residual
                # Relative to exp(-left*RSS/2), the chord is (1-t)+t*exp(-delta).
                with np.errstate(divide='ignore'):
                    log_chord = np.logaddexp(np.log1p(-t), np.log(t) - delta)
                log_weights = log_gamma(s) - log_caps[cell] - t * delta - log_chord
                if not np.isfinite(log_weights).all() or np.any(log_weights > 1e-10):
                    raise ValueError('Invalid interpolation envelope weights')
                weights = np.exp(np.minimum(0., log_weights))
                zero_weights += int(np.count_nonzero(weights == 0))
                variance = 1 / s
                probabilities = _probes(model, values, variance, labels, predictive, state, variance_points)
                template = {key: array.shape[1] for key, array in probabilities.items()}
                numerators = np.column_stack([weights, *(weights[:, None] * array for array in probabilities.values())])
                subtotal = numerators.sum(axis=0)
                second = np.square(numerators).sum(axis=0)
                if len(values) <= 8:
                    deterministic = model.jump_probability in (0., 1.)
                    indices = (np.zeros(count_chunk, dtype=int) if deterministic
                               else labels @ (2**np.arange(len(values) - 1, -1, -1)))
                    size = 1 if deterministic else 2**len(values)
                    template['sequence_probabilities'] = size
                    subtotal = np.r_[subtotal, np.bincount(indices, weights=weights, minlength=size)]
                    second = np.r_[second, np.bincount(indices, weights=weights**2, minlength=size)]
                    observed = np.bincount(indices, minlength=size)
                    visits = observed if visits is None else visits + observed
                sums = subtotal if sums is None else sums + subtotal
                squares = second if squares is None else squares + second
                if weights.sum() > 0:
                    raw_template = _raw(model, values, variance, weights / weights.sum(), labels,
                        _condition(model, values, labels, variance), scale, predictive, state, variance_points)
                    raw = _flatten(raw_template) * weights.sum()
                    raw_total = raw if raw_total is None else raw_total + raw
    means = sums / settings.samples
    sample_variances = np.maximum(0., squares - sums * means) / (settings.samples - 1)
    lower, upper, denominator, unseen_radius = _probability_intervals(
        means, sample_variances, settings, lattice.mass_ratio, scaled_tail)
    posterior = (_decode(_unflatten(raw_total / sums[0], raw_template), model, scale)
                 if raw_total is not None else None)
    if posterior is not None:
        posterior['log_evidence'] = float(log_envelope + math.log(means[0]))
    if not np.isfinite(np.r_[means, sample_variances, lower, upper, denominator,
                              [] if posterior is None else _flatten(posterior)]).all():
        raise ValueError('Nonfinite lattice probability calculation')
    intervals = {}
    offset = 0
    for key, size in template.items():
        part = slice(offset, offset + size)
        intervals[key] = dict(lower=lower[part].tolist(), upper=upper[part].tolist(),
            estimate=(means[1:][part] / means[0]).tolist() if means[0] > 0 else None)
        offset += size
    maximum_width = float(np.max(upper - lower)) if len(lower) else 0.
    checks = dict(positive_normalizer=denominator[0] > 0, probability_width=maximum_width <= 2 * settings.tolerance)
    return dict(status='completed' if all(checks.values()) else 'unresolved', model=model.configuration,
        settings=asdict(settings), values=list(values), predictive_points=list(predictive),
        state_points=list(state), variance_points=list(variance_points), intervals=intervals,
        posterior_estimates=posterior, maximum_probability_width=maximum_width, checks=checks,
        normalizer=dict(log_envelope=log_envelope, scaled_estimate=float(means[0]),
            scaled_lower=denominator[0], scaled_upper=denominator[1]),
        proposal=dict(kind='observation_conditioned_state_lattice', precision_limits=limits.tolist(),
            precision_cells=cells, state_nodes=len(lattice.values),
            state_limits=(lattice.values[[0, -1]] + model.initial_mean).tolist(),
            lattice_mass_ratio=list(lattice.mass_ratio), scaled_tail_upper=float(scaled_tail),
            log_tail_evidence_upper=float(log_tail), zero_mass_knots=int(np.isneginf(knot_mass).sum()),
            sampled_knots=int(np.count_nonzero(np.r_[0, components[:, 1]] + np.r_[components[:, 0], 0]))),
        sufficient_statistics=dict(template=template, means=means.tolist(), sample_variances=sample_variances.tolist()),
        zero_weight_draws=zero_weights, sequence_proposal_visits=None if visits is None else visits.tolist(),
        zero_variance_numerator_radius=unseen_radius,
        uncertainty_kind='Finite-sample probability intervals with analytic lattice and tail allowances in exact arithmetic; moments and densities are descriptive')
