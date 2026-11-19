"""Independent continuous-state Gibbs calculation for the innovation study."""

import math

import numpy as np
from scipy.linalg.lapack import dpttrf, dpttrs
from scipy.special import logsumexp, ndtr

from model_pipelining.innovation_reference import enumerate_sequences


class _PathGaussian:
    """Condition a Gaussian path using its tridiagonal precision."""

    def __init__(self, model, values, labels):
        increments = np.where(labels, model.large_variance, model.small_variance)
        positive = increments > 0
        self.positive_edges = positive
        self.increments = increments[positive]
        self.initial_variance = model.initial_variance
        # Zero-variance transitions identify adjacent states. With a known
        # initial value, the first group is fixed at zero in centered units.
        offset = int(model.initial_variance == 0)
        self.indices = np.r_[0, np.cumsum(positive)] - offset
        size = int(positive.sum()) + 1 - offset
        self.counts = np.bincount(self.indices[1:] + offset, minlength=size + offset)[offset:]
        self.totals = np.bincount(self.indices[1:] + offset, weights=values,
                                  minlength=size + offset)[offset:]
        self.diagonal = np.zeros(size)
        edges = 1 / increments[positive]
        self.off_diagonal = -edges[offset:]
        self.diagonal[:-1] += edges[offset:]
        self.diagonal[1:] += edges[offset:]
        if offset and size:
            self.diagonal[0] += edges[0]
        if not offset:
            self.diagonal[0] += 1 / model.initial_variance
        self.log_prior_determinant = float(np.log(increments[positive]).sum())
        if not offset:
            self.log_prior_determinant += math.log(model.initial_variance)
        self.values = values

    def _factor(self, variance):
        diagonal = self.diagonal + self.counts / variance
        if len(diagonal) < 2:
            if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                raise ValueError("Invalid Gaussian state precision")
            return diagonal, self.off_diagonal
        diagonal, lower, status = dpttrf(diagonal, self.off_diagonal)
        if status != 0 or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise ValueError("Invalid Gaussian state precision")
        return diagonal, lower

    @staticmethod
    def _solve(diagonal, lower, rhs):
        if len(diagonal) < 2:
            return rhs / diagonal
        result, status = dpttrs(diagonal, lower, rhs[:, None])
        if status != 0 or not np.isfinite(result).all():
            raise ValueError("Gaussian state precision solve failed")
        return result[:, 0]

    def _expand(self, free):
        return np.r_[free, 0.][self.indices]

    def draw(self, variance, rng):
        diagonal, lower = self._factor(variance)
        noise = np.sqrt(diagonal) * rng.standard_normal(len(diagonal))
        rhs = self.totals / variance + noise
        rhs[1:] += lower * noise[:-1]
        return self._expand(self._solve(diagonal, lower, rhs))

    def log_likelihood(self, variance):
        diagonal, lower = self._factor(variance)
        mean = self._solve(diagonal, lower, self.totals / variance)
        path = self._expand(mean)
        residual = self.values - path[1:]
        # Evaluate the minimum quadratic directly without subtracting nearly
        # equal marginal and conditional sums of squares.
        prior_cost = float(np.sum(np.diff(path)[self.positive_edges]**2 / self.increments))
        if self.initial_variance > 0:
            prior_cost += path[0]**2 / self.initial_variance
        return -.5 * (len(residual) * (math.log(2 * math.pi) + math.log(variance))
                     + self.log_prior_determinant + float(np.log(diagonal).sum())
                     + float(residual @ residual) / variance + prior_cost)


def sample_innovation_gibbs(values, model, *, seeds, warmup, draws):
    """Sample x, V and labels from separate continuous-state Gibbs chains.

    Histories with mixed zero/positive transition variances require an exact
    blocked label update, limited here to the existing small oracle cases.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not 1 <= len(values) <= 512 or not np.isfinite(values).all()
            or type(warmup) is not int or warmup < 0 or type(draws) is not int or not 8 <= draws <= 65536
            or not 2 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Invalid history or independent-chain settings")
    values = values - model.initial_mean
    n = len(values)
    probability = model.jump_probability
    fixed = probability in (0., 1.)
    blocked = not fixed and model.small_variance == 0 and model.large_variance > 0
    if blocked:
        sequences, prior = enumerate_sequences(model, n)
        systems = [_PathGaussian(model, values, labels) for labels in sequences]
    elif not fixed and model.small_variance > 0:
        log_odds = (math.log(probability / (1 - probability))
                    + .5 * math.log(model.small_variance / model.large_variance))
        variance_contrast = .5 * (1 / model.small_variance - 1 / model.large_variance)

    samples = np.empty((len(seeds), draws, 3))
    labels_saved = np.empty((len(seeds), draws, n), dtype=np.uint8)
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        v = model.prior_scale / (model.prior_shape - 1) * (2. ** (chain % 4 - 1))
        labels = np.full(n, int(probability == 1 or not fixed and chain % 2), dtype=np.uint8)
        system = _PathGaussian(model, values, labels)
        path = system.draw(v, rng)
        for step in range(warmup + draws):
            residual = values - path[1:]
            v = (model.prior_scale + .5 * (residual @ residual)) / rng.gamma(model.prior_shape + n / 2)
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Invalid observation variance draw")
            if blocked:
                log_masses = prior + np.array([system.log_likelihood(v) for system in systems])
                probabilities = np.exp(log_masses - logsumexp(log_masses))
                selected = rng.choice(len(sequences), p=probabilities)
                labels, system = sequences[selected], systems[selected]
            elif not fixed:
                if model.small_variance == 0:
                    labels = (rng.random(n) < probability).astype(np.uint8)
                else:
                    log_p = -np.logaddexp(0., -log_odds - variance_contrast * np.diff(path)**2)
                    labels = (rng.random(n) < np.exp(log_p)).astype(np.uint8)
                system = _PathGaussian(model, values, labels)
            path = system.draw(v, rng)
            if step >= warmup:
                samples[chain, step-warmup] = path[0], path[-1], v
                labels_saved[chain, step-warmup] = labels
    return samples, labels_saved


def innovation_gibbs_probes(samples, labels, model, *, predictive_points, state_points, variance_points):
    """Per-draw contributions for joint moments and predictive probes."""
    initial, last, v = np.moveaxis(samples, -1, 0)
    counts = labels.sum(axis=-1)
    initial_centered = initial - initial.mean()
    last_centered = last - last.mean()
    features = dict(initial_mean=initial + model.initial_mean, last_mean=last + model.initial_mean,
        initial_variance=initial_centered**2, last_variance=last_centered**2,
        initial_last_covariance=initial_centered * last_centered, observation_variance_mean=v,
        variance_jump_count_covariance=(v - v.mean()) * (counts - counts.mean()),
        jump_count=counts, predictive_mean=last + model.initial_mean,
        predictive_variance=last_centered**2 + v +
            (1-model.jump_probability)*model.small_variance + model.jump_probability*model.large_variance,
        next_jump_probability=np.full(v.shape, model.jump_probability))
    for point in predictive_points:
        cdf, density = np.zeros(v.shape), np.zeros(v.shape)
        for p, increment in ((1-model.jump_probability, model.small_variance),
                             (model.jump_probability, model.large_variance)):
            sd = np.sqrt(v + increment)
            z = (point - model.initial_mean - last) / sd
            cdf += p * ndtr(z)
            density += p * np.exp(-.5*z*z) / (math.sqrt(2*math.pi)*sd)
        features[f"predictive_cdf:{float(point)}"] = cdf
        features[f"predictive_density:{float(point)}"] = density
    for point in state_points:
        for name, state in (("initial", initial), ("last", last)):
            features[f"{name}_cdf_lower:{float(point)}"] = state < point - model.initial_mean
            features[f"{name}_cdf_upper:{float(point)}"] = state <= point - model.initial_mean
    for point in variance_points:
        features[f"variance_cdf:{float(point)}"] = v <= point
    return features
