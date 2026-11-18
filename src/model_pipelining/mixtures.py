"""Numerical summaries of a finite weighted Gaussian predictive mixture."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from model_pipelining.models import HighestDensityRegion, Interval
from model_pipelining.uncertainty import interval_probabilities


@dataclass(frozen=True)
class NormalMixture:
    weights: tuple[float, ...]
    means: tuple[float, ...]
    variances: tuple[float, ...]

    def __post_init__(self):
        weights, means, variances = map(lambda x: np.asarray(x, dtype=float),
                                        (self.weights, self.means, self.variances))
        if (weights.ndim != 1 or not len(weights) or means.shape != weights.shape or variances.shape != weights.shape
                or not all(np.isfinite(x).all() for x in (weights, means, variances))
                or np.any(weights < 0) or np.any(variances <= 0) or not math.isfinite(float(weights.sum()))
                or weights.sum() <= 0):
            raise ValueError("Need finite matching Gaussian components, positive variances and nonnegative weights")
        positive = weights > 0
        object.__setattr__(self, "weights", tuple(map(float, weights[positive]/weights.sum())))
        object.__setattr__(self, "means", tuple(map(float, means[positive])))
        object.__setattr__(self, "variances", tuple(map(float, variances[positive])))

    @property
    def mean(self):
        return math.fsum(w*m for w, m in zip(self.weights, self.means))

    @property
    def variance(self):
        mean = self.mean
        return math.fsum(w*(v+(m-mean)**2) for w, m, v in zip(self.weights, self.means, self.variances))

    def cdf(self, x):
        values = np.asarray(x, dtype=float)
        result = np.sum(np.asarray(self.weights)*ndtr((values[..., None]-self.means)/np.sqrt(self.variances)), axis=-1)
        return float(result) if values.ndim == 0 else result

    def pdf(self, x):
        values = np.asarray(x, dtype=float)
        with np.errstate(over="ignore", under="ignore"):
            z = (values[..., None]-self.means)/np.sqrt(self.variances)
            result = np.sum(np.asarray(self.weights)*np.exp(-0.5*z*z)/(math.sqrt(2*math.pi)*np.sqrt(self.variances)), axis=-1)
        return float(result) if values.ndim == 0 else result

    def ppf(self, probability):
        if isinstance(probability, bool) or not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("Probability must be finite and strictly between zero and one")
        component = np.asarray(self.means)+np.sqrt(self.variances)*ndtri(probability)
        lower, upper = float(component.min()), float(component.max())
        if not all(math.isfinite(x) for x in (lower, upper)):
            raise ValueError("Unrepresentable mixture quantile bracket")
        if lower == upper:
            if abs(self.cdf(lower)-probability) > 1e-8:
                raise ValueError("Requested quantile is not representable at this location/scale")
            return lower
        # Tail components can make the bracket enormous while the requested
        # quantile remains near the bulk. Never scale absolute root tolerance
        # by that bracket: it could then skip the whole bulk distribution.
        scale = max(math.sqrt(min(self.variances)), 1e-150)
        root = brentq(lambda x: self.cdf(x)-probability, lower, upper,
                      xtol=scale*1e-10, rtol=1e-13, maxiter=1024)
        if abs(self.cdf(root)-probability) > 1e-8:
            raise ValueError("Mixture quantile precision failed")
        return float(root)

    def highest_density(self, level, *, tolerance=1e-7):
        """Return a numerically resolved density superlevel set, including gaps.

        Component-scale knots plus a doubled grid locate crossings. Mass is
        evaluated with the mixture CDF, not a histogram. A second resolution
        must agree on segments, mass and boundaries; otherwise fail visibly.
        This is a numerical resolution check, not a proof for arbitrary mixtures.
        """
        interval_probabilities(level)
        tail = min(1e-10, (1-level)*1e-4)
        if tail <= 0 or 1-tail == 1:
            raise ValueError("HPD level is beyond representable tail resolution")
        lower, upper = self.ppf(tail), self.ppf(1-tail)
        means, sds = np.asarray(self.means), np.sqrt(self.variances)
        # Tiny-weight components still participate in every PDF/CDF evaluation
        # and contribute their centers. Their scale grids can be omitted within
        # this explicitly saved mass budget, keeping large quadratures practical.
        selected = np.asarray(self.weights) > tolerance/(1000*len(self.weights))
        previous = None
        evidence = None
        for resolution in (1, 2, 4):
            offsets = np.linspace(-8, 8, 65*resolution)
            knots = np.concatenate(([lower, upper], means, (means[selected, None]+sds[selected, None]*offsets).ravel(),
                                    np.linspace(float(means.min()), float(means.max()), 257*resolution)))
            grid = np.unique(knots[(knots >= lower)&(knots <= upper)])
            # Bound temporary arrays when a posterior has many quadrature nodes.
            density = np.concatenate([self.pdf(grid[i:i+256]) for i in range(0, len(grid), 256)])
            def region_at(threshold):
                above = density >= threshold
                crossings = np.flatnonzero(above[:-1] != above[1:])
                boundaries = [lower] if above[0] else []
                for index in crossings:
                    boundaries.append(float(brentq(lambda x: self.pdf(x)-threshold,
                                                   grid[index], grid[index+1], xtol=max(sds.min()*1e-10, 1e-100), rtol=1e-13)))
                if above[-1]:
                    boundaries.append(upper)
                segments = tuple(Interval(a, b) for a, b in zip(boundaries[::2], boundaries[1::2]))
                mass = math.fsum(self.cdf(item.upper)-self.cdf(item.lower) for item in segments)
                return segments, mass
            top = math.fsum(w/(math.sqrt(2*math.pi)*math.sqrt(v)) for w, v in zip(self.weights, self.variances))
            threshold = brentq(lambda h: region_at(h)[1]-level, 0, top,
                               xtol=max(top*1e-12, 1e-300), rtol=1e-12)
            segments, mass = region_at(threshold)
            boundary_error = max((abs(self.pdf(x)-threshold)/max(threshold, 1e-300)
                                  for item in segments for x in (item.lower, item.upper)), default=math.inf)
            evidence = {"level": level, "mass": mass, "density_threshold": threshold,
                        "segments": len(segments), "grid_resolution": resolution,
                        "boundary_relative_density_error": boundary_error,
                        "outside_search_probability": 2*tail, "resolution_verified": False}
            evidence["components_without_scale_knots_mass"] = float(np.asarray(self.weights)[~selected].sum())
            if previous is not None and len(previous) == len(segments):
                agreement = max((abs(a-b) for old, new in zip(previous, segments)
                                 for a, b in ((old.lower, new.lower), (old.upper, new.upper))), default=math.inf)
                if (abs(mass-level) <= tolerance and boundary_error <= tolerance*10
                        and agreement <= tolerance*max(1, math.sqrt(self.variance))):
                    evidence["resolution_verified"] = True
                    return HighestDensityRegion(level, segments), evidence
            previous = segments
        raise ValueError(f"Mixture HPD resolution failed: {evidence}")
