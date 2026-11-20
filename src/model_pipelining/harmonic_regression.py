"""Point forecasts from a least-squares intercept plus one weekly Fourier pair.

y[t] = b0 + b1 sin(2 pi d/7) + b2 cos(2 pi d/7) + error, where d is the number
of days since the first eligible observation. Coefficients are refit on every
eligible history; the forecast evaluates the fitted curve at the target time.
This small adapter exercises the shared point-forecast contract for a second
model family. It declares no uncertainty outputs.
"""

import math

import numpy as np


class WeeklyHarmonicRegression:
    """One fixed seven-day Fourier pair and an intercept."""

    name = "weekly-harmonic-example"
    supports_quantiles = False
    supports_highest_density = False

    @property
    def configuration(self):
        return {"period_days": 7, "harmonics": 1}

    @classmethod
    def from_configuration(cls, configuration):
        """Rebuild the adapter; its period and harmonic count are fixed."""
        if dict(configuration) != {"period_days": 7, "harmonics": 1}:
            raise ValueError("WeeklyHarmonicRegression supports only period_days=7 and harmonics=1")
        return cls()

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))

    def forecast(self, request):
        start = request.history[0].timestamp

        def features(timestamp):
            phase = 2 * math.pi * ((timestamp - start).total_seconds() / 86400 % 7) / 7
            return [1., math.sin(phase), math.cos(phase)]

        design = np.array([features(observation.timestamp) for observation in request.history])
        values = [observation.value for observation in request.history]
        coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank != len(coefficients):
            raise ValueError("The supplied history does not identify the weekly Fourier pair")
        return float(features(request.target_time) @ coefficients)
