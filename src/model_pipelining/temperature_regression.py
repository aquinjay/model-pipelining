"""Scalar temperature forecasts with the frozen Central Park regression settings."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from model_pipelining.models import ForecastRequest


ANNUAL_PERIOD_DAYS = 365.2425
PHASE_ORIGIN = date(2023, 1, 1)
TEMPERATURE_REGRESSIONS = {
    "annual-only": (True, False),
    "lag-only": (False, True),
    "annual-plus-lag": (True, True),
}


@dataclass(frozen=True)
class TemperatureRegression:
    """Ordinary least squares with an annual Fourier pair, a one-day lag, or both."""

    name: str
    supports_quantiles = False
    supports_highest_density = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in TEMPERATURE_REGRESSIONS:
            raise ValueError(f"Choose a temperature regression from {', '.join(TEMPERATURE_REGRESSIONS)}")

    @property
    def configuration(self) -> dict:
        annual, lag = TEMPERATURE_REGRESSIONS[self.name]
        return dict(intercept=True, annual_period_days=ANNUAL_PERIOD_DAYS if annual else None,
                    phase_origin=PHASE_ORIGIN.isoformat() if annual else None, lag_days=1 if lag else None)

    @classmethod
    def from_configuration(cls, configuration: dict) -> "TemperatureRegression":
        """Rebuild the variant described by its saved JSON settings."""
        if (not isinstance(configuration, dict) or configuration.get("intercept") is not True
                or isinstance(configuration.get("lag_days"), bool)):
            raise ValueError("Expected the saved configuration of a supported temperature regression")
        for name in TEMPERATURE_REGRESSIONS:
            model = cls(name)
            if configuration == model.configuration:
                return model
        raise ValueError("Expected the saved configuration of a supported temperature regression")

    def forecast(self, request: ForecastRequest) -> float:
        values = np.array([row.value for row in request.history])
        ordinals = np.array([row.timestamp.toordinal() for row in request.history])
        target = request.target_time.toordinal()
        if len(values) < 2 or target != ordinals[-1] + 1 or not np.all(np.diff(ordinals) == 1):
            raise ValueError("Temperature regression requires a contiguous one-day forecast")
        # Fit all variants on the same response dates. The first eligible value
        # is available as a lag but has no earlier observation of its own.
        design = [np.ones(len(values) - 1)]
        future = [1.]
        annual, lag = TEMPERATURE_REGRESSIONS[self.name]
        if annual:
            origin = PHASE_ORIGIN.toordinal()
            phase = 2 * np.pi * (ordinals[1:] - origin) / ANNUAL_PERIOD_DAYS
            target_phase = 2 * np.pi * (target - origin) / ANNUAL_PERIOD_DAYS
            design.extend((np.sin(phase), np.cos(phase)))
            future.extend((np.sin(target_phase), np.cos(target_phase)))
        if lag:
            design.append(values[:-1])
            future.append(values[-1])
        coefficients, _, rank, _ = np.linalg.lstsq(np.column_stack(design), values[1:], rcond=None)
        if rank != len(design):
            raise ValueError("Temperature history does not identify the regression coefficients")
        return float(np.dot(future, coefficients))
