"""The first point-forecast boundary and a training-mean baseline.

The runner owns historical selection and scoring. Models receive a fresh,
immutable request containing eligible observations and the target time only.
"""

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class ForecastRequest:
    history: tuple[Observation, ...]
    issued_at: datetime
    target_time: datetime


class PointForecaster(Protocol):
    """A small boundary for deterministic models returning a predictive mean.

    Refit from the supplied history on every call. Do not retain training data
    between calls. configuration must describe all settings as JSON values.
    This is a trusted Python interface, not a security sandbox.
    """

    name: str

    @property
    def configuration(self) -> dict: ...

    def forecast(self, request: ForecastRequest) -> float: ...


class TrainingMean:
    """Predict the next value with the arithmetic mean of eligible history."""

    name = "training-mean"

    @property
    def configuration(self) -> dict:
        return {}

    def forecast(self, request: ForecastRequest) -> float:
        return fmean(observation.value for observation in request.history)
