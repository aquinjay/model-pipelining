"""Versioned forecasting tasks, release-aware histories, and evaluation cases."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from model_pipelining.ingest import read_local_table
from model_pipelining.models import BENCHMARK_MODELS, Observation
from model_pipelining.snapshots import retrieve_snapshot
from model_pipelining.tags import resolve_tag


# Separate evidence workflows an experiment can require, in report order.
EVIDENCE_WORKFLOWS = ("predictive_checks", "prior_influence", "sequential_checks")


def _utc(value, name):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


@dataclass(frozen=True, kw_only=True)
class ForecastTask:
    name: str = "daily-value"
    target_description: str = "Value at the future daily timestamp"
    source_description: str = "Caller-supplied preserved CSV"
    timestamp_column: str = "timestamp"
    target_column: str = "value"
    units: str = "Input value units; no transformations"
    calendar: str = "daily_utc"
    missing_policy: str = "reject"
    horizon_days: int = 1
    minimum_history: int = 30
    issuance_delay_seconds: int = 0
    availability_column: str | None = None
    revision_policy: str = "reject"
    actual_policy: str = "first_release"

    def __post_init__(self):
        for name in ("name", "target_description", "source_description", "timestamp_column", "target_column", "units"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonblank string")
        if self.calendar != "daily_utc" or self.missing_policy != "reject":
            raise ValueError("Only daily_utc with missing_policy=reject is implemented")
        for name in ("horizon_days", "minimum_history"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.issuance_delay_seconds) is not int or not 0 <= self.issuance_delay_seconds < 86400:
            raise ValueError("issuance_delay_seconds must be an integer in [0, 86400)")
        if self.availability_column is not None and (not isinstance(self.availability_column, str) or not self.availability_column.strip()):
            raise ValueError("availability_column must be a nonblank string or None")
        columns = [self.timestamp_column, self.target_column] + ([self.availability_column] if self.availability_column is not None else [])
        if len(set(columns)) != len(columns):
            raise ValueError("Timestamp, target, and availability columns must be distinct")
        if self.revision_policy not in ("reject", "latest_available"):
            raise ValueError("revision_policy must be reject or latest_available")
        if self.revision_policy == "latest_available" and self.availability_column is None:
            raise ValueError("latest_available requires an explicit availability_column")
        if self.actual_policy not in ("first_release", "latest_in_snapshot"):
            raise ValueError("actual_policy must be first_release or latest_in_snapshot")


@dataclass(frozen=True, kw_only=True)
class EvaluationPlan:
    role: str = "development"
    final_start: datetime | None = None
    final_end: datetime | None = None
    benchmarks: tuple[str, ...] = ()

    def __post_init__(self):
        if self.role not in ("development", "final"):
            raise ValueError("role must be development or final")
        if (self.final_start is None) != (self.final_end is None):
            raise ValueError("Supply both final_start and final_end, or neither")
        if self.final_start is not None:
            object.__setattr__(self, "final_start", _utc(self.final_start, "final_start"))
            object.__setattr__(self, "final_end", _utc(self.final_end, "final_end"))
            if self.final_start >= self.final_end:
                raise ValueError("final_start must precede final_end")
        if self.role == "final" and self.final_start is None:
            raise ValueError("Final evaluation requires explicit start and end boundaries")
        if (not isinstance(self.benchmarks, tuple)
                or any(name not in BENCHMARK_MODELS for name in self.benchmarks)
                or len(set(self.benchmarks)) != len(self.benchmarks)):
            raise ValueError(f"benchmarks must be a tuple of unique {'/'.join(BENCHMARK_MODELS)} names")

    def includes(self, timestamp):
        if self.final_start is None:
            return True
        if self.role == "development":
            return timestamp < self.final_start
        return self.final_start <= timestamp < self.final_end


@dataclass(frozen=True, kw_only=True)
class EvidencePlan:
    """Caller-owned computational requirements; no model-approval thresholds.

    Separate workflows run only through explicit evidence execution. Fit-check
    names apply at every primary-model forecast origin, including final runs.
    """

    required_workflows: tuple[str, ...] = ()
    required_fit_checks: tuple[str, ...] = ()
    replications: int = 1000
    seed: int = 0
    interval_level: float = 0.95

    def __post_init__(self):
        for name in ("required_workflows", "required_fit_checks"):
            value = getattr(self, name)
            if (not isinstance(value, tuple)
                    or any(not isinstance(item, str) or not item.strip() for item in value)
                    or len(set(value)) != len(value)):
                raise ValueError(f"{name} must be a tuple of unique nonblank names")
        if any(name not in EVIDENCE_WORKFLOWS for name in self.required_workflows):
            raise ValueError("Required evidence workflows must be "
                             f"{', '.join(EVIDENCE_WORKFLOWS[:-1])} or {EVIDENCE_WORKFLOWS[-1]}")
        if type(self.replications) is not int or self.replications < 1:
            raise ValueError("replications must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (isinstance(self.interval_level, bool) or not isinstance(self.interval_level, Real)
                or not math.isfinite(self.interval_level) or not 0 < self.interval_level < 1):
            raise ValueError("interval_level must be finite and in (0, 1)")


@dataclass(frozen=True, kw_only=True)
class ExperimentSpec:
    task: ForecastTask = field(default_factory=ForecastTask)
    evaluation: EvaluationPlan = field(default_factory=EvaluationPlan)
    evidence: EvidencePlan = field(default_factory=EvidencePlan)

    def __post_init__(self):
        if (not isinstance(self.task, ForecastTask) or not isinstance(self.evaluation, EvaluationPlan)
                or not isinstance(self.evidence, EvidencePlan)):
            raise ValueError("ExperimentSpec requires a ForecastTask, EvaluationPlan and EvidencePlan")

    def to_dict(self):
        evaluation = asdict(self.evaluation)
        for name in ("final_start", "final_end"):
            evaluation[name] = evaluation[name].isoformat() if evaluation[name] is not None else None
        evaluation["benchmarks"] = list(evaluation["benchmarks"])
        evidence = asdict(self.evidence)
        for name in ("required_workflows", "required_fit_checks"):
            evidence[name] = list(evidence[name])
        return {"format_version": 2, "task": asdict(self.task), "evaluation": evaluation, "evidence": evidence}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or type(value.get("format_version")) is not int
                or value["format_version"] not in (1, 2)
                or set(value) != ({"format_version", "task", "evaluation"} |
                                 ({"evidence"} if value["format_version"] == 2 else set()))
                or not isinstance(value["task"], dict) or not isinstance(value["evaluation"], dict)):
            raise ValueError("Expected a version-1/2 experiment with task/evaluation and version-2 evidence")
        evaluation = dict(value["evaluation"])
        for name in ("final_start", "final_end"):
            if evaluation.get(name) is not None:
                evaluation[name] = datetime.fromisoformat(evaluation[name])
        if "benchmarks" in evaluation:
            if not isinstance(evaluation["benchmarks"], list):
                raise ValueError("JSON benchmarks must be a list")
            evaluation["benchmarks"] = tuple(evaluation["benchmarks"])
        evidence = value.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("Experiment evidence must be a dictionary")
        evidence = dict(evidence)
        for name in ("required_workflows", "required_fit_checks"):
            if name in evidence:
                if not isinstance(evidence[name], list):
                    raise ValueError(f"JSON {name} must be a list")
                evidence[name] = tuple(evidence[name])
        return cls(task=ForecastTask(**value["task"]), evaluation=EvaluationPlan(**evaluation),
                   evidence=EvidencePlan(**evidence))

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ObservationVersion:
    timestamp: datetime
    value: float
    available_at: datetime


@dataclass(frozen=True)
class ForecastCase:
    origin_index: int
    issued_at: datetime
    target_time: datetime
    versions: tuple[ObservationVersion, ...]

    @property
    def history(self):
        return tuple(Observation(item.timestamp, item.value) for item in self.versions)

    @property
    def history_sha256(self):
        content = [(item.timestamp.isoformat(), item.value, item.available_at.isoformat()) for item in self.versions]
        return hashlib.sha256(json.dumps(content, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_task_data(path: Path, task: ForecastTask) -> tuple[ObservationVersion, ...]:
    """Validate the daily long table; releases for each period must be ordered.

    Records must be grouped by increasing period timestamp, then by strictly
    increasing availability. No sorting, gap filling, or invented vintages.
    """
    if path.suffix.lower() != ".csv":
        raise ValueError("The backtest supports CSV snapshots only")
    table = read_local_table(path)
    columns = [task.timestamp_column, task.target_column]
    if task.availability_column is not None:
        columns.append(task.availability_column)
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Input must contain {', '.join(columns)} columns")
    versions = []
    for record in table[columns].itertuples(index=False, name=None):
        timestamp = _utc(datetime.fromisoformat(str(record[0])), "Observation timestamp")
        raw_value = record[1]
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError("Every value must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("Every value must be finite and nonmissing")
        available_at = _utc(datetime.fromisoformat(str(record[2])), "Availability timestamp") if task.availability_column is not None else timestamp
        if available_at < timestamp:
            raise ValueError("An observation cannot be available before its period timestamp")
        if versions:
            previous = versions[-1]
            if timestamp == previous.timestamp:
                if task.revision_policy != "latest_available" or available_at <= previous.available_at:
                    raise ValueError("Duplicate periods require latest_available and strictly ordered unique releases")
            elif timestamp - previous.timestamp != timedelta(days=1):
                raise ValueError("Timestamps must be ordered and exactly one UTC day apart; gaps are rejected")
        versions.append(ObservationVersion(timestamp, value, available_at))
    return tuple(versions)


def read_observations(path: Path) -> tuple[Observation, ...]:
    """Read a daily CSV with immediate, unrevised availability as plain observations."""
    return tuple(Observation(item.timestamp, item.value) for item in read_task_data(path, ForecastTask()))


def select_history_as_of(versions: tuple[ObservationVersion, ...], *, issued_at: datetime,
                         period_end: datetime | None = None) -> tuple[ObservationVersion, ...]:
    """Select latest eligible releases from an ordered, validated release table.

    Period timestamps must be <= period_end (default issued_at); releases must
    be <= issued_at. This selects training history, never scoring outcomes.
    """
    issuance = _utc(issued_at, "issued_at")
    end = issuance if period_end is None else _utc(period_end, "period_end")
    if end > issuance:
        raise ValueError("period_end must be no later than issued_at")
    selected = {}
    for item in versions:
        if item.timestamp <= end and item.available_at <= issuance:
            selected[item.timestamp] = item
    return tuple(selected.values())


def plan_cases(versions: tuple[ObservationVersion, ...], experiment: ExperimentSpec):
    """Plan target-based splits and as-of histories before any model calls."""
    periods = tuple(dict.fromkeys(item.timestamp for item in versions))
    task = experiment.task
    cases, schedule = [], []
    for target_index in range(task.horizon_days, len(periods)):
        origin_index = target_index - task.horizon_days
        origin = periods[origin_index]
        issued_at = origin + timedelta(seconds=task.issuance_delay_seconds)
        history = select_history_as_of(versions, issued_at=issued_at, period_end=origin)
        included = experiment.evaluation.includes(periods[target_index])
        reason = "selected" if included else "outside_evaluation_role"
        if included and len(history) < task.minimum_history:
            reason = "insufficient_available_history"
        schedule.append({"origin_index": origin_index, "issued_at": issued_at.isoformat(),
                         "target_time": periods[target_index].isoformat(), "available_history_count": len(history),
                         "selection": reason})
        if reason == "selected":
            cases.append(ForecastCase(origin_index, issued_at, periods[target_index], history))
    if not cases:
        raise ValueError("No eligible forecast cases for this experiment and minimum_history")
    return tuple(cases), schedule


def select_actuals(versions: tuple[ObservationVersion, ...], task: ForecastTask):
    """Choose scoring outcomes separately from the as-of training selection."""
    selected = {}
    for item in versions:
        key = item.timestamp.isoformat()
        if task.actual_policy == "latest_in_snapshot" or key not in selected:
            selected[key] = item
    return selected


def select_development_history(*, dataset: str, issued_at: datetime, snapshot_id: str | None = None,
                         tag: str | None = None, store: str | Path = "data/versioned",
                         experiment: ExperimentSpec | None = None):
    """Common verified end-history/context for predictive and prior comparisons."""
    if (snapshot_id is None) == (tag is None):
        raise ValueError("Supply exactly one of snapshot_id or tag")
    if not isinstance(issued_at, datetime) or issued_at.utcoffset() is None:
        raise ValueError("issued_at must be a timezone-aware datetime")
    issued_at = issued_at.astimezone(UTC)
    specification = ExperimentSpec() if experiment is None else experiment
    if not isinstance(specification, ExperimentSpec) or specification.evaluation.role != "development":
        raise ValueError("A development ExperimentSpec is required")
    boundary = specification.evaluation.final_start
    if boundary is not None and issued_at >= boundary:
        raise ValueError("issued_at must precede the experiment's final_start")
    target_time = issued_at + timedelta(days=specification.task.horizon_days)
    selected_id = resolve_tag(dataset=dataset, tag=tag, store=store) if tag is not None else snapshot_id
    versions = read_task_data(retrieve_snapshot(selected_id, dataset=dataset, store=store), specification.task)
    selected = select_history_as_of(versions, issued_at=issued_at)
    if len(selected) < specification.task.minimum_history:
        raise ValueError("Insufficient available development history")
    case = ForecastCase(0, issued_at, target_time, selected)
    context = {"input": {"dataset": dataset, "snapshot_id": selected_id, "selected_tag": tag},
               "issued_at": issued_at.isoformat(), "history_sha256": case.history_sha256,
               "experiment": specification.to_dict(),
               "selection": "Latest available release per period with period and availability <= issued_at",
               "unused_forecast_settings": "horizon/issuance delay/actual policy/benchmarks do not score or select check datasets; issued_at is explicit",
               "selected_history": [{"timestamp": item.timestamp.isoformat(), "available_at": item.available_at.isoformat(), "value": item.value}
                                    for item in selected]}
    return specification, versions, case, context
