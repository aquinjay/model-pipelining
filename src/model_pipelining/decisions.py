"""Model-independent saved-only decision issuance, outcome scoring, and comparison.

Issuance reads only forecast-time artifacts (frozen predictive distributions or
saved event probabilities) and writes immutable per-origin recommendations before
consulting any outcomes. Scoring joins outcomes in a separate output directory
without mutating issued records.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.special import ndtr
from scipy.stats import t

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.decision_value import (
    asymmetric_expense,
    asymmetric_quantile_level,
    decide,
    evaluate_asymmetric_decisions,
    expense,
    summarize_expenses,
    value_curve,
)
from model_pipelining.distribution_scoring import validate_distribution
from model_pipelining.mixtures import NormalMixture


VALID_POLICIES = ("binary_cost_loss", "continuous_asymmetric")
VALID_COMPARISONS = (">=", ">", "<=", "<")
BINARY_FALLBACKS = ("act", "wait", "climatology")
DEFAULT_SENSITIVITY_ALPHAS = tuple(percent / 100 for percent in range(5, 100, 5))

RECOMMENDATION_FIELDS = (
    "origin_index",
    "issued_at",
    "target_time",
    "dataset",
    "snapshot_id",
    "history_sha256",
    "model_name",
    "policy",
    "event_probability",
    "cost_loss_ratio",
    "under_cost",
    "over_cost",
    "action",
    "decision_source",
    "forecast_status",
    "forecast_error",
)

EVALUATION_FIELDS = (
    "origin_index",
    "issued_at",
    "target_time",
    "policy",
    "event_probability",
    "action",
    "decision_source",
    "actual",
    "actual_available_at",
    "event",
    "outcome_category",
    "expense",
    "baseline_expense",
    "perfect_expense",
)


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonblank(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path: Path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if len(set(fields)) != len(fields):
            raise ValueError(f"Duplicate CSV header columns in {path}")
        return fields, list(reader)


def _normalize_target_time(value: str, name: str = "target_time") -> str:
    text = _nonblank(value, name).strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        parsed_day = date.fromisoformat(text)
        return datetime(parsed_day.year, parsed_day.month, parsed_day.day, tzinfo=UTC).isoformat()
    parsed = datetime.fromisoformat(text)
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp or YYYY-MM-DD date")
    return parsed.astimezone(UTC).isoformat()


def _source_hashes() -> dict[str, str]:
    here = Path(__file__)
    files = (
        here,
        here.with_name("decision_value.py"),
        here.with_name("distribution_scoring.py"),
        here.with_name("mixtures.py"),
        here.with_name("artifacts.py"),
    )
    return {path.name: _sha256_file(path) for path in files}


def event_indicator(value: float, *, threshold: float, comparison: str = ">=") -> int:
    """Return 1 if value satisfies the scalar threshold comparison, else 0."""
    number = _finite_number(value, "value")
    cutoff = _finite_number(threshold, "threshold")
    if comparison == ">=":
        return 1 if number >= cutoff else 0
    if comparison == ">":
        return 1 if number > cutoff else 0
    if comparison == "<=":
        return 1 if number <= cutoff else 0
    if comparison == "<":
        return 1 if number < cutoff else 0
    raise ValueError(f"comparison must be one of {', '.join(VALID_COMPARISONS)}")


def distribution_event_probability(record: dict, *, threshold: float, comparison: str = ">=") -> float:
    """Return the exact event probability from a validated frozen predictive distribution."""
    law = validate_distribution(record)
    cutoff = _finite_number(threshold, "threshold")
    if comparison not in VALID_COMPARISONS:
        raise ValueError(f"comparison must be one of {', '.join(VALID_COMPARISONS)}")
    upper = comparison in (">=", ">")
    if law["family"] == "student_t":
        dist = t(df=law["df"], loc=law["location"], scale=law["scale"])
        probability = float(dist.sf(cutoff) if upper else dist.cdf(cutoff))
    else:
        mixture = NormalMixture(tuple(law["weights"]), tuple(law["means"]), tuple(law["variances"]))
        weights = np.asarray(mixture.weights, dtype=float)
        means = np.asarray(mixture.means, dtype=float)
        sds = np.sqrt(np.asarray(mixture.variances, dtype=float))
        z = (means - cutoff) / sds if upper else (cutoff - means) / sds
        probability = float(np.sum(weights * ndtr(z)))
    if not math.isfinite(probability):
        raise ValueError("Nonfinite event probability")
    # Guard against 1-ulp overshoot from weighted sums.
    if -1e-15 <= probability < 0.0:
        probability = 0.0
    elif 1.0 < probability <= 1.0 + 1e-15:
        probability = 1.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Event probability out of [0, 1]: {probability}")
    return probability


def distribution_quantile(record: dict, probability: float) -> float:
    """Return the exact predictive quantile at probability in (0, 1) from a frozen distribution."""
    law = validate_distribution(record)
    prob = _finite_number(probability, "probability")
    if not 0.0 < prob < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    if law["family"] == "student_t":
        value = float(t.ppf(prob, df=law["df"], loc=law["location"], scale=law["scale"]))
    else:
        mixture = NormalMixture(tuple(law["weights"]), tuple(law["means"]), tuple(law["variances"]))
        value = mixture.ppf(prob)
    if not math.isfinite(value):
        raise ValueError("Nonfinite predictive quantile")
    return value


@dataclass(frozen=True, kw_only=True)
class DecisionSpec:
    """Predeclared decision problem, policy parameters, calibration reference, and fallback.

    Choose policy parameters and calibration reference before scoring evaluation
    outcomes so decision evaluation remains reproducible and free of outcome tuning.
    """

    name: str
    policy: str
    event_description: str
    action_description: str
    target_description: str
    horizon_days: int
    calibration_reference: dict
    fallback: str | float
    cost_loss_ratio: float | None = None
    threshold: float | None = None
    comparison: str = ">="
    under_cost: float | None = None
    over_cost: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "event_description", "action_description", "target_description"):
            _nonblank(getattr(self, field_name), field_name)
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {', '.join(VALID_POLICIES)}")
        if type(self.horizon_days) is not int or self.horizon_days < 1:
            raise ValueError("horizon_days must be a positive integer")
        if self.notes is not None:
            _nonblank(self.notes, "notes")
        if not isinstance(self.calibration_reference, dict) or not self.calibration_reference:
            raise ValueError("calibration_reference must be a nonempty dictionary")
        reference = json.loads(json.dumps(self.calibration_reference, allow_nan=False))
        _nonblank(reference.get("description"), "calibration_reference.description")
        if self.policy == "binary_cost_loss":
            if self.cost_loss_ratio is None:
                raise ValueError("binary_cost_loss requires cost_loss_ratio")
            alpha = _finite_number(self.cost_loss_ratio, "cost_loss_ratio")
            if not 0.0 < alpha < 1.0:
                raise ValueError("cost_loss_ratio must be strictly between 0 and 1")
            if self.under_cost is not None or self.over_cost is not None:
                raise ValueError("under_cost and over_cost are only valid for continuous_asymmetric")
            if self.comparison not in VALID_COMPARISONS:
                raise ValueError(f"comparison must be one of {', '.join(VALID_COMPARISONS)}")
            if self.threshold is not None:
                object.__setattr__(self, "threshold", _finite_number(self.threshold, "threshold"))
            base_rate = _finite_number(reference.get("base_rate"), "calibration_reference.base_rate")
            if not 0.0 <= base_rate <= 1.0:
                raise ValueError("calibration_reference.base_rate must be in [0, 1]")
            reference["base_rate"] = base_rate
            if self.fallback not in BINARY_FALLBACKS:
                raise ValueError(f"binary_cost_loss fallback must be one of {', '.join(BINARY_FALLBACKS)}")
            object.__setattr__(self, "cost_loss_ratio", alpha)
        else:
            if self.under_cost is None or self.over_cost is None:
                raise ValueError("continuous_asymmetric requires under_cost and over_cost")
            if self.cost_loss_ratio is not None or self.threshold is not None:
                raise ValueError("cost_loss_ratio and threshold are only valid for binary_cost_loss")
            under = _finite_number(self.under_cost, "under_cost")
            over = _finite_number(self.over_cost, "over_cost")
            asymmetric_quantile_level(under, over)
            baseline = _finite_number(reference.get("baseline_action"), "calibration_reference.baseline_action")
            reference["baseline_action"] = baseline
            if self.fallback != "baseline":
                object.__setattr__(self, "fallback", _finite_number(self.fallback, "fallback"))
            object.__setattr__(self, "under_cost", under)
            object.__setattr__(self, "over_cost", over)
        object.__setattr__(self, "calibration_reference", reference)

    @property
    def base_rate(self) -> float | None:
        return self.calibration_reference.get("base_rate") if self.policy == "binary_cost_loss" else None

    @property
    def baseline_action(self) -> float | None:
        return self.calibration_reference.get("baseline_action") if self.policy == "continuous_asymmetric" else None

    def resolve_fallback_action(self) -> str | float:
        if self.policy == "binary_cost_loss":
            if self.fallback == "climatology":
                return "act" if decide(self.base_rate, self.cost_loss_ratio) else "wait"
            return str(self.fallback)
        if self.fallback == "baseline":
            return float(self.baseline_action)
        return float(self.fallback)

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "name": self.name,
            "policy": self.policy,
            "event_description": self.event_description,
            "action_description": self.action_description,
            "target_description": self.target_description,
            "horizon_days": self.horizon_days,
            "cost_loss_ratio": self.cost_loss_ratio,
            "threshold": self.threshold,
            "comparison": self.comparison if self.policy == "binary_cost_loss" else None,
            "under_cost": self.under_cost,
            "over_cost": self.over_cost,
            "calibration_reference": json.loads(json.dumps(self.calibration_reference, allow_nan=False)),
            "fallback": self.fallback,
            "notes": self.notes,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return _sha256_bytes(payload)

    @classmethod
    def from_dict(cls, value: dict) -> "DecisionSpec":
        if not isinstance(value, dict) or value.get("format_version") != 1:
            raise ValueError("Expected a version-1 DecisionSpec dictionary")
        allowed = {
            "format_version", "name", "policy", "event_description", "action_description",
            "target_description", "horizon_days", "cost_loss_ratio", "threshold", "comparison",
            "under_cost", "over_cost", "calibration_reference", "fallback", "notes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown DecisionSpec fields: {', '.join(sorted(unknown))}")
        kwargs = {key: value[key] for key in value if key != "format_version"}
        if kwargs.get("comparison") is None:
            kwargs["comparison"] = ">="
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "DecisionSpec":
        return cls.from_dict(_load_json(Path(path).expanduser()))


def _write_issued_artifacts(
    output: Path,
    *,
    spec: DecisionSpec,
    recommendations: list[dict],
    source_kind: str,
    source_directory: Path,
    input_identity: dict,
    model_identity: dict,
    source_files_sha256: dict[str, str],
) -> Path:
    if not recommendations:
        raise ValueError("At least one recommendation is required")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "decision_spec.json", spec.to_dict())
    write_csv(output / "recommendations.csv", RECOMMENDATION_FIELDS, recommendations)
    forecast_count = sum(1 for row in recommendations if row["decision_source"] == "forecast")
    fallback_count = len(recommendations) - forecast_count
    manifest = {
        "format_version": 1,
        "kind": "decision_issuance",
        "status": "issued",
        "issued_bundle_created_at": datetime.now(UTC).isoformat(),
        "source_kind": source_kind,
        "source_directory": str(source_directory.resolve()),
        "input": input_identity,
        "model": model_identity,
        "spec_name": spec.name,
        "policy": spec.policy,
        "spec_sha256": spec.sha256,
        "case_count": len(recommendations),
        "target_start": recommendations[0]["target_time"],
        "target_end": recommendations[-1]["target_time"],
        "forecast_decision_count": forecast_count,
        "fallback_count": fallback_count,
        "source_files_sha256": source_files_sha256,
        "artifact_sha256": {
            "decision_spec.json": _sha256_file(output / "decision_spec.json"),
            "recommendations.csv": _sha256_file(output / "recommendations.csv"),
        },
        "environment": {"python": platform.python_version()},
        "source_sha256": _source_hashes(),
        "limitations": [
            "Recommendations are frozen from forecast-time outputs only; no outcomes were read during issuance",
            "Score outcomes separately with score_decisions; modifying later outcomes does not change these records",
        ],
    }
    write_json(output / "decision_manifest.json", manifest)
    return output


def issue_decisions_from_run(
    *,
    run_directory: str | Path,
    spec: DecisionSpec,
    output_directory: str | Path,
) -> Path:
    """Issue frozen per-origin recommendations from a saved distribution backtest run.

    Reads only manifest.json, predictions.csv, and predictive_distributions.json.
    Never reads evaluation.csv or scores.json so recommendations cannot depend on
    realized outcomes.
    """
    if not isinstance(spec, DecisionSpec):
        raise ValueError("spec must be a DecisionSpec")
    if spec.policy == "binary_cost_loss" and spec.threshold is None:
        raise ValueError("Extracting a binary event from a scalar predictive distribution requires spec.threshold")
    source = Path(run_directory).expanduser().absolute()
    manifest = _load_json(source / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format_version") != 3:
        raise ValueError("Expected a version-3 backtest run manifest")
    if manifest.get("status") not in ("completed", "failed"):
        raise ValueError("Decision issuance requires a finished backtest run")
    if not manifest.get("model", {}).get("supports_distribution", False):
        raise ValueError("Run model does not declare frozen full predictive distributions")
    if manifest.get("task", {}).get("horizon_days") != spec.horizon_days:
        raise ValueError("Run horizon_days does not match DecisionSpec.horizon_days")

    fields, predictions = _read_csv_rows(source / "predictions.csv")
    required_columns = {"origin_index", "issued_at", "target_time", "history_sha256", "status", "error"}
    if not required_columns.issubset(fields):
        raise ValueError("predictions.csv is missing required forecast-time columns")
    distributions = _load_json(source / "predictive_distributions.json")
    if not isinstance(distributions, list) or len(distributions) != len(predictions) or not predictions:
        raise ValueError("predictive_distributions.json does not align with predictions.csv")

    dataset = manifest["input"]["dataset"]
    snapshot_id = manifest["input"]["snapshot_id"]
    model_name = manifest["model"]["name"]
    quantile_level = (
        asymmetric_quantile_level(spec.under_cost, spec.over_cost)
        if spec.policy == "continuous_asymmetric"
        else None
    )
    recommendations = []
    seen_targets = set()
    for prediction, dist_row in zip(predictions, distributions):
        for key in ("origin_index", "issued_at", "target_time", "history_sha256", "status"):
            if str(dist_row.get(key)) != str(prediction[key]):
                raise ValueError(f"Distribution record mismatch on {key}")
        target_time = _normalize_target_time(prediction["target_time"])
        issued_at = _normalize_target_time(prediction["issued_at"], "issued_at")
        if target_time in seen_targets:
            raise ValueError(f"Duplicate target_time in predictions.csv: {target_time}")
        seen_targets.add(target_time)
        status = prediction["status"]
        error_text = prediction["error"] or ""
        event_prob = None
        if status == "passed":
            if spec.policy == "binary_cost_loss":
                event_prob = distribution_event_probability(
                    dist_row["distribution"], threshold=spec.threshold, comparison=spec.comparison,
                )
                action = "act" if decide(event_prob, spec.cost_loss_ratio) else "wait"
            else:
                action = distribution_quantile(dist_row["distribution"], quantile_level)
            decision_source = "forecast"
        else:
            action = spec.resolve_fallback_action()
            decision_source = "fallback"
        recommendations.append({
            "origin_index": int(prediction["origin_index"]),
            "issued_at": issued_at,
            "target_time": target_time,
            "dataset": dataset,
            "snapshot_id": snapshot_id,
            "history_sha256": prediction["history_sha256"],
            "model_name": model_name,
            "policy": spec.policy,
            "event_probability": event_prob,
            "cost_loss_ratio": spec.cost_loss_ratio,
            "under_cost": spec.under_cost,
            "over_cost": spec.over_cost,
            "action": action,
            "decision_source": decision_source,
            "forecast_status": status,
            "forecast_error": error_text,
        })

    source_files_sha256 = {
        name: _sha256_file(source / name)
        for name in ("manifest.json", "predictions.csv", "predictive_distributions.json")
    }
    return _write_issued_artifacts(
        Path(output_directory).expanduser().absolute(),
        spec=spec,
        recommendations=recommendations,
        source_kind="backtest_run",
        source_directory=source,
        input_identity=manifest["input"],
        model_identity=manifest["model"],
        source_files_sha256=source_files_sha256,
    )


def issue_decisions_from_probabilities(
    *,
    source_directory: str | Path,
    method: str,
    spec: DecisionSpec,
    output_directory: str | Path,
    probabilities_file: str = "matched-range.csv",
    dataset: str | None = None,
    model_name: str | None = None,
) -> Path:
    """Issue frozen per-origin recommendations from a saved probability CSV (e.g., NOAA range study).

    Reads only forecast-time columns (target date/time, issued_at, and the named
    probability column); outcome columns such as event or actual values are ignored.
    """
    if not isinstance(spec, DecisionSpec):
        raise ValueError("spec must be a DecisionSpec")
    if spec.policy != "binary_cost_loss":
        raise ValueError("Saved event probabilities only support binary_cost_loss decisions")
    method_name = _nonblank(method, "method")
    source = Path(source_directory).expanduser().absolute()
    csv_path = source / probabilities_file
    fields, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    target_column = "target_time" if "target_time" in fields else "date" if "date" in fields else None
    if target_column is None:
        raise ValueError("Probability CSV must contain a target_time or date column")
    probability_column = (
        f"{method_name}_probability" if f"{method_name}_probability" in fields
        else method_name if method_name in fields
        else None
    )
    if probability_column is None:
        raise ValueError(f"Probability CSV does not contain {method_name}_probability or {method_name}")
    if "issued_at" not in fields:
        raise ValueError("Probability CSV must contain an issued_at column")

    source_files_sha256 = {probabilities_file: _sha256_file(csv_path)}
    inputs_meta = {}
    if (source / "inputs.json").is_file():
        inputs_meta = _load_json(source / "inputs.json")
        source_files_sha256["inputs.json"] = _sha256_file(source / "inputs.json")

    resolved_dataset = dataset or (
        inputs_meta.get("dataset") if isinstance(inputs_meta, dict) else None
    ) or source.name
    snapshot_id = (
        inputs_meta.get("snapshot_id") if isinstance(inputs_meta, dict) and inputs_meta.get("snapshot_id")
        else source_files_sha256[probabilities_file]
    )
    resolved_model = model_name or f"saved-probabilities:{method_name}"

    recommendations = []
    seen_targets = set()
    for index, row in enumerate(rows):
        target_time = _normalize_target_time(row[target_column], target_column)
        issued_at = _normalize_target_time(row["issued_at"], "issued_at")
        if target_time in seen_targets:
            raise ValueError(f"Duplicate {target_column} in {csv_path}: {target_time}")
        seen_targets.add(target_time)
        raw_prob = row[probability_column]
        status = row.get("status", "passed") or "passed"
        error_text = row.get("error", "") or ""
        if status == "passed":
            prob = _finite_number(float(raw_prob), probability_column)
            action = "act" if decide(prob, spec.cost_loss_ratio) else "wait"
            decision_source = "forecast"
        else:
            prob = None
            action = spec.resolve_fallback_action()
            decision_source = "fallback"
        recommendations.append({
            "origin_index": int(row["origin_index"]) if row.get("origin_index") else index,
            "issued_at": issued_at,
            "target_time": target_time,
            "dataset": resolved_dataset,
            "snapshot_id": snapshot_id,
            "history_sha256": row.get("history_sha256") or "",
            "model_name": resolved_model,
            "policy": spec.policy,
            "event_probability": prob,
            "cost_loss_ratio": spec.cost_loss_ratio,
            "under_cost": None,
            "over_cost": None,
            "action": action,
            "decision_source": decision_source,
            "forecast_status": status,
            "forecast_error": error_text,
        })

    return _write_issued_artifacts(
        Path(output_directory).expanduser().absolute(),
        spec=spec,
        recommendations=recommendations,
        source_kind="saved_probabilities",
        source_directory=source,
        input_identity={
            "dataset": resolved_dataset,
            "snapshot_id": snapshot_id,
            "probabilities_file": probabilities_file,
            "probability_column": probability_column,
        },
        model_identity={"name": resolved_model, "method": method_name},
        source_files_sha256=source_files_sha256,
    )


def load_outcomes(outcomes_source: str | Path) -> tuple[dict[str, dict], dict]:
    """Load realized outcomes by UTC target_time from a run directory, NOAA directory, or CSV file."""
    path = Path(outcomes_source).expanduser().absolute()
    if path.is_dir():
        if (path / "evaluation.csv").is_file():
            csv_path = path / "evaluation.csv"
            kind = "backtest_run_evaluation"
        elif (path / "matched-range.csv").is_file():
            csv_path = path / "matched-range.csv"
            kind = "noaa_range_outcomes"
        else:
            raise ValueError(f"Directory {path} does not contain evaluation.csv or matched-range.csv")
    elif path.is_file():
        csv_path = path
        kind = "outcomes_csv"
    else:
        raise FileNotFoundError(path)

    fields, rows = _read_csv_rows(csv_path)
    target_column = "target_time" if "target_time" in fields else "date" if "date" in fields else None
    if target_column is None:
        raise ValueError("Outcomes CSV must contain a target_time or date column")
    actual_column = (
        "actual" if "actual" in fields
        else "actual_range_celsius" if "actual_range_celsius" in fields
        else None
    )
    has_event = "event" in fields
    if actual_column is None and not has_event:
        raise ValueError("Outcomes CSV must contain actual, actual_range_celsius, or event")

    outcomes = {}
    for row in rows:
        target_time = _normalize_target_time(row[target_column], target_column)
        if target_time in outcomes:
            raise ValueError(f"Duplicate outcome target_time: {target_time}")
        actual_value = None
        if actual_column is not None and row[actual_column] != "":
            actual_value = _finite_number(float(row[actual_column]), actual_column)
        event_value = None
        if has_event and row["event"] != "":
            parsed = int(row["event"])
            if parsed not in (0, 1) or str(parsed) != row["event"].strip():
                raise ValueError(f"Invalid event value for {target_time}: {row['event']!r}")
            event_value = parsed
        outcomes[target_time] = {
            "actual": actual_value,
            "event": event_value,
            "actual_available_at": row.get("actual_available_at") or "",
        }
    provenance = {
        "kind": kind,
        "source": str(path),
        "file": str(csv_path),
        "sha256": _sha256_file(csv_path),
        "row_count": len(outcomes),
    }
    return outcomes, provenance


def _load_verified_issuance(issued_directory: str | Path) -> tuple[Path, dict, DecisionSpec, list[dict]]:
    issued = Path(issued_directory).expanduser().absolute()
    manifest = _load_json(issued / "decision_manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("format_version") != 1
            or manifest.get("kind") != "decision_issuance" or manifest.get("status") != "issued"):
        raise ValueError("Expected a valid version-1 decision_issuance manifest")
    expected_hashes = manifest.get("artifact_sha256", {})
    for filename in ("decision_spec.json", "recommendations.csv"):
        if expected_hashes.get(filename) != _sha256_file(issued / filename):
            raise ValueError(f"Issued artifact digest mismatch for {filename}")
    spec = DecisionSpec.load(issued / "decision_spec.json")
    if spec.sha256 != manifest.get("spec_sha256"):
        raise ValueError("Issued DecisionSpec digest mismatch")
    fields, rows = _read_csv_rows(issued / "recommendations.csv")
    if tuple(fields) != RECOMMENDATION_FIELDS or len(rows) != manifest.get("case_count") or not rows:
        raise ValueError("Issued recommendations.csv schema or row count mismatch")
    return issued, manifest, spec, rows


def score_decisions(
    *,
    issued_directory: str | Path,
    outcomes_source: str | Path,
    output_directory: str | Path,
    sensitivity_alphas: tuple[float, ...] | None = None,
) -> Path:
    """Link later outcomes to frozen recommendations and save expense evaluation to a new directory.

    Never modifies issued_directory. Changing outcomes produces new expenses while
    leaving the issued recommendations untouched.
    """
    issued, manifest, spec, recommendations = _load_verified_issuance(issued_directory)
    outcomes, outcome_provenance = load_outcomes(outcomes_source)

    eval_rows = []
    forecast_expenses = []
    baseline_expenses = []
    perfect_expenses = []
    binary_prob_event_pairs = []
    continuous_action_actual_pairs = []
    hits = false_alarms = misses = correct_negatives = 0
    fallback_count = 0

    for row in recommendations:
        target_time = row["target_time"]
        if target_time not in outcomes:
            raise ValueError(f"Missing realized outcome for issued target_time {target_time}")
        outcome = outcomes[target_time]
        actual = outcome["actual"]
        if row["decision_source"] == "fallback":
            fallback_count += 1

        if spec.policy == "binary_cost_loss":
            event = outcome["event"]
            if event is None:
                if spec.threshold is None or actual is None:
                    raise ValueError(
                        f"Binary scoring for {target_time} needs an explicit event or both actual and spec.threshold"
                    )
                event = event_indicator(actual, threshold=spec.threshold, comparison=spec.comparison)
            elif spec.threshold is not None and actual is not None:
                derived = event_indicator(actual, threshold=spec.threshold, comparison=spec.comparison)
                if derived != event:
                    raise ValueError(f"Outcome event disagrees with actual and spec.threshold for {target_time}")
            if row["action"] not in ("act", "wait"):
                raise ValueError(f"Invalid binary action in recommendations.csv: {row['action']!r}")
            acted = row["action"] == "act"
            base_acted = decide(spec.base_rate, spec.cost_loss_ratio)
            case_expense = expense(acted, event, spec.cost_loss_ratio)
            base_expense = expense(base_acted, event, spec.cost_loss_ratio)
            perf_expense = expense(bool(event), event, spec.cost_loss_ratio)
            if acted and event:
                category = "hit"
                hits += 1
            elif acted and not event:
                category = "false_alarm"
                false_alarms += 1
            elif not acted and event:
                category = "miss"
                misses += 1
            else:
                category = "correct_negative"
                correct_negatives += 1
            if row["decision_source"] == "forecast":
                binary_prob_event_pairs.append((float(row["event_probability"]), event))
            eval_rows.append({
                "origin_index": int(row["origin_index"]),
                "issued_at": row["issued_at"],
                "target_time": target_time,
                "policy": spec.policy,
                "event_probability": float(row["event_probability"]) if row["event_probability"] != "" else None,
                "action": row["action"],
                "decision_source": row["decision_source"],
                "actual": actual,
                "actual_available_at": outcome["actual_available_at"],
                "event": event,
                "outcome_category": category,
                "expense": case_expense,
                "baseline_expense": base_expense,
                "perfect_expense": perf_expense,
            })
        else:
            if actual is None:
                raise ValueError(f"Continuous asymmetric scoring requires a numeric actual for {target_time}")
            action_value = _finite_number(float(row["action"]), "action")
            case_expense = asymmetric_expense(
                action_value, actual, under_cost=spec.under_cost, over_cost=spec.over_cost,
            )
            base_expense = asymmetric_expense(
                spec.baseline_action, actual, under_cost=spec.under_cost, over_cost=spec.over_cost,
            )
            perf_expense = 0.0
            category = "under" if actual > action_value else "over" if actual < action_value else "exact"
            continuous_action_actual_pairs.append((action_value, actual))
            eval_rows.append({
                "origin_index": int(row["origin_index"]),
                "issued_at": row["issued_at"],
                "target_time": target_time,
                "policy": spec.policy,
                "event_probability": None,
                "action": action_value,
                "decision_source": row["decision_source"],
                "actual": actual,
                "actual_available_at": outcome["actual_available_at"],
                "event": None,
                "outcome_category": category,
                "expense": case_expense,
                "baseline_expense": base_expense,
                "perfect_expense": perf_expense,
            })
        forecast_expenses.append(case_expense)
        baseline_expenses.append(base_expense)
        perfect_expenses.append(perf_expense)

    count = len(eval_rows)
    expense_summary = summarize_expenses(forecast_expenses)
    baseline_summary = summarize_expenses(baseline_expenses)
    mean_forecast = expense_summary["mean_expense"]
    mean_baseline = baseline_summary["mean_expense"]
    mean_perfect = math.fsum(perfect_expenses) / count
    denominator = mean_baseline - mean_perfect

    sensitivity = None
    if spec.policy == "binary_cost_loss":
        observed_events = hits + misses
        summary_block = {
            "alpha": spec.cost_loss_ratio,
            "base_rate": spec.base_rate,
            "cases": count,
            "observed_events": observed_events,
            "observed_event_rate": observed_events / count,
            "action_count": hits + false_alarms,
            "action_rate": (hits + false_alarms) / count,
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "correct_negatives": correct_negatives,
            "mean_forecast_expense": mean_forecast,
            "mean_climatology_expense": mean_baseline,
            "mean_perfect_expense": mean_perfect,
            "relative_value": (mean_baseline - mean_forecast) / denominator if denominator else None,
            "relative_value_reason": None if denominator else "Climatology and perfect forecast have equal mean expense.",
            "fallback_count": fallback_count,
        }
        if fallback_count == 0:
            grid = DEFAULT_SENSITIVITY_ALPHAS if sensitivity_alphas is None else tuple(sensitivity_alphas)
            alphas = tuple(sorted({float(a) for a in grid} | {spec.cost_loss_ratio}))
            sensitivity = {
                "status": "completed",
                "predeclared_alpha": spec.cost_loss_ratio,
                "note": "Descriptive sensitivity across cost-loss ratios; alpha was predeclared in DecisionSpec",
                "curve": value_curve(binary_prob_event_pairs, alphas=alphas, base_rate=spec.base_rate),
            }
        else:
            sensitivity = {
                "status": "omitted_due_to_fallbacks",
                "predeclared_alpha": spec.cost_loss_ratio,
                "note": "Sensitivity curve requires forecast probabilities on all origins",
                "curve": [],
            }
    else:
        asymmetric_result = evaluate_asymmetric_decisions(
            continuous_action_actual_pairs,
            under_cost=spec.under_cost,
            over_cost=spec.over_cost,
            baseline_action=spec.baseline_action,
        )
        signed_errors = [action - actual for action, actual in continuous_action_actual_pairs]
        ordered_errors = sorted(signed_errors)
        mid = count // 2
        median_bias = (
            ordered_errors[mid] if count % 2 == 1
            else 0.5 * (ordered_errors[mid - 1] + ordered_errors[mid])
        )
        summary_block = {
            **{key: value for key, value in asymmetric_result.items()
               if key not in ("expense_distribution", "baseline_expense_distribution")},
            "mean_action_minus_actual": math.fsum(signed_errors) / count,
            "median_action_minus_actual": median_bias,
            "fallback_count": fallback_count,
        }

    checks = [
        {"name": "issued_artifacts_valid", "required": True, "status": "passed"},
        {"name": "outcomes_matched_complete", "required": True, "status": "passed"},
        {"name": "all_decisions_scored", "required": True, "status": "passed"},
        {"name": "all_decisions_from_forecasts_without_fallback", "required": False,
         "status": "passed" if fallback_count == 0 else "failed"},
    ]
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "decision_evaluation.csv", EVALUATION_FIELDS, eval_rows)
    write_json(output / "checks.json", checks)
    write_json(output / "review.json", {
        "status": "pending",
        "decision": None,
        "limitations": [
            "Realized expense is descriptive for this cohort, threshold/cost ratio, and policy",
            "Passing computational checks does not approve the forecast model or decision rule for live use",
            "Inspect top-5% expense concentration and sensitivity before interpreting mean savings",
        ],
    })
    summary_payload = {
        "format_version": 1,
        "kind": "decision_evaluation",
        "status": "completed",
        "scored_at": datetime.now(UTC).isoformat(),
        "issued_directory": str(issued),
        "issued_sha256": {
            "decision_manifest.json": _sha256_file(issued / "decision_manifest.json"),
            "decision_spec.json": _sha256_file(issued / "decision_spec.json"),
            "recommendations.csv": _sha256_file(issued / "recommendations.csv"),
        },
        "outcomes": outcome_provenance,
        "input": manifest["input"],
        "model": manifest["model"],
        "spec": spec.to_dict(),
        "spec_sha256": spec.sha256,
        "target_start": eval_rows[0]["target_time"],
        "target_end": eval_rows[-1]["target_time"],
        "summary": summary_block,
        "expense_distribution": expense_summary,
        "baseline_expense_distribution": baseline_summary,
        "sensitivity": sensitivity,
        "environment": {"python": platform.python_version()},
        "source_sha256": _source_hashes(),
    }
    write_json(output / "decision_summary.json", summary_payload)
    return output


def _newey_west_se(differences: list[float]) -> tuple[float | None, float | None, int]:
    """Return (naive_se, bartlett_hac_se, max_lag) for a time-ordered paired difference series."""
    n = len(differences)
    if n < 2:
        return None, None, 0
    avg = math.fsum(differences) / n
    centered = [value - avg for value in differences]
    gamma0 = math.fsum(value * value for value in centered) / n
    naive_var = math.fsum(value * value for value in centered) / (n - 1)
    naive_se = math.sqrt(max(naive_var, 0.0) / n)
    max_lag = min(n - 1, int(n ** (1.0 / 3.0)))
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_lag = math.fsum(centered[i] * centered[i - lag] for i in range(lag, n)) / n
        long_run_var += 2.0 * weight * gamma_lag
    hac_se = math.sqrt(max(long_run_var, 0.0) / n)
    return naive_se, hac_se, max_lag


def plot_decision_scores(scored_directory: str | Path) -> Path:
    """Draw decision_scores.png from a saved scored decision directory; refuse overwrite."""
    directory = Path(scored_directory).expanduser().absolute()
    summary = _load_json(directory / "decision_summary.json")
    if summary.get("kind") != "decision_evaluation" or summary.get("status") != "completed":
        raise ValueError("Expected a completed decision_evaluation directory")
    _, rows = _read_csv_rows(directory / "decision_evaluation.csv")
    times = [datetime.fromisoformat(row["target_time"]) for row in rows]
    cum_forecast = np.cumsum([float(row["expense"]) for row in rows])
    cum_baseline = np.cumsum([float(row["baseline_expense"]) for row in rows])
    cum_perfect = np.cumsum([float(row["perfect_expense"]) for row in rows])

    sensitivity = summary.get("sensitivity")
    has_curve = isinstance(sensitivity, dict) and sensitivity.get("status") == "completed" and sensitivity.get("curve")
    figure = Figure(figsize=(9.5, 7.2 if has_curve else 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(2 if has_curve else 1, 1)
    top_ax = axes_list[0] if has_curve else axes_list
    top_ax.plot(times, cum_forecast, label=f"Model: {summary['model']['name']}", linewidth=1.6, color="#28547B")
    top_ax.plot(times, cum_baseline, label="Calibration baseline", linestyle="--", linewidth=1.2, color="#485865")
    top_ax.plot(times, cum_perfect, label="Perfect foresight", linestyle=":", linewidth=1.2, color="#2D7A4D")
    top_ax.set(
        title=f"Cumulative decision expense — {summary['spec']['name']} ({summary['model']['name']})",
        xlabel="Target date (UTC)",
        ylabel="Cumulative expense",
    )
    top_ax.grid(axis="y", alpha=0.22)
    top_ax.legend(frameon=False)

    if has_curve:
        bottom_ax = axes_list[1]
        curve = sensitivity["curve"]
        alphas = [row["alpha"] for row in curve]
        values = [row["relative_value"] if row["relative_value"] is not None else math.nan for row in curve]
        bottom_ax.plot(alphas, values, marker="o", markersize=4, color="#28547B", label="Relative economic value")
        predeclared = sensitivity["predeclared_alpha"]
        predeclared_row = next((row for row in curve if math.isclose(row["alpha"], predeclared)), None)
        if predeclared_row and predeclared_row["relative_value"] is not None:
            bottom_ax.scatter(
                [predeclared], [predeclared_row["relative_value"]],
                color="#B65D24", s=65, zorder=5, label=f"Predeclared ratio ({predeclared:g})",
            )
        bottom_ax.axhline(0.0, color="#485865", linestyle="--", linewidth=1, label="Climatology")
        bottom_ax.axvline(summary["summary"]["base_rate"], color="#777777", linestyle=":", linewidth=1,
                          label=f"Calibration base rate ({summary['summary']['base_rate']:.1%})")
        bottom_ax.set(
            title="Descriptive sensitivity across cost-loss ratios (ratio predeclared in DecisionSpec)",
            xlabel="Cost-loss ratio (C / L)",
            ylabel="Relative economic value",
            xlim=(0, 1),
        )
        bottom_ax.grid(axis="y", alpha=0.22)
        bottom_ax.legend(frameon=False)

    figure.autofmt_xdate()
    destination = directory / "decision_scores.png"
    with destination.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return destination


def compare_decision_scores(
    first_scored: str | Path,
    second_scored: str | Path,
    *,
    output_directory: str | Path,
) -> Path:
    """Compare two scored decision directories on identical specs, dates, and outcomes."""
    dirs = [Path(path).expanduser().absolute() for path in (first_scored, second_scored)]
    summaries, tables = [], []
    for directory in dirs:
        summary = _load_json(directory / "decision_summary.json")
        if (not isinstance(summary, dict) or summary.get("format_version") != 1
                or summary.get("kind") != "decision_evaluation" or summary.get("status") != "completed"):
            raise ValueError(f"Expected a completed decision_evaluation directory: {directory}")
        _, rows = _read_csv_rows(directory / "decision_evaluation.csv")
        if not rows:
            raise ValueError(f"Empty decision_evaluation.csv in {directory}")
        summaries.append(summary)
        tables.append(rows)

    if summaries[0]["spec_sha256"] != summaries[1]["spec_sha256"]:
        raise ValueError("Decision comparison requires identical DecisionSpec definitions")
    if len(tables[0]) != len(tables[1]):
        raise ValueError("Decision comparison requires matching evaluation cohorts")
    for first_row, second_row in zip(tables[0], tables[1]):
        if (first_row["target_time"] != second_row["target_time"]
                or first_row["actual"] != second_row["actual"]
                or first_row["event"] != second_row["event"]):
            raise ValueError("Decision comparison targets and observed outcomes must match exactly")

    differences = []
    first_better = second_better = ties = action_agree = 0
    for first_row, second_row in zip(tables[0], tables[1]):
        diff = float(first_row["expense"]) - float(second_row["expense"])
        differences.append(diff)
        if diff < 0:
            first_better += 1
        elif diff > 0:
            second_better += 1
        else:
            ties += 1
        if first_row["action"] == second_row["action"]:
            action_agree += 1

    count = len(differences)
    mean_diff = math.fsum(differences) / count
    ordered_diff = sorted(differences)
    mid = count // 2
    median_diff = ordered_diff[mid] if count % 2 == 1 else 0.5 * (ordered_diff[mid - 1] + ordered_diff[mid])
    abs_diffs = sorted((abs(value) for value in differences), reverse=True)
    total_abs = math.fsum(abs_diffs)
    top_count = max(1, math.ceil(0.05 * count))
    top_5pct_abs_share = math.fsum(abs_diffs[:top_count]) / total_abs if total_abs > 0 else 0.0
    naive_se, hac_se, hac_lag = _newey_west_se(differences)

    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    comparison = {
        "format_version": 1,
        "kind": "decision_comparison",
        "created_at": datetime.now(UTC).isoformat(),
        "spec": summaries[0]["spec"],
        "spec_sha256": summaries[0]["spec_sha256"],
        "case_count": count,
        "target_start": tables[0][0]["target_time"],
        "target_end": tables[0][-1]["target_time"],
        "difference_direction": "first_minus_second; negative means first has lower expense",
        "runs": [
            {
                "path": str(directory),
                "model": summary["model"],
                "summary": summary["summary"],
                "expense_distribution": summary["expense_distribution"],
            }
            for directory, summary in zip(dirs, summaries)
        ],
        "paired_differences": {
            "mean_expense_difference": mean_diff,
            "median_expense_difference": median_diff,
            "mean_difference_naive_se": naive_se,
            "mean_difference_hac_se": hac_se,
            "hac_bartlett_lag": hac_lag,
            "first_better_count": first_better,
            "second_better_count": second_better,
            "tie_count": ties,
            "first_win_rate": first_better / count,
            "second_win_rate": second_better / count,
            "tie_rate": ties / count,
            "action_agreement_count": action_agree,
            "action_agreement_rate": action_agree / count,
            "top_5pct_case_count": top_count,
            "top_5pct_abs_difference_share": top_5pct_abs_share,
        },
        "source_sha256": _source_hashes(),
        "interpretation": "Descriptive paired decision comparison; inspect concentration and uncertainty before selecting a policy",
    }
    write_json(output / "decision_comparison.json", comparison)

    times = [datetime.fromisoformat(row["target_time"]) for row in tables[0]]
    cumulative_diff = np.cumsum(differences)
    figure = Figure(figsize=(9.5, 6.4), layout="constrained")
    FigureCanvasAgg(figure)
    top_ax, bottom_ax = figure.subplots(2, 1)
    first_label = f"1: {summaries[0]['model']['name']}"
    second_label = f"2: {summaries[1]['model']['name']}"
    top_ax.plot(times, cumulative_diff, color="#28547B", linewidth=1.6)
    top_ax.axhline(0.0, color="#485865", linestyle="--", linewidth=1)
    top_ax.set(
        title=f"Cumulative expense difference ({first_label} minus {second_label}; below 0 favors 1)",
        xlabel="Target date (UTC)",
        ylabel="Cumulative expense diff",
    )
    top_ax.grid(axis="y", alpha=0.22)

    labels = [first_label, second_label, "Calibration baseline"]
    means = [
        summaries[0]["expense_distribution"]["mean_expense"],
        summaries[1]["expense_distribution"]["mean_expense"],
        summaries[0]["baseline_expense_distribution"]["mean_expense"],
    ]
    p95s = [
        summaries[0]["expense_distribution"]["p95_expense"],
        summaries[1]["expense_distribution"]["p95_expense"],
        summaries[0]["baseline_expense_distribution"]["p95_expense"],
    ]
    positions = np.arange(len(labels))
    width = 0.35
    bottom_ax.bar(positions - width / 2, means, width, label="Mean expense", color="#28547B")
    bottom_ax.bar(positions + width / 2, p95s, width, label="95th percentile expense", color="#B65D24")
    bottom_ax.set_xticks(positions, labels)
    bottom_ax.set(title="Mean vs. tail (95th percentile) decision expense", ylabel="Expense")
    bottom_ax.grid(axis="y", alpha=0.22)
    bottom_ax.legend(frameon=False)
    figure.autofmt_xdate()
    with (output / "decision_comparison.png").open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return output
