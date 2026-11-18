"""Optional per-request fit evidence, independent of model family.

Adapters return evidence; the runner supplies input/history identity and writes
it before checking acceptance. Evidence is trusted Python data, not a sandbox.
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

from model_pipelining.models import Forecast


@dataclass(frozen=True)
class FitEvidence:
    metadata: dict
    checks: tuple[dict, ...]
    diagnostics: tuple[dict, ...] = ()
    draws: tuple[dict, ...] = ()


@dataclass(frozen=True)
class FittedForecast:
    forecast: Forecast
    evidence: FitEvidence


def assess_checks(checks, *, required_names: tuple[str, ...] = ()) -> tuple[bool, tuple[dict, ...]]:
    """Validate supplied checks and independently enforce caller-required names.

    Missing checks fail requirements. A provider marking a requested check as
    optional or not applicable cannot bypass the caller's requirement.
    """
    if not isinstance(checks, (tuple, list)) or not checks or not any(
            isinstance(item, dict) and item.get("required") is True for item in checks):
        raise ValueError("Evidence needs at least one required check")
    if (not isinstance(required_names, tuple)
            or any(not isinstance(name, str) or not name.strip() for name in required_names)
            or len(set(required_names)) != len(required_names)):
        raise ValueError("Required check names must be unique nonblank names")
    names = {}
    for item in checks:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip()
                or item["name"] in names or type(item.get("required")) is not bool
                or item.get("status") not in ("passed", "failed", "not_applicable")
                or (item["required"] and item["status"] == "not_applicable")):
            raise ValueError("Malformed evidence check")
        names[item["name"]] = item
    requirements = tuple({"name": name, "required": True,
        "status": "passed" if names.get(name, {}).get("status") == "passed" else "failed",
        "reason": "Missing required check" if name not in names else None} for name in required_names)
    passed = (all(item["status"] == "passed" for item in checks if item["required"])
              and all(item["status"] == "passed" for item in requirements))
    return passed, requirements


def save_fit_evidence(result: FittedForecast, directory: Path, *, identity: dict,
                      required_checks: tuple[str, ...] = ()) -> bool:
    """Save finite JSON/CSV evidence and return whether required checks passed.

    Missing/empty required checks cannot pass. Unsupported or malformed evidence
    raises. I/O errors propagate to the runner. Nonfinite diagnostic estimates
    must be represented as null plus a failed check by the adapter.
    """
    if not isinstance(result, FittedForecast) or not isinstance(result.evidence, FitEvidence):
        raise ValueError("fit() must return a FittedForecast with FitEvidence")
    evidence = result.evidence
    if not isinstance(evidence.metadata, dict):
        raise ValueError("Fit metadata must be a dictionary")
    if any(not isinstance(item, dict) for item in evidence.checks + evidence.diagnostics + evidence.draws):
        raise ValueError("Fit checks, diagnostics, and draws must contain dictionaries")
    passed, requirements = assess_checks(evidence.checks, required_names=required_checks)
    summary = {"format_version": 1, "status": "completed" if passed else "failed",
               "identity": identity, "metadata": evidence.metadata,
               "checks": evidence.checks, "diagnostics": evidence.diagnostics}
    if requirements:
        summary["experiment_requirements"] = requirements
    # Validate all serialization before creating the directory.
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    json.dumps(evidence.draws, allow_nan=False)
    if evidence.draws and any(set(row) != set(evidence.draws[0]) for row in evidence.draws):
        raise ValueError("Fit draw rows must share one schema")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "fit.json").write_text(summary_text, encoding="utf-8")
    if evidence.draws:
        with (directory / "draws.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evidence.draws[0]))
            writer.writeheader()
            writer.writerows(evidence.draws)
    return passed
