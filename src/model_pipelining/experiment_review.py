"""Read-only assembly of compatible forecast, evidence and calibration artifacts."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path

from model_pipelining.artifacts import write_json
from model_pipelining.evidence_workflows import REQUIRED_WORKFLOW_CHECKS, WORKFLOW_MANIFESTS
from model_pipelining.experiments import ExperimentSpec, ForecastCase, ObservationVersion
from model_pipelining.inference import assess_checks
from model_pipelining.distribution_scoring import read_saved_distribution_scores


_FORECAST_CHECKS = ("input_schema_and_daily_calendar", "eligible_history_and_target_alignment",
                    "forecast_outputs_valid", "all_expected_cases_scored",
                    "declared_case_plan_and_release_eligibility")


def _json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(directory):
    """Snapshot current files without following links into another artifact tree."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Not a regular artifact directory: {directory}")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in artifact directory: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = _digest(path)
    return files


def _inside(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact link must be a relative path")
    result = directory / relative
    if ".." in Path(relative).parts or not result.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Artifact link escapes its source directory")
    return result


def _aware(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("Artifact timestamps must be timezone-aware")
    return result.astimezone(UTC)


def _context(context, forecast, specification):
    if not isinstance(context, dict):
        raise ValueError("Missing selected-history context")
    for key in ("dataset", "snapshot_id"):
        if context.get("input", {}).get(key) != forecast["input"][key]:
            raise ValueError(f"Evidence {key} differs from forecast input")
    expected = replace(specification, evaluation=replace(specification.evaluation, role="development"))
    if ExperimentSpec.from_dict(context["experiment"]) != expected:
        raise ValueError("Evidence task/evaluation/evidence requirements differ from forecast")
    cutoff = _aware(context["issued_at"])
    if expected.evaluation.final_start is not None and cutoff >= expected.evaluation.final_start:
        raise ValueError("Evidence cutoff enters the final period")
    history = []
    for item in context["selected_history"]:
        observation = ObservationVersion(_aware(item["timestamp"]), item["value"], _aware(item["available_at"]))
        if (isinstance(observation.value, bool) or not isinstance(observation.value, (int, float))
                or not math.isfinite(observation.value)
                or observation.available_at < observation.timestamp
                or max(observation.timestamp, observation.available_at) > cutoff
                or (history and observation.timestamp <= history[-1].timestamp)):
            raise ValueError("Evidence selected history violates its declared cutoff/order")
        history.append(observation)
    if len(history) < specification.task.minimum_history:
        raise ValueError("Evidence selected history is too short")
    case = ForecastCase(0, cutoff, cutoff + timedelta(days=specification.task.horizon_days), tuple(history))
    if case.history_sha256 != context["history_sha256"]:
        raise ValueError("Evidence selected-history hash mismatch")
    return cutoff


def _checks(directory, names):
    passed, requirements = assess_checks(_json(directory / "checks.json"), required_names=tuple(names))
    return passed, requirements


def _model_identity(record, model):
    if not isinstance(record, dict) or record.get("name") != model["name"] or record.get("configuration") != model["configuration"]:
        raise ValueError("Model name or expanded configuration differs from forecast")


def _forecast(directory, manifest, specification):
    raw = json.dumps(manifest["experiment"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != manifest["experiment_sha256"]:
        raise ValueError("Forecast experiment hash mismatch")
    predictions = _csv(directory / "predictions.csv")
    if not predictions or len({row["target_time"] for row in predictions}) != len(predictions):
        raise ValueError("Missing or duplicated forecast cases")
    names = list(_FORECAST_CHECKS)
    fit_required = manifest["model"].get("supports_fit_evidence", False) or bool(specification.evidence.required_fit_checks)
    if fit_required:
        names.append("required_fit_evidence_and_checks")
    if specification.evaluation.benchmarks:
        names.append("all_requested_benchmarks_scored")
    if manifest["model"].get("supports_quantiles"):
        names.extend(("requested_observation_quantiles_valid", "all_uncertainty_cases_scored"))
    if manifest["model"].get("supports_highest_density"):
        names.append("requested_highest_density_regions_valid")
        if "all_uncertainty_cases_scored" not in names:
            names.append("all_uncertainty_cases_scored")
    if manifest["model"].get("supports_distribution"):
        names.extend(("full_predictive_distributions_valid", "all_distribution_cases_scored"))
    passed, requirements = _checks(directory, names)
    passed = passed and all(row["status"] == "passed" for row in predictions)
    if manifest["model"].get("supports_distribution") and passed:
        read_saved_distribution_scores(directory, manifest, predictions, _csv(directory/"evaluation.csv"))
    if fit_required:
        fits = _csv(directory / "fit_index.csv")
        if (len(fits) != len(predictions) or len({row["origin_index"] for row in fits}) != len(fits)
                or [row["target_time"] for row in fits] != [row["target_time"] for row in predictions]):
            raise ValueError("Fit index does not cover every forecast case")
        for row, prediction in zip(fits, predictions):
            if not row["fit_directory"]:
                passed = False
                continue
            fit = _json(_inside(directory, row["fit_directory"]) / "fit.json")
            identity = fit["identity"]
            for key in ("dataset", "snapshot_id"):
                if identity.get(key) != manifest["input"][key]:
                    raise ValueError("Fit input identity mismatch")
            for key in ("issued_at", "target_time", "history_sha256"):
                if identity.get(key) != prediction[key]:
                    raise ValueError("Fit forecast/history identity mismatch")
            if (identity.get("origin_index") != int(prediction["origin_index"])
                    or identity.get("training_count") != int(prediction["training_count"])
                    or row["origin_index"] != prediction["origin_index"]
                    or row["history_sha256"] != prediction["history_sha256"]):
                raise ValueError("Fit index/origin identity mismatch")
            if fit["metadata"].get("configuration") != manifest["model"]["configuration"]:
                raise ValueError("Fit model configuration mismatch")
            fit_passed, _ = assess_checks(fit["checks"], required_names=specification.evidence.required_fit_checks)
            passed = passed and fit_passed and fit.get("status") == "completed" and row["status"] == "passed"
    return {"passed": passed and manifest["status"] == "completed", "check_requirements": requirements,
            "case_count": len(predictions), "target_start": predictions[0]["target_time"],
            "target_end": predictions[-1]["target_time"], "scores": _json(directory / "scores.json")}


def _evidence(directory, manifest, forecast, specification):
    if manifest.get("format_version") != 1 or manifest.get("kind") != "experiment_evidence":
        raise ValueError("Expected version-1 experiment evidence")
    _model_identity(manifest["model"], forecast["model"])
    _context(manifest["context"], forecast, specification)
    plan = specification.to_dict()["evidence"]
    if manifest.get("plan") != plan:
        raise ValueError("Evidence plan differs from forecast")
    outcomes = manifest.get("workflows", [])
    if [row["name"] for row in outcomes] != list(specification.evidence.required_workflows):
        raise ValueError("Evidence outcomes omit, duplicate or reorder required workflows")
    bundle_passed, _ = _checks(directory, specification.evidence.required_workflows)
    results, versions = {}, [manifest]
    for row in outcomes:
        name = row["name"]
        if name not in WORKFLOW_MANIFESTS or row.get("directory") != name or row.get("manifest") != WORKFLOW_MANIFESTS[name]:
            raise ValueError("Invalid evidence workflow artifact link")
        child_directory = _inside(directory, row["directory"])
        if not child_directory.exists():
            results[name] = {"status": "failed", "error": row.get("error") or "Child evidence is missing"}
            continue
        receipts = row.get("artifact_sha256")
        if not isinstance(receipts, dict) or set(receipts) != {WORKFLOW_MANIFESTS[name], "checks.json"}:
            raise ValueError("Missing child manifest/check receipts")
        for filename, expected in receipts.items():
            if _digest(_inside(child_directory, filename)) != expected:
                raise ValueError("Child evidence receipt mismatch")
        child = _json(child_directory / WORKFLOW_MANIFESTS[name])
        versions.append(child)
        if child.get("format_version") != 1:
            raise ValueError("Unexpected child evidence format version")
        if child.get("kind") != f"{child.get('family')}_{name}":
            # Pre-provider artifacts use the same kind without family metadata.
            if child.get("kind") not in (f"normal_mean_variance_{name}", f"local_level_variance_{name}"):
                raise ValueError("Unexpected child evidence format/kind")
        _context(child["context"], forecast, specification)
        if (child["context"]["history_sha256"] != manifest["context"]["history_sha256"]
                or child["context"]["issued_at"] != manifest["context"]["issued_at"]):
            raise ValueError("Child evidence history/cutoff differs from bundle")
        configuration = child.get("configuration") if name != "prior_influence" else child.get("configurations", {}).get("reference")
        _model_identity({"name": forecast["model"]["name"], "configuration": configuration}, forecast["model"])
        if name == "predictive_checks" and (child.get("replications") != plan["replications"] or child.get("seed") != plan["seed"]):
            raise ValueError("Predictive simulation settings mismatch")
        if name == "prior_influence" and child.get("interval_level") != plan["interval_level"]:
            raise ValueError("Prior-comparison interval setting mismatch")
        passed, requirements = _checks(child_directory, REQUIRED_WORKFLOW_CHECKS[name])
        results[name] = {"status": "passed" if passed and child.get("status") == "completed" and row.get("status") == "passed" else "failed",
                         "check_requirements": requirements, "cutoff": child["context"]["issued_at"]}
    return results, versions, bundle_passed and manifest["status"] == "completed" and all(
        item["status"] == "passed" for item in results.values())


def _calibration(directory, manifest, forecast):
    if manifest.get("format_version") != 1 or manifest.get("kind") != "simulation_calibration":
        raise ValueError("Expected version-1 simulation calibration")
    _model_identity(manifest["model"], forecast["model"])
    passed, requirements = _checks(directory, ("all_calibration_trials_computed",))
    parameters, trials = manifest["parameters"], manifest["trials"]
    if (type(trials) is not int or trials < 1 or not isinstance(parameters, list) or not parameters
            or any(not isinstance(name, str) or not name for name in parameters) or len(set(parameters)) != len(parameters)):
        raise ValueError("Invalid calibration trial/parameter declaration")
    rows = _csv(directory / "calibration.csv")
    identities = {(int(row["trial"]), row["parameter"]) for row in rows}
    expected = {(index, name) for index in range(trials) for name in parameters}
    if len(rows) != len(expected) or identities != expected:
        raise ValueError("Calibration rows omit or duplicate trials/parameters")
    passed = passed and all(row["status"] == "passed" for row in rows)
    return {"passed": passed and manifest.get("status") == "completed", "check_requirements": requirements,
            "trials": trials, "seed": manifest["seed"], "observation_count": manifest["observation_count"],
            "method": manifest["method"], "summary": _json(directory / "summary.json"),
            "scope": "Prior/data model-level exercise; not external-data or every-history-length validation"}


def build_experiment_review(*, forecast_directory, output_directory, evidence_directory=None,
                            calibration_directories=(), require_calibration=False):
    """Build JSON/Markdown from saved artifacts; never execute a model or mutate a run.

    A version-3 forecast is the identity anchor. Evidence may describe an earlier
    development cutoff, including when the forecast has final role. Calibration
    must match the exact model/configuration but has no observed-input identity.
    Missing/failed/incompatible supplied evidence stays visible in the report.
    """
    if type(require_calibration) is not bool or not isinstance(calibration_directories, (tuple, list)):
        raise ValueError("Use a boolean calibration requirement and a list/tuple of directories")
    forecast_path = Path(forecast_directory).expanduser().absolute()
    output = Path(output_directory).expanduser().absolute()
    paths = [("forecast", forecast_path)]
    if evidence_directory is not None:
        paths.append(("evidence", Path(evidence_directory).expanduser().absolute()))
    paths.extend((f"calibration-{index}", Path(path).expanduser().absolute()) for index, path in enumerate(calibration_directories))
    if len({path.resolve() for _, path in paths}) != len(paths):
        raise ValueError("Artifact directories must be distinct")
    if any(output.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(output.resolve()) for _, path in paths):
        raise ValueError("Review output must be separate from every source artifact tree")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if forecast_path.is_symlink():
        raise ValueError("Forecast artifact root cannot be a symlink")
    forecast = _json(forecast_path / "manifest.json")
    if not isinstance(forecast, dict) or forecast.get("format_version") != 3 or forecast.get("status") not in ("completed", "failed"):
        raise ValueError("A finished version-3 forecast with explicit experiment metadata is required")
    specification = ExperimentSpec.from_dict(forecast["experiment"])
    sources, versions, inventories = [], [], []
    workflow_results = {}
    for kind, directory in paths:
        record = {"kind": kind, "path": str(directory), "validation_status": "failed", "error": None,
                  "computation_status": None, "details": None}
        try:
            inventory = _inventory(directory)
            inventories.append((directory, inventory))
            record["artifact_sha256"] = inventory
            manifest_name = "manifest.json" if kind == "forecast" else "evidence.json" if kind == "evidence" else "calibration.json"
            manifest = _json(directory / manifest_name)
            record["computation_status"] = manifest.get("status")
            if manifest.get("status") not in ("completed", "failed"):
                raise ValueError("Source computation is not finished")
            versions.append((kind, manifest))
            if kind == "forecast":
                if manifest != forecast:
                    raise ValueError("Forecast manifest changed during review initialization")
                record["details"] = _forecast(directory, manifest, specification)
            elif kind == "evidence":
                workflow_results, children, passed = _evidence(directory, manifest, forecast, specification)
                versions.extend((f"{kind}-child-{index}", child) for index, child in enumerate(children[1:]))
                record["details"] = {"passed": passed, "workflows": workflow_results, "cutoff": manifest["context"]["issued_at"],
                                     "scope": "Evidence applies to its explicit selected development history/cutoff"}
            else:
                record["details"] = _calibration(directory, manifest, forecast)
            record["validation_status"] = "passed"
            record["source_review"] = _json(directory / "review.json")
        except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
            record.update(validation_status="failed", error=f"{type(error).__name__}: {error}")
        sources.append(record)
    conflicts, known, omissions = [], {}, []
    for kind, manifest in versions:
        hashes = manifest.get("source_sha256")
        if (not isinstance(hashes, dict) or not hashes
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in "0123456789abcdef" for c in value) for value in hashes.values())):
            conflicts.append({"source": kind, "field": "source_sha256", "error": "Missing or invalid source hashes"})
        environment = manifest.get("environment", {})
        if not isinstance(environment, dict) or not isinstance(environment.get("python"), str):
            # Existing bundle/sequential schemas do not record an environment.
            entry = {"source": kind, "field": "environment.python", "error": "Missing Python version"}
            if kind == "evidence" or manifest.get("kind", "").endswith("_sequential_checks"):
                omissions.append(entry)
            else:
                conflicts.append(entry)
        for group in ("source_sha256", "environment"):
            declarations = manifest.get(group, {})
            if not isinstance(declarations, dict):
                conflicts.append({"source": kind, "field": group, "error": "Malformed version declaration"})
                continue
            for name, value in declarations.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    conflicts.append({"source": kind, "field": group, "error": "Invalid version/hash value"})
                    continue
                key = (group, name)
                if key in known and known[key][1] != value:
                    conflicts.append({"source": kind, "field": f"{group}.{name}", "other_source": known[key][0]})
                else:
                    known[key] = (kind, value)
    checks = [{"name": "provided_artifacts_compatible", "required": True,
               "status": "passed" if not conflicts and all(row["validation_status"] == "passed" for row in sources) else "failed"},
              {"name": "forecast_computation", "required": True,
               "status": "passed" if sources[0]["details"] and sources[0]["details"]["passed"] else "failed"}]
    for name in specification.evidence.required_workflows:
        checks.append({"name": f"required_workflow:{name}", "required": True,
                       "status": workflow_results.get(name, {}).get("status", "missing")})
    for row in sources:
        if row["kind"] == "evidence":
            checks.append({"name": "evidence_computation", "required": True,
                           "status": "passed" if row["validation_status"] == "passed" and row["details"]["passed"] else "failed"})
    calibrations = [row for row in sources if row["kind"].startswith("calibration-")]
    if require_calibration and not calibrations:
        checks.append({"name": "required_calibration", "required": True, "status": "missing"})
    for row in calibrations:
        checks.append({"name": row["kind"], "required": True,
                       "status": "passed" if row["validation_status"] == "passed" and row["details"]["passed"] else "failed"})
    if any(row["status"] == "failed" for row in checks):
        status = "failed"
    elif any(row["status"] == "missing" for row in checks):
        status = "incomplete"
    else:
        status = "ready_for_review"
    for directory, before in inventories:
        if _inventory(directory) != before:
            raise ValueError("Source artifacts changed during report construction")
    output.mkdir(parents=True, exist_ok=False)
    report = {"format_version": 1, "kind": "experiment_review", "status": status,
              "created_at": datetime.now(UTC).isoformat(), "input": forecast["input"], "model": forecast["model"],
              "experiment": specification.to_dict(), "sources": sources, "checks": checks,
              "version_conflicts": conflicts, "version_omissions": omissions, "human_review_status": "pending",
              "omitted_evidence": {"historical_evidence": evidence_directory is None, "calibration": not calibrations},
              "processor_sha256": _digest(Path(__file__)),
              "limitations": ["Ready for review means compatible computations and requested artifacts, not model approval",
                  "Current file hashes snapshot observed bytes; only existing child receipts check earlier completion bytes",
                  "Input release claims and model calculations are not rerun or independently authenticated",
                  "Only declared versions are compared; older bundle/sequential formats omit environment versions",
                  "Development evidence and calibration retain their own scopes; final-role selection is not proof of untouched data"]}
    write_json(output / "report.json", report)
    write_json(output / "checks.json", checks)
    write_json(output / "review.json", {"status": "pending", "decision": None, "notes": None,
                                       "questions": ["Are the stated assumptions and evidence sufficient for the intended use?"]})
    lines = ["# Experiment review", "", f"Computational readiness: **{status}**. Human review: **pending**.", "",
             f"Model: `{forecast['model']['name']}`", f"Dataset: `{forecast['input']['dataset']}`",
             f"Evaluation role: `{specification.evaluation.role}`",
             f"Historical evidence supplied: {evidence_directory is not None}. Calibration exercises: {len(calibrations)} (required: {require_calibration}).",
             "", "[Complete machine-readable record](report.json)", "", "## Requirements", "",
             "| Requirement | Status |", "|---|---|"]
    lines.extend(f"| {row['name']} | {row['status']} |" for row in checks)
    lines.extend(("", "## Saved evidence", ""))
    for row in sources:
        relative = Path(os.path.relpath(row["path"], output)).as_posix()
        lines.append(f"- [{row['kind']}](<{relative}>) — identity/receipts: {row['validation_status']}; computation: {row['computation_status']}.")
        if row["error"]:
            lines.append(f"  - {row['error']}")
        if row["details"]:
            detail = row["details"]
            if row["kind"] == "forecast":
                scores = detail["scores"]
                lines.append(f"  - {detail['case_count']} forecast cases from {detail['target_start']} through {detail['target_end']}; MAE: {scores.get('mae')}; RMSE: {scores.get('rmse')}.")
                if scores.get("distribution", {}).get("status") == "completed":
                    loss = scores["distribution"]
                    lines.append(f"  - Mean CRPS: {loss['mean_crps']}; mean negative log density: {loss['mean_log_score']} (lower losses are better).")
            elif row["kind"] == "evidence":
                lines.append(f"  - Selected development cutoff: {detail['cutoff']}. This does not assert coverage of every forecast origin.")
                lines.extend(f"  - {name}: {outcome['status']}" for name, outcome in detail["workflows"].items())
            else:
                lines.extend((f"  - {detail['trials']} prior/data trials of {detail['observation_count']} observations; seed {detail['seed']}.",
                              "", "| Parameter | CDF distance | Uniform reference | Interval coverage | Mean interval mass |",
                              "|---|---:|---:|---:|---:|"))
                for name, summary in detail["summary"].items():
                    lines.append(f"| {name} | {summary.get('cdf_uniform_distance')} | {summary.get('cdf_simultaneous_95_band')} | {summary.get('coverage', {}).get('fraction')} | {summary.get('mean_interval_posterior_mass')} |")
        lines.append("")
        for name in row.get("artifact_sha256", {}):
            if name.endswith(".png"):
                lines.append(f"  - [Chart: {name}](<{relative}/{name}>)")
        lines.append("")
    if conflicts:
        lines.extend(("## Code/environment conflicts", "", "```json", json.dumps(conflicts, indent=2), "```", ""))
    lines.extend(("## Interpretation", "", *report["limitations"], ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output
