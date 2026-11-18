"""Explicit execution of the separate workflows required by an experiment."""

from datetime import UTC, datetime
import json
from pathlib import Path

from model_pipelining.evidence import get_evidence_provider, source_hashes, _write_json
from model_pipelining.experiments import ExperimentSpec, select_development_history
from model_pipelining.inference import assess_checks
from model_pipelining.predictive_checks import run_predictive_checks, plot_predictive_checks
from model_pipelining.prior_influence import run_prior_influence, plot_prior_influence


_MANIFESTS = {"predictive_checks": "predictive_checks.json", "prior_influence": "prior_influence.json"}
_REQUIRED_CHECKS = {
    "predictive_checks": ("prior_simulation_and_statistics", "posterior_simulation_and_statistics",
                          "observed_statistics_computed"),
    "prior_influence": ("development_history_selection", "all_requested_prior_and_posterior_updates"),
}


def run_experiment_evidence(*, dataset: str, model, issued_at: datetime,
                            experiment: ExperimentSpec, output_directory: str | Path,
                            snapshot_id: str | None = None, tag: str | None = None,
                            store: str | Path = "data/versioned", scenarios: dict | None = None) -> Path:
    """Run the experiment's required development evidence, retaining failures.

    Resolve a tag once and pin its permanent snapshot for every child workflow.
    Missing capability/scenarios or failed/missing required checks fail the
    evidence run; later requested workflows still run. This does not forecast,
    score, approve a model, execute per-origin fits or alter existing runs.
    Invalid selection/plan raises before output creation. I/O errors propagate.
    """
    if not isinstance(experiment, ExperimentSpec) or not experiment.evidence.required_workflows:
        raise ValueError("An ExperimentSpec with required evidence workflows is needed")
    _, _, case, context = select_development_history(dataset=dataset, issued_at=issued_at,
        snapshot_id=snapshot_id, tag=tag, store=store, experiment=experiment)
    configuration = json.loads(json.dumps(model.configuration, allow_nan=False))
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1, "kind": "experiment_evidence", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "context": context,
        "model": {"name": model.name, "configuration": configuration},
        "plan": experiment.to_dict()["evidence"], "workflows": [],
        "scope": "Required separate evidence workflows only; per-origin fit requirements belong to forecasting; human review remains pending",
        "source_sha256": source_hashes(*(Path(__file__).with_name(name) for name in
            ("evidence_workflows.py", "evidence.py", "experiments.py", "inference.py"))),
    }
    _write_json(output / "evidence.json", manifest)
    for workflow in experiment.evidence.required_workflows:
        directory = output / workflow
        row = {"name": workflow, "required": True, "status": "failed", "error": None,
               "directory": workflow, "manifest": _MANIFESTS[workflow]}
        try:
            provider = get_evidence_provider(model, workflow)
            common = dict(dataset=dataset, snapshot_id=context["input"]["snapshot_id"], store=store,
                          experiment=experiment, issued_at=case.issued_at, output_directory=directory)
            if workflow == "predictive_checks":
                run_predictive_checks(**common, model=model, replications=experiment.evidence.replications,
                                      seed=experiment.evidence.seed)
            else:
                if not scenarios:
                    raise ValueError("Required prior_influence needs explicit named scenarios")
                run_prior_influence(**common, reference=model, scenarios=scenarios,
                                    interval_level=experiment.evidence.interval_level)
            child = json.loads((directory / _MANIFESTS[workflow]).read_text(encoding="utf-8"))
            if child.get("format_version") != 1 or child.get("kind") != f"{provider.metadata.family}_{workflow}":
                raise ValueError("Child evidence has an incompatible format or workflow kind")
            if ((workflow == "predictive_checks" and
                 (child.get("replications") != experiment.evidence.replications or child.get("seed") != experiment.evidence.seed))
                    or (workflow == "prior_influence" and child.get("interval_level") != experiment.evidence.interval_level)):
                raise ValueError("Child evidence does not match experiment evidence settings")
            checks = json.loads((directory / "checks.json").read_text(encoding="utf-8"))
            passed, requirements = assess_checks(checks, required_names=_REQUIRED_CHECKS[workflow])
            row["check_requirements"] = requirements
            identity = child.get("context", {})
            child_configuration = (child.get("configuration") if workflow == "predictive_checks"
                                   else child.get("configurations", {}).get("reference"))
            if (identity.get("input", {}).get("dataset") != dataset
                    or identity.get("input", {}).get("snapshot_id") != context["input"]["snapshot_id"]
                    or identity.get("history_sha256") != case.history_sha256
                    or identity.get("issued_at") != case.issued_at.isoformat()
                    or identity.get("experiment") != experiment.to_dict()
                    or child_configuration != configuration):
                raise ValueError("Child evidence does not match the pinned experiment/model/history")
            row.update(status="passed" if passed and child.get("status") == "completed" else "failed",
                       artifact_sha256=source_hashes(directory / _MANIFESTS[workflow], directory / "checks.json"))
            if row["status"] == "failed":
                row["error"] = "Required workflow computation/checks failed; inspect saved child evidence"
        except (ValueError, TypeError, OverflowError, FloatingPointError, ZeroDivisionError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        manifest["workflows"].append(row)
        # Keep completed/failed child outcomes if a later I/O operation interrupts.
        (output / "evidence.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    checks = [{"name": row["name"], "required": True, "status": row["status"]}
              for row in manifest["workflows"]]
    _write_json(output / "checks.json", checks)
    _write_json(output / "review.json", {"status": "pending", "decision": None,
        "limitations": ["Completion means requested computations/checks passed, not model approval",
                        "Prior plausibility, discrepancies and influence require human interpretation",
                        "This is development evidence, not a final forecast evaluation"]})
    manifest.update(status="completed" if all(row["status"] == "passed" for row in checks) else "failed",
                    finished_at=datetime.now(UTC).isoformat())
    (output / "evidence.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return output


def plot_experiment_evidence(evidence_directory: str | Path) -> tuple[Path, ...]:
    """Draw finished child evidence from saved files only; never refit or simulate."""
    directory = Path(evidence_directory).expanduser()
    manifest = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1 or manifest.get("kind") != "experiment_evidence"
            or manifest.get("status") not in ("completed", "failed")):
        raise ValueError("Plotting requires finished experiment evidence")
    outputs = []
    for row in manifest["workflows"]:
        workflow = row["name"]
        if workflow not in _MANIFESTS or row["directory"] != workflow:
            raise ValueError("Invalid saved evidence workflow")
        child_path = directory / workflow / _MANIFESTS[workflow]
        if not child_path.is_file():
            continue  # Missing capability/scenarios can fail before child creation.
        child = json.loads(child_path.read_text(encoding="utf-8"))
        if child.get("status") not in ("completed", "failed"):
            continue
        plot = plot_predictive_checks if workflow == "predictive_checks" else plot_prior_influence
        outputs.extend(plot(directory / workflow))
    return tuple(outputs)
