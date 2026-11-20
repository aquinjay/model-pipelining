"""Extend a saved model-review design with predictive criticism for every prior."""

import argparse
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from model_pipelining.experiments import ExperimentSpec
from model_pipelining.predictive_checks import run_predictive_checks, plot_predictive_checks
from review_model_study import MODEL_CONFIGS, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    study = args.study_directory.expanduser().absolute()
    output = args.output_directory.expanduser().absolute()
    plan = read_json(study / "study.json")
    if plan.get("kind") != "model_review_study" or plan.get("status") not in ("running", "completed"):
        raise ValueError("Use an active or complete model-review design with prepared snapshots")
    experiment = ExperimentSpec.from_dict(plan["experiment"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "model_review_prior_checks", "status": "running", "parent_study": str(study),
        "parent_design": {key: plan[key] for key in ("models", "experiment", "generator", "snapshots")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": [],
        "scope": "All declared alternative-prior predictive checks; no alternative calibration or model approval"}
    write_json(output / "study.json", manifest)
    for scenario in plan["scenario_order"]:
        for name, settings in plan["models"].items():
            kind = MODEL_CONFIGS[name][0]
            for prior, configuration in settings["alternatives"].items():
                print(f"Prior criticism: {scenario}/{name}/{prior}", flush=True)
                path = run_predictive_checks(dataset=scenario, snapshot_id=plan["snapshots"][scenario],
                    store=study / "store", experiment=experiment, model=kind.from_configuration(configuration),
                    issued_at=datetime.fromisoformat(plan["experiment"]["evaluation"]["final_start"]) - timedelta(seconds=1),
                    replications=experiment.evidence.replications, seed=experiment.evidence.seed,
                    output_directory=output / scenario / name / prior)
                plot_predictive_checks(path)
                result = {"scenario": scenario, "model": name, "prior": prior,
                          "status": read_json(path / "predictive_checks.json")["status"]}
                manifest["results"].append(result)
                write_json(output / "study.json", manifest)
    manifest["status"] = "completed" if all(row["status"] == "completed" for row in manifest["results"]) else "failed"
    write_json(output / "study.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
