"""Run an experiment's required evidence workflows, then draw saved results."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from model_pipelining.adapters import EVIDENCE_MODEL_NAMES, build_model, build_scenarios
from model_pipelining.evidence_workflows import run_experiment_evidence, plot_experiment_evidence
from model_pipelining.experiments import ExperimentSpec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", choices=EVIDENCE_MODEL_NAMES, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Reference model settings and named prior overrides")
    parser.add_argument("--output-directory", required=True)
    options = vars(parser.parse_args())
    configuration = json.loads(options.pop("config").read_text(encoding="utf-8"))
    if (not isinstance(configuration, dict) or set(configuration) != {"reference", "scenarios"}
            or not isinstance(configuration["reference"], dict) or not isinstance(configuration["scenarios"], dict)
            or any(not isinstance(value, dict) for value in configuration["scenarios"].values())):
        raise ValueError("Config needs reference settings and named scenario override dictionaries")
    model = build_model(options.pop("model"), configuration["reference"])
    scenarios = build_scenarios(model, configuration["scenarios"])
    experiment = ExperimentSpec.load(options.pop("experiment"))
    options["issued_at"] = options.pop("as_of")
    output = run_experiment_evidence(**options, experiment=experiment, model=model, scenarios=scenarios)
    print(f"Saved experiment evidence: {output}")
    print(f"Charts: {plot_experiment_evidence(output)}")
    status = json.loads((output / "evidence.json").read_text(encoding="utf-8"))["status"]
    print(f"Computation: {status}; human review: pending")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
