"""Compare explicit mean/variance priors; strong influence is descriptive evidence."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from model_pipelining.adapters import EVIDENCE_MODEL_NAMES, build_model, build_scenarios
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.prior_influence import plot_prior_influence, run_prior_influence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=EVIDENCE_MODEL_NAMES, default="normal-mean-variance")
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    parser.add_argument("--experiment", help="Development experiment with release rules/final boundary")
    parser.add_argument("--config", type=Path, required=True, help="Reference prior and named scenario overrides")
    parser.add_argument("--interval-level", type=float, default=0.95)
    parser.add_argument("--output-directory", required=True)
    options = vars(parser.parse_args())
    config = json.loads(options.pop("config").read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or set(config) != {"reference", "scenarios"}
            or not isinstance(config["reference"], dict) or not isinstance(config["scenarios"], dict)
            or any(not isinstance(value, dict) for value in config["scenarios"].values())):
        raise ValueError("Config must contain reference settings and named scenario override dictionaries")
    reference = build_model(options.pop("model"), config["reference"])
    scenarios = build_scenarios(reference, config["scenarios"])
    experiment = options.pop("experiment")
    options["issued_at"] = options.pop("as_of")
    output = run_prior_influence(**options, reference=reference, scenarios=scenarios,
                                experiment=ExperimentSpec.load(experiment) if experiment else None)
    print(f"Saved prior comparisons: {output}")
    print(f"Charts: {plot_prior_influence(output)}")
    manifest = json.loads((output / "prior_influence.json").read_text(encoding="utf-8"))
    print(f"Computation: {manifest['status']}; human review: pending")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
