"""Build a saved experiment review without fitting, simulating or changing runs."""

import argparse
import json

from model_pipelining.experiment_review import build_experiment_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-directory", required=True)
    parser.add_argument("--evidence-directory")
    parser.add_argument("--calibration-directories", nargs="*", default=())
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--decision-directories", nargs="*", default=())
    parser.add_argument("--require-decisions", action="store_true")
    parser.add_argument("--output-directory", required=True)
    output = build_experiment_review(**vars(parser.parse_args()))
    status = json.loads((output / "report.json").read_text())["status"]
    print(f"Saved review: {output / 'report.md'}; computation: {status}; human review: pending")
    return 0 if status == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
