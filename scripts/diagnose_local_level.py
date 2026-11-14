"""Inspect caller-chosen assumptions; strong prior influence is not a failure."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from model_pipelining.diagnostics import plot_local_level_diagnostics, run_local_level_diagnostics
from model_pipelining.local_level import LocalLevel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--config", type=Path, required=True, help="JSON reference settings and named scenario overrides")
    parser.add_argument("--output-directory", required=True)
    options = vars(parser.parse_args())
    configuration = json.loads(options.pop("config").read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or set(configuration) != {"reference", "scenarios"}:
        raise ValueError("Config must contain exactly reference and scenarios")
    if not isinstance(configuration["reference"], dict) or not isinstance(configuration["scenarios"], dict):
        raise ValueError("reference and scenarios must be dictionaries")
    reference = LocalLevel(**configuration["reference"])
    scenarios = {name: replace(reference, **overrides) for name, overrides in configuration["scenarios"].items()}
    output = run_local_level_diagnostics(**options, reference=reference, scenarios=scenarios)
    print(f"Saved diagnostics: {output}")
    print(f"Charts: {plot_local_level_diagnostics(output)}")
    manifest = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    print(f"Computation: {manifest['status']}; human review: pending")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
