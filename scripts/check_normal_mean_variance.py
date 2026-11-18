"""Simulate prior datasets or check a development history with exact replications."""

import argparse
from datetime import datetime
import json

from model_pipelining.experiments import ExperimentSpec
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings
from model_pipelining.predictive_checks import plot_predictive_checks, run_predictive_checks, run_prior_predictive_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("normal-mean-variance", "local-level-variance"), default="normal-mean-variance")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prior-only", action="store_true")
    mode.add_argument("--dataset")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--as-of", type=datetime.fromisoformat, help="Required aware as-of time for observed checks")
    parser.add_argument("--experiment", help="Optional development ExperimentSpec with release rules/final boundary")
    parser.add_argument("--observation-count", type=int, help="Required dataset length for --prior-only")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-mean", type=float)
    parser.add_argument("--prior-mean-precision", type=float)
    parser.add_argument("--initial-mean", type=float)
    parser.add_argument("--initial-variance", type=float)
    parser.add_argument("--evolution-variance", type=float)
    parser.add_argument("--quadrature-config")
    parser.add_argument("--prior-shape", type=float, default=3)
    parser.add_argument("--prior-scale", type=float, default=2)
    parser.add_argument("--output-directory", required=True)
    options = vars(parser.parse_args())
    prior_only = options.pop("prior_only")
    dataset, snapshot_id, tag = (options.pop(name) for name in ("dataset", "snapshot_id", "tag"))
    store, issued_at, experiment_path = (options.pop(name) for name in ("store", "as_of", "experiment"))
    count = options.pop("observation_count")
    if prior_only:
        if count is None or any(value is not None for value in (snapshot_id, tag, issued_at, experiment_path)):
            parser.error("--prior-only requires --observation-count and cannot use input/as-of/experiment options")
    elif count is not None or issued_at is None or (snapshot_id is None) == (tag is None):
        parser.error("Observed checks require --as-of and one of --snapshot-id/--tag; omit --observation-count")
    kind = options.pop("model")
    static = {name: options.pop(name) for name in ("prior_mean", "prior_mean_precision")}
    dynamic = {name: options.pop(name) for name in ("initial_mean", "initial_variance", "evolution_variance")}
    numerical = options.pop("quadrature_config")
    common = {name: options.pop(name) for name in ("prior_shape", "prior_scale")}
    if kind == "local-level-variance":
        if any(value is not None for value in static.values()):
            parser.error("Constant-mean prior flags do not configure the local-level model")
        settings = {}
        if numerical:
            with open(numerical, encoding="utf-8") as stream:
                settings = json.load(stream)
        model = LocalLevelVariance(**common, **{key: value for key, value in dynamic.items() if value is not None},
                                   quadrature=QuadratureSettings(**settings))
    else:
        if numerical or any(value is not None for value in dynamic.values()):
            parser.error("State/numerical flags require --model local-level-variance")
        model = NormalMeanVariance(**common, **{key: value for key, value in static.items() if value is not None})
    if prior_only:
        output = run_prior_predictive_checks(model=model, observation_count=count, **options)
    else:
        output = run_predictive_checks(model=model, dataset=dataset, snapshot_id=snapshot_id, tag=tag,
            store=store, issued_at=issued_at,
            experiment=ExperimentSpec.load(experiment_path) if experiment_path else None, **options)
    print(f"Saved checks: {output}")
    print(f"Charts: {plot_predictive_checks(output)}")
    manifest = json.loads((output / "predictive_checks.json").read_text(encoding="utf-8"))
    print(f"Computation: {manifest['status']}; human review: pending")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
