"""Run local-level mean/uncertainty forecasts and save prior draws for review.

Launch from the repository root with PYTHONPATH=src; see docs/local-level.md.
"""

import argparse
import csv
import json
import platform

from model_pipelining.backtest import run_backtest
from model_pipelining.local_level import LocalLevel
from model_pipelining.reporting import plot_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-id")
    selection.add_argument("--tag")
    parser.add_argument("--store", default="data/versioned")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--minimum-history", type=int, default=30)
    parser.add_argument("--interval-levels", type=float, nargs="+", default=(0.8, 0.95))
    parser.add_argument("--initial-mean", type=float, default=0.0)
    parser.add_argument("--initial-variance", type=float, default=100.0)
    parser.add_argument("--observation-variance", type=float, default=1.0)
    parser.add_argument("--evolution-variance", type=float, default=0.1)
    parser.add_argument("--prior-seed", type=int, default=0)
    options = vars(parser.parse_args())
    model = LocalLevel(**{name: options.pop(name) for name in (
        "initial_mean", "initial_variance", "observation_variance", "evolution_variance",
    )})
    seed = options.pop("prior_seed")
    # Simulation is deliberately before fitting. Save it after the runner has
    # created its new output directory, without modifying runner-owned artifacts.
    prior = model.simulate_prior(count=200, seed=seed)
    run = run_backtest(**options, model=model)
    with (run / "prior_predictive.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "level", "value"))
        writer.writerows((index, draw.level, draw.value) for index, draw in enumerate(prior, start=1))
    with (run / "prior_predictive.json").open("x", encoding="utf-8") as stream:
        json.dump({"seed": seed, "count": len(prior), "python": platform.python_version(),
                   "model_configuration": model.configuration, "review_status": "pending",
                   "scope": "One prior trajectory; not held-out forecasts or evidence of prior plausibility"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved run: {run}")
    print(f"Chart: {plot_run(run)}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    print((run / "scores.json").read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
