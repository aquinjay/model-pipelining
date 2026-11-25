"""Run a probabilistic baseline backtest (gaussian-climatology or residual-persistence) and plot it."""

import argparse
from dataclasses import replace
from pathlib import Path

from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.probabilistic_baselines import PROBABILISTIC_BENCHMARKS
from model_pipelining.reporting import plot_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PROBABILISTIC_BENCHMARKS), required=True)
    parser.add_argument("--window", type=int, default=None,
                        help="Optional trailing observation/difference window; omit for expanding history")
    parser.add_argument("--dataset", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-id")
    source.add_argument("--tag")
    parser.add_argument("--store", type=Path, default=Path("data/versioned"))
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--evaluation-role", choices=("development", "final"), default=None)
    parser.add_argument("--minimum-history", type=int, default=None)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)

    experiment = ExperimentSpec.load(args.experiment) if args.experiment is not None else None
    if args.evaluation_role is not None:
        if experiment is None:
            parser.error("--evaluation-role requires --experiment")
        experiment = replace(experiment, evaluation=replace(experiment.evaluation, role=args.evaluation_role))
    model = PROBABILISTIC_BENCHMARKS[args.model](window=args.window)
    output = run_backtest(
        dataset=args.dataset,
        snapshot_id=args.snapshot_id,
        tag=args.tag,
        minimum_history=args.minimum_history,
        experiment=experiment,
        model=model,
        store=args.store,
        output_directory=args.output_directory,
    )
    plot_run(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
