"""Issue frozen decision recommendations, score later outcomes, and compare saved runs."""

import argparse
from pathlib import Path

from model_pipelining.decisions import (
    DecisionSpec,
    compare_decision_scores,
    issue_decisions_from_probabilities,
    issue_decisions_from_run,
    plot_decision_scores,
    score_decisions,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    issue_parser = subparsers.add_parser("issue", help="Freeze per-origin recommendations from forecast outputs only")
    source_group = issue_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run-directory", type=Path, help="Saved version-3 distribution backtest directory")
    source_group.add_argument("--probabilities-directory", type=Path, help="Directory with saved event probabilities")
    issue_parser.add_argument("--method", help="Method name for --probabilities-directory (e.g. independent, same_day)")
    issue_parser.add_argument("--probabilities-file", default="matched-range.csv")
    issue_parser.add_argument("--dataset", default=None)
    issue_parser.add_argument("--model-name", default=None)
    issue_parser.add_argument("--spec", type=Path, required=True, help="Path to a version-1 DecisionSpec JSON file")
    issue_parser.add_argument("--output-directory", type=Path, required=True)

    score_parser = subparsers.add_parser("score", help="Score frozen recommendations against realized outcomes")
    score_parser.add_argument("--issued-directory", type=Path, required=True)
    score_parser.add_argument("--outcomes", type=Path, required=True,
                              help="Backtest directory, NOAA range directory, or outcomes CSV")
    score_parser.add_argument("--output-directory", type=Path, required=True)
    score_parser.add_argument("--no-plot", action="store_true", help="Skip generating decision_scores.png")

    compare_parser = subparsers.add_parser("compare", help="Compare two scored decision directories on identical cohorts")
    compare_parser.add_argument("--first", type=Path, required=True)
    compare_parser.add_argument("--second", type=Path, required=True)
    compare_parser.add_argument("--output-directory", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "issue":
        spec = DecisionSpec.load(args.spec)
        if args.run_directory is not None:
            destination = issue_decisions_from_run(
                run_directory=args.run_directory, spec=spec, output_directory=args.output_directory,
            )
        else:
            if not args.method:
                parser.error("--method is required with --probabilities-directory")
            destination = issue_decisions_from_probabilities(
                source_directory=args.probabilities_directory,
                method=args.method,
                spec=spec,
                output_directory=args.output_directory,
                probabilities_file=args.probabilities_file,
                dataset=args.dataset,
                model_name=args.model_name,
            )
        print(destination)
        return 0

    if args.command == "score":
        destination = score_decisions(
            issued_directory=args.issued_directory,
            outcomes_source=args.outcomes,
            output_directory=args.output_directory,
        )
        if not args.no_plot:
            plot_decision_scores(destination)
        print(destination)
        return 0

    destination = compare_decision_scores(
        args.first, args.second, output_directory=args.output_directory,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
