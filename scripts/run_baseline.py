"""Run the training-mean baseline against a preserved CSV, then draw its chart.

Launch from the repository root with PYTHONPATH=src; see docs/forecasting.md.
"""

import argparse
import json

from model_pipelining.backtest import run_backtest
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
    arguments = parser.parse_args()
    run = run_backtest(**vars(arguments))
    chart = plot_run(run)
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    print(f"Saved run: {run}")
    print(f"Chart: {chart}")
    print(json.dumps(scores, indent=2))
    return 0 if scores["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
