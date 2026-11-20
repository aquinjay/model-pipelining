"""Run the weekly harmonic adapter and the training mean on a generated weekly series."""

import argparse
import csv
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
import random

from model_pipelining.backtest import run_backtest
from model_pipelining.harmonic_regression import WeeklyHarmonicRegression
from model_pipelining.reporting import compare_runs
from model_pipelining.snapshots import register_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    out = parser.parse_args(argv).output_directory
    out.mkdir(parents=True, exist_ok=False)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    rng = random.Random(176)
    source = out / "weekly.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        rows = csv.writer(stream)
        rows.writerow(("timestamp", "value"))
        for day in range(100):
            timestamp = start + timedelta(days=day)
            value = 10 + 3 * math.sin(2 * math.pi * day / 7) + rng.gauss(0, .4)
            rows.writerow((timestamp.isoformat(), value))
    store = out / "snapshots"
    snapshot = register_snapshot(source, dataset="weekly-example", store=store)
    common = dict(dataset=snapshot.dataset, snapshot_id=snapshot.snapshot_id, store=store, minimum_history=28)
    baseline = run_backtest(**common, output_directory=out / "mean")
    harmonic = run_backtest(**common, model=WeeklyHarmonicRegression(), output_directory=out / "harmonic")
    print(compare_runs(baseline, harmonic, output_directory=out / "comparison"))


if __name__ == "__main__":
    main()
