"""Evaluate the cost of acting on saved NOAA range probabilities."""

import argparse
import csv
from datetime import date
import hashlib
import io
import json
from math import nan
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model_pipelining.artifacts import write_csv, write_json
from model_pipelining.decision_value import value_curve
from scripts.study_noaa_range import METHODS


ALPHAS = tuple(percent / 100 for percent in range(5, 100, 5))


def _cohort_counts(summary, cohort):
    cases = summary.get("cases")
    events = summary.get("observed_events")
    if type(cases) is not int or type(events) is not int or not 0 <= events <= cases or cases == 0:
        raise ValueError(f"Expected valid cases and observed_events counts for the {cohort} cohort")
    return cases, events


def _read_cases(raw, summary):
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"date", "event", *(f"{method}_probability" for method in METHODS)}
        if len(set(fields)) != len(fields) or not required.issubset(fields):
            raise ValueError("Expected unique date, event, independent_probability and same_day_probability columns")
        rows = list(reader)

    counts = _cohort_counts(summary, "evaluation")
    cases = {method: [] for method in METHODS}
    dates = []
    for row in rows:
        try:
            day = date.fromisoformat(row["date"])
            event = int(row["event"])
            probabilities = {method: float(row[f"{method}_probability"]) for method in METHODS}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid saved range case for date {row['date']!r}") from error
        if event not in (0, 1):
            raise ValueError(f"Invalid event for {day}. Expected 0 or 1.")
        dates.append(day)
        for method, probability in probabilities.items():
            cases[method].append((probability, event))

    if (not dates or len(dates) != counts[0] or sum(event for _, event in cases[METHODS[0]]) != counts[1]
            or dates != sorted(set(dates)) or (dates[-1] - dates[0]).days + 1 != len(dates)
            or dates[0].isoformat() != summary.get("evaluation_start")
            or dates[-1].isoformat() != summary.get("evaluation_end")):
        raise ValueError("Saved range cases do not match the complete evaluation cohort in summary.json")
    return cases


def _plot(path, methods, base_rate, *, start, end):
    figure = Figure(figsize=(8.6, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for method, label, color, marker in (
        ("independent", "Independent pairing", "#28547B", "o"),
        ("same_day", "Same-day pairing", "#B65D24", "s"),
    ):
        rows = methods[method]
        values = [row["relative_value"] if row["relative_value"] is not None else nan for row in rows]
        axis.plot([row["alpha"] for row in rows], values, label=label, color=color, marker=marker, markersize=4)
    axis.axhline(0, color="#485865", linestyle="--", linewidth=1, label="Climatology")
    axis.axvline(base_rate, color="#777777", linestyle=":", linewidth=1,
                label=f"Calibration event rate ({base_rate:.1%})")
    axis.set(title=f"Economic value of NOAA range forecasts\nCentral Park · {start} to {end}",
             xlabel="Cost-loss ratio (C / L)",
             ylabel="Relative economic value", xlim=(0, 1))
    axis.grid(axis="y", alpha=.22)
    axis.legend(loc="upper right", frameon=False)
    figure.savefig(path, dpi=170)


def run_study(input_directory, output_directory) -> Path:
    """Read one completed range evaluation and write decision artifacts to a new directory."""
    source, output = Path(input_directory), Path(output_directory)
    inputs = {name: (source / name).read_bytes() for name in ("matched-range.csv", "summary.json")}
    summary = json.loads(inputs["summary.json"])
    calibration = summary.get("calibration") if isinstance(summary, dict) else None
    if not isinstance(calibration, dict):
        raise ValueError("summary.json must record the frozen 2025 calibration cohort")
    count, events = _cohort_counts(calibration, "2025 calibration")
    base_rate = events / count
    cases = _read_cases(inputs["matched-range.csv"], summary)
    methods = {method: value_curve(cases[method], alphas=ALPHAS, base_rate=base_rate) for method in METHODS}
    rows = [dict(method=method, **row) for method, curve in methods.items() for row in curve]

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "decision-value.csv", rows[0], rows)
    _plot(output / "decision-value.png", methods, base_rate,
          start=summary["evaluation_start"], end=summary["evaluation_end"])
    result = output / "decision-value-summary.json"
    write_json(result, {
        "format_version": 1,
        "input_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        "evaluation_start": summary["evaluation_start"],
        "evaluation_end": summary["evaluation_end"],
        "threshold_celsius": summary.get("threshold_celsius"),
        "evaluation_cases": summary["cases"],
        "evaluation_events": summary["observed_events"],
        "alpha_grid": ALPHAS,
        "base_rate": base_rate,
        "base_rate_source": {
            "cohort": "2025 calibration",
            "file": "summary.json",
            "fields": ["calibration.observed_events", "calibration.cases"],
            "cases": count,
            "observed_events": events,
        },
        "methods": methods,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_study(args.input_directory, args.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
