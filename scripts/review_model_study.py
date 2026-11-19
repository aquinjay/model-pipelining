"""Run a bounded three-model development review using existing workflows.

This is an explicit research exercise, not an automatic model selection stage.
Run from the repository root with PYTHONPATH=src. Outputs must be new.
"""

import argparse
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import StatisticsError, correlation, mean

from model_pipelining.adapters import build_scenarios
from model_pipelining.backtest import run_backtest
from model_pipelining.calibration import run_calibration, plot_calibration
from model_pipelining.distribution_scoring import read_saved_distribution_scores
from model_pipelining.evidence_workflows import run_experiment_evidence, plot_experiment_evidence
from model_pipelining.experiment_review import build_experiment_review
from model_pipelining.experiments import ExperimentSpec, ForecastTask, EvaluationPlan, EvidencePlan
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.reporting import plot_run, compare_runs
from model_pipelining.sequential import predictive_law
from model_pipelining.snapshots import register_snapshot


ROOT = Path(__file__).resolve().parents[1]
START = datetime(2020, 1, 1, tzinfo=UTC)
MODEL_CONFIGS = {
    "constant": (NormalMeanVariance, "normal_mean_variance_priors.json"),
    "learned-noise": (LocalLevelVariance, "local_level_variance_priors.json"),
    "joint": (LocalLevelJointVariance, "local_level_joint_priors.json"),
}
WINDOWS = {"all": (30, 160), "before": (30, 100), "onset": (100, 120), "later": (120, 160)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def models():
    result = {}
    for name, (kind, filename) in MODEL_CONFIGS.items():
        config = read_json(ROOT / "configs" / filename)
        reference = kind.from_configuration(config["reference"])
        result[name] = (reference, build_scenarios(reference, config["scenarios"]))
    return result


def generate_sources(output):
    """Preserve original stable/shift bytes; add a ramp with the same noise."""
    source = output / "source"
    source.mkdir()
    generator = random.Random(42)
    noise = [generator.gauss(0, 1) for _ in range(200)]
    result = {}
    for scenario in ("mean-shift", "stable", "gradual"):
        path = source / f"{scenario}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("timestamp", "value"))
            for index, error in enumerate(noise):
                change = (5 * (index >= 100) if scenario == "mean-shift" else
                          .05 * max(0, index - 99) if scenario == "gradual" else 0)
                value = 10 + error + change
                timestamp = (START + timedelta(days=index)).strftime("%Y-%m-%dT00:00:00Z")
                writer.writerow((timestamp, format(value, ".17g")))
        result[scenario] = register_snapshot(path, dataset=scenario, store=output / "store").snapshot_id
    return result


def case_metrics(directory):
    """Validate complete saved cohorts; derive PIT from already frozen laws."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Study summaries require a complete forecast")
    predictions = read_csv(directory / "predictions.csv")
    evaluations = read_csv(directory / "evaluation.csv")
    _, scores = read_saved_distribution_scores(directory, manifest, predictions, evaluations)
    laws = read_json(directory / "predictive_distributions.json")
    regions = read_csv(directory / "highest_density_evaluation.csv")
    regions = [row for row in regions if float(row["level"]) == .95]
    targets = [row["target_time"] for row in predictions]
    if (not targets or len(set(targets)) != len(targets) or targets != sorted(targets)
            or [row["target_time"] for row in regions] != targets):
        raise ValueError("Incomplete, duplicate or unordered 95% HPD cohort")
    result = []
    for prediction, evaluation, score, law, region in zip(predictions, evaluations, scores, laws, regions):
        actual = float(evaluation["actual"])
        if (region["status"] != "passed" or float(region["actual"]) != actual
                or region["covered"] not in ("True", "False")):
            raise ValueError("HPD actual/status differs from forecast cohort")
        width = float(region["total_width"])
        distribution = predictive_law(law["distribution"])
        pit = float(distribution.cdf(actual))
        variance = float(distribution.var()) if law["distribution"]["family"] == "student_t" else distribution.variance
        innovation = actual - float(prediction["prediction"])
        standardized = innovation / math.sqrt(variance) if variance is not None and math.isfinite(variance) else None
        if not math.isfinite(width) or width <= 0 or not math.isfinite(pit) or not 0 <= pit <= 1:
            raise ValueError("Invalid saved region/PIT")
        result.append({"target_time": prediction["target_time"], "issued_at": prediction["issued_at"],
            "history_sha256": prediction["history_sha256"], "actual": actual,
            "actual_available_at": evaluation["actual_available_at"],
            "day": (datetime.fromisoformat(prediction["target_time"]) - START).days,
            "prediction": float(prediction["prediction"]), "innovation": innovation,
            "standardized_innovation": standardized, "pit": pit, "crps": score["crps"],
            "log_score": score["log_score"], "covered_95": region["covered"] == "True", "width_95": width})
    return result


def matched(first, second):
    keys = ("target_time", "issued_at", "history_sha256", "actual", "actual_available_at")
    if not first or [tuple(row[key] for key in keys) for row in first] != [tuple(row[key] for key in keys) for row in second]:
        raise ValueError("Study cases, training histories or actual vintages differ")


def summarize_window(rows, bounds):
    selected = [row for row in rows if bounds[0] <= row["day"] < bounds[1]]
    if [row["day"] for row in selected] != list(range(*bounds)):
        raise ValueError("Study window must contain every declared daily target")
    standardized = [row["standardized_innovation"] for row in selected]
    pairs = [(a["standardized_innovation"], b["standardized_innovation"])
             for a, b in zip(selected, selected[1:]) if b["day"] == a["day"] + 1
             and a["standardized_innovation"] is not None and b["standardized_innovation"] is not None]
    try:
        acf = correlation(*zip(*pairs)) if len(pairs) >= 3 else None
    except StatisticsError:
        acf = None
    return {"cases": len(selected), "mean_crps": mean(row["crps"] for row in selected),
            "mean_log_score": mean(row["log_score"] for row in selected),
            "coverage_95": mean(row["covered_95"] for row in selected),
            "mean_width_95": mean(row["width_95"] for row in selected),
            "mean_innovation": mean(row["innovation"] for row in selected),
            "mean_pit": mean(row["pit"] for row in selected),
            "pit_lower_025": sum(row["pit"] <= .025 for row in selected),
            "pit_upper_975": sum(row["pit"] >= .975 for row in selected),
            "standardized_count": sum(value is not None for value in standardized),
            "residual_lag_one": acf}


def summarize(output):
    """Read saved runs only. Preserve signed paired losses and all alternatives."""
    output = Path(output)
    plan = read_json(output / "study.json")
    all_cases, summaries, differences = [], [], []
    for scenario in plan["scenario_order"]:
        reference_cases = {}
        for model, settings in plan["models"].items():
            for prior in ("reference", *settings["alternatives"]):
                rows = case_metrics(output / scenario / model / prior)
                if [row["day"] for row in rows] != list(range(30, 160)):
                    raise ValueError("Study requires exactly the 130 development cases")
                if prior == "reference":
                    reference_cases[model] = rows
                matched(reference_cases[model], rows)
                for row in rows:
                    all_cases.append({"scenario": scenario, "model": model, "prior": prior, **row})
                for window, bounds in WINDOWS.items():
                    summaries.append({"scenario": scenario, "model": model, "prior": prior,
                                      "window": window, **summarize_window(rows, bounds)})
                for a, b in zip(rows, reference_cases[model]):
                    differences.append({"scenario": scenario, "model": model, "prior": prior,
                        "reference_model": model, "reference_prior": "reference", "target_time": a["target_time"],
                        "crps_difference": a["crps"] - b["crps"], "log_score_difference": a["log_score"] - b["log_score"]})
        for first, second in (("constant", "learned-noise"), ("constant", "joint"), ("learned-noise", "joint")):
            matched(reference_cases[first], reference_cases[second])
            for a, b in zip(reference_cases[second], reference_cases[first]):
                differences.append({"scenario": scenario, "model": second, "prior": "reference",
                    "reference_model": first, "reference_prior": "reference", "target_time": a["target_time"],
                    "crps_difference": a["crps"] - b["crps"], "log_score_difference": a["log_score"] - b["log_score"]})
    write_csv(output / "case_metrics.csv", all_cases)
    write_csv(output / "window_metrics.csv", summaries)
    write_csv(output / "paired_losses.csv", differences)
    write_json(output / "summary.json", summaries)
    return summaries


def plot_summary(output, summaries):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)
    for row_axes, scenario in zip(axes, ("mean-shift", "stable", "gradual")):
        for axis, key in zip(row_axes, ("mean_crps", "mean_log_score", "coverage_95")):
            for model in MODEL_CONFIGS:
                rows = [next(row for row in summaries if row["scenario"] == scenario and row["model"] == model
                             and row["prior"] == "reference" and row["window"] == window)
                        for window in ("before", "onset", "later")]
                axis.plot(("before", "onset", "later"), [row[key] for row in rows], marker="o", label=model)
            axis.set(title=f"{scenario}: {key.replace('_', ' ')}")
            if key == "coverage_95":
                axis.axhline(.95, color="grey", linestyle="--")
    axes[0][0].legend()
    figure.suptitle("Matched development cases; descriptive evidence, human review pending")
    figure.savefig(output / "scenario_comparison.png", dpi=150)


def run(output, *, replications=1000, calibration_trials=1000):
    for value in (replications, calibration_trials):
        if type(value) is not int or value < 1:
            raise ValueError("Replication/trial counts must be positive integers")
    output = Path(output).expanduser().absolute()
    selected_models = models()
    spec = ExperimentSpec(task=ForecastTask(
        target_description="Next daily observation in a controlled stable, shift or ramp fixture",
        source_description="Seed-42 Gaussian noise; original stable/shift bytes and a matched new ramp",
        units="Arbitrary synthetic units", minimum_history=30),
        evaluation=EvaluationPlan(final_start=START + timedelta(days=160), final_end=START + timedelta(days=200),
                                  benchmarks=("training-mean", "last-value")),
        evidence=EvidencePlan(required_workflows=("predictive_checks", "prior_influence", "sequential_checks"),
                              replications=replications, seed=1910, interval_level=.95))
    output.mkdir(parents=True, exist_ok=False)
    plan = {"format_version": 1, "kind": "model_review_study", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "scenario_order": ["mean-shift", "stable", "gradual"],
        "windows": WINDOWS, "experiment": spec.to_dict(), "interval_levels": [.8, .95],
        "generator": {"seed": 42, "count": 200, "noise": "random.Random(42).gauss(0,1), shared across scenarios",
            "stable": "10 + noise[t]", "mean-shift": "10 + 5*I[t>=100] + noise[t]",
            "gradual": "10 + .05*max(0,t-99) + noise[t]"},
        "calibration": {"trials": calibration_trials, "observation_count": 20, "seed": 1911,
            "scope": "Fresh model-level prior/data exercises, not historical or per-origin calibration"},
        "models": {name: {"reference": model.configuration,
                          "alternatives": {key: alt.configuration for key, alt in alternatives.items()}}
                   for name, (model, alternatives) in selected_models.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": {filename: hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest()
                          for _, filename in MODEL_CONFIGS.values()},
        "scope": "Known synthetic development exercise; all alternatives descriptive; no automatic approval or tuning",
        "results": []}
    write_json(output / "study.json", plan)
    plan["snapshots"] = generate_sources(output)
    write_json(output / "study.json", plan)
    calibration = {}
    for name, (model, _) in selected_models.items():
        print(f"Calibration: {name}, {calibration_trials} trials", flush=True)
        calibration[name] = run_calibration(model=model, observation_count=20, trials=calibration_trials,
            seed=1911, output_directory=output / "calibration" / name)
        plot_calibration(calibration[name])
    for scenario in plan["scenario_order"]:
        common = dict(dataset=scenario, snapshot_id=plan["snapshots"][scenario], store=output / "store", experiment=spec)
        for name, (model, alternatives) in selected_models.items():
            base = output / scenario / name
            print(f"Review: {scenario}/{name}", flush=True)
            forecast = run_backtest(**common, model=model, interval_levels=(.8, .95), output_directory=base / "reference")
            plot_run(forecast)
            evidence = run_experiment_evidence(**common, model=model, scenarios=alternatives,
                issued_at=START + timedelta(days=160) - timedelta(seconds=1), output_directory=base / "evidence")
            plot_experiment_evidence(evidence)
            review = build_experiment_review(forecast_directory=forecast, evidence_directory=evidence,
                calibration_directories=(calibration[name],), require_calibration=True, output_directory=base / "review")
            status = read_json(review / "report.json")["status"]
            plan["results"].append({"scenario": scenario, "model": name, "review_status": status})
            write_json(output / "study.json", plan)
            for prior, alternative in alternatives.items():
                print(f"Prior forecast: {scenario}/{name}/{prior}", flush=True)
                path = run_backtest(**common, model=alternative, interval_levels=(.8, .95), output_directory=base / prior)
                if read_json(path / "manifest.json")["status"] != "completed":
                    plan["results"].append({"scenario": scenario, "model": name, "prior": prior, "forecast_status": "failed"})
                    write_json(output / "study.json", plan)
        for first, second in (("constant", "learned-noise"), ("constant", "joint"), ("learned-noise", "joint")):
            compare_runs(output / scenario / first / "reference", output / scenario / second / "reference",
                         output_directory=output / scenario / "comparisons" / f"{first}-vs-{second}")
    # Failed/partial forecasts cannot silently enter the complete-cohort summary.
    try:
        summary = summarize(output)
        plot_summary(output, summary)
    except ValueError as error:
        plan["summary_error"] = str(error)
    plan["status"] = ("completed" if len(plan["results"]) == 9 and "summary_error" not in plan
                      and all(row.get("review_status") == "ready_for_review" for row in plan["results"]) else "failed")
    plan["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output / "study.json", plan)
    print(f"Study: {plan['status']}; saved {output}", flush=True)
    return 0 if plan["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--calibration-trials", type=int, default=1000)
    args = parser.parse_args()
    return run(args.output_directory, replications=args.replications, calibration_trials=args.calibration_trials)


if __name__ == "__main__":
    raise SystemExit(main())
