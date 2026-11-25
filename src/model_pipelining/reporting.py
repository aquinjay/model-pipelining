"""Draw a chart from saved evaluation results; never fit or call a model."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import NormalDist

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from model_pipelining.decision_value import mean_murphy_curve, murphy_dominance
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.distribution_scoring import read_saved_distribution_scores
from model_pipelining.robust_scoring import (
    summarize_loss_series,
    summarize_paired_differences,
    summarize_point_errors,
)
from model_pipelining.uncertainty import interval_probabilities


def plot_fit(run_directory: str | Path, *, origin_index: int) -> Path:
    """Draw saved constant-mean audit traces/densities with warmup separation.

Reads finished run/fit evidence only; no fitting, input-store access, or sampler
imports. A partial failed fit can be drawn if it retained some draws. Exact-only
fits have no trace. Existing output is rejected.
    """
    if type(origin_index) is not int or origin_index < 0:
        raise ValueError("origin_index must be a nonnegative integer")
    _read_run(run_directory)
    directory = Path(run_directory).expanduser() / "fits" / f"{origin_index:06d}"
    summary = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    if summary["format_version"] != 1:
        raise ValueError("Unsupported fit format")
    metadata = summary["metadata"]
    if metadata.get("inference") != "analytic_with_mcmc_audit":
        raise ValueError("This fit has no MCMC audit trace")
    output = directory / "trace.png"
    if output.exists():
        raise FileExistsError(output)
    with (directory / "draws.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("This fit retained no draws")
    warmup = metadata["sampler"]["settings"]["warmup"]
    posterior = metadata["posterior"]
    references = metadata.get("reference_distributions")
    if references is None:
        references = [
            {"column": "mu", "family": "normal", "label": "mu (parameter)",
             "loc": posterior["mean"], "scale": math.sqrt(posterior["variance"])},
            {"column": "future_value", "family": "normal", "label": "future observation",
             "loc": posterior["mean"], "scale": math.sqrt(metadata["predictive_variance"])},
        ]
    figure = Figure(figsize=(12, 3.5 * len(references)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(references), 2, squeeze=False)
    for chain in sorted({int(row["chain"]) for row in rows}):
        selected = [row for row in rows if int(row["chain"]) == chain]
        for index, reference in enumerate(references):
            column = reference["column"]
            times = [int(row["draw"]) + (0 if row["warmup"] == "True" else warmup) for row in selected]
            values = [float(row[column]) for row in selected]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Invalid saved draws")
            axes[index, 0].plot(times, values, linewidth=0.5, alpha=0.65, label=f"chain {chain}")
            retained = [float(row[column]) for row in selected if row["warmup"] == "False"]
            if retained:
                axes[index, 1].hist(retained, bins=40, density=True, histtype="step", alpha=0.7)
    for index, reference in enumerate(references):
        family = reference["family"]
        if family == "normal":
            distribution = NormalDist(reference["loc"], reference["scale"])
            lower, upper = distribution.inv_cdf(0.001), distribution.inv_cdf(0.999)
        else:
            from scipy.stats import invgamma, t
            if family == "student_t":
                distribution = t(df=reference["df"], loc=reference["loc"], scale=reference["scale"])
            elif family == "inverse_gamma":
                distribution = invgamma(a=reference["shape"], scale=reference["scale"])
            else:
                raise ValueError(f"Unsupported saved reference distribution: {family}")
            lower, upper = map(float, distribution.ppf([0.001, 0.999]))
        if not all(math.isfinite(value) for value in (lower, upper)) or lower >= upper:
            raise ValueError("Unrepresentable saved reference plotting range")
        grid = [lower + (upper - lower) * i / 200 for i in range(201)]
        axes[index, 1].plot(grid, [distribution.pdf(value) for value in grid], color="black", label="exact reference")
        axes[index, 0].axvspan(0, warmup, alpha=0.1, color="grey", label="excluded warmup")
        axes[index, 0].set(xlabel="iteration", ylabel=reference["label"])
        axes[index, 1].set(xlabel=reference["label"], ylabel="retained-draw density")
        axes[index, 0].legend(fontsize="small")
        axes[index, 1].legend(fontsize="small")
    figure.suptitle(f"Origin {origin_index}: {summary['status']} MCMC audit; forecasts use exact integration")
    figure.savefig(output, dpi=150)
    return output


def _read_run(run_directory: str | Path) -> tuple[dict, dict, list[dict]]:
    """Read finished evidence without touching the input store or any model."""
    run = Path(run_directory).expanduser()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] not in (1, 2, 3):
        raise ValueError("Unsupported run format version")
    if manifest["status"] not in ("completed", "failed"):
        raise ValueError("Reporting requires a finished run")
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    with (run / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return manifest, scores, rows


def _saved_interval(run: str | Path, manifest: dict, rows: list[dict], level: float | None = None):
    """Read one central interval from frozen quantiles, including failed gaps."""
    if not manifest["model"].get("supports_quantiles", False):
        if level is not None:
            raise ValueError("This saved run has no observation quantiles")
        return None
    levels = manifest["evaluation"]["interval_levels"]
    chosen = max(levels) if level is None else level
    if chosen not in levels:
        raise ValueError("The requested interval level was not saved by this run")
    lower_probability, upper_probability = interval_probabilities(chosen)
    with (Path(run).expanduser() / "quantiles.csv").open(encoding="utf-8", newline="") as stream:
        quantiles = {(row["target_time"], float(row["probability"])): row for row in csv.DictReader(stream)}
    lower, upper = [], []
    for row in rows:
        first = quantiles[(row["target_time"], lower_probability)]
        second = quantiles[(row["target_time"], upper_probability)]
        if first["status"] == second["status"] == "passed":
            first_value, second_value = float(first["value"]), float(second["value"])
            if not math.isfinite(first_value) or not math.isfinite(second_value) or first_value > second_value:
                raise ValueError("Invalid saved interval bounds")
            lower.append(first_value)
            upper.append(second_value)
        else:
            lower.append(float("nan"))
            upper.append(float("nan"))
    return chosen, lower, upper


def _draw_uncertainty(axes, times, run, manifest, rows, *, level=None, method="auto", prefix="", color=None):
    """Prefer supplied highest-density regions; never fill their internal gaps."""
    if method not in ("auto", "highest_density", "equal_tail"):
        raise ValueError("interval_method must be auto, highest_density, or equal_tail")
    model = manifest["model"]
    if method == "auto":
        if model.get("supports_highest_density", False):
            method = "highest_density"
        elif model.get("supports_quantiles", False):
            method = "equal_tail"
        elif level is None:
            return
        else:
            raise ValueError("This saved run has no observation uncertainty")
    if method == "equal_tail":
        if not model.get("supports_quantiles", False):
            raise ValueError("This saved run has no observation quantiles")
        chosen, lower, upper = _saved_interval(run, manifest, rows, level)
        axes.fill_between(times, lower, upper, alpha=0.15, color=color,
                          label=f"{prefix}{100 * chosen:g}% equal-tail prediction interval")
        return
    if not model.get("supports_highest_density", False):
        raise ValueError("This saved run has no highest-density prediction regions")
    levels = manifest["evaluation"]["highest_density_levels"]
    chosen = max(levels) if level is None else level
    if chosen not in levels:
        raise ValueError("The requested interval level was not saved by this run")
    regions = {row["target_time"]: {} for row in rows}
    with (Path(run).expanduser() / "highest_density.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["level"]) != chosen or row["status"] != "passed":
                continue
            lower, upper = float(row["lower"]), float(row["upper"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("Invalid saved highest-density segment")
            regions[row["target_time"]][int(row["segment_index"])] = (lower, upper)
    for index in range(max((len(segments) for segments in regions.values()), default=0)):
        pairs = [regions[row["target_time"]].get(index, (float("nan"), float("nan"))) for row in rows]
        axes.fill_between(times, [pair[0] for pair in pairs], [pair[1] for pair in pairs], alpha=0.15, color=color,
                          label=f"{prefix}{100 * chosen:g}% highest-density prediction region" if index == 0 else None)


def plot_run(
    run_directory: str | Path, *, interval_level: float | None = None, interval_method: str = "auto",
) -> Path:
    """Save forecast.png from an existing run, refusing to overwrite a chart.

    Failed predictions appear as gaps. The title includes run status and scored
    coverage so a partial run is not presented as a complete success. Draw the
    widest saved highest-density region by default, falling back to equal-tail
    intervals only when highest-density output is unavailable.
    """
    run = Path(run_directory).expanduser()
    manifest, scores, rows = _read_run(run)
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    times = [datetime.fromisoformat(row["target_time"]) for row in rows]
    axes.plot(times, [float(row["actual"]) for row in rows], label="Actual", linewidth=1)
    axes.plot(times, [float(row["prediction"]) if row["prediction"] else float("nan")
                      for row in rows], label="Forecast", linewidth=1.5)
    _draw_uncertainty(axes, times, run, manifest, rows, level=interval_level, method=interval_method, color="C1")
    if scores.get("benchmarks", {}).get("runs"):
        with (run / "benchmark_predictions.csv").open(encoding="utf-8", newline="") as stream:
            benchmark_rows = list(csv.DictReader(stream))
        for index, benchmark in enumerate(scores["benchmarks"]["runs"], start=2):
            selected = [row for row in benchmark_rows if row["benchmark"] == benchmark["name"]]
            axes.plot([datetime.fromisoformat(row["target_time"]) for row in selected],
                      [float(row["prediction"]) if row["prediction"] else float("nan") for row in selected],
                      label=f"Benchmark: {benchmark['name']}", linestyle="--", linewidth=1, color=f"C{index % 10}")
    case_counts = f"points {scores['scored_count']}/{scores['expected_count']}"
    uncertainty = scores.get("uncertainty", {})
    if uncertainty.get("status") not in (None, "not_applicable"):
        case_counts += f"; uncertainty {uncertainty['scored_count']}/{uncertainty['expected_count']}"
    axes.set(title=f"{manifest['model']['name']} — {manifest['status']} ({case_counts})",
             xlabel="Target date (UTC)", ylabel="Value (input units)")
    axes.legend()
    figure.autofmt_xdate()
    destination = run / "forecast.png"
    with destination.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    return destination


def compare_runs(
    first_run: str | Path,
    second_run: str | Path,
    *,
    output_directory: str | Path,
) -> Path:
    """Compare complete runs on identical inputs and cases; support old point runs.

    Save comparison.json and comparison.png in a new directory and return its
    absolute path. Require the same permanent input, task, training selections,
    target dates, and actuals. Refuse failed/partial runs rather than comparing
    selectively successful cases. Source artifacts are never rewritten.
    """
    manifests, scores, evaluations, coordinates = [], [], [], []
    for run in (first_run, second_run):
        manifest, score, rows = _read_run(run)
        if (
            manifest["status"] != "completed" or not score["complete"]
            or score["failed_count"] != 0 or not rows
            or score["scored_count"] != len(rows)
            or score["expected_count"] != len(rows)
            or any(row["status"] != "passed" for row in rows)
        ):
            raise ValueError("Comparisons require two complete runs with all cases scored")
        with (Path(run).expanduser() / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        keys = ("origin_index", "issued_at", "training_cutoff", "training_start", "training_count", "target_time")
        optional_keys = ("training_end", "history_sha256")
        coordinates.append([tuple(row[key] for key in keys) + tuple(row.get(key) for key in optional_keys)
                            for row in predictions])
        if (
            [row["target_time"] for row in predictions] != [row["target_time"] for row in rows]
            or len({row["target_time"] for row in rows}) != len(rows)
        ):
            raise ValueError("Saved forecast and evaluation cases do not align")
        manifests.append(manifest)
        scores.append(score)
        evaluations.append(rows)
    first_input, second_input = (manifest["input"] for manifest in manifests)
    # Version-1 specs have the same meaning as version-2 specs with default
    # evidence requirements. Missing experiment metadata stays missing.
    experiments = [ExperimentSpec.from_dict(manifest["experiment"]).to_dict()
                   if manifest.get("experiment") is not None else None for manifest in manifests]
    if (
        (first_input["dataset"], first_input["snapshot_id"])
        != (second_input["dataset"], second_input["snapshot_id"])
        or manifests[0]["task"] != manifests[1]["task"]
        or experiments[0] != experiments[1]
        or coordinates[0] != coordinates[1]
        or [row["actual"] for row in evaluations[0]] != [row["actual"] for row in evaluations[1]]
        or [row.get("actual_available_at") for row in evaluations[0]] != [row.get("actual_available_at") for row in evaluations[1]]
    ):
        raise ValueError("Comparison inputs, tasks, training selections, and actuals must match")

    point_summaries = [summarize_point_errors(rows) for rows in evaluations]
    abs_diffs = [
        abs(float(first["prediction"]) - float(first["actual"]))
        - abs(float(second["prediction"]) - float(second["actual"]))
        for first, second in zip(evaluations[0], evaluations[1])
    ]
    sq_diffs = [
        (float(first["prediction"]) - float(first["actual"])) ** 2
        - (float(second["prediction"]) - float(second["actual"])) ** 2
        for first, second in zip(evaluations[0], evaluations[1])
    ]
    first_pairs = [(float(row["prediction"]), float(row["actual"])) for row in evaluations[0]]
    second_pairs = [(float(row["prediction"]), float(row["actual"])) for row in evaluations[1]]
    all_point_values = [value for pairs in (first_pairs, second_pairs) for pair in pairs for value in pair]
    low_val, high_val = min(all_point_values), max(all_point_values)
    murphy_grid = (
        [low_val] if low_val == high_val
        else [low_val + (high_val - low_val) * i / 40.0 for i in range(41)]
    )
    point_murphy = {
        "kind": "mean_elementary_score",
        **murphy_dominance(
            mean_murphy_curve(first_pairs, thresholds=murphy_grid),
            mean_murphy_curve(second_pairs, thresholds=murphy_grid),
        ),
    }
    point_comparison = {
        "status": "completed",
        "difference_direction": "first_minus_second; negative means first has lower error",
        "runs": point_summaries,
        "paired_absolute_error_differences": summarize_paired_differences(abs_diffs),
        "paired_squared_error_differences": summarize_paired_differences(sq_diffs),
        "murphy_comparison": point_murphy,
    }

    distribution_runs = []
    for run, manifest, rows in zip((first_run, second_run), manifests, evaluations):
        if manifest["model"].get("supports_distribution", False):
            with (Path(run)/"predictions.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            distribution_runs.append(read_saved_distribution_scores(run, manifest, predictions, rows))
        else:
            distribution_runs.append(None)
    distribution = {"status": "not_applicable", "reason": "Both runs must declare complete full predictive distributions"}
    if all(item is not None for item in distribution_runs):
        summaries, case_scores = zip(*distribution_runs)
        both_have_log = all(summary["mean_log_score"] is not None for summary in summaries)
        differences = [
            {
                "target_time": first["target_time"],
                "crps_first_minus_second": first["crps"] - second["crps"],
                "log_score_first_minus_second": (
                    first["log_score"] - second["log_score"] if both_have_log else None
                ),
            }
            for first, second in zip(*case_scores)
        ]
        if any(
            value is not None and not math.isfinite(value)
            for row in differences
            for key, value in row.items()
            if key != "target_time"
        ):
            raise ValueError("Nonfinite distribution score difference")
        crps_distributions = [summarize_loss_series([row["crps"] for row in rows], nonnegative=True)
                              for rows in case_scores]
        log_score_distributions = [
            summarize_loss_series([row["log_score"] for row in rows], nonnegative=False)
            if summary["mean_log_score"] is not None else None
            for summary, rows in zip(summaries, case_scores)
        ]
        paired_crps = summarize_paired_differences([row["crps_first_minus_second"] for row in differences])
        paired_log = (
            summarize_paired_differences([row["log_score_first_minus_second"] for row in differences])
            if both_have_log else None
        )
        distribution = {"status": "completed", "method": summaries[0]["method"], "runs": list(summaries),
                        "crps_distributions": crps_distributions,
                        "log_score_distributions": log_score_distributions,
                        "case_differences": differences, "difference_direction": "first_minus_second; lower losses are better",
                        "mean_crps_first_minus_second": summaries[0]["mean_crps"] - summaries[1]["mean_crps"],
                        "mean_log_score_first_minus_second": (
                            summaries[0]["mean_log_score"] - summaries[1]["mean_log_score"]
                            if both_have_log else None
                        ),
                        "paired_crps_differences": paired_crps,
                        "paired_log_score_differences": paired_log,
                        "chart": "distribution_scores.png"}

    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    times = [datetime.fromisoformat(row["target_time"]) for row in evaluations[0]]
    axes.plot(times, [float(row["actual"]) for row in evaluations[0]], label="Actual", linewidth=1, alpha=0.65)
    for index, (manifest, score, rows) in enumerate(zip(manifests, scores, evaluations), start=1):
        line, = axes.plot(times, [float(row["prediction"]) for row in rows], linewidth=1.5,
                          label=f"{index}: {manifest['model']['name']}")
        _draw_uncertainty(axes, times, (first_run, second_run)[index - 1], manifest, rows,
                          prefix=f"{index}: ", color=line.get_color())
    role = manifests[0]["task"].get("evaluation_role", "development")
    axes.set(title=f"{role.capitalize()} comparison — {len(times)} common forecast dates",
             xlabel="Target date (UTC)", ylabel="Value (input units)")
    axes.legend()
    figure.autofmt_xdate()
    output = Path(output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    uncertainty = {"status": "not_applicable", "intervals": [], "highest_density_regions": [],
                   "reason": "Both runs must supply complete uncertainty with common methods and levels"}
    uncertainty_scores = [score.get("uncertainty", {}) for score in scores]
    if all(score.get("status") == "completed" for score in uncertainty_scores):
        for kind in ("highest_density_regions", "intervals"):
            first_levels = {row["level"]: row for row in uncertainty_scores[0].get(kind, [])}
            second_levels = {row["level"]: row for row in uncertainty_scores[1].get(kind, [])}
            common = sorted(first_levels.keys() & second_levels.keys())
            uncertainty[kind] = [{"level": level, "runs": [first_levels[level], second_levels[level]]}
                                 for level in common]
        if uncertainty["intervals"] or uncertainty["highest_density_regions"]:
            uncertainty["status"] = "completed"
            uncertainty.pop("reason")
    summary = {
        "format_version": 2,
        "input": {"dataset": first_input["dataset"], "snapshot_id": first_input["snapshot_id"]},
        "task": manifests[0]["task"],
        "experiment": manifests[0].get("experiment"),
        "case_count": len(times),
        "runs": [{"path": str(Path(run).expanduser().absolute()), "model": manifest["model"], "scores": score}
                 for run, manifest, score in zip((first_run, second_run), manifests, scores)],
        "point_comparison": point_comparison,
        "uncertainty_comparison": uncertainty,
        "distribution_comparison": distribution,
        "interpretation": f"Declared {role} comparison; no automatic selection, tuning, or human approval",
    }
    with (output / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    first_name = manifests[0]["model"]["name"]
    second_name = manifests[1]["model"]["name"]
    md_lines = [
        f"# Saved forecast comparison: `{first_name}` vs `{second_name}`",
        "",
        f"Dataset: `{first_input['dataset']}` · Evaluation role: `{role}` · Common cases: **{len(times)}** "
        f"({evaluations[0][0]['target_time']} to {evaluations[0][-1]['target_time']}).",
        "",
        "[Machine-readable comparison](comparison.json) · [Forecast chart](comparison.png)",
        "",
        "## Point error & bias summary",
        "",
        "| Model | Mean bias | Median bias | MdAE | MAE | p95 abs error | Top-5% abs share | RMSE (sensitive) | Top-5% sq share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, (label, rp) in enumerate(((f"1: {first_name}", point_summaries[0]), (f"2: {second_name}", point_summaries[1])), 1):
        md_lines.append(
            f"| `{label}` | {rp['mean_error']:+.4g} | {rp['median_error']:+.4g} | {rp['mdae']:.4g} | "
            f"{rp['mae']:.4g} | {rp['p95_absolute_error']:.4g} | {rp['top_5pct_abs_error_share']:.1%} | "
            f"{rp['rmse']:.4g} | {rp['top_5pct_sq_error_share']:.1%} |"
        )
    pa = point_comparison["paired_absolute_error_differences"]
    ps = point_comparison["paired_squared_error_differences"]
    pm = point_comparison["murphy_comparison"]
    md_lines.extend([
        "",
        f"- **Paired |error| diff (1 − 2; negative favors 1):** mean `{pa['mean_difference']:+.4g}` "
        f"(Bartlett HAC SE `{pa['mean_difference_hac_se']:.4g}` at lag {pa['hac_bartlett_lag']}; "
        f"median `{pa['median_difference']:+.4g}`; 1 won {pa['first_win_rate']:.1%}, 2 won {pa['second_win_rate']:.1%}; "
        f"top-5% abs diff share {pa['top_5pct_abs_difference_share']:.1%})."
        if pa["mean_difference_hac_se"] is not None else
        f"- **Paired |error| diff (1 − 2):** mean `{pa['mean_difference']:+.4g}`, median `{pa['median_difference']:+.4g}`.",
        f"- **Paired squared-error diff (1 − 2):** mean `{ps['mean_difference']:+.4g}` "
        f"(Bartlett HAC SE `{ps['mean_difference_hac_se']:.4g}`; top-5% abs diff share {ps['top_5pct_abs_difference_share']:.1%})."
        if ps["mean_difference_hac_se"] is not None else
        f"- **Paired squared-error diff (1 − 2):** mean `{ps['mean_difference']:+.4g}`.",
        f"- **Mean Murphy dominance (41 thresholds):** `{pm['dominance_status']}` "
        f"(1 better on {pm['first_better_count']}/{pm['threshold_count']}, 2 better on {pm['second_better_count']}/{pm['threshold_count']}).",
    ])
    if distribution["status"] == "completed":
        md_lines.extend([
            "",
            "## Full predictive distribution summary",
            "",
            "[Distribution & cumulative CRPS chart](distribution_scores.png)",
            "",
            "| Model | Mean CRPS | Median CRPS | p95 CRPS | Top-5% CRPS share | Mean log score | Median log score | Top-5% log mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for idx, label in enumerate((f"1: {first_name}", f"2: {second_name}")):
            cd = distribution["crps_distributions"][idx]
            ld = distribution["log_score_distributions"][idx]
            log_cells = (
                f"{ld['mean']:.4g} | {ld['median']:.4g} | {ld['top_5pct_mean']:.4g}"
                if ld is not None else "n/a (empirical) | n/a | n/a"
            )
            md_lines.append(
                f"| `{label}` | {cd['mean']:.4g} | {cd['median']:.4g} | {cd['p95']:.4g} | "
                f"{cd['top_5pct_share']:.1%} | {log_cells} |"
            )
        pc_diff = distribution["paired_crps_differences"]
        md_lines.append("")
        md_lines.append(
            f"- **Paired CRPS diff (1 − 2; negative favors 1):** mean `{pc_diff['mean_difference']:+.4g}` "
            f"(Bartlett HAC SE `{pc_diff['mean_difference_hac_se']:.4g}` at lag {pc_diff['hac_bartlett_lag']}; "
            f"median `{pc_diff['median_difference']:+.4g}`; 1 won {pc_diff['first_win_rate']:.1%}, "
            f"2 won {pc_diff['second_win_rate']:.1%}; top-5% abs diff share {pc_diff['top_5pct_abs_difference_share']:.1%})."
            if pc_diff["mean_difference_hac_se"] is not None else
            f"- **Paired CRPS diff (1 − 2):** mean `{pc_diff['mean_difference']:+.4g}`, median `{pc_diff['median_difference']:+.4g}`."
        )
        pl_diff = distribution["paired_log_score_differences"]
        if pl_diff is not None and pl_diff["mean_difference_hac_se"] is not None:
            md_lines.append(
                f"- **Paired log-score diff (1 − 2; outlier-sensitive):** mean `{pl_diff['mean_difference']:+.4g}` "
                f"(Bartlett HAC SE `{pl_diff['mean_difference_hac_se']:.4g}`; median `{pl_diff['median_difference']:+.4g}`; "
                f"top-5% abs diff share {pl_diff['top_5pct_abs_difference_share']:.1%})."
            )
    if uncertainty["status"] == "completed":
        md_lines.extend(["", "## Common uncertainty summary", ""])
        for item in uncertainty.get("highest_density_regions", []):
            r1, r2 = item["runs"]
            md_lines.append(
                f"- **{int(round(100 * item['level']))}% HPD:** "
                f"1 cov={r1['coverage']:.1%} (width {r1['mean_total_width']:.4g}) vs "
                f"2 cov={r2['coverage']:.1%} (width {r2['mean_total_width']:.4g})."
            )
        for item in uncertainty.get("intervals", []):
            r1, r2 = item["runs"]
            md_lines.append(
                f"- **{int(round(100 * item['level']))}% equal-tail:** "
                f"1 cov={r1['coverage']:.1%} (Winkler {r1['mean_interval_score']:.4g}) vs "
                f"2 cov={r2['coverage']:.1%} (Winkler {r2['mean_interval_score']:.4g})."
            )
    md_lines.extend(["", f"*{summary['interpretation']}.*", ""])
    with (output / "comparison.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(md_lines))
    with (output / "comparison.png").open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    if distribution["status"] == "completed":
        score_figure = Figure(figsize=(10, 8.2), layout="constrained")
        FigureCanvasAgg(score_figure)
        score_axes = score_figure.subplots(3, 1, sharex=True)
        for index, (_, rows) in enumerate(distribution_runs):
            score_axes[0].plot(times, [row["crps"] for row in rows], label=f"{index+1}: {manifests[index]['model']['name']}")
        score_axes[0].set(ylabel="CRPS (target units)", title="Saved distribution losses — lower is better")
        score_axes[0].legend()
        plotted_log = False
        for index, (summary_item, rows) in enumerate(distribution_runs):
            if summary_item["mean_log_score"] is not None:
                score_axes[1].plot(times, [row["log_score"] for row in rows],
                                   label=f"{index+1}: {manifests[index]['model']['name']}")
                plotted_log = True
        score_axes[1].set(ylabel="Negative log density", title="Saved negative log density (continuous laws only)")
        if plotted_log:
            score_axes[1].legend()
        else:
            score_axes[1].text(0.5, 0.5, "Not applicable: discrete empirical_samples laws have no continuous density",
                               ha="center", va="center", transform=score_axes[1].transAxes)
        cumulative_crps_diff = []
        running_total = 0.0
        for row in distribution["case_differences"]:
            running_total += row["crps_first_minus_second"]
            cumulative_crps_diff.append(running_total)
        score_axes[2].plot(times, cumulative_crps_diff, color="#28547B", linewidth=1.6)
        score_axes[2].axhline(0.0, color="#485865", linestyle="--", linewidth=1)
        score_axes[2].set(
            ylabel="Cumulative CRPS diff",
            xlabel="Target date (UTC)",
            title=(f"Cumulative CRPS difference (1: {manifests[0]['model']['name']} minus "
                   f"2: {manifests[1]['model']['name']}; below 0 favors 1)"),
        )
        score_figure.autofmt_xdate()
        with (output / "distribution_scores.png").open("xb") as stream:
            score_figure.savefig(stream, format="png", dpi=150)
    return output
