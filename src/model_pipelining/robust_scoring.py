"""Robust, tail-concentration, and paired-stability summaries for saved forecast losses.

All functions operate on finite numeric sequences from saved evaluation files
without calling models or mutating artifacts. Standard errors use a deterministic
Bartlett (Newey-West) kernel so serially correlated forecast errors are not
treated as independent observations.
"""

import math
from numbers import Real


def _finite_list(values, name: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} values must be finite real numbers")
        result.append(float(value))
    if not result:
        raise ValueError(f"At least one {name} value is required")
    return result


def linear_quantile(ordered: list[float], probability: float) -> float:
    """Return a deterministic linear-interpolated quantile from a sorted nonempty list."""
    if not ordered:
        raise ValueError("Cannot compute quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def newey_west_se(differences) -> tuple[float | None, float | None, int]:
    """Return (naive_se, bartlett_hac_se, max_lag) for a time-ordered paired difference series."""
    values = _finite_list(differences, "difference")
    n = len(values)
    if n < 2:
        return None, None, 0
    avg = math.fsum(values) / n
    centered = [value - avg for value in values]
    gamma0 = math.fsum(value * value for value in centered) / n
    naive_var = math.fsum(value * value for value in centered) / (n - 1)
    naive_se = math.sqrt(max(naive_var, 0.0) / n)
    max_lag = min(n - 1, int(n ** (1.0 / 3.0)))
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_lag = math.fsum(centered[i] * centered[i - lag] for i in range(lag, n)) / n
        long_run_var += 2.0 * weight * gamma_lag
    hac_se = math.sqrt(max(long_run_var, 0.0) / n)
    return naive_se, hac_se, max_lag


def summarize_loss_series(losses, *, nonnegative: bool = True) -> dict:
    """Summarize a loss series with median, upper tails, and worst-5% concentration.

    For nonnegative losses (CRPS, pinball, absolute/squared error, expense),
    top_5pct_share gives the fraction of total loss coming from the worst 5% of
    origins. For signed losses (such as negative log density, which can be
    negative when densities exceed 1), top_5pct_share is omitted in favor of
    top_5pct_mean and mean_minus_median.
    """
    values = _finite_list(losses, "loss")
    if nonnegative and any(value < 0.0 for value in values):
        raise ValueError("Nonnegative loss series cannot contain negative values")
    ordered = sorted(values)
    count = len(ordered)
    total = math.fsum(ordered)
    mean_value = total / count
    median_value = linear_quantile(ordered, 0.5)
    top_count = max(1, math.ceil(0.05 * count))
    top_total = math.fsum(ordered[-top_count:])
    summary = {
        "count": count,
        "mean": mean_value,
        "median": median_value,
        "p90": linear_quantile(ordered, 0.9),
        "p95": linear_quantile(ordered, 0.95),
        "max": ordered[-1],
        "mean_minus_median": mean_value - median_value,
        "top_5pct_case_count": top_count,
        "top_5pct_mean": top_total / top_count,
    }
    if nonnegative:
        summary["total"] = total
        summary["top_5pct_share"] = (top_total / total) if total > 0.0 else 0.0
    return summary


def summarize_point_errors(rows) -> dict:
    """Compute robust point-error, bias, and concentration summaries from saved evaluation rows."""
    errors = []
    abs_errors = []
    sq_errors = []
    for row in rows:
        if row.get("status") != "passed":
            continue
        raw_error = row.get("error")
        if raw_error is None or raw_error == "":
            error = float(row["prediction"]) - float(row["actual"])
        else:
            error = float(raw_error)
        if not math.isfinite(error):
            raise ValueError("Nonfinite saved forecast error")
        sq = error * error
        if not math.isfinite(sq):
            raise ValueError("Nonfinite saved squared error")
        errors.append(error)
        abs_errors.append(abs(error))
        sq_errors.append(sq)
    if not errors:
        raise ValueError("At least one passed evaluation row is required")
    ordered_errors = sorted(errors)
    abs_summary = summarize_loss_series(abs_errors, nonnegative=True)
    sq_summary = summarize_loss_series(sq_errors, nonnegative=True)
    count = len(errors)
    return {
        "count": count,
        "mean_error": math.fsum(errors) / count,
        "median_error": linear_quantile(ordered_errors, 0.5),
        "mae": abs_summary["mean"],
        "mdae": abs_summary["median"],
        "rmse": math.sqrt(sq_summary["mean"]),
        "p90_absolute_error": abs_summary["p90"],
        "p95_absolute_error": abs_summary["p95"],
        "max_absolute_error": abs_summary["max"],
        "top_5pct_case_count": abs_summary["top_5pct_case_count"],
        "top_5pct_abs_error_share": abs_summary["top_5pct_share"],
        "top_5pct_sq_error_share": sq_summary["top_5pct_share"],
    }


def summarize_paired_differences(differences) -> dict:
    """Summarize time-ordered paired loss differences (first minus second; negative favors first)."""
    values = _finite_list(differences, "difference")
    count = len(values)
    mean_diff = math.fsum(values) / count
    ordered = sorted(values)
    median_diff = linear_quantile(ordered, 0.5)
    first_better = sum(1 for value in values if value < 0.0)
    second_better = sum(1 for value in values if value > 0.0)
    ties = count - first_better - second_better
    abs_sorted = sorted((abs(value) for value in values), reverse=True)
    total_abs = math.fsum(abs_sorted)
    top_count = max(1, math.ceil(0.05 * count))
    top_abs_share = math.fsum(abs_sorted[:top_count]) / total_abs if total_abs > 0.0 else 0.0
    naive_se, hac_se, hac_lag = newey_west_se(values)
    return {
        "count": count,
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "mean_difference_naive_se": naive_se,
        "mean_difference_hac_se": hac_se,
        "hac_bartlett_lag": hac_lag,
        "first_better_count": first_better,
        "second_better_count": second_better,
        "tie_count": ties,
        "first_win_rate": first_better / count,
        "second_win_rate": second_better / count,
        "tie_rate": ties / count,
        "top_5pct_case_count": top_count,
        "top_5pct_abs_difference_share": top_abs_share,
    }
