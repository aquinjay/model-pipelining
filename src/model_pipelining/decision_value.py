"""Cost-loss decisions for binary and continuous asymmetric tasks.

Binary expenses are expressed in units of the unprotected loss L, with
cost-loss ratio alpha = C / L in (0, 1). Continuous asymmetric (newsvendor)
expenses use explicit positive unit under-forecast and over-forecast costs.
"""

import math
from numbers import Integral, Real


def _alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(alpha)


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return float(value)


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_cost(value, name):
    result = _finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _validate_event(event):
    if isinstance(event, bool) or not isinstance(event, Integral) or event not in (0, 1):
        raise ValueError("event must be 0 or 1")


def _iterate(values, name):
    try:
        return iter(values)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error


def _linear_quantile(ordered: list[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_expenses(expenses) -> dict:
    """Return robust summary and tail-concentration diagnostics for case expenses."""
    values = [_finite_real(value, "expense") for value in _iterate(expenses, "expenses")]
    if not values:
        raise ValueError("At least one expense is required")
    if any(value < 0 for value in values):
        raise ValueError("Expenses must be nonnegative")
    ordered = sorted(values)
    count = len(ordered)
    total = math.fsum(ordered)
    top_count = max(1, math.ceil(0.05 * count))
    top_total = math.fsum(ordered[-top_count:])
    return {
        "count": count,
        "total_expense": total,
        "mean_expense": total / count,
        "median_expense": _linear_quantile(ordered, 0.5),
        "p90_expense": _linear_quantile(ordered, 0.9),
        "p95_expense": _linear_quantile(ordered, 0.95),
        "max_expense": ordered[-1],
        "top_5pct_case_count": top_count,
        "top_5pct_expense_share": (top_total / total) if total > 0 else 0.0,
    }


def decide(probability, alpha) -> bool:
    """Take protective action when the event probability exceeds the cost-loss ratio."""
    return _probability(probability, "probability") > _alpha(alpha)


def expense(acted, event, alpha) -> float:
    """Return the expense of one binary decision in units of the unprotected loss."""
    if type(acted) is not bool:
        raise ValueError("acted must be a boolean")
    _validate_event(event)
    alpha = _alpha(alpha)
    return alpha if acted else float(event)


def evaluate_decisions(cases, *, alpha, base_rate) -> dict:
    """Evaluate (probability, event) pairs against calibration climatology and perfect foresight."""
    alpha = _alpha(alpha)
    base_rate = _probability(base_rate, "base_rate")
    hits = false_alarms = misses = correct_negatives = 0
    for case in _iterate(cases, "cases"):
        try:
            probability, event = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be a (probability, event) pair") from error
        _validate_event(event)
        if decide(probability, alpha):
            if event:
                hits += 1
            else:
                false_alarms += 1
        elif event:
            misses += 1
        else:
            correct_negatives += 1

    count = hits + false_alarms + misses + correct_negatives
    if not count:
        raise ValueError("At least one case is required")
    observed_event_rate = (hits + misses) / count
    forecast_expense = alpha * ((hits + false_alarms) / count) + misses / count
    climatology_acted = decide(base_rate, alpha)
    climatology_expense = alpha if climatology_acted else observed_event_rate
    perfect_expense = alpha * observed_event_rate
    denominator = climatology_expense - perfect_expense
    return {
        "alpha": alpha,
        "base_rate": base_rate,
        "cases": count,
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct_negatives,
        "mean_forecast_expense": forecast_expense,
        "mean_climatology_expense": climatology_expense,
        "mean_perfect_expense": perfect_expense,
        "relative_value": (climatology_expense - forecast_expense) / denominator if denominator else None,
        "relative_value_reason": None if denominator else "Climatology and perfect forecast have equal mean expense.",
    }


def value_curve(cases, *, alphas, base_rate) -> list[dict]:
    """Evaluate the same cases at each cost-loss ratio in the supplied order."""
    cases = tuple(_iterate(cases, "cases"))
    alphas = tuple(_iterate(alphas, "alphas"))
    if not alphas:
        raise ValueError("At least one alpha is required")
    return [evaluate_decisions(cases, alpha=alpha, base_rate=base_rate) for alpha in alphas]


def asymmetric_quantile_level(under_cost, over_cost) -> float:
    """Return the Bayes-optimal predictive quantile level under_cost / (under_cost + over_cost)."""
    under = _positive_cost(under_cost, "under_cost")
    over = _positive_cost(over_cost, "over_cost")
    total = under + over
    if not math.isfinite(total) or total <= 0:
        raise ValueError("under_cost + over_cost must be finite and positive")
    level = under / total
    if not 0 < level < 1:
        raise ValueError("Asymmetric cost ratio must yield a quantile level strictly between 0 and 1")
    return level


def asymmetric_expense(action, actual, *, under_cost, over_cost) -> float:
    """Return asymmetric linear expense: under_cost * max(actual - action, 0) + over_cost * max(action - actual, 0)."""
    action_value = _finite_real(action, "action")
    actual_value = _finite_real(actual, "actual")
    under = _positive_cost(under_cost, "under_cost")
    over = _positive_cost(over_cost, "over_cost")
    difference = actual_value - action_value
    cost = under * difference if difference >= 0 else over * (-difference)
    if not math.isfinite(cost):
        raise ValueError("Nonfinite asymmetric expense")
    return cost


def evaluate_asymmetric_decisions(cases, *, under_cost, over_cost, baseline_action) -> dict:
    """Evaluate (action, actual) pairs against a constant baseline action and perfect foresight."""
    under = _positive_cost(under_cost, "under_cost")
    over = _positive_cost(over_cost, "over_cost")
    quantile_level = asymmetric_quantile_level(under, over)
    baseline = _finite_real(baseline_action, "baseline_action")
    forecast_expenses = []
    baseline_expenses = []
    under_expenses = []
    over_expenses = []
    under_count = over_count = exact_count = 0
    for case in _iterate(cases, "cases"):
        try:
            action, actual = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be an (action, actual) pair") from error
        action_value = _finite_real(action, "action")
        actual_value = _finite_real(actual, "actual")
        cost = asymmetric_expense(action_value, actual_value, under_cost=under, over_cost=over)
        base_cost = asymmetric_expense(baseline, actual_value, under_cost=under, over_cost=over)
        forecast_expenses.append(cost)
        baseline_expenses.append(base_cost)
        if actual_value > action_value:
            under_count += 1
            under_expenses.append(cost)
        elif actual_value < action_value:
            over_count += 1
            over_expenses.append(cost)
        else:
            exact_count += 1

    count = len(forecast_expenses)
    if not count:
        raise ValueError("At least one case is required")
    mean_forecast = math.fsum(forecast_expenses) / count
    mean_baseline = math.fsum(baseline_expenses) / count
    mean_perfect = 0.0
    denominator = mean_baseline - mean_perfect
    return {
        "under_cost": under,
        "over_cost": over,
        "quantile_level": quantile_level,
        "baseline_action": baseline,
        "cases": count,
        "under_count": under_count,
        "over_count": over_count,
        "exact_count": exact_count,
        "mean_under_expense": math.fsum(under_expenses) / count,
        "mean_over_expense": math.fsum(over_expenses) / count,
        "mean_forecast_expense": mean_forecast,
        "mean_baseline_expense": mean_baseline,
        "mean_perfect_expense": mean_perfect,
        "relative_value": (mean_baseline - mean_forecast) / denominator if denominator > 0 else None,
        "relative_value_reason": None if denominator > 0 else "Baseline and perfect forecast have equal mean expense.",
        "expense_distribution": summarize_expenses(forecast_expenses),
        "baseline_expense_distribution": summarize_expenses(baseline_expenses),
    }


def brier_decomposition(cases, *, base_rate, bins: int = 10) -> dict:
    """Decompose Brier score into reliability, resolution, uncertainty, and within-bin residual.

    Satisfies the exact identity:
        brier_score == reliability - resolution + uncertainty + within_bin_term

    Also reports Expected Calibration Error (ECE), probability bias, and Brier
    Skill Score relative to the predeclared calibration base_rate (never using
    in-sample evaluation prevalence as the baseline forecast).
    """
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    base = _probability(base_rate, "base_rate")
    pairs = []
    for case in _iterate(cases, "cases"):
        try:
            probability, event = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be a (probability, event) pair") from error
        _validate_event(event)
        pairs.append((_probability(probability, "probability"), int(event)))
    if not pairs:
        raise ValueError("At least one case is required")

    count = len(pairs)
    brier = math.fsum((p - y) ** 2 for p, y in pairs) / count
    clim_brier = math.fsum((base - y) ** 2 for _, y in pairs) / count
    mean_prob = math.fsum(p for p, _ in pairs) / count
    obs_rate = math.fsum(y for _, y in pairs) / count
    uncertainty = obs_rate * (1.0 - obs_rate)

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for prob, event in pairs:
        index = min(bins - 1, int(prob * bins))
        buckets[index].append((prob, event))

    reliability_terms = []
    resolution_terms = []
    ece_terms = []
    within_terms = []
    bin_records = []
    for index, items in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        n_k = len(items)
        if n_k == 0:
            bin_records.append({
                "lower": lower, "upper": upper, "count": 0,
                "mean_probability": None, "observed_rate": None, "abs_calibration_gap": None,
            })
            continue
        p_k = math.fsum(p for p, _ in items) / n_k
        o_k = math.fsum(y for _, y in items) / n_k
        gap = abs(p_k - o_k)
        reliability_terms.append(n_k * ((p_k - o_k) ** 2))
        resolution_terms.append(n_k * ((o_k - obs_rate) ** 2))
        ece_terms.append(n_k * gap)
        for p, y in items:
            dp = p - p_k
            within_terms.append(dp * dp - 2.0 * dp * (y - o_k))
        bin_records.append({
            "lower": lower, "upper": upper, "count": n_k,
            "mean_probability": p_k, "observed_rate": o_k, "abs_calibration_gap": gap,
        })

    reliability = math.fsum(reliability_terms) / count
    resolution = math.fsum(resolution_terms) / count
    within_bin = math.fsum(within_terms) / count
    ece = math.fsum(ece_terms) / count
    bss = (1.0 - brier / clim_brier) if clim_brier > 0.0 else None
    return {
        "cases": count,
        "base_rate": base,
        "brier_score": brier,
        "climatology_brier_score": clim_brier,
        "brier_skill_score": bss,
        "brier_skill_reason": None if clim_brier > 0.0 else "Climatology Brier score is zero.",
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "within_bin_term": within_bin,
        "ece": ece,
        "mean_probability": mean_prob,
        "observed_event_rate": obs_rate,
        "probability_bias": mean_prob - obs_rate,
        "bin_count": bins,
        "bins": bin_records,
    }


def binary_murphy_curve(cases, *, alphas) -> list[dict]:
    """Evaluate elementary cost-loss regret S_alpha(p, y) across cost-loss ratios alpha in (0, 1).

    S_alpha(p, y) = expense(p > alpha, y, alpha) - perfect_expense(y, alpha):
    equals alpha on false alarms, (1 - alpha) on misses, and 0 otherwise.
    Integrating S_alpha over alpha in (0, 1) recovers 0.5 * Brier score.
    """
    pairs = []
    for case in _iterate(cases, "cases"):
        try:
            probability, event = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be a (probability, event) pair") from error
        _validate_event(event)
        pairs.append((_probability(probability, "probability"), int(event)))
    if not pairs:
        raise ValueError("At least one case is required")
    grid = [_alpha(value) for value in _iterate(alphas, "alphas")]
    if not grid:
        raise ValueError("At least one alpha is required")
    count = len(pairs)
    curve = []
    for ratio in grid:
        scores = []
        for prob, event in pairs:
            acted = prob > ratio
            if acted and event == 0:
                scores.append(ratio)
            elif (not acted) and event == 1:
                scores.append(1.0 - ratio)
            else:
                scores.append(0.0)
        curve.append({"threshold": ratio, "mean_elementary_score": math.fsum(scores) / count})
    return curve


def quantile_murphy_curve(cases, *, quantile_level, thresholds) -> list[dict]:
    """Evaluate Ehm et al. (2016) elementary quantile score S_{theta, tau}(action, actual).

    For quantile level tau in (0, 1) and threshold theta:
        S_{theta, tau}(x, y) = (1 - tau) if y <= theta < x
                               tau       if x <= theta < y
                               0         otherwise
    Integrating over theta in (-inf, inf) recovers unscaled pinball loss at tau.
    """
    tau = _alpha(quantile_level)
    pairs = []
    for case in _iterate(cases, "cases"):
        try:
            action, actual = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be an (action, actual) pair") from error
        pairs.append((_finite_real(action, "action"), _finite_real(actual, "actual")))
    if not pairs:
        raise ValueError("At least one case is required")
    grid = [_finite_real(value, "threshold") for value in _iterate(thresholds, "thresholds")]
    if not grid:
        raise ValueError("At least one threshold is required")
    count = len(pairs)
    curve = []
    for theta in grid:
        scores = []
        for action, actual in pairs:
            if actual <= theta < action:
                scores.append(1.0 - tau)
            elif action <= theta < actual:
                scores.append(tau)
            else:
                scores.append(0.0)
        curve.append({"threshold": theta, "mean_elementary_score": math.fsum(scores) / count})
    return curve


def mean_murphy_curve(cases, *, thresholds) -> list[dict]:
    """Evaluate Ehm et al. (2016) elementary mean score S_theta(prediction, actual).

    For threshold theta:
        S_theta(x, y) = |theta - y|  if min(x, y) <= theta < max(x, y)  (sign-flipped across theta)
                        0            otherwise
    More precisely, S_theta(x, y) = (theta - y) * (1{x > theta} - 1{y > theta}) >= 0.
    Integrating S_theta over theta in (-inf, inf) recovers 0.5 * (x - y)^2.
    """
    pairs = []
    for case in _iterate(cases, "cases"):
        try:
            prediction, actual = case
        except (TypeError, ValueError) as error:
            raise ValueError("Each case must be a (prediction, actual) pair") from error
        pairs.append((_finite_real(prediction, "prediction"), _finite_real(actual, "actual")))
    if not pairs:
        raise ValueError("At least one case is required")
    grid = [_finite_real(value, "threshold") for value in _iterate(thresholds, "thresholds")]
    if not grid:
        raise ValueError("At least one threshold is required")
    count = len(pairs)
    curve = []
    for theta in grid:
        scores = []
        for prediction, actual in pairs:
            indicator_diff = (1.0 if prediction > theta else 0.0) - (1.0 if actual > theta else 0.0)
            scores.append((theta - actual) * indicator_diff if indicator_diff else 0.0)
        curve.append({"threshold": theta, "mean_elementary_score": math.fsum(scores) / count})
    return curve


def murphy_dominance(first_curve: list[dict], second_curve: list[dict], *, tolerance: float = 1e-12) -> dict:
    """Compare two aligned Murphy curves (first minus second; lower elementary score is better)."""
    if not first_curve or len(first_curve) != len(second_curve):
        raise ValueError("Murphy dominance requires two nonempty aligned curves")
    first_better = second_better = ties = 0
    max_first_adv = 0.0
    max_second_adv = 0.0
    differences = []
    for first_pt, second_pt in zip(first_curve, second_curve):
        if not math.isclose(first_pt["threshold"], second_pt["threshold"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Murphy curves must share identical threshold grids")
        diff = first_pt["mean_elementary_score"] - second_pt["mean_elementary_score"]
        differences.append({
            "threshold": first_pt["threshold"],
            "first_score": first_pt["mean_elementary_score"],
            "second_score": second_pt["mean_elementary_score"],
            "first_minus_second": diff,
        })
        if diff < -tolerance:
            first_better += 1
            max_first_adv = max(max_first_adv, -diff)
        elif diff > tolerance:
            second_better += 1
            max_second_adv = max(max_second_adv, diff)
        else:
            ties += 1
    total = len(differences)
    if first_better > 0 and second_better == 0:
        status = "first_dominates"
    elif second_better > 0 and first_better == 0:
        status = "second_dominates"
    elif first_better == 0 and second_better == 0:
        status = "tied"
    else:
        status = "crossing"
    return {
        "dominance_status": status,
        "threshold_count": total,
        "first_better_count": first_better,
        "second_better_count": second_better,
        "tie_count": ties,
        "first_better_fraction": first_better / total,
        "second_better_fraction": second_better / total,
        "max_first_advantage": max_first_adv,
        "max_second_advantage": max_second_adv,
        "curve": differences,
    }
