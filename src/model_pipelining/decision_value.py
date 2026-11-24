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
