"""Cost-loss decisions for binary events, with expenses in units of the loss."""

from numbers import Integral, Real


def _alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(alpha)


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return float(value)


def _validate_event(event):
    if not isinstance(event, Integral) or event not in (0, 1):
        raise ValueError("event must be 0 or 1")


def _iterate(values, name):
    try:
        return iter(values)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error


def decide(probability, alpha) -> bool:
    """Take protective action when the event probability exceeds the cost-loss ratio."""
    return _probability(probability, "probability") > _alpha(alpha)


def expense(acted, event, alpha) -> float:
    """Return the expense of one decision in units of the unprotected loss."""
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
