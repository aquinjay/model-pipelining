"""Proper losses for saved observation laws; no fitting or sampling.

Both scores are losses (lower is better). CRPS requires a finite first moment in
this interface. Continuous laws (`student_t`, `gaussian_mixture`) compute exact
closed-form CRPS and negative log density. Discrete weighted empirical laws
(`empirical_samples`) compute exact O(k log k) step-integral CRPS and leave
negative log density as None because no continuous Lebesgue density exists.
See Jordan et al. (2019), scoringRules appendix, for the Student-t/mixture formulas.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import betaln, erf, logsumexp
from scipy.stats import t

from model_pipelining.mixtures import NormalMixture
from model_pipelining.models import Forecast


METHOD = {"version": 1, "crps": "cdf_integral_loss_closed_form",
          "log_score": "negative_natural_log_density", "direction": "lower_is_better",
          "target": "future_observation", "density_measure": "Lebesgue_in_target_units",
          "crps_relative_roundoff_budget": 1e-7}


def _sorted_empirical_pairs(record: dict) -> tuple[list[float], list[float]]:
    """Aggregate equal sample values in ascending order with normalized positive weights."""
    by_value: dict[float, float] = {}
    for value, weight in zip(record["values"], record["weights"]):
        by_value[value] = by_value.get(value, 0.0) + weight
    ordered_values = sorted(by_value)
    raw_weights = [by_value[value] for value in ordered_values]
    total = math.fsum(raw_weights)
    weights = [weight / total for weight in raw_weights]
    return ordered_values, weights


def empirical_cdf_bounds(record: dict, x: float) -> tuple[float, float]:
    """Return (P(Y < x), P(Y <= x)) for a validated empirical_samples law."""
    ordered_values, weights = _sorted_empirical_pairs(record)
    strictly_less = math.fsum(w for v, w in zip(ordered_values, weights) if v < x)
    at_most = math.fsum(w for v, w in zip(ordered_values, weights) if v <= x)
    return min(max(strictly_less, 0.0), 1.0), min(max(at_most, 0.0), 1.0)


def empirical_quantile(record: dict, probability: float) -> float:
    """Return the exact left-continuous inverse CDF inf{x : F(x) >= p}."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability):
        raise ValueError("probability must be finite")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    ordered_values, weights = _sorted_empirical_pairs(record)
    cumulative = 0.0
    for value, weight in zip(ordered_values, weights):
        cumulative += weight
        if cumulative >= probability - 1e-15:
            return value
    return ordered_values[-1]


def validate_distribution(record):
    """Return an independent finite JSON law; never infer one from quantiles."""
    if not isinstance(record, dict):
        raise ValueError("A declared distribution model must supply a distribution dictionary")
    record = json.loads(json.dumps(record, allow_nan=False))
    family = record.get("family")
    if family == "student_t":
        values = [record[key] for key in ("location", "scale", "df")]
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)
                or record["scale"] <= 0 or record["df"] <= 1):
            raise ValueError("Student-t scoring requires finite location/scale and df > 1 (finite first moment)")
        return {key: record[key] for key in ("family", "location", "scale", "df")}
    if family == "gaussian_mixture":
        components = [record[key] for key in ("weights", "means", "variances")]
        if any(not isinstance(items, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in items)
               for items in components):
            raise ValueError("Gaussian components must be numeric lists")
        law = NormalMixture(*map(tuple, components))
        if any(weight <= 0 for weight in law.weights):
            raise ValueError("A positive mixture weight is unrepresentable after normalization")
        return {"family": "gaussian_mixture", "weights": list(law.weights),
                "means": list(law.means), "variances": list(law.variances)}
    if family == "empirical_samples":
        values = record.get("values")
        if (not isinstance(values, list) or not values
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)):
            raise ValueError("empirical_samples requires a nonempty list of finite numeric values")
        weights = record.get("weights")
        if weights is None:
            weights = [1.0 / len(values)] * len(values)
        elif (not isinstance(weights, list) or len(weights) != len(values)
              or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
                     for w in weights)):
            raise ValueError("empirical_samples weights must be a matching list of finite nonnegative numbers")
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("empirical_samples weights must have a finite positive sum")
        kept_values, kept_weights = [], []
        for value, weight in zip(values, weights):
            if weight > 0.0:
                normalized = float(weight / total)
                if normalized <= 0.0:
                    raise ValueError("A positive empirical sample weight is unrepresentable after normalization")
                kept_values.append(float(value))
                kept_weights.append(normalized)
        return {"family": "empirical_samples", "values": kept_values, "weights": kept_weights}
    raise ValueError("Unsupported predictive distribution family")


def validate_distribution_forecast(record, forecast):
    """Check the serialized law agrees with the supplied observation summaries."""
    record = validate_distribution(record)
    family = record["family"]
    if family == "student_t":
        law = t(df=record["df"], loc=record["location"], scale=record["scale"])
        location = record["location"]
        rounding_scale = abs(location)
    elif family == "gaussian_mixture":
        law = NormalMixture(tuple(record["weights"]), tuple(record["means"]), tuple(record["variances"]))
        location = law.mean
        rounding_scale = max(map(abs, law.means))
    else:
        location = math.fsum(w * v for v, w in zip(record["values"], record["weights"]))
        rounding_scale = max(map(abs, record["values"]))
        tolerance = max(128 * math.ulp(max(rounding_scale, abs(forecast.mean))), 1e-12 * max(1.0, rounding_scale))
        if abs(location - forecast.mean) > tolerance:
            raise ValueError("Saved distribution mean differs from forecast mean")
        for q in forecast.quantiles:
            left_cdf, right_cdf = empirical_cdf_bounds(record, q.value)
            if not (left_cdf - 1e-7 <= q.probability <= right_cdf + 1e-7):
                raise ValueError("Saved distribution differs from forecast quantiles")
        if forecast.highest_density_regions:
            raise ValueError("Discrete empirical_samples laws do not define continuous highest-density regions")
        return record
    tolerance = 128*math.ulp(max(rounding_scale, abs(forecast.mean)))
    if abs(location-forecast.mean) > tolerance:
        raise ValueError("Saved distribution mean differs from forecast mean")
    if any(abs(float(law.cdf(q.value))-q.probability) > 1e-7 for q in forecast.quantiles):
        raise ValueError("Saved distribution differs from forecast quantiles")
    for region in forecast.highest_density_regions:
        mass = math.fsum(float(law.cdf(s.upper))-float(law.cdf(s.lower)) for s in region.segments)
        if abs(mass-region.level) > 1e-6:
            raise ValueError("Saved distribution differs from forecast region mass")
    return record


def _normal_absolute(delta, sd):
    """E|N(delta, sd**2)|, allowing saturated tails without density floors."""
    with np.errstate(over="ignore", under="ignore"):
        z = delta / sd
        return sd*math.sqrt(2/math.pi)*np.exp(-.5*z*z) + delta*erf(z/math.sqrt(2))


def _empirical_crps(record: dict, actual: float) -> float:
    """Exact nonnegative step-function integral int (F(x) - 1{x >= actual})^2 dx in O(k log k)."""
    ordered_values, weights = _sorted_empirical_pairs(record)
    terms = []
    if actual < ordered_values[0]:
        terms.append(ordered_values[0] - actual)
    cumulative = 0.0
    for index in range(len(ordered_values) - 1):
        cumulative += weights[index]
        left = ordered_values[index]
        right = ordered_values[index + 1]
        if actual <= left:
            terms.append(((1.0 - cumulative) ** 2) * (right - left))
        elif actual >= right:
            terms.append((cumulative ** 2) * (right - left))
        else:
            terms.append((cumulative ** 2) * (actual - left))
            terms.append(((1.0 - cumulative) ** 2) * (right - actual))
    if actual > ordered_values[-1]:
        terms.append(actual - ordered_values[-1])
    crps = math.fsum(terms)
    if not math.isfinite(crps) or crps < 0.0:
        raise ValueError("Nonfinite or negative empirical CRPS")
    return crps


def score_distribution(record, actual):
    """Return finite CRPS and negative log density (or None for discrete empirical laws).

    No Monte Carlo approximation, moment-matched Gaussian, tail truncation or
    density floor is used. Continuous families require finite log density.
    Discrete empirical_samples laws have no continuous Lebesgue density, so
    log_score is None while CRPS is exact.
    """
    record = validate_distribution(record)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError("Scoring actual must be finite and numeric")
    family = record["family"]
    if family == "empirical_samples":
        return {"crps": _empirical_crps(record, float(actual)), "log_score": None}
    if family == "student_t":
        df, scale, location = (record[key] for key in ("df", "scale", "location"))
        delta = actual-location
        if not math.isfinite(delta):
            raise ValueError("Unrepresentable observation distance")
        log_z = math.log(abs(delta))-math.log(scale) if delta else -math.inf
        log_ratio = float(np.logaddexp(0., 2*log_z-math.log(df)))
        log_pdf_standard = -.5*math.log(df)-float(betaln(.5, df/2))-(df+1)/2*log_ratio
        log_score = math.log(scale)-log_pdf_standard
        z = delta/scale
        cdf = float(t.cdf(z, df))
        absolute_term = delta*(2*cdf-1) + math.exp(
            math.log(scale)+math.log(2)+log_pdf_standard+math.log(df)+log_ratio-math.log(df-1))
        pair_term = math.exp(math.log(scale)+math.log(2)+.5*math.log(df)-math.log(df-1)
                             +float(betaln(.5, df-.5))-2*float(betaln(.5, df/2)))
    else:
        weights, means, variances = (np.asarray(record[key], dtype=float) for key in ("weights", "means", "variances"))
        sd = np.sqrt(variances)
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            delta = actual-means
            z = delta/sd
            log_components = np.log(weights)-np.log(sd)-.5*math.log(2*math.pi)-.5*z*z
            log_score = -float(logsumexp(log_components))
            absolute_term = float(np.dot(weights, _normal_absolute(delta, sd)))
            blocks = []
            for start in range(0, len(weights), 128):
                # hypot avoids overflow from adding large finite variances.
                distances = means[start:start+128, None]-means
                scales = np.hypot(sd[start:start+128, None], sd)
                expectations = _normal_absolute(distances, scales)
                blocks.append(float(weights[start:start+128] @ (expectations @ weights)))
            pair_term = .5*math.fsum(blocks)
    crps = absolute_term-pair_term
    if not math.isfinite(log_score) or not math.isfinite(crps) or crps < 0:
        raise ValueError("Nonfinite or negative distribution score; no density floor is applied")
    if 128*math.ulp(max(abs(absolute_term), abs(pair_term))) > METHOD["crps_relative_roundoff_budget"]*crps:
        raise ValueError("CRPS precision is unresolved at this scale")
    return {"crps": crps, "log_score": log_score}


def evaluate_distributions(records, actuals):
    """Score already-frozen origin records against explicitly selected actuals."""
    rows = []
    for record in records:
        actual = actuals[record["target_time"]]
        row = {key: record[key] for key in ("origin_index", "issued_at", "target_time", "history_sha256", "status")}
        row.update(actual=actual.value, actual_available_at=actual.available_at.isoformat(),
                   crps=None, log_score=None, failure=record["error"])
        if record["status"] == "passed":
            try:
                row.update(score_distribution(record["distribution"], actual.value))
            except (ValueError, KeyError, OverflowError, FloatingPointError) as error:
                row.update(status="failed", failure=f"{type(error).__name__}: {error}")
        rows.append(row)
    return rows


def summarize_distribution_scores(rows, *, supported):
    if not supported:
        return {"status": "not_applicable", "reason": "Model does not declare a full predictive distribution"}
    scored = [row for row in rows if row["status"] == "passed"]
    has_log_scores = bool(scored) and all(row["log_score"] is not None for row in scored)
    return {"status": "completed" if len(scored) == len(rows) else "failed", "method": dict(METHOD),
            "expected_count": len(rows), "scored_count": len(scored), "failed_count": len(rows)-len(scored),
            "mean_crps": mean(row["crps"] for row in scored) if scored else None,
            "mean_log_score": mean(row["log_score"] for row in scored) if has_log_scores else None,
            "scope": "Successful cases only; incomplete scores cannot enter a full-cohort comparison"}


def read_saved_distribution_scores(directory, manifest, predictions, evaluations):
    """Validate complete saved score cohorts for comparison/review; no model calls.

    Recompute aggregates from the saved per-case scores, rather than trusting a
    completion flag or summary alone. This is consistency checking, not artifact
    authentication or an independent recomputation of model calculations.
    """
    directory = Path(directory)
    def load(name):
        def reject(value):
            raise ValueError(f"Nonfinite saved JSON: {value}")
        return json.loads((directory/name).read_text(encoding="utf-8"), parse_constant=reject)
    if manifest.get("evaluation", {}).get("distribution_scoring") != METHOD:
        raise ValueError("Incompatible distribution scoring method/units")
    checks = load("checks.json")
    for name in ("full_predictive_distributions_valid", "all_distribution_cases_scored"):
        matches = [item for item in checks if item.get("name") == name]
        if len(matches) != 1 or matches[0].get("required") is not True or matches[0].get("status") != "passed":
            raise ValueError("Missing or failed distribution scoring check")
    frozen = load("predictive_distributions.json")
    with (directory/"distribution_evaluation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(predictions) or len(frozen) != len(rows) or len(rows) != len(evaluations):
        raise ValueError("Distribution score cohort is incomplete")
    for law, row, prediction, evaluation in zip(frozen, rows, predictions, evaluations):
        for key in ("origin_index", "issued_at", "target_time", "history_sha256"):
            if str(law[key]) != prediction[key] or row[key] != prediction[key]:
                raise ValueError("Distribution origin/history cohort differs from point forecasts")
        if any(item["status"] != "passed" for item in (law, row, prediction, evaluation)):
            raise ValueError("Distribution comparisons require every case scored")
        if (not math.isfinite(float(row["actual"])) or float(row["actual"]) != float(evaluation["actual"])
                or row["actual_available_at"] != evaluation["actual_available_at"]):
            raise ValueError("Distribution scoring actual/vintage mismatch")
        validated_law = validate_distribution_forecast(law["distribution"], Forecast(float(prediction["prediction"])))
        row["crps"] = float(row["crps"])
        if not math.isfinite(row["crps"]) or row["crps"] < 0:
            raise ValueError("Invalid saved distribution loss")
        if validated_law["family"] == "empirical_samples":
            if row["log_score"] not in ("", None):
                raise ValueError("Discrete empirical_samples law should not have a continuous log_score")
            row["log_score"] = None
        else:
            row["log_score"] = float(row["log_score"])
            if not math.isfinite(row["log_score"]):
                raise ValueError("Invalid saved distribution loss")
    summary = summarize_distribution_scores(rows, supported=True)
    saved = load("scores.json").get("distribution", {})
    for key in ("status", "method", "expected_count", "scored_count", "failed_count"):
        if saved.get(key) != summary[key]:
            raise ValueError("Saved distribution score summary is incomplete/inconsistent")
    for key in ("mean_crps", "mean_log_score"):
        value = saved.get(key)
        expected = summary[key]
        if expected is None:
            if value is not None:
                raise ValueError("Saved distribution loss aggregate differs from case scores")
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Saved distribution loss aggregate differs from case scores")
    return summary, rows
