"""Proper losses for saved continuous observation laws; no fitting or sampling.

Both scores are losses (lower is better). CRPS requires a finite first moment in
this interface. Gaussian mixtures retain every positive-weight component. See
Jordan et al. (2019), scoringRules appendix, for the Student-t/mixture formulas.
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


def validate_distribution(record):
    """Return an independent finite JSON law; never infer one from quantiles."""
    if not isinstance(record, dict):
        raise ValueError("A declared distribution model must supply a distribution dictionary")
    record = json.loads(json.dumps(record, allow_nan=False))
    if record.get("family") == "student_t":
        values = [record[key] for key in ("location", "scale", "df")]
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)
                or record["scale"] <= 0 or record["df"] <= 1):
            raise ValueError("Student-t scoring requires finite location/scale and df > 1 (finite first moment)")
        return {key: record[key] for key in ("family", "location", "scale", "df")}
    if record.get("family") == "gaussian_mixture":
        components = [record[key] for key in ("weights", "means", "variances")]
        if any(not isinstance(items, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in items)
               for items in components):
            raise ValueError("Gaussian components must be numeric lists")
        law = NormalMixture(*map(tuple, components))
        if any(weight <= 0 for weight in law.weights):
            raise ValueError("A positive mixture weight is unrepresentable after normalization")
        return {"family": "gaussian_mixture", "weights": list(law.weights),
                "means": list(law.means), "variances": list(law.variances)}
    raise ValueError("Unsupported predictive distribution family")


def validate_distribution_forecast(record, forecast):
    """Check the serialized law agrees with the supplied observation summaries."""
    record = validate_distribution(record)
    if record["family"] == "student_t":
        law = t(df=record["df"], loc=record["location"], scale=record["scale"])
        location = record["location"]
        rounding_scale = abs(location)
    else:
        law = NormalMixture(tuple(record["weights"]), tuple(record["means"]), tuple(record["variances"]))
        location = law.mean
        rounding_scale = max(map(abs, law.means))
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


def score_distribution(record, actual):
    """Return finite CRPS and negative log density, or fail explicitly.

    No Monte Carlo approximation, moment-matched Gaussian, tail truncation or
    density floor is used. Mixture pair sums use blocks to bound memory; runtime
    remains quadratic in component count. Unrepresentable results raise.
    """
    record = validate_distribution(record)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError("Scoring actual must be finite and numeric")
    if record["family"] == "student_t":
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
    return {"status": "completed" if len(scored) == len(rows) else "failed", "method": dict(METHOD),
            "expected_count": len(rows), "scored_count": len(scored), "failed_count": len(rows)-len(scored),
            "mean_crps": mean(row["crps"] for row in scored) if scored else None,
            "mean_log_score": mean(row["log_score"] for row in scored) if scored else None,
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
        validate_distribution_forecast(law["distribution"], Forecast(float(prediction["prediction"])))
        for key in ("crps", "log_score"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or (key == "crps" and row[key] < 0):
                raise ValueError("Invalid saved distribution loss")
    summary = summarize_distribution_scores(rows, supported=True)
    saved = load("scores.json").get("distribution", {})
    for key in ("status", "method", "expected_count", "scored_count", "failed_count"):
        if saved.get(key) != summary[key]:
            raise ValueError("Saved distribution score summary is incomplete/inconsistent")
    for key in ("mean_crps", "mean_log_score"):
        value = saved.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Saved distribution loss aggregate differs from case scores")
    return summary, rows
