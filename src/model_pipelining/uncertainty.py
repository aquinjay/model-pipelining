"""Shared observation-quantile/region validation and uncertainty evaluation.

No model family or distribution is assumed here. Pinball loss is unscaled;
interval score uses the conventional 2 / (1 - coverage) tail penalty.
"""

import math
from decimal import Decimal, localcontext
from numbers import Real
from statistics import mean

from model_pipelining.models import Forecast, HighestDensityRegion, Interval, Quantile


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def interval_probabilities(level: float) -> tuple[float, float]:
    """Return the equal-tail quantile probabilities for a central interval."""
    if not _finite_number(level) or not 0 < level < 1:
        raise ValueError("Interval levels must be finite probabilities strictly between zero and one")
    # Treat a declared decimal level such as 0.8 as 80%, so its probability
    # keys are the ordinary floats 0.1 and 0.9 rather than subtraction artifacts.
    with localcontext() as context:
        context.prec = 50
        coverage = Decimal(str(float(level)))
        lower = float((1 - coverage) / 2)
        upper = float((1 + coverage) / 2)
    if not 0 < lower < upper < 1:
        raise ValueError("Interval level is too close to one for finite-precision quantiles")
    return lower, upper


def validate_interval_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    """Validate caller-selected levels before run creation; sort for storage."""
    if not isinstance(levels, (tuple, list)) or not levels:
        raise ValueError("interval_levels must be a nonempty tuple or list")
    for level in levels:
        interval_probabilities(level)
    if len(set(levels)) != len(levels):
        raise ValueError("interval_levels must not contain duplicates")
    return tuple(sorted(float(level) for level in levels))


def required_quantiles(levels: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted({probability for level in levels for probability in interval_probabilities(level)}))


def validate_forecast(
    output: Forecast | float,
    probabilities: tuple[float, ...],
    highest_density_levels: tuple[float, ...] = (),
) -> Forecast:
    """Normalize legacy means; require finite, complete, noncrossing quantiles.

    A missing or unexpected quantile fails the entire forecast. Quantiles may
    coincide for discrete distributions. No symmetry or Gaussian shape is
    imposed, and a skewed distribution's mean need not lie inside its interval.
    """
    if not isinstance(output, Forecast):
        if not _finite_number(output):
            raise ValueError("Model must return a finite mean or a Forecast record")
        output = Forecast(float(output))
    if not _finite_number(output.mean):
        raise ValueError("Forecast mean must be finite and numeric")
    if not isinstance(output.quantiles, tuple) or any(not isinstance(item, Quantile) for item in output.quantiles):
        raise ValueError("Forecast quantiles must be a tuple of Quantile records")
    for item in output.quantiles:
        if not _finite_number(item.probability) or not 0 < item.probability < 1 or not _finite_number(item.value):
            raise ValueError("Quantile probabilities and values must be finite, with probabilities in (0, 1)")
    items = sorted(output.quantiles, key=lambda item: item.probability)
    if tuple(item.probability for item in items) != probabilities:
        raise ValueError("Model must return exactly the requested quantile probabilities without duplicates")
    if any(first.value > second.value for first, second in zip(items, items[1:])):
        raise ValueError("Forecast quantiles must not cross")
    regions = output.highest_density_regions
    if not isinstance(regions, tuple) or any(not isinstance(region, HighestDensityRegion) for region in regions):
        raise ValueError("Highest-density regions must be a tuple of HighestDensityRegion records")
    for region in regions:
        if not _finite_number(region.level) or not 0 < region.level < 1:
            raise ValueError("Highest-density region levels must be probabilities in (0, 1)")
        if not isinstance(region.segments, tuple) or not region.segments:
            raise ValueError("Each highest-density region must contain a nonempty tuple of segments")
        for segment in region.segments:
            if (not isinstance(segment, Interval) or not _finite_number(segment.lower)
                or not _finite_number(segment.upper) or segment.lower > segment.upper):
                raise ValueError("Highest-density segments must have finite ordered bounds")
        if any(first.upper >= second.lower for first, second in zip(region.segments, region.segments[1:])):
            raise ValueError("Highest-density segments must be ordered, disjoint, and non-touching")
    regions = sorted(regions, key=lambda region: region.level)
    if tuple(region.level for region in regions) != highest_density_levels:
        raise ValueError("Model must supply exactly the requested highest-density region levels")
    # Highest-density sets for different masses must be nested; check set
    # inclusion without filling gaps between disjoint components.
    for smaller, larger in zip(regions, regions[1:]):
        if any(not any(outer.lower <= inner.lower <= inner.upper <= outer.upper for outer in larger.segments)
               for inner in smaller.segments):
            raise ValueError("Highest-density regions must be nested across levels")
    return Forecast(
        float(output.mean), tuple(Quantile(float(item.probability), float(item.value)) for item in items),
        tuple(HighestDensityRegion(float(region.level), tuple(Interval(float(segment.lower), float(segment.upper))
                                                               for segment in region.segments)) for region in regions),
    )


def evaluate_highest_density(regions: tuple[HighestDensityRegion, ...], actual: float) -> list[dict]:
    """Describe coverage and total set width without treating gaps as covered.

    The equal-tail interval score is not applied to highest-density regions:
    it targets equal-tail quantiles and generally does not assess HPD optimality.
    """
    rows = []
    for region in regions:
        width = math.fsum(segment.upper - segment.lower for segment in region.segments)
        if not math.isfinite(width):
            raise ValueError("Nonfinite highest-density region width")
        rows.append({"level": region.level,
                     "covered": any(segment.lower <= actual <= segment.upper for segment in region.segments),
                     "total_width": width, "segment_count": len(region.segments)})
    return rows


def pinball_loss(actual: float, value: float, probability: float) -> float:
    """Unscaled quantile loss: p*(y-q) above q, (1-p)*(q-y) below q."""
    if not all(_finite_number(item) for item in (actual, value, probability)) or not 0 < probability < 1:
        raise ValueError("Pinball loss requires finite values and a probability in (0, 1)")
    loss = probability * (actual - value) if actual >= value else (1 - probability) * (value - actual)
    if not math.isfinite(loss):
        raise ValueError("Nonfinite pinball loss")
    return loss


def evaluate_intervals(quantiles: dict[float, float], actual: float, levels: tuple[float, ...]) -> list[dict]:
    """Score all requested intervals for one frozen forecast; fail as one case."""
    rows = []
    for level in levels:
        lower_probability, upper_probability = interval_probabilities(level)
        lower, upper = quantiles[lower_probability], quantiles[upper_probability]
        width = upper - lower
        interval_score = width
        if actual < lower:
            interval_score += 2 * (lower - actual) / (1 - level)
        elif actual > upper:
            interval_score += 2 * (actual - upper) / (1 - level)
        if not all(math.isfinite(value) for value in (width, interval_score)):
            raise ValueError("Nonfinite interval scoring result")
        rows.append({
            "level": level, "lower_probability": lower_probability, "upper_probability": upper_probability,
            "lower": lower, "upper": upper, "covered": lower <= actual <= upper,
            "width": width, "interval_score": interval_score,
            "lower_pinball_loss": pinball_loss(actual, lower, lower_probability),
            "upper_pinball_loss": pinball_loss(actual, upper, upper_probability),
        })
    return rows


def summarize_uncertainty(
    rows: list[dict], levels: tuple[float, ...], *, expected_count: int, supported: bool,
    highest_density_rows: list[dict] | None = None, supports_highest_density: bool = False,
) -> dict:
    """Summarize successful common uncertainty cases, with explicit coverage."""
    highest_density_rows = [] if highest_density_rows is None else highest_density_rows
    if not supported and not supports_highest_density:
        return {"status": "not_applicable", "expected_count": 0, "scored_count": 0,
                "failed_count": 0, "intervals": [], "quantiles": [], "highest_density_regions": [],
                "reason": "Model declares point-only output; uncertainty was not supplied"}
    intervals, quantiles, highest_density = [], [], []
    scored_count = 0
    for level in levels:
        if supports_highest_density:
            scored_density = [row for row in highest_density_rows if row["level"] == level and row["status"] == "passed"]
            scored_count = len(scored_density)
            highest_density.append({
                "level": level, "scored_count": scored_count,
                "covered_count": sum(row["covered"] for row in scored_density),
                "coverage": mean(row["covered"] for row in scored_density) if scored_density else None,
                "mean_total_width": mean(row["total_width"] for row in scored_density) if scored_density else None,
                "mean_segment_count": mean(row["segment_count"] for row in scored_density) if scored_density else None,
            })
        if not supported:
            continue
        scored = [row for row in rows if row["level"] == level and row["status"] == "passed"]
        scored_count = len(scored)
        intervals.append({
            "level": level, "scored_count": scored_count,
            "covered_count": sum(row["covered"] for row in scored),
            "coverage": mean(row["covered"] for row in scored) if scored else None,
            "mean_width": mean(row["width"] for row in scored) if scored else None,
            "mean_interval_score": mean(row["interval_score"] for row in scored) if scored else None,
        })
        for side, probability in zip(("lower", "upper"), interval_probabilities(level)):
            quantiles.append({"probability": probability, "scored_count": scored_count,
                              "mean_pinball_loss": mean(row[f"{side}_pinball_loss"] for row in scored) if scored else None})
    return {"status": "completed" if scored_count == expected_count else "failed",
            "expected_count": expected_count, "scored_count": scored_count,
            "failed_count": expected_count - scored_count,
            "intervals": intervals, "quantiles": sorted(quantiles, key=lambda row: row["probability"]),
            "highest_density_regions": highest_density,
            "scope": "Successful uncertainty cases only; coverage is descriptive, not an approval criterion"}
