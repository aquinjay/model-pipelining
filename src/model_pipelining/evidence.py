"""Small model evidence boundary and shared artifact helpers.

Providers own distribution calculations. Workflows own selection, discrepancies,
persistence and computational requirements. This is a trusted Python interface.
"""

import csv
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Protocol

from model_pipelining.models import ForecastRequest


@dataclass(frozen=True)
class ReplicatedDataset:
    replication: int
    mean: float | None
    observation_variance: float
    values: tuple[float, ...]
    initial_level: float | None = None
    latent_levels: tuple[float, ...] = ()


@dataclass(frozen=True)
class PredictiveReplications:
    kind: str
    seed: int
    requested_replications: int
    observation_count: int
    distribution: dict
    datasets: tuple[ReplicatedDataset, ...]
    failures: tuple[dict, ...]


@dataclass(frozen=True, kw_only=True)
class EvidenceMetadata:
    family: str
    implementation_files: tuple[Path, ...]
    predictive_scope: dict
    influence_scope: dict
    presentation: dict
    workflows: tuple[str, ...] = ("predictive_checks", "prior_influence")
    parameter_fields: tuple[str, ...] = ("mean", "observation_variance")
    latent_paths: bool = False
    integration_checks: bool = False
    mean_prior_settings: tuple[str, ...] = ("prior_mean", "prior_mean_precision")
    variance_prior_settings: tuple[str, ...] = ("prior_shape", "prior_scale")


class EvidenceProvider(Protocol):
    metadata: EvidenceMetadata

    def simulate(self, *, observation_count: int, replications: int, seed: int,
                 request: ForecastRequest | None = None) -> PredictiveReplications: ...

    def summarize(self, case, level: float) -> dict: ...

    def validate_scenarios(self, models: tuple) -> None: ...


def get_evidence_provider(model, workflow: str) -> EvidenceProvider:
    factory = getattr(model, "evidence_provider", None)
    if not callable(factory):
        raise ValueError(f"Model does not supply {workflow} evidence")
    provider = factory()
    metadata = getattr(provider, "metadata", None)
    if (not isinstance(metadata, EvidenceMetadata)
            or not isinstance(metadata.family, str) or not metadata.family.isidentifier()
            or any(not callable(getattr(provider, name, None)) for name in ("simulate", "summarize", "validate_scenarios"))):
        raise ValueError("Malformed model evidence provider")
    if workflow not in metadata.workflows:
        raise ValueError(f"Model does not supply {workflow} evidence")
    if (not isinstance(metadata.workflows, tuple)
            or any(name not in ("predictive_checks", "prior_influence") for name in metadata.workflows)
            or len(set(metadata.workflows)) != len(metadata.workflows)):
        raise ValueError("Malformed evidence workflow capabilities")
    if (not isinstance(metadata.implementation_files, tuple)
            or any(not isinstance(path, Path) for path in metadata.implementation_files)
            or not isinstance(metadata.predictive_scope, dict) or not isinstance(metadata.influence_scope, dict)):
        raise ValueError("Malformed evidence source/scope metadata")
    allowed_parameters = {"mean", "observation_variance", "initial_level"}
    if (not isinstance(metadata.parameter_fields, tuple)
            or not {"mean", "observation_variance"}.issubset(metadata.parameter_fields)
            or len(set(metadata.parameter_fields)) != len(metadata.parameter_fields)
            or not set(metadata.parameter_fields).issubset(allowed_parameters)
            or type(metadata.latent_paths) is not bool or type(metadata.integration_checks) is not bool):
        raise ValueError("Malformed evidence parameter metadata")
    _validate_presentation(metadata.presentation)
    json.dumps((metadata.predictive_scope, metadata.influence_scope, metadata.presentation), allow_nan=False)
    return provider


def validate_replications(result, *, observation_count, replications, seed, kind, latent_paths):
    """Reject omitted, duplicated, malformed or nonfinite provider output."""
    if (not isinstance(result, PredictiveReplications) or result.kind != kind or result.seed != seed
            or result.requested_replications != replications or result.observation_count != observation_count
            or not isinstance(result.distribution, dict)
            or not isinstance(result.datasets, tuple) or not isinstance(result.failures, tuple)):
        raise ValueError("Evidence provider returned incompatible replication metadata")
    json.dumps(result.distribution, allow_nan=False)
    def finite(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    indices = []
    for item in result.datasets:
        if (not isinstance(item, ReplicatedDataset) or not isinstance(item.values, tuple)
                or len(item.values) != observation_count or not isinstance(item.latent_levels, tuple)
                or not finite(item.observation_variance) or item.observation_variance <= 0
                or any(not finite(value) for value in item.values)
                or any(value is not None and not finite(value) for value in (item.mean, item.initial_level))
                or (latent_paths and (len(item.latent_levels) != observation_count or item.initial_level is None))
                or any(not finite(value) for value in item.latent_levels)):
            raise ValueError("Malformed replicated dataset from evidence provider")
        indices.append(item.replication)
    for item in result.failures:
        if not isinstance(item, dict) or not isinstance(item.get("error"), str) or not item["error"].strip():
            raise ValueError("Malformed replication failure from evidence provider")
        indices.append(item.get("replication"))
    if (any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(replications))):
        raise ValueError("Evidence provider omitted or duplicated requested replications")
    return result


def _validate_presentation(presentation):
    if (not isinstance(presentation, dict)
            or any(not isinstance(presentation.get(name), str) or not presentation[name].strip()
                   for name in ("generation", "intervals"))
            or any(type(presentation.get(name)) is not bool for name in
                   ("predictive_segments", "prior_location_coefficient", "degrees_of_freedom"))):
        raise ValueError("Malformed evidence presentation metadata")


def saved_presentation(manifest, workflow):
    """Read presentation metadata; recognize the two pre-interface artifact kinds."""
    if manifest.get("format_version") != 1 or manifest.get("status") not in ("completed", "failed"):
        raise ValueError("Plotting requires finished version-1 evidence")
    if "presentation" in manifest:
        family = manifest.get("family")
        if not isinstance(family, str) or not family.isidentifier() or manifest.get("kind") != f"{family}_{workflow}":
            raise ValueError("Incompatible saved evidence kind")
        result = manifest["presentation"]
    else:
        legacy = {
            f"normal_mean_variance_{workflow}": {
                "generation": "one mean/variance pair per dataset",
                "intervals": "mean/predictive HPD; variance equal-tail credible intervals",
                "predictive_segments": False, "prior_location_coefficient": True, "degrees_of_freedom": True},
            f"local_level_variance_{workflow}": {
                "generation": "one variance and fresh latent path per dataset",
                "intervals": "state/variance equal-tail credible; predictive mixture HPD",
                "predictive_segments": True, "prior_location_coefficient": False, "degrees_of_freedom": False},
        }
        if manifest.get("kind") not in legacy:
            raise ValueError("Unsupported legacy evidence kind")
        result = legacy[manifest["kind"]]
    _validate_presentation(result)
    return result


def source_hashes(*files):
    import hashlib
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def _write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
