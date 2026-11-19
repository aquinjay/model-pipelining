"""Name-to-class catalog for building repository model adapters from JSON settings.

Command-line scripts use this module to turn a model name and a saved
configuration into an adapter. The shared runner, evidence workflows and
reporting never consult it: they accept any object that satisfies the
Forecaster or EvidenceProvider contract. A new adapter registers here only so
scripts can construct it by name.
"""

from model_pipelining.local_level import LocalLevel
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.normal_mean import NormalMean
from model_pipelining.normal_mean_variance import NormalMeanVariance


MODEL_TYPES = {kind.name: kind for kind in
               (NormalMean, NormalMeanVariance, LocalLevel, LocalLevelVariance, LocalLevelJointVariance)}
EVIDENCE_MODEL_NAMES = tuple(name for name, kind in MODEL_TYPES.items() if hasattr(kind, "evidence_provider"))


def build_model(name, configuration):
    """Return the named adapter rebuilt by its own from_configuration method."""
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(MODEL_TYPES)}")
    if not isinstance(configuration, dict):
        raise ValueError("Model configuration must be a dictionary")
    return MODEL_TYPES[name].from_configuration(configuration)


def build_scenarios(reference, overrides):
    """Return named alternatives that change only the given reference settings."""
    if not isinstance(overrides, dict) or any(not isinstance(value, dict) for value in overrides.values()):
        raise ValueError("Scenarios must map names to override dictionaries")
    restore = type(reference).from_configuration
    return {name: restore({**reference.configuration, **values}) for name, values in overrides.items()}
