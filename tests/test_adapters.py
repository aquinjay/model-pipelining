"""Saved model configurations rebuild the same adapters; scripts share one catalog."""

import json
import unittest

from model_pipelining.adapters import EVIDENCE_MODEL_NAMES, MODEL_TYPES, build_model, build_scenarios
from model_pipelining.evidence import get_evidence_provider
from model_pipelining.evidence_normal import NormalMeanVarianceEvidence
from model_pipelining.local_level import LocalLevel
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings
from model_pipelining.normal_mean import NormalMean
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.sampling import MCMCSettings


def json_round_trip(model):
    """Restore a model the way scripts read a saved manifest or config file."""
    return type(model).from_configuration(json.loads(json.dumps(model.configuration)))


class ConfigurationRestoreTests(unittest.TestCase):
    examples = (
        NormalMean(), NormalMean(prior_mean=2, prior_variance=5, mcmc=MCMCSettings(draws=40, seed=7)),
        NormalMeanVariance(), NormalMeanVariance(prior_shape=4, mcmc=MCMCSettings(chains=3, proposal_scale=1.7)),
        LocalLevel(), LocalLevel(initial_mean=3, evolution_variance=0.5),
        LocalLevelVariance(), LocalLevelVariance(prior_scale=4, quadrature=QuadratureSettings(minimum_nodes=16)),
        LocalLevelJointVariance(),
        LocalLevelJointVariance(evolution_variances=(0., .2), evolution_weights=(.3, .7),
                                quadrature=QuadratureSettings(maximum_nodes=512)),
    )

    def test_every_catalog_adapter_restores_from_saved_json(self):
        self.assertEqual({type(model) for model in self.examples}, set(MODEL_TYPES.values()))
        for model in self.examples:
            with self.subTest(model=model):
                restored = json_round_trip(model)
                self.assertEqual(restored, model)
                self.assertEqual(restored.configuration, model.configuration)

    def test_mcmc_settings_are_rebuilt_rather_than_rejected(self):
        for kind in (NormalMean, NormalMeanVariance):
            restored = json_round_trip(kind(mcmc=MCMCSettings(draws=40)))
            self.assertIsInstance(restored.mcmc, MCMCSettings)
            self.assertEqual(restored.mcmc.draws, 40)
            self.assertIsNone(json_round_trip(kind()).mcmc)

    def test_invalid_saved_settings_still_fail_validation(self):
        with self.assertRaisesRegex(ValueError, "draws"):
            NormalMeanVariance.from_configuration({"mcmc": {"draws": 1}})
        with self.assertRaises(TypeError):
            NormalMean.from_configuration({"unknown_setting": 1})


class CatalogTests(unittest.TestCase):
    def test_names_come_from_adapter_classes(self):
        self.assertEqual(MODEL_TYPES, {kind.name: kind for kind in MODEL_TYPES.values()})
        self.assertEqual(EVIDENCE_MODEL_NAMES,
                         ("normal-mean-variance", "local-level-variance", "local-level-joint-variance"))

    def test_build_model_uses_each_adapter_restore(self):
        model = build_model("normal-mean-variance", {"prior_mean": 1, "mcmc": {"draws": 40}})
        self.assertEqual(model, NormalMeanVariance(prior_mean=1, mcmc=MCMCSettings(draws=40)))
        joint = build_model("local-level-joint-variance", {"evolution_variances": [0, 1]})
        self.assertEqual(joint.evolution_variances, (0., 1.))
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("missing", {})
        with self.assertRaisesRegex(ValueError, "dictionary"):
            build_model("local-level", [])

    def test_scenarios_change_only_their_overrides(self):
        reference = NormalMeanVariance(prior_shape=4, mcmc=MCMCSettings(draws=40))
        scenarios = build_scenarios(reference, {"wide": {"prior_scale": 9}})
        self.assertEqual(scenarios, {"wide": NormalMeanVariance(prior_shape=4, prior_scale=9,
                                                                 mcmc=MCMCSettings(draws=40))})
        self.assertEqual(build_scenarios(LocalLevel(), {}), {})
        with self.assertRaisesRegex(ValueError, "override dictionaries"):
            build_scenarios(reference, {"bad": 3})


class CalibrationProviderTests(unittest.TestCase):
    def test_calibration_parameters_are_checked_when_the_provider_is_selected(self):
        for parameters in (None, (), ("mean", "mean"), ["mean"], ("mean", " ")):
            class Provider(NormalMeanVarianceEvidence):
                calibration_parameters = parameters

            class Model(NormalMeanVariance):
                def evidence_provider(self):
                    return Provider(self)

            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, "unique unknown parameter"):
                get_evidence_provider(Model(), "calibration")
        self.assertEqual(get_evidence_provider(NormalMeanVariance(), "calibration").calibration_parameters,
                         ("mean", "observation_variance"))


if __name__ == "__main__":
    unittest.main()
