"""Independent probability/conditioning references for the bounded prototype."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import invgamma, norm, t

from model_pipelining.innovation_reference import (InnovationMixtureReference, ReferenceSettings,
    ReferencePrecisionError, enumerate_sequences, run_innovation_reference, _BatchGaussian, _sequential_condition)
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.models import Observation, ForecastRequest


class InnovationReferenceTests(unittest.TestCase):
    def test_batch_conditioning_matches_independent_forward_backward_recursions(self):
        model = InnovationMixtureReference(initial_mean=-.4, initial_variance=.7, jump_probability=.2)
        values = (-1., .2, 2.3)
        labels, _ = enumerate_sequences(model, 3)
        batch = _BatchGaussian(model, values, labels)
        for variance in (.03, .5, 3.):
            a = batch.condition(variance)
            b = _sequential_condition(model, values, labels, np.array([variance]))
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-11, err_msg=key)

    def test_one_observation_bayes_factors_and_prediction_match_direct_variance_integrals(self):
        model = InnovationMixtureReference(initial_mean=-.4, initial_variance=.7,
            small_variance=.02, large_variance=3., jump_probability=.1)
        y = 2.5
        result = run_innovation_reference((y,), model=model, predictive_points=(0., 2.), variance_points=(1.,))['batch_adaptive']
        evidences, variance_moments, state_means = [], [], []
        for w in (model.small_variance, model.large_variance):
            def density(v):
                return invgamma.pdf(v, 3, scale=2) * norm.pdf(y, loc=-.4, scale=math.sqrt(.7 + w + v))
            evidences.append(quad(density, 0, math.inf, epsabs=1e-12)[0])
            variance_moments.append(quad(lambda v: v * density(v), 0, math.inf, epsabs=1e-12)[0])
            state_means.append(quad(lambda v: (-.4 + (.7 + w) / (.7 + w + v) * (y + .4)) * density(v),
                                    0, math.inf, epsabs=1e-12)[0])
        priors = np.array([.9, .1])
        z = priors @ evidences
        expected = priors * evidences / z
        np.testing.assert_allclose(result['sequence_probabilities'], expected, atol=1e-9)
        self.assertAlmostEqual(result['log_evidence'], math.log(z), places=9)
        self.assertAlmostEqual(result['observation_variance_mean'], priors @ variance_moments / z, places=8)
        self.assertAlmostEqual(result['last_mean'], priors @ state_means / z, places=8)
        covariance = priors[1] * variance_moments[1] / z - (priors @ variance_moments / z) * expected[1]
        self.assertAlmostEqual(result['variance_jump_count_covariance'], covariance, places=8)
        def state_integral(quantity):
            def integrand(v):
                total = 0.
                for prior, w in zip(priors, (model.small_variance, model.large_variance)):
                    q = .7 + w + v
                    m0 = -.4 + .7 / q * (y + .4)
                    ml = -.4 + (.7 + w) / q * (y + .4)
                    moment = {'initial_mean': m0, 'initial_second': .7 * (w + v) / q + m0**2,
                              'cross': .7 * v / q + m0 * ml}[quantity]
                    total += prior * norm.pdf(y, loc=-.4, scale=math.sqrt(q)) * moment
                return total * invgamma.pdf(v, 3, scale=2)
            return quad(integrand, 0, math.inf, epsabs=1e-12)[0] / z
        initial_mean = state_integral('initial_mean')
        self.assertAlmostEqual(result['initial_mean'], initial_mean, places=8)
        self.assertAlmostEqual(result['initial_variance'], state_integral('initial_second') - initial_mean**2, places=8)
        self.assertAlmostEqual(result['initial_last_covariance'], state_integral('cross') - initial_mean * result['last_mean'], places=8)
        for i, point in enumerate((0., 2.)):
            def integrand(v, density=False):
                total = 0.
                for prior, w in zip(priors, (model.small_variance, model.large_variance)):
                    q = .7 + w + v
                    location = -.4 + (.7 + w) / q * (y + .4)
                    c = v * (.7 + w) / q
                    likelihood = norm.pdf(y, loc=-.4, scale=math.sqrt(q))
                    function = norm.pdf if density else norm.cdf
                    law = sum(prob * function(point, loc=location, scale=math.sqrt(c + v + future))
                              for prob, future in zip(priors, (model.small_variance, model.large_variance)))
                    total += prior * likelihood * law
                return total * invgamma.pdf(v, 3, scale=2)
            self.assertAlmostEqual(result['predictive_cdf'][i], quad(integrand, 0, math.inf, epsabs=1e-12)[0] / z, places=8)
            self.assertAlmostEqual(result['predictive_density'][i], quad(lambda v: integrand(v, True), 0, math.inf, epsabs=1e-12)[0] / z, places=8)

    def test_zero_one_and_equal_variances_recover_existing_gaussian_model(self):
        values = (-.1, .2, 1.)
        start = datetime(2020, 1, 1, tzinfo=UTC)
        history = tuple(Observation(start + timedelta(days=i), value) for i, value in enumerate(values))
        request = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=1))
        base = InnovationMixtureReference()
        for model, w in ((replace(base, jump_probability=0), .01),
                         (replace(base, jump_probability=1), 4.),
                         (replace(base, large_variance=.01), .01)):
            with self.subTest(model=model):
                actual = run_innovation_reference(values, model=model)['batch_adaptive']
                reference = LocalLevelVariance(initial_mean=0, initial_variance=1, evolution_variance=w).posterior(request)
                self.assertAlmostEqual(actual['log_evidence'], reference.log_evidence, places=6)
                self.assertAlmostEqual(actual['last_mean'], reference.level_mean, places=6)
                self.assertAlmostEqual(actual['last_variance'], reference.level_variance, places=6)
                self.assertAlmostEqual(actual['predictive_variance'], reference.predictive().variance, places=6)
                for point, probability in zip(actual['predictive_points'], actual['predictive_cdf']):
                    self.assertAlmostEqual(probability, reference.predictive().cdf(point), places=6)

    def test_known_constant_level_recovers_exact_inverse_gamma_and_student_t(self):
        values = (.5, -1., .2)
        model = InnovationMixtureReference(initial_variance=0, small_variance=0, large_variance=0, jump_probability=.2)
        result = run_innovation_reference(values, model=model)['batch_adaptive']
        shape, scale = 3 + len(values) / 2, 2 + sum(y * y for y in values) / 2
        law = t(df=2 * shape, loc=0, scale=math.sqrt(scale / shape))
        evidence = 3 * math.log(2) - float(gammaln(3)) + float(gammaln(shape)) - shape * math.log(scale) - len(values) / 2 * math.log(2 * math.pi)
        self.assertAlmostEqual(result['log_evidence'], evidence, places=9)
        self.assertAlmostEqual(result['observation_variance_mean'], scale / (shape - 1), places=9)
        np.testing.assert_allclose(result['variance_cdf'], invgamma.cdf(result['variance_points'], shape, scale=scale), atol=1e-9)
        np.testing.assert_allclose(result['predictive_cdf'], law.cdf(result['predictive_points']), atol=1e-9)
        np.testing.assert_allclose(result['predictive_density'], law.pdf(result['predictive_points']), atol=1e-9)
        self.assertEqual(result['initial_variance'], 0)
        self.assertEqual(result['last_variance'], 0)

    def test_coincident_scales_leave_labels_at_their_independent_prior(self):
        model = InnovationMixtureReference(small_variance=.1, large_variance=.1, jump_probability=.2)
        result = run_innovation_reference((0., 2., 1.), model=model)
        np.testing.assert_allclose(result['batch_adaptive']['sequence_probabilities'], result['sequence_prior_probabilities'], atol=1e-10)
        np.testing.assert_allclose(result['batch_adaptive']['jump_probabilities'], [.2] * 3, atol=1e-10)
        self.assertAlmostEqual(result['batch_adaptive']['variance_jump_count_covariance'], 0, places=10)

    def test_empty_history_has_one_past_sequence_and_an_independent_future_jump(self):
        model = InnovationMixtureReference(initial_variance=0, small_variance=0, large_variance=9, jump_probability=.15)
        result = run_innovation_reference((), model=model)
        self.assertEqual(result['sequences'], [[]])
        actual = result['batch_adaptive']
        self.assertEqual(actual['jump_probabilities'], [])
        self.assertAlmostEqual(actual['log_evidence'], 0, places=10)
        self.assertAlmostEqual(actual['predictive_variance'], 1 + .15 * 9, places=8)
        self.assertEqual(actual['next_jump_probability'], .15)
        self.assertAlmostEqual(actual['predictive_cdf'][1], .5, places=10)

    def test_shift_updates_local_labels_preserving_variance_dependence_and_future_prior(self):
        result = run_innovation_reference((0., .1, -.1, 3.8, 4.1, 3.9))
        actual = result['batch_adaptive']
        self.assertGreater(actual['jump_probabilities'][3], .9)
        self.assertLess(max(actual['jump_probabilities'][:3]), .1)
        self.assertEqual(actual['next_jump_probability'], .05)
        self.assertLess(actual['variance_jump_count_covariance'], 0)
        self.assertLess(result['maximum_scaled_difference'], 1e-7)

    def test_known_initial_state_and_no_jump_path_preserve_real_state_atoms(self):
        model = InnovationMixtureReference(initial_variance=0, small_variance=0, jump_probability=.1)
        result = run_innovation_reference((.1, .2), model=model, state_points=(-1., 0., 1.))
        for method in ('batch_adaptive', 'sequential_quadrature'):
            actual = result[method]
            self.assertAlmostEqual(actual['initial_cdf_lower'][1], 0, places=12)
            self.assertAlmostEqual(actual['initial_cdf_upper'][1], 1, places=12)
            self.assertAlmostEqual(actual['last_cdf_upper'][1] - actual['last_cdf_lower'][1], actual['sequence_probabilities'][0], places=9)
            self.assertTrue(all(math.isfinite(x) and x > 0 for x in actual['predictive_density']))

    def test_cap_eight_retains_every_sequence_and_quadrature_node(self):
        result = run_innovation_reference((0., .1, -.1, .05, .1, -.1, .15, 0.), predictive_points=(0.,), variance_points=(1.,))
        self.assertEqual(len(result['sequences']), 256)
        self.assertEqual(len({tuple(x) for x in result['sequences']}), 256)
        self.assertTrue(all(x > 0 for x in result['batch_adaptive']['sequence_probabilities']))
        self.assertEqual(result['sequential_quadrature']['numerics']['discarded_sequences'], 0)
        with self.assertRaisesRegex(ValueError, 'at most 8'):
            run_innovation_reference((0.,) * 9)

    def test_invalid_models_data_settings_and_query_grids_fail(self):
        for kwargs in ({'initial_variance': -1}, {'small_variance': 5}, {'jump_probability': 2},
                       {'prior_shape': 1}, {'prior_scale': 0}, {'large_variance': math.nan}, {'initial_mean': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InnovationMixtureReference(**kwargs)
        for values in ((math.nan,), (True,), (math.inf,)):
            with self.assertRaises(ValueError):
                run_innovation_reference(values)
        for kwargs in ({'predictive_points': (1., 0.)}, {'state_points': (0., 0.)}, {'variance_points': (0.,)}):
            with self.assertRaises(ValueError):
                run_innovation_reference((), **kwargs)
        with self.assertRaises(ValueError):
            ReferenceSettings(maximum_nodes=4096)

    def test_unresolved_quadrature_retains_candidate_and_does_not_succeed(self):
        settings = ReferenceSettings(minimum_nodes=8, maximum_nodes=8, maximum_expansions=1)
        with self.assertRaises(ReferencePrecisionError) as caught:
            run_innovation_reference((0., .1, 2.), settings=settings)
        self.assertTrue(caught.exception.evidence['refinement_history'])
        self.assertIn('candidate', caught.exception.evidence)

    def test_unresolved_adaptive_integration_fails_explicitly(self):
        with self.assertRaisesRegex(ReferencePrecisionError, 'Adaptive batch integration'):
            run_innovation_reference((0., .1, 2.), settings=ReferenceSettings(adaptive_limit=1))

    def test_numerical_replay_does_not_change_global_random_state(self):
        before = np.random.get_state()
        first = run_innovation_reference((0., .1, 2.))
        self.assertEqual(first, run_innovation_reference((0., .1, 2.)))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_cli_freezes_design_replays_and_retains_failed_cases(self):
        root = Path(__file__).resolve().parents[1]
        environment = {**os.environ, 'PYTHONPATH': str(root / 'src')}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = json.loads((root / 'configs/innovation_reference.json').read_text())
            config['cases'] = {'prior': [], 'short': [0., .1, 3.]}
            path = directory / 'config.json'
            path.write_text(json.dumps(config))
            def execute(name):
                return subprocess.run([sys.executable, str(root / 'scripts/run_innovation_reference.py'),
                    '--config', str(path), '--output-directory', str(directory / name)],
                    env=environment, capture_output=True, text=True)
            first, second = execute('first'), execute('second')
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            for name in ('prior', 'short'):
                self.assertEqual((directory / f'first/cases/{name}.json').read_bytes(),
                                 (directory / f'second/cases/{name}.json').read_bytes())
            self.assertTrue((directory / 'first/design.json').is_file())
            self.assertTrue((directory / 'first/reference.png').is_file())
            self.assertTrue(all(row['status'] == 'passed' for row in json.loads((directory / 'first/gaussian_limits.json').read_text())))
            config['settings'] = {'minimum_nodes': 8, 'maximum_nodes': 8, 'maximum_expansions': 1}
            path.write_text(json.dumps(config))
            failed = execute('failed')
            self.assertEqual(failed.returncode, 1)
            manifest = json.loads((directory / 'failed/study.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual([row['name'] for row in manifest['cases']], ['prior', 'short'])
            self.assertTrue(all(row['status'] == 'failed' for row in manifest['cases']))
            self.assertIn('evidence', json.loads((directory / 'failed/cases/short.json').read_text()))
            config['cases']['too-long'] = [0.] * 9
            path.write_text(json.dumps(config))
            self.assertNotEqual(execute('invalid').returncode, 0)
            self.assertFalse((directory / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
