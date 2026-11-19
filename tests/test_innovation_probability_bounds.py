"""Check the envelope independently and retain unseen-event uncertainty."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import gammaln
from scipy.stats import invgamma

from model_pipelining.innovation_probability_bounds import (
    ProbabilityBoundSettings, bound_innovation_probabilities, _draw, _ratio_bounds)
from model_pipelining.innovation_reference import (
    InnovationMixtureReference, _BatchGaussian, run_innovation_reference)


class InnovationProbabilityBoundTests(unittest.TestCase):
    def test_importance_envelope_matches_independent_dense_likelihood_and_densities(self):
        model = InnovationMixtureReference(initial_mean=-.3, initial_variance=.7)
        values = (-.1, .2, 2.)
        v, labels, weights = _draw(model, values, 12, np.random.default_rng(32))
        batch = _BatchGaussian(model, values, labels)
        log_constant = (gammaln(model.prior_shape + len(values) / 2) - gammaln(model.prior_shape)
            - len(values) / 2 * math.log(2 * math.pi * model.prior_scale))
        for index, variance in enumerate(v):
            # Batch conditioning uses a dense covariance, independent of the
            # scalar recursion used to calculate the bounded importance weight.
            reference = batch.condition(variance)
            log_likelihood = np.asarray(reference['log_likelihood']).reshape(-1)[index]
            expected = math.exp(log_likelihood + invgamma.logpdf(variance, model.prior_shape, scale=model.prior_scale)
                - invgamma.logpdf(variance, model.prior_shape + len(values) / 2, scale=model.prior_scale)
                - log_constant)
            self.assertAlmostEqual(weights[index], expected, places=13)
        self.assertTrue(np.all((weights >= 0) & (weights <= 1)))

    def test_empirical_bernstein_has_nonzero_radius_for_unseen_event(self):
        count, confidence = 10000, .95
        lo, hi, denominator, radius = _ratio_bounds(np.array([.2, 0.]), np.zeros(2), count, confidence)
        expected = 7 * math.log(8 / (1 - confidence)) / (3 * (count - 1))
        self.assertAlmostEqual(radius[1], expected)
        self.assertEqual(lo[0], 0)
        self.assertAlmostEqual(hi[0], expected / (.2 - expected))
        self.assertEqual(denominator, (.2 - expected, .2 + expected))

    def test_unresolved_normalizer_yields_vacuous_bounds_even_for_zero_draws(self):
        lo, hi, denominator, _ = _ratio_bounds(np.zeros(3), np.zeros(3), 32, .95)
        self.assertEqual(denominator[0], 0)
        np.testing.assert_array_equal(lo, [0., 0.])
        np.testing.assert_array_equal(hi, [1., 1.])
        result = bound_innovation_probabilities([1000.] * 9,
            settings=ProbabilityBoundSettings(samples=32))
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['zero_weight_draws'], 32)
        self.assertIsNone(result['intervals']['jump_probabilities']['estimate'])
        json.dumps(result, allow_nan=False)

    def test_retained_quiet_and_gradual_rare_path_failures_have_positive_intervals(self):
        for values, paths in (((0., .1, -.1, .05, .1, -.05), ('011100', '101001')),
                ((0., .1, .3, .7, 1., 1.3), ('000111',))):
            result = bound_innovation_probabilities(values, settings=ProbabilityBoundSettings(samples=65536))
            oracle = run_innovation_reference(values)['batch_adaptive']
            for key, interval in result['intervals'].items():
                expected = np.asarray(oracle[key])
                self.assertTrue(np.all(expected >= np.array(interval['lower']) - 1e-9), key)
                self.assertTrue(np.all(expected <= np.array(interval['upper']) + 1e-9), key)
            for path in paths:
                interval, index = result['intervals']['sequence_probabilities'], int(path, 2)
                self.assertGreater(interval['upper'][index] - interval['lower'][index], 0)
                self.assertGreater(oracle['sequence_probabilities'][index], 0)
            unseen = np.flatnonzero(np.array(result['sequence_proposal_visits']) == 0)
            self.assertGreater(len(unseen), 0)
            for index in unseen:
                self.assertEqual(result['intervals']['sequence_probabilities']['estimate'][index], 0)
                self.assertGreater(result['intervals']['sequence_probabilities']['upper'][index], 0)

    def test_gaussian_limits_label_independence_and_real_atoms(self):
        base = InnovationMixtureReference()
        for model in (replace(base, jump_probability=0), replace(base, jump_probability=1),
                replace(base, small_variance=.1, large_variance=.1),
                replace(base, initial_variance=0, small_variance=0)):
            values = (.1, .2)
            result = bound_innovation_probabilities(values, model=model,
                settings=ProbabilityBoundSettings(samples=32768))
            oracle = run_innovation_reference(values, model=model)['batch_adaptive']
            for key, interval in result['intervals'].items():
                self.assertTrue(np.all(np.asarray(oracle[key]) >= np.array(interval['lower']) - 1e-9), key)
                self.assertTrue(np.all(np.asarray(oracle[key]) <= np.array(interval['upper']) + 1e-9), key)
            if model.initial_variance == 0:
                self.assertEqual(result['intervals']['initial_cdf_lower']['estimate'][0], 0)
                self.assertAlmostEqual(result['intervals']['initial_cdf_upper']['estimate'][0], 1)
                self.assertGreater(result['intervals']['last_cdf_upper']['estimate'][0],
                                   result['intervals']['last_cdf_lower']['estimate'][0])

    def test_seed_replay_rng_isolation_and_empty_history(self):
        np.random.seed(98)
        before = np.random.get_state()
        settings = ProbabilityBoundSettings(samples=4096)
        first = bound_innovation_probabilities((), settings=settings)
        self.assertEqual(first, bound_innovation_probabilities((), settings=settings))
        self.assertNotEqual(first, bound_innovation_probabilities((), settings=replace(settings, seed=1)))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(first['normalizer']['log_envelope'], 0)
        self.assertEqual(first['normalizer']['scaled_estimate'], 1)

    def test_nearly_dominant_prior_keeps_unvisited_positive_path_mass_possible(self):
        model = InnovationMixtureReference(jump_probability=1 - 1e-10)
        values = (.1, .2, .3)
        result = bound_innovation_probabilities(values, model=model,
            settings=ProbabilityBoundSettings(samples=4096))
        oracle = run_innovation_reference(values, model=model)['batch_adaptive']
        interval = result['intervals']['sequence_probabilities']
        self.assertEqual(result['sequence_proposal_visits'][-1], 4096)
        for index in range(7):
            self.assertGreater(oracle['sequence_probabilities'][index], 0)
            self.assertEqual(interval['estimate'][index], 0)
            self.assertGreater(interval['upper'][index], oracle['sequence_probabilities'][index])

    def test_long_history_never_enumerates_paths_or_treats_zero_weights_as_precision(self):
        result = bound_innovation_probabilities([0.] * 160,
            settings=ProbabilityBoundSettings(samples=64))
        self.assertNotIn('sequence_probabilities', result['intervals'])
        self.assertEqual(len(result['intervals']['jump_probabilities']['lower']), 160)
        self.assertEqual(result['status'], 'unresolved')
        self.assertFalse(result['checks']['positive_normalizer'])

    def test_saved_statistics_recalculate_intervals(self):
        result = bound_innovation_probabilities((0., .1), settings=ProbabilityBoundSettings(samples=8192))
        stats = result['sufficient_statistics']
        lower, upper, _, _ = _ratio_bounds(np.array(stats['means']), np.array(stats['sample_variances']),
            result['settings']['samples'], result['settings']['confidence'])
        np.testing.assert_array_equal(lower, np.concatenate([item['lower'] for item in result['intervals'].values()]))
        np.testing.assert_array_equal(upper, np.concatenate([item['upper'] for item in result['intervals'].values()]))

    def test_input_and_budget_validation(self):
        for kwargs in ({'samples': True}, {'samples': 1}, {'samples': 4194305}, {'seed': -1},
                {'confidence': .4}, {'confidence': 1}, {'tolerance': 0}, {'tolerance': float('nan')}):
            with self.assertRaises(ValueError):
                ProbabilityBoundSettings(**kwargs)
        for values in ([True], [float('inf')], [0.] * 513, np.array([0.])):
            with self.assertRaises(ValueError):
                bound_innovation_probabilities(values)
        with self.assertRaises(ValueError):
            bound_innovation_probabilities((), variance_points=(0.,))
        extreme = bound_innovation_probabilities((),
            model=InnovationMixtureReference(prior_shape=100, prior_scale=1e308),
            settings=ProbabilityBoundSettings(samples=32))
        self.assertEqual(extreme['normalizer']['log_envelope'], 0)
        json.dumps(extreme, allow_nan=False)
        with np.errstate(invalid='ignore'):
            with self.assertRaisesRegex(ValueError, 'envelope'):
                bound_innovation_probabilities((0.,),
                    model=InnovationMixtureReference(prior_shape=1e308),
                    settings=ProbabilityBoundSettings(samples=32))

    def test_cli_freezes_parent_failures_replays_and_rejects_overwrite_or_wrong_oracle(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OPENBLAS_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / 'config.json'
            config.write_text(json.dumps(dict(model=InnovationMixtureReference().configuration,
                settings=dict(minimum_particles=32, maximum_particles=64, replicates=8, tolerance=1e-12),
                predictive_points=[0., 2.], state_points=[0.], variance_points=[1.],
                short_cases={'prior': []},
                long_history=dict(count=9, onset=4, seeds=[42], baseline=0., shift=2., slope=.1, noise_sd=1.))))
            parent, output = directory / 'parent', directory / 'bounds'
            made = subprocess.run([sys.executable, str(root / 'scripts/run_innovation_particles.py'),
                '--config', str(config), '--output-directory', str(parent)], env=env, capture_output=True, text=True)
            self.assertEqual(made.returncode, 1, made.stderr)
            original = (parent / 'summary.json').read_bytes()
            command = [sys.executable, str(root / 'scripts/check_innovation_probability_bounds.py'),
                '--source-directory', str(parent), '--output-directory', str(output), '--samples', '32']
            checked = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 1, checked.stderr)
            self.assertEqual((parent / 'summary.json').read_bytes(), original)
            manifest = json.loads((output / 'study.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['precision_status'], 'unresolved')
            self.assertEqual(manifest['study_confidence'], .95)
            self.assertGreater(manifest['case_confidence'], .95)
            self.assertEqual(json.loads((output / 'validation.json').read_text())['status'], 'passed')
            self.assertEqual((output / 'inputs/summary.json').read_bytes(), original)
            self.assertTrue((output / 'sources/src/model_pipelining/innovation_probability_bounds.py').is_file())
            record = json.loads((output / 'cases/stable-42.json').read_text())
            replay = bound_innovation_probabilities(record['values'],
                model=InnovationMixtureReference(**record['model']), settings=ProbabilityBoundSettings(**record['settings']),
                **{key: record[key] for key in ('predictive_points', 'state_points', 'variance_points')})
            self.assertEqual(record, replay)
            namespace = runpy.run_path(str(root / 'scripts/check_innovation_probability_bounds.py'))
            with patch.dict(namespace['plot_saved'].__globals__, bound_innovation_probabilities=lambda *a, **k: self.fail('Saved-only plot')):
                namespace['plot_saved'](output)
            repeated = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / 'study.json').read_text(), json.dumps(manifest, indent=2) + '\n')
            oracle = parent / 'oracles/short-prior.json'
            wrong = json.loads(oracle.read_text())
            wrong['model']['initial_mean'] += 1
            oracle.write_text(json.dumps(wrong))
            rejected = directory / 'rejected'
            command[command.index('--output-directory') + 1] = str(rejected)
            mismatch = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn('oracle identity', mismatch.stderr)
            self.assertFalse(rejected.exists())
