"""Independent numerical and reproducibility checks for particle approximation."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from model_pipelining.innovation_particles import (ParticleSettings, ParticlePrecisionError,
    run_innovation_particles, _filter, _condition, _raw, _aggregate, _systematic,
    _flatten, _multipliers)
from model_pipelining.innovation_reference import (InnovationMixtureReference,
    run_innovation_reference, enumerate_sequences, _BatchGaussian)


class InnovationParticleTests(unittest.TestCase):
    def test_per_particle_smoothing_matches_independent_dense_conditioning(self):
        model = InnovationMixtureReference(initial_mean=-.4, initial_variance=.7)
        values = (-1., .2, 2.3)
        labels, _ = enumerate_sequences(model, 3)
        variances = np.geomspace(.01, 10, len(labels))
        actual = _condition(model, values, labels, variances)
        batch = _BatchGaussian(model, values, labels)
        for index, variance in enumerate(variances):
            reference = batch.condition(variance)
            for key in actual:
                self.assertAlmostEqual(actual[key][index], reference[key][index, 0], places=10, msg=key)

    def test_one_step_optimal_proposal_and_evidence_match_hand_calculation(self):
        model = InnovationMixtureReference(jump_probability=.25)
        y, count, seed = 2., 10000, 32
        rng = np.random.default_rng(seed)
        variance = model.prior_scale / rng.gamma(model.prior_shape, size=count)
        pred = model.initial_variance + np.array([model.small_variance, model.large_variance])
        q = pred + variance[:, None]
        likelihood = np.exp(-.5 * y**2 / q) / np.sqrt(2 * math.pi * q)
        increments = likelihood @ [.75, .25]
        probability = .25 * likelihood[:, 1] / increments
        expected_labels = rng.random(count) < probability
        v, weights, labels, state, logz, diagnostics = _filter(model, (y,), count, np.random.default_rng(seed), .5)
        np.testing.assert_array_equal(v, variance)
        np.testing.assert_array_equal(labels[:, 0], expected_labels)
        np.testing.assert_allclose(weights, increments / increments.sum(), atol=1e-16)
        self.assertAlmostEqual(logz, math.log(increments.mean()), places=13)
        self.assertFalse(diagnostics['steps'][0]['resampled'])
        self.assertTrue(np.isfinite(state['last_mean']).all())

    def test_systematic_resampling_is_unbiased_and_preserves_zero_mass(self):
        weights = np.array([0., .1, .2, .7])
        rng = np.random.default_rng(99)
        counts = sum((np.bincount(_systematic(weights, rng), minlength=4) for _ in range(5000)))
        np.testing.assert_allclose(counts / counts.sum(), weights, atol=.004)
        self.assertEqual(counts[0], 0)

    def test_resampling_keeps_variance_labels_and_states_aligned(self):
        model = InnovationMixtureReference()
        values = tuple([0.] * 12 + [4.] * 12)
        variance, weights, labels, state, _, diagnostics = _filter(model, values, 256, np.random.default_rng(2), .95)
        self.assertTrue(any(row['resampled'] for row in diagnostics['steps']))
        self.assertLess(diagnostics['distinct_variance_ancestors'], 256)
        batch = _BatchGaussian(model, values, labels[:4])
        for i in range(4):
            reference = batch.condition(variance[i])
            for key in state:
                self.assertAlmostEqual(state[key][i], reference[key][i, 0], places=9)
        self.assertAlmostEqual(weights.sum(), 1, places=12)

    def test_evidence_weighting_and_joint_moments_use_unnormalized_estimates(self):
        model = InnovationMixtureReference(initial_variance=0, small_variance=0, large_variance=0)
        replicates = []
        for evidence, variance in ((1., 1.), (3., 5.), (2., 2.)):
            raw = _raw(model, (), np.array([variance]), np.ones(1), np.empty((1, 0), dtype=int),
                _condition(model, (), np.empty((1, 0), dtype=int), np.array([variance])), 1., (), (), ())
            replicates.append(dict(log_evidence=math.log(evidence), raw=raw))
        summary, mcse, ess = _aggregate(replicates, model, 1.)
        self.assertAlmostEqual(summary['observation_variance_mean'], 20 / 6)
        self.assertNotAlmostEqual(summary['observation_variance_mean'], 8 / 3)
        self.assertAlmostEqual(summary['log_evidence'], math.log(2))
        deleted = np.array([19 / 5, 5 / 3, 16 / 4])
        expected_mcse = math.sqrt(2 / 3 * np.sum((deleted - deleted.mean())**2))
        self.assertAlmostEqual(mcse['observation_variance_mean'], expected_mcse)
        self.assertAlmostEqual(ess, 36 / 14)

    def test_joint_moments_and_mcse_include_between_run_dependence(self):
        model = InnovationMixtureReference()
        replicates = []
        for v, label, mean in ((1., 0, -1.), (4., 1, 2.), (2., 1, 0.)):
            labels = np.array([[label]])
            condition = dict(initial_mean=np.array([mean]), last_mean=np.array([2 * mean]),
                initial_variance=np.array([.2]), last_variance=np.array([.3]), state_covariance=np.array([.1]))
            raw = _raw(model, (0.,), np.array([v]), np.ones(1), labels, condition, 1., (), (), ())
            replicates.append(dict(log_evidence=0., raw=raw))
        summary, mcse, _ = _aggregate(replicates, model, 1.)
        self.assertAlmostEqual(summary['initial_variance'], .2 + 5 / 3 - (1 / 3)**2)
        self.assertAlmostEqual(summary['initial_last_covariance'], .1 + 10 / 3 - 2 / 9)
        self.assertAlmostEqual(summary['variance_jump_count_covariance'], 2 - 14 / 9)
        # Delete-one covariances are .0, .25, .75, rather than per-run zeroes.
        deleted = np.array([0., .25, .75])
        self.assertAlmostEqual(mcse['variance_jump_count_covariance'],
            math.sqrt(2 / 3 * np.sum((deleted - deleted.mean())**2)))

    def test_short_history_matches_all_oracle_probabilities_and_laws(self):
        values = (0., .1, 3.)
        settings = ParticleSettings(minimum_particles=1024, maximum_particles=4096, replicates=24)
        actual = run_innovation_particles(values, settings=settings)
        target = run_innovation_reference(values)['batch_adaptive']
        summary, mcse = actual['posterior'], actual['mcse']
        for key, value in summary.items():
            if key == 'jump_count':
                expected = sum(target['jump_probabilities'])
            else:
                expected = target[key]
            error = np.abs(np.asarray(value) - expected)
            self.assertTrue(np.all(error <= 8 * np.asarray(mcse[key]) + 2e-4), key)
        self.assertLess(max(abs(np.array(summary['sequence_probabilities']) - target['sequence_probabilities'])), .015)
        self.assertEqual(summary['next_jump_probability'], .05)
        self.assertLess(summary['variance_jump_count_covariance'], 0)

    def test_gaussian_limits_and_empty_history_use_checked_quadrature(self):
        base = InnovationMixtureReference()
        for values, model in (((), base), ((0., 2.), replace(base, jump_probability=0)),
                ((0., 2.), replace(base, jump_probability=1)),
                ((0., 2.), replace(base, small_variance=.1, large_variance=.1, jump_probability=.2))):
            actual = run_innovation_particles(values, model=model)
            target = run_innovation_reference(values, model=model)['batch_adaptive']
            self.assertEqual(actual['mode'], 'gaussian_quadrature')
            self.assertIsNone(actual['mcse'])
            self.assertTrue(actual['numerics']['converged'])
            for key, value in actual['posterior'].items():
                expected = sum(target['jump_probabilities']) if key == 'jump_count' else target[key]
                np.testing.assert_allclose(value, expected, atol=2e-6, rtol=2e-6, err_msg=key)

    def test_real_state_atoms_and_independent_future_jump_survive(self):
        model = InnovationMixtureReference(initial_variance=0, small_variance=0)
        result = run_innovation_particles((.1, .2), model=model, state_points=(-1., 0., 1.),
            settings=ParticleSettings(tolerance=.1))
        summary = result['posterior']
        self.assertEqual(summary['initial_cdf_lower'][1], 0)
        self.assertAlmostEqual(summary['initial_cdf_upper'][1], 1)
        self.assertAlmostEqual(summary['last_cdf_upper'][1] - summary['last_cdf_lower'][1], summary['sequence_probabilities'][0])
        self.assertEqual(summary['next_jump_probability'], .05)
        self.assertTrue(all(x > 0 for x in summary['predictive_density']))

    def test_seed_replay_and_global_rng_isolation(self):
        settings = ParticleSettings(tolerance=.1)
        np.random.seed(81)
        before = np.random.get_state()
        first = run_innovation_particles((0., .1, 3.), settings=settings)
        second = run_innovation_particles((0., .1, 3.), settings=settings)
        self.assertEqual(first, second)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        changed = run_innovation_particles((0., .1, 3.), settings=replace(settings, seed=1))
        self.assertNotEqual(first['posterior'], changed['posterior'])

    def test_long_history_is_bounded_without_enumeration_and_retains_failure(self):
        settings = ParticleSettings(minimum_particles=32, maximum_particles=64, replicates=8, tolerance=1e-12)
        values = np.random.default_rng(42).normal(0, .7, 160).tolist()
        with patch('model_pipelining.innovation_reference.enumerate_sequences', side_effect=AssertionError('No enumeration')):
            with self.assertRaises(ParticlePrecisionError) as raised:
                run_innovation_particles(values, settings=settings)
        evidence = raised.exception.evidence
        self.assertEqual(evidence['status'], 'unresolved')
        self.assertEqual([level['particles'] for level in evidence['levels']], [32, 64])
        for level in evidence['levels']:
            self.assertEqual(len(level['replicates']), 8)
            self.assertEqual(len(level['posterior']['jump_probabilities']), 160)
            self.assertNotIn('sequence_probabilities', level['posterior'])
            self.assertEqual(len(level['replicates'][0]['diagnostics']['steps']), 160)

    def test_failed_replicate_is_preserved_without_replacement(self):
        with patch('model_pipelining.innovation_particles._filter', side_effect=ValueError('injected')) as mocked:
            with self.assertRaises(ParticlePrecisionError) as raised:
                run_innovation_particles((1.,))
        self.assertEqual(mocked.call_count, 1)
        row = raised.exception.evidence['levels'][0]['replicates'][0]
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error'], 'injected')

    def test_validation_of_inputs_resource_limits_and_moment_assumptions(self):
        settings = ParticleSettings(resample_fraction=np.float32(.5), tolerance=np.float32(.05))
        self.assertIs(type(settings.tolerance), float)
        json.dumps(run_innovation_particles((), settings=settings), allow_nan=False)
        for kwargs in ({'minimum_particles': 16}, {'maximum_particles': 512}, {'maximum_particles': 1536},
                {'replicates': 7}, {'seed': True}, {'tolerance': float('nan')}, {'confidence': .4}):
            with self.assertRaises(ValueError):
                ParticleSettings(**kwargs)
        for values in ([0.] * 513, [float('inf')], [True], np.array([1.])):
            with self.assertRaises(ValueError):
                run_innovation_particles(values)
        with self.assertRaises(ValueError):
            run_innovation_particles((1.,), model=InnovationMixtureReference(prior_shape=2))
        with self.assertRaises(ValueError):
            run_innovation_particles((1.,), state_points=(1., 0.))
        with self.assertRaises(ValueError):
            run_innovation_particles((1.,), variance_points=(0.,))

    def test_refinement_requires_two_independent_levels_and_no_false_certificate(self):
        result = run_innovation_particles((0., .1, 3.), settings=ParticleSettings(tolerance=.1))
        self.assertGreaterEqual(len(result['levels']), 2)
        self.assertIsNone(result['levels'][0]['refinement_envelope'])
        self.assertTrue(all(result['levels'][-1]['checks'].values()))
        self.assertIn('no certified mass bound', result['uncertainty_kind'])
        self.assertGreater(result['levels'][-1]['critical_value'], 1.96)
        self.assertTrue(np.all(_flatten(result['mcse']) * _multipliers(result['posterior'], result['comparison_scale']) >= 0))

    def _cli(self, config, output, root):
        path = root / 'config.json'
        path.write_text(json.dumps(config))
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        return subprocess.run([sys.executable, 'scripts/run_innovation_particles.py', '--config', str(path),
            '--output-directory', str(output)], env=env, capture_output=True, text=True)

    def _config(self):
        return dict(model=InnovationMixtureReference().configuration,
            settings=dict(minimum_particles=32, maximum_particles=64, replicates=8, tolerance=1e-12),
            predictive_points=[0., 2.], state_points=[0.], variance_points=[1.], short_cases={'prior': []},
            long_history=dict(count=9, onset=4, seeds=[42], baseline=0., shift=2., slope=.1, noise_sd=1.))

    def test_cli_freezes_inputs_retains_unresolved_outputs_and_draws_saved_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'run'
            completed = self._cli(self._config(), output, root)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            design = json.loads((output / 'design.json').read_text())
            self.assertEqual(len(design['cases']), 8)
            self.assertEqual(len(design['cases'][-1]['values']), 9)
            manifest = json.loads((output / 'study.json').read_text())
            self.assertEqual(manifest['status'], 'unresolved')
            self.assertEqual(len(manifest['cases']), 8)
            saved = json.loads((output / 'cases/stable-42.json').read_text())
            self.assertEqual(saved['status'], 'unresolved')
            self.assertEqual(len(saved['levels']), 2)
            self.assertTrue((output / 'particles.png').is_file())
            existing = self._cli(self._config(), output, root)
            self.assertNotEqual(existing.returncode, 0)
            invalid = self._config()
            invalid['long_history']['count'] = 513
            rejected = self._cli(invalid, root / 'invalid', root)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / 'invalid').exists())

    def test_cli_successful_gaussian_exercise_replays_numerical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config()
            config['model']['jump_probability'] = 0
            for name in ('first', 'second'):
                result = self._cli(config, root / name, root)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / 'first/checks.json').read_text())['status'], 'passed')
            for path in (root / 'first/cases').glob('*.json'):
                self.assertEqual(path.read_bytes(), (root / 'second/cases' / path.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
