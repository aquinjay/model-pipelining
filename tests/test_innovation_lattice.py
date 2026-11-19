"""Independent checks of the new grid envelope and probability bounds."""

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.special import logsumexp

from model_pipelining.innovation_lattice import (
    _StateLattice, _probability_intervals, bound_innovation_probabilities_lattice,
)
from model_pipelining.innovation_probability_bounds import ProbabilityBoundSettings
from model_pipelining.innovation_reference import (
    InnovationMixtureReference, _BatchGaussian, enumerate_sequences, run_innovation_reference,
)


class InnovationLatticeTests(unittest.TestCase):
    def test_grid_likelihood_bound_against_dense_gaussian_with_and_without_atoms(self):
        base = InnovationMixtureReference(initial_mean=.7)
        values = (-.2, .1, 3.1)
        limits = (.2, 12.)
        for model in (base, replace(base, initial_variance=0., small_variance=0.),
                      replace(base, jump_probability=0.),
                      replace(base, jump_probability=1.),
                      replace(base, large_variance=base.small_variance)):
            centered = tuple(y - model.initial_mean for y in values)
            labels, log_prior = enumerate_sequences(model, len(values))
            batch = _BatchGaussian(model, values, labels)
            grid = _StateLattice(model, centered, limits)
            for s in (.2, 1.3, 12.):
                exact = logsumexp(batch.condition(1 / s)['log_likelihood'].ravel() + log_prior)
                discrete, _ = grid.filter(s)
                ratio = math.exp(discrete + len(values) / 2 * math.log(s / (2 * math.pi)) - exact)
                self.assertGreaterEqual(ratio, grid.mass_ratio[0])
                self.assertLessEqual(ratio, grid.mass_ratio[1])

    def test_zero_visits_and_omitted_tail_retain_positive_uncertainty(self):
        settings = ProbabilityBoundSettings(samples=4096)
        inner = _probability_intervals(np.array([.8, 0.]), np.zeros(2), settings, (1., 1.), 0.)
        expanded = _probability_intervals(np.array([.8, 0.]), np.zeros(2), settings, (.99, 1.01), .01)
        self.assertEqual(expanded[0][0], 0.)
        self.assertGreater(expanded[1][0], inner[1][0])
        self.assertGreater(inner[1][0], 0.)
        self.assertLess(expanded[2][0], inner[2][0])
        self.assertGreater(expanded[2][1], inner[2][1])

    def test_dominant_paths_preserve_atoms_and_probability_oracle_and_replay(self):
        model = InnovationMixtureReference(initial_variance=0., small_variance=0., jump_probability=1 - 1e-10)
        values = (.1, .2, 2.9)
        settings = ProbabilityBoundSettings(samples=8192, seed=13)
        result = bound_innovation_probabilities_lattice(values, model=model, settings=settings)
        self.assertEqual(result, bound_innovation_probabilities_lattice(values, model=model, settings=settings))
        self.assertEqual(result['status'], 'completed')
        oracle = run_innovation_reference(values, model=model)['batch_adaptive']
        for key, interval in result['intervals'].items():
            self.assertTrue(np.all(np.asarray(oracle[key]) >= np.asarray(interval['lower']) - 5e-6), key)
            self.assertTrue(np.all(np.asarray(oracle[key]) <= np.asarray(interval['upper']) + 5e-6), key)
        self.assertEqual(result['sequence_proposal_visits'][:-1], [0] * 7)
        interval = result['intervals']['sequence_probabilities']
        self.assertTrue(all(bound > 0. for bound in interval['upper'][:-1]))
        self.assertEqual(oracle['initial_cdf_lower'], [0.])
        self.assertAlmostEqual(oracle['initial_cdf_upper'][0], 1.)
        self.assertAlmostEqual(result['posterior_estimates']['initial_cdf_upper'][0], 1.)
        self.assertTrue(all(value > 0. for value in result['posterior_estimates']['predictive_density']))

    def test_extreme_prior_precision_hits_explicit_grid_budget(self):
        settings = ProbabilityBoundSettings(samples=32)
        with self.assertRaisesRegex(ValueError, 'lattice exceeds'):
            bound_innovation_probabilities_lattice((0., 1.), settings=settings,
                model=InnovationMixtureReference(initial_variance=1e-12))
