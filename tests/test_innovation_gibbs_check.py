"""Independent checks for the research sampler's Gaussian block."""

import unittest

import numpy as np

from model_pipelining.innovation_gibbs_check import (
    _PathGaussian, innovation_gibbs_probes, sample_innovation_gibbs,
)
from model_pipelining.innovation_reference import InnovationMixtureReference


class InnovationGibbsCheckTests(unittest.TestCase):
    def test_precision_likelihood_matches_dense_conditioning_with_identified_states(self):
        for initial_variance in (0., .7):
            for small_variance in (0., .01):
                with self.subTest(initial_variance=initial_variance, small_variance=small_variance):
                    model = InnovationMixtureReference(initial_variance=initial_variance,
                        small_variance=small_variance)
                    history = np.array([.1, .2, 2.])
                    labels = np.array([0, 1, 0])
                    variance = .65
                    process = initial_variance + np.cumsum(np.where(labels, model.large_variance, small_variance))
                    covariance = np.minimum.outer(process, process) + variance * np.eye(3)
                    expected = -.5 * (3 * np.log(2*np.pi) + np.linalg.slogdet(covariance)[1]
                                      + history @ np.linalg.solve(covariance, history))
                    actual = _PathGaussian(model, history, labels).log_likelihood(variance)
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_precision_draw_matches_dense_gaussian_when_transition_is_zero(self):
        model = InnovationMixtureReference(initial_variance=.7, small_variance=0.)
        values = np.array([1.2, -.4])
        labels = np.array([0, 1])
        noise = .8
        sampler = _PathGaussian(model, values, labels)
        rng = np.random.default_rng(511)
        draws = np.stack([sampler.draw(noise, rng) for _ in range(6000)])
        np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
        prior_cov = np.array([[.7, .7], [.7, .7 + model.large_variance]])
        precision = np.linalg.inv(prior_cov) + np.eye(2)/noise
        covariance = np.linalg.inv(precision)
        mean = covariance @ values / noise
        np.testing.assert_allclose(draws[:, 1:].mean(axis=0), mean, atol=.03)
        np.testing.assert_allclose(np.cov(draws[:, 1:].T), covariance, atol=.03)

    def test_blocked_labels_can_enter_and_leave_a_known_state_atom(self):
        model = InnovationMixtureReference(initial_mean=2., initial_variance=0.,
                                         small_variance=0., jump_probability=.5)
        draws, labels = sample_innovation_gibbs([2.2], model, seeds=[32, 62], warmup=32, draws=256)
        repeated, repeated_labels = sample_innovation_gibbs([2.2], model,
            seeds=[32, 62], warmup=32, draws=256)
        np.testing.assert_array_equal(draws, repeated)
        np.testing.assert_array_equal(labels, repeated_labels)
        self.assertTrue(np.all((draws[..., 1] == 0.) == (labels[..., 0] == 0)))
        self.assertTrue(np.all(np.ptp(labels[..., 0], axis=1) > 0))
        result = innovation_gibbs_probes(draws, labels, model, predictive_points=[2.],
            state_points=[2.], variance_points=[1.])
        np.testing.assert_array_equal(result["initial_cdf_lower:2.0"], False)
        np.testing.assert_array_equal(result["initial_cdf_upper:2.0"], True)
        np.testing.assert_allclose(
            (result["last_cdf_upper:2.0"] & ~result["last_cdf_lower:2.0"]).mean(),
            (labels == 0).mean())
