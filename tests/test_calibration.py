"""Independent calibration references, failure visibility and simulation replay."""
from dataclasses import replace
import csv
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import invgamma,t
from model_pipelining.calibration import run_calibration,plot_calibration
from model_pipelining.evidence_normal import NormalMeanVarianceEvidence
from model_pipelining.local_level_variance import LocalLevelVariance,QuadratureSettings
from model_pipelining.normal_mean_variance import NormalMeanVariance
from test_normal_mean import request
import test_predictive_checks as fixtures


class CalibrationTests(unittest.TestCase):
    setUp=fixtures.PredictiveWorkflowTests.setUp
    register=fixtures.PredictiveWorkflowTests.register
    read_json=fixtures.PredictiveWorkflowTests.read_json

    def run_cal(self,name='calibration',**changes):
        values=dict(model=self.model,observation_count=4,trials=5,seed=9,output_directory=self.root/name)
        values.update(changes)
        return run_calibration(**values)

    def test_exact_posterior_cdfs_match_hand_updates(self):
        model=NormalMeanVariance(prior_mean=0,prior_mean_precision=1,prior_shape=3,prior_scale=2)
        result,_=model.evidence_provider().calibration_fit(request((1,3)),{'mean':2.,'observation_variance':2.},.8)
        self.assertAlmostEqual(result['mean']['cdf'],t.cdf(2,df=8,loc=4/3,scale=(13/36)**.5))
        self.assertAlmostEqual(result['observation_variance']['cdf'],invgamma.cdf(2,a=4,scale=13/3))

    def test_known_level_calibration_has_conjugate_variance_reference(self):
        model=LocalLevelVariance(initial_mean=2,initial_variance=0,evolution_variance=0)
        provider=model.evidence_provider()
        self.assertEqual(provider.calibration_parameters,('observation_variance',))
        result,_=provider.calibration_fit(request((1,3,4)),{'observation_variance':1.5},.95)
        expected=invgamma(a=4.5,scale=5.)
        self.assertAlmostEqual(result['observation_variance']['cdf'],expected.cdf(1.5),places=7)
        self.assertAlmostEqual(result['observation_variance']['median'],expected.ppf(.5),places=6)

    def test_local_generator_independence_and_covariance(self):
        provider=LocalLevelVariance(initial_mean=2,initial_variance=2,evolution_variance=.2).evidence_provider()
        rng=np.random.default_rng(71)
        draws=[provider.calibration_draw(3,rng) for _ in range(6000)]
        values=np.asarray([item[1] for item in draws])
        self.assertAlmostEqual(values[:,0].mean(),2,delta=.08)
        self.assertAlmostEqual(values[:,0].var(),3.2,delta=.16)
        self.assertAlmostEqual(np.cov(values[:,0],values[:,2])[0,1],2.2,delta=.16)
        self.assertTrue(all(set(item[0])==set(provider.calibration_parameters) for item in draws))

    def test_replay_prefix_and_global_rng_isolation(self):
        state=np.random.get_state()
        first=self.run_cal('first')
        second=self.run_cal('second')
        larger=self.run_cal('larger',trials=7)
        for name in ('calibration.csv','summary.json','inference.json'):
            self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
        for path in (first/'trials').iterdir():
            self.assertEqual(path.read_bytes(),(larger/'trials'/path.name).read_bytes())
        np.testing.assert_array_equal(state[1],np.random.get_state()[1])

    def test_seeded_calibration_and_biased_cdf_control(self):
        output=self.run_cal(trials=400)
        summaries=self.read_json(output,'summary.json')
        for summary in summaries.values():
            self.assertLess(abs(summary['cdf_mean']-.5),.075)
            self.assertLess(summary['cdf_uniform_distance'],.12)
            self.assertGreater(summary['coverage']['fraction'],.9)
        original=NormalMeanVarianceEvidence.calibration_fit
        def biased(provider,*args):
            results,evidence=original(provider,*args)
            for result in results.values():result['cdf']=.999
            return results,evidence
        with patch.object(NormalMeanVarianceEvidence,'calibration_fit',new=biased):
            bad=self.run_cal('biased',trials=20)
        self.assertGreater(self.read_json(bad,'summary.json')['mean']['cdf_uniform_distance'],.99)
        self.assertEqual(self.read_json(bad,'review.json')['status'],'pending')
        self.assertEqual(self.read_json(bad,'calibration.json')['status'],'completed')

    def test_missing_parameter_fails_trial_and_later_trials_continue(self):
        original=NormalMeanVarianceEvidence.calibration_fit
        count=0
        def missing(provider,*args):
            nonlocal count
            count+=1
            results,evidence=original(provider,*args)
            if count==1:results.pop('mean')
            return results,evidence
        with patch.object(NormalMeanVarianceEvidence,'calibration_fit',new=missing):
            output=self.run_cal()
        self.assertEqual(self.read_json(output,'calibration.json')['status'],'failed')
        summary=self.read_json(output,'summary.json')['mean']
        self.assertEqual(summary['completed_trials'],4)
        self.assertFalse(summary['band_applies_to_all_trials'])

    def test_dynamic_numerical_failures_retain_truth_and_precision_evidence(self):
        model=LocalLevelVariance(quadrature=QuadratureSettings(minimum_nodes=8,maximum_nodes=16,maximum_expansions=1))
        output=self.run_cal(model=model)
        self.assertEqual(self.read_json(output,'calibration.json')['status'],'failed')
        self.assertEqual(len(list((output/'trials').iterdir())),5)
        self.assertIsNotNone(self.read_json(output,'inference.json')[0]['inference'])
        with patch.object(LocalLevelVariance,'posterior',side_effect=AssertionError('No fitting')):
            self.assertTrue(plot_calibration(output).is_file())

    def test_invalid_options_raise_before_creation_and_outputs_cannot_be_reused(self):
        for values in ({'trials':0},{'observation_count':False},{'seed':-1},{'interval_level':1}):
            with self.assertRaises(ValueError):self.run_cal(**values)
        self.assertFalse((self.root/'calibration').exists())
        self.run_cal()
        with self.assertRaises(FileExistsError):self.run_cal()

    def test_cli_both_models_and_saved_only_chart(self):
        for kind in ('normal-mean-variance','local-level-variance'):
            output=self.root/kind
            result=subprocess.run([sys.executable,'scripts/calibrate_model.py','--model',kind,
                '--config',f'configs/{kind.replace("-","_")}_priors.json','--trials','3','--observation-count','4',
                '--output-directory',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((output/'calibration.png').is_file())

    def test_dynamic_continuous_parameters_have_finite_recovery_intervals(self):
        output=self.run_cal(model=LocalLevelVariance())
        self.assertEqual(self.read_json(output,'calibration.json')['status'],'completed')
        with (output/'calibration.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual({row['parameter'] for row in rows},{'initial_level','last_level','observation_variance'})
        self.assertTrue(all(float(row['lower'])<float(row['upper']) for row in rows))
