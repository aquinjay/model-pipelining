"""Discrete/mixed posterior calibration with genuine CDF jumps and closed masses."""
from dataclasses import replace
import csv,json,math,subprocess,sys
import unittest
from unittest.mock import patch
import numpy as np
from model_pipelining.calibration import run_calibration,plot_calibration,calibration_transform
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import QuadratureSettings
from test_normal_mean import request
import test_predictive_checks as fixtures


class DistributionalTransformTests(unittest.TestCase):
    def transform(self,item,**changes):
        options=dict(seed=72,trial=0,parameter='category',interval_level=.95);options.update(changes)
        return calibration_transform(item,**options)

    def test_pure_atomic_distribution_is_uniform_after_randomization(self):
        rng=np.random.default_rng(43)
        truth=rng.choice(2,size=6000,p=(.3,.7))
        values=[];original=[]
        for index,state in enumerate(truth):
            left,right=(0.,.3) if state==0 else (.3,1.)
            item=self.transform({'cdf':right,'cdf_lower':left,'cdf_upper':right,'interval_mass':1.},trial=index)
            self.assertTrue(left<=item['cdf']<=right)
            values.append(item['cdf']);original.append(right)
        values=np.sort(values)
        distance=max(np.max(np.arange(1,len(values)+1)/len(values)-values),np.max(values-np.arange(len(values))/len(values)))
        self.assertLess(distance,.025)
        self.assertGreater(np.mean(original),.75)

    def test_continuous_transform_is_identity_without_random_draws(self):
        with patch.object(np.random,'Generator',side_effect=AssertionError('No randomization for a continuous CDF')):
            result=self.transform({'cdf':.2})
        self.assertEqual(result['cdf'],.2)
        self.assertFalse(result['randomized'])
        self.assertEqual(result['interval_mass'],.95)

    def test_parameter_labels_and_trial_prefix_are_stable(self):
        item={'cdf':.7,'cdf_lower':.1,'cdf_upper':.7}
        first={name:self.transform(item,parameter=name,trial=5)['cdf'] for name in ('a','b')}
        second={name:self.transform(item,parameter=name,trial=5)['cdf'] for name in ('b','a')}
        self.assertEqual(first,second)
        self.assertNotEqual(first['a'],first['b'])

    def test_invalid_bounds_or_masses_are_rejected(self):
        for item in ({'cdf':.2,'cdf_lower':.3},{'cdf':1.2},{'cdf':.2,'interval_mass':0},
                     {'cdf':.2,'cdf_upper':.4},{'cdf':.2,'cdf_lower':float('nan')}):
            with self.assertRaises(ValueError):self.transform(item)

    def test_joint_categorical_cdf_bounds_and_closed_interval_mass(self):
        model=LocalLevelJointVariance(initial_variance=2,evolution_variances=(0.,.2))
        provider=model.evidence_provider();truth,values=provider.calibration_draw(4,np.random.default_rng(4))
        posterior=model.posterior(request(values));results,_=provider.calibration_fit(request(values),truth,.95)
        result=results['evolution_variance'];index=model.evolution_variances.index(truth['evolution_variance'])
        self.assertAlmostEqual(result['cdf_upper']-result['cdf_lower'],posterior.evolution_weights[index])
        expected=sum(w for value,w in zip(model.evolution_variances,posterior.evolution_weights)
                     if result['lower']<=value<=result['upper'])
        self.assertAlmostEqual(result['interval_mass'],expected)
        self.assertGreaterEqual(result['interval_mass'],.95-1e-8)
        known=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,)).evidence_provider()
        self.assertEqual(known.calibration_parameters,('observation_variance',))

    def test_final_state_atom_has_correct_posterior_jump(self):
        model=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,.2),evolution_weights=(9.,1.))
        provider=model.evidence_provider();truth,values=provider.calibration_draw(4,np.random.default_rng(5))
        self.assertEqual(truth['evolution_variance'],0)
        self.assertEqual(truth['last_level'],0)
        posterior=model.posterior(request(values));results,_=provider.calibration_fit(request(values),truth,.95)
        result=results['last_level']
        self.assertAlmostEqual(result['cdf_upper']-result['cdf_lower'],posterior.evolution_weights[0])
        transformed=self.transform(result,parameter='last_level')
        self.assertTrue(result['cdf_lower']<=transformed['cdf']<=result['cdf_upper'])


class MixedCalibrationWorkflowTests(unittest.TestCase):
    setUp=fixtures.PredictiveWorkflowTests.setUp
    register=fixtures.PredictiveWorkflowTests.register
    read_json=fixtures.PredictiveWorkflowTests.read_json

    def run_cal(self,name='calibration',**changes):
        options=dict(model=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,.2),evolution_weights=(3.,1.)),
            observation_count=4,trials=8,seed=9,output_directory=self.root/name)
        options.update(changes)
        return run_calibration(**options)

    def test_mixed_calibration_replay_prefix_and_saved_only_plot(self):
        first=self.run_cal('first');second=self.run_cal('second');larger=self.run_cal('larger',trials=10)
        for name in ('calibration.csv','summary.json','inference.json'):
            self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
        with (first/'calibration.csv').open() as stream:a=list(csv.DictReader(stream))
        with (larger/'calibration.csv').open() as stream:b=list(csv.DictReader(stream))
        self.assertEqual(a,b[:len(a)])
        summary=self.read_json(first,'summary.json')
        self.assertEqual(summary['evolution_variance']['randomized_trials'],8)
        self.assertGreater(summary['last_level']['randomized_trials'],0)
        self.assertEqual(summary['observation_variance']['randomized_trials'],0)
        self.assertNotIn('initial_level',summary)
        with patch.object(LocalLevelJointVariance,'posterior',side_effect=AssertionError('Saved records only')):
            self.assertTrue(plot_calibration(first).is_file())

    def test_numerical_failures_keep_simulation_truth_and_unconditional_warning(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.2),quadrature=QuadratureSettings(minimum_nodes=8,maximum_nodes=16,maximum_expansions=1))
        output=self.run_cal(model=model,trials=3)
        self.assertEqual(self.read_json(output,'calibration.json')['status'],'failed')
        self.assertEqual(len(list((output/'trials').iterdir())),3)
        self.assertFalse(self.read_json(output,'summary.json')['evolution_variance']['band_applies_to_all_trials'])
        self.assertIsNotNone(self.read_json(output,'inference.json')[0]['inference'])

    def test_seeded_joint_calibration_smoke_and_conservative_discrete_coverage(self):
        output=self.run_cal(model=LocalLevelJointVariance(initial_variance=1,evolution_variances=(0.,.2)),trials=120,observation_count=8,seed=21)
        self.assertEqual(self.read_json(output,'calibration.json')['status'],'completed')
        summaries=self.read_json(output,'summary.json')
        for summary in summaries.values():
            self.assertLess(abs(summary['cdf_mean']-.5),.11)
            self.assertLess(summary['cdf_uniform_distance'],.18)
        evolution=summaries['evolution_variance']
        self.assertGreaterEqual(evolution['mean_interval_posterior_mass'],.95-1e-8)
        self.assertGreaterEqual(evolution['coverage']['fraction'],.95)

    def test_joint_calibration_cli(self):
        config=self.root/'config.json';model=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,.2))
        config.write_text(json.dumps({'reference':model.configuration,'scenarios':{}}))
        result=subprocess.run([sys.executable,'scripts/calibrate_model.py','--model','local-level-joint-variance',
            '--config',str(config),'--trials','3','--observation-count','4','--output-directory',str(self.root/'cli')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.root/'cli/calibration.png').is_file())
