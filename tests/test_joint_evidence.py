"""Joint parameter replication, categorical prior influence and atomic states."""
from dataclasses import replace
from datetime import timedelta
import csv,json,math,subprocess,sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import norm
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import QuadratureSettings
from model_pipelining.state_mixtures import StateMixture
from model_pipelining import predictive_checks as pc,prior_influence as pi
from model_pipelining.evidence_joint_level import JointLevelEvidence
from model_pipelining.experiments import EvaluationPlan,EvidencePlan
from test_normal_mean import request
import test_predictive_checks as fixtures


class JointSimulationTests(unittest.TestCase):
    def test_state_mixture_preserves_atoms_and_generalized_quantiles(self):
        law=StateMixture((.6,.4),(0.,0.),(0.,1.))
        self.assertAlmostEqual(law.cdf(0,inclusive=False),.2)
        self.assertAlmostEqual(law.cdf(0),.8)
        self.assertEqual(law.ppf(.5),0)
        self.assertAlmostEqual(law.ppf(.1),norm.ppf(.25),places=8)
        atoms=StateMixture((.3,.7),(0.,2.),(0.,0.))
        self.assertEqual(atoms.ppf(.3),0)
        self.assertEqual(atoms.ppf(.3001),2)
        with self.assertRaises(ValueError):StateMixture((1.,),(1e200,),(1.,)).ppf(.1)

    def test_joint_prior_frequency_covariance_and_data_free_generation(self):
        model=LocalLevelJointVariance(initial_mean=2,initial_variance=2,evolution_variances=(0.,.4),evolution_weights=(3.,1.))
        with patch.object(LocalLevelJointVariance,'posterior',side_effect=AssertionError('Prior only')):
            result=pc.simulate_prior_predictive(model,observation_count=3,replications=8000,seed=51)
        values=np.asarray([row.values for row in result.datasets]);w=np.asarray([row.evolution_variance for row in result.datasets])
        self.assertAlmostEqual((w==.4).mean(),.25,delta=.025)
        self.assertAlmostEqual(np.cov(values[:,0],values[:,2])[0,1],2.1,delta=.13)
        self.assertAlmostEqual(values[:,0].var(),3.1,delta=.17)
        self.assertFalse(result.failures)

    def test_joint_posterior_preserves_variance_dependence_and_fresh_innovations(self):
        model=LocalLevelJointVariance(initial_variance=1,evolution_variances=(0.,10.))
        req=request((-3.,3.,-3.,3.));posterior=model.posterior(req)
        result=pc.simulate_posterior_predictive(model,req,replications=12000,seed=29)
        w=np.array([row.evolution_variance for row in result.datasets]);v=np.array([row.observation_variance for row in result.datasets])
        self.assertAlmostEqual((w==10).mean(),posterior.evolution_weights[1],delta=.02)
        products=(w-posterior.evolution_variance_mean)*(v-posterior.observation_variance_mean)
        self.assertAlmostEqual(products.mean(),posterior.variance_covariance,delta=6*products.std()/len(w)**.5)
        initial=np.array([row.initial_level for row in result.datasets])
        levels=np.array([row.latent_levels for row in result.datasets])
        increments=np.diff(np.column_stack((initial,levels)),axis=1)
        self.assertTrue(np.all(increments[w==0]==0))
        self.assertAlmostEqual(increments[w==10].mean(),0,delta=.12)
        self.assertAlmostEqual(increments[w==10].var(),10,delta=.5)

    def test_declared_evolution_parameter_cannot_be_omitted(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1))
        result=pc.simulate_prior_predictive(model,observation_count=3,replications=2)
        bad=replace(result,datasets=tuple(replace(row,evolution_variance=None) for row in result.datasets))
        with patch.object(JointLevelEvidence,'simulate',return_value=bad),self.assertRaises(ValueError):
            pc.simulate_prior_predictive(model,observation_count=3,replications=2)


class JointEvidenceWorkflowTests(unittest.TestCase):
    setUp=fixtures.PredictiveWorkflowTests.setUp
    register=fixtures.PredictiveWorkflowTests.register
    read_json=fixtures.PredictiveWorkflowTests.read_json

    def prior_run(self,name='priors',**changes):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1))
        options=dict(dataset='example',reference=model,scenarios={'stable':replace(model,evolution_weights=(8.,1.))},
            snapshot_id=self.snapshot.snapshot_id,store=self.store,experiment=self.spec,
            issued_at=self.history[-1].timestamp,output_directory=self.root/name)
        options.update(changes)
        return pi.run_prior_influence(**options)

    def test_prior_paths_save_discrete_probabilities_covariance_and_differences(self):
        output=self.prior_run()
        with (output/'learning_paths.csv').open() as stream:rows=list(csv.DictReader(stream))
        a=next(row for row in rows if row['configuration']=='reference' and row['stage']=='prior')
        b=next(row for row in rows if row['configuration']=='stable' and row['stage']=='prior')
        self.assertAlmostEqual(json.loads(a['evolution_probabilities_json'])[0][1],.5,places=6)
        self.assertAlmostEqual(json.loads(b['evolution_probabilities_json'])[0][1],8/9,places=6)
        self.assertTrue(all(row['degrees_of_freedom']=='' and row['prior_mean_weight']=='' for row in rows))
        self.assertTrue(all(row['variance_covariance']!='' for row in rows))
        with patch.object(LocalLevelJointVariance,'posterior',side_effect=AssertionError('Saved only')):
            self.assertEqual(len(pi.plot_prior_influence(output)),3)

    def test_known_state_and_zero_evolution_mass_do_not_invent_state_spread(self):
        model=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,.1))
        output=self.prior_run(reference=model,scenarios={'stable':replace(model,evolution_weights=(8.,1.))})
        self.assertEqual(self.read_json(output,'prior_influence.json')['status'],'completed')
        with (output/'learning_paths.csv').open() as stream:rows=list(csv.DictReader(stream))
        prior=next(row for row in rows if row['stage']=='prior')
        self.assertEqual(float(prior['mean_lower']),0)
        self.assertEqual(float(prior['mean_upper']),0)

    def test_changed_support_is_rejected_before_output(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1))
        with self.assertRaises(ValueError):self.prior_run(scenarios={'different':replace(model,evolution_variances=(0.,1.))})
        self.assertFalse((self.root/'priors').exists())

    def test_future_final_isolation_replay_and_saved_predictive_plot(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1))
        boundary=self.history[4].timestamp
        spec=replace(self.spec,evaluation=EvaluationPlan(final_start=boundary,final_end=boundary+timedelta(days=1)))
        options=dict(dataset='example',model=model,store=self.store,experiment=spec,
                     issued_at=boundary-timedelta(seconds=1),replications=12)
        first=pc.run_predictive_checks(**options,snapshot_id=self.snapshot.snapshot_id,output_directory=self.root/'first')
        changed=self.register((1,3,5,7,900))
        second=pc.run_predictive_checks(**options,snapshot_id=changed.snapshot_id,output_directory=self.root/'second')
        for path in first.rglob('*'):
            if path.is_file() and path.name!='predictive_checks.json':
                self.assertEqual(path.read_bytes(),(second/path.relative_to(first)).read_bytes())
        with patch.object(LocalLevelJointVariance,'posterior',side_effect=AssertionError('No fit')):
            self.assertEqual(len(pc.plot_predictive_checks(first)),2)
        with (first/'posterior/parameter_draws.csv').open() as stream:
            self.assertTrue(all(row['evolution_variance']!='' for row in csv.DictReader(stream)))

    def test_numerical_failure_keeps_prior_and_component_failure_evidence(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1),quadrature=QuadratureSettings(minimum_nodes=8,maximum_nodes=16,maximum_expansions=1))
        output=pc.run_predictive_checks(dataset='example',model=model,snapshot_id=self.snapshot.snapshot_id,
            store=self.store,experiment=self.spec,issued_at=self.history[-1].timestamp,replications=3,output_directory=self.root/'failed')
        self.assertEqual(self.read_json(output/'prior','summary.json')['status'],'completed')
        self.assertIsNotNone(self.read_json(output/'posterior','summary.json')['inference_failure'])

    def test_joint_evidence_cli_accepts_weight_overrides_and_all_three_workflows(self):
        model=LocalLevelJointVariance(evolution_variances=(0.,.1))
        config=self.root/'joint.json';config.write_text(json.dumps({'reference':model.configuration,'scenarios':{'stable':{'evolution_weights':[8,1]}}}))
        spec=self.root/'experiment.json';spec.write_text(json.dumps(replace(self.spec,evidence=EvidencePlan(
            required_workflows=('predictive_checks','prior_influence','sequential_checks'),replications=5)).to_dict()))
        output=self.root/'cli'
        result=subprocess.run([sys.executable,'scripts/run_evidence.py','--model','local-level-joint-variance','--dataset','example',
            '--snapshot-id',self.snapshot.snapshot_id,'--store',str(self.store),'--as-of',self.history[-1].timestamp.isoformat(),
            '--experiment',str(spec),'--config',str(config),'--output-directory',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.read_json(output,'evidence.json')['status'],'completed')
        self.assertTrue((output/'prior_influence/evolution_priors.png').is_file())
