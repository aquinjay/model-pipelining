"""Independent marginal likelihood, joint mixtures, limits and evidence replay."""
import json,math,subprocess,sys
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln,ndtr
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance,plot_joint_variances
from model_pipelining.local_level_variance import LocalLevelVariance,QuadratureSettings,NumericalIntegrationError
from model_pipelining.backtest import run_backtest
from model_pipelining.dynamics_study import run_dynamics_study,plot_dynamics_study
from model_pipelining.sequential import run_sequential_checks
from test_normal_mean import request
import test_predictive_checks as fixtures


class JointNumericalTests(unittest.TestCase):
    def test_symmetric_mixture_median_handles_cdf_roundoff(self):
        from model_pipelining.mixtures import NormalMixture
        law=NormalMixture((.1,.2,.7),(1e-18,0.,-1e-18),(1.,2.,3.))
        self.assertAlmostEqual(law.cdf(law.ppf(.5)),.5,places=12)

    def test_single_support_matches_scalar_model_forecast_and_hpd(self):
        scalar=LocalLevelVariance(evolution_variance=.2)
        joint=LocalLevelJointVariance(evolution_variances=(.2,))
        req=replace(request((1,3,2)),quantile_levels=(.1,.9),highest_density_levels=(.8,))
        self.assertEqual(scalar.forecast(req),joint.forecast(req))
        self.assertEqual(joint.posterior(req).evolution_weights,(1.,))

    def test_weights_predictive_cdf_and_confounding_match_independent_batch_integration(self):
        model=LocalLevelJointVariance(initial_variance=2,evolution_variances=(0.,.3),evolution_weights=(2.,1.))
        y=np.array([-1.,0.,1.]);times=np.arange(1,4);n=len(y)
        evidences=[];first_moments=[];cdfs=[];variance_cdfs=[]
        for w in model.evolution_variances:
            def density(z,mode):
                v=math.exp(z);cov=2+w*np.minimum.outer(times,times)+v*np.eye(n)
                log_likelihood=-.5*(n*math.log(2*math.pi)+np.linalg.slogdet(cov)[1]+y@np.linalg.solve(cov,y))
                log_prior=3*math.log(2)-gammaln(3)-3*z-2/v
                value=math.exp(log_prior+log_likelihood)
                if mode=='v':value*=v
                if mode=='cdf':
                    cross=2+w*times;mean=cross@np.linalg.solve(cov,y)
                    variance=2+w*4+v-cross@np.linalg.solve(cov,cross)
                    value*=ndtr(-mean/math.sqrt(variance))
                return value
            evidences.append(quad(lambda z:density(z,'normal'),-25,25,epsabs=1e-12)[0])
            first_moments.append(quad(lambda z:density(z,'v'),-25,25,epsabs=1e-12)[0])
            cdfs.append(quad(lambda z:density(z,'cdf'),-25,25,epsabs=1e-12)[0])
            variance_cdfs.append(quad(lambda z:density(z,'normal'),-25,math.log(1.5),epsabs=1e-12)[0])
        weights=np.asarray(model.evolution_weights)*evidences;weights/=weights.sum()
        posterior=model.posterior(request(tuple(y)))
        np.testing.assert_allclose(posterior.evolution_weights,weights,atol=1e-7)
        normalizer=np.asarray(model.evolution_weights)@evidences
        expected_cdf=np.asarray(model.evolution_weights)@cdfs/normalizer
        self.assertAlmostEqual(posterior.predictive().cdf(0),expected_cdf,places=7)
        self.assertAlmostEqual(posterior.cdf(1.5),np.asarray(model.evolution_weights)@variance_cdfs/normalizer,places=7)
        expected_v=np.asarray(model.evolution_weights)@first_moments/normalizer
        expected_vw=np.asarray(model.evolution_weights)*model.evolution_variances@np.asarray(first_moments)/normalizer
        expected_w=weights@model.evolution_variances
        self.assertAlmostEqual(posterior.variance_covariance,expected_vw-expected_v*expected_w,places=7)

    def test_prior_keeps_explicit_discrete_mass_and_independent_variances(self):
        model=LocalLevelJointVariance(evolution_weights=(4.,1.,2.,1.))
        posterior=model.prior()
        np.testing.assert_allclose(posterior.evolution_weights,model.evolution_weights,atol=1e-7)
        self.assertAlmostEqual(posterior.variance_covariance,0,places=7)
        self.assertEqual(posterior.evolution_ppf(.1),0)
        self.assertAlmostEqual(posterior.evolution_cdf(0),.5,places=7)
        self.assertEqual(posterior.evolution_cdf(0,inclusive=False),0)

    def test_observation_variance_cdf_quantile_and_total_predictive_variance(self):
        posterior=LocalLevelJointVariance().posterior(request((1,3,2)))
        for p in (.1,.5,.9):self.assertAlmostEqual(posterior.cdf(posterior.ppf(p)),p,places=6)
        means=[p.predictive().mean for p in posterior.components]
        expected=sum(w*(p.predictive().variance+(m-posterior.predictive().mean)**2)
                     for w,p,m in zip(posterior.evolution_weights,posterior.components,means))
        self.assertAlmostEqual(posterior.predictive().variance,expected)

    def test_precision_failure_is_required_and_keeps_all_component_evidence(self):
        model=LocalLevelJointVariance(quadrature=QuadratureSettings(minimum_nodes=8,maximum_nodes=16,maximum_expansions=1))
        result=model.fit(request((1,3,2)))
        self.assertEqual(result.evidence.checks[0]['status'],'failed')
        self.assertEqual(len(result.evidence.metadata['quadrature']['components']),4)
        with self.assertRaises(NumericalIntegrationError):model.posterior(request((1,3,2)))

    def test_configuration_validation_and_roundtrip(self):
        model=LocalLevelJointVariance()
        self.assertEqual(LocalLevelJointVariance.from_configuration(json.loads(json.dumps(model.configuration))),model)
        for values in ({'evolution_variances':()}, {'evolution_variances':(.1,0.)},
                       {'evolution_variances':(0.,0.)},{'evolution_weights':()},
                       {'evolution_weights':(1.,0.,1.,1.)},{'prior_shape':-1}):
            with self.assertRaises(ValueError):LocalLevelJointVariance(**values)

    def test_known_state_zero_evolution_component_still_has_continuous_observations(self):
        model=LocalLevelJointVariance(initial_variance=0,evolution_variances=(0.,.1))
        law=model.posterior(request((1,3,2))).predictive()
        self.assertTrue(all(v>0 for v in law.variances))
        self.assertTrue(0<law.cdf(2)<1)


class JointWorkflowTests(unittest.TestCase):
    setUp=fixtures.PredictiveWorkflowTests.setUp
    register=fixtures.PredictiveWorkflowTests.register
    read_json=fixtures.PredictiveWorkflowTests.read_json

    def run_model(self,name,**changes):
        values=dict(dataset='example',model=LocalLevelJointVariance(evolution_variances=(0.,.1)),
            snapshot_id=self.snapshot.snapshot_id,store=self.store,experiment=self.spec,
            interval_levels=(.8,),output_directory=self.root/name)
        values.update(changes)
        return run_backtest(**values)

    def test_replay_saved_variance_chart_and_node_identities(self):
        first=self.run_model('first');second=self.run_model('second')
        for filename in ('predictions.csv','scores.json','fit_index.csv'):
            self.assertEqual((first/filename).read_bytes(),(second/filename).read_bytes())
        for path in (first/'fits').glob('*/fit.json'):
            self.assertEqual(path.read_bytes(),(second/path.relative_to(first)).read_bytes())
            data=json.loads(path.read_text())['metadata']
            self.assertTrue(all('evolution_variance' in node for node in data['quadrature_nodes']))
        with patch.object(LocalLevelJointVariance,'posterior',side_effect=AssertionError('No fit')):
            self.assertTrue(plot_joint_variances(first).is_file())

    def test_future_target_exclusion_and_sequential_provider(self):
        first=self.run_model('first')
        changed=self.register((1,3,500,7,9))
        second=self.run_model('second',snapshot_id=changed.snapshot_id)
        a=sorted((first/'fits').glob('*/fit.json'))[0];b=second/a.relative_to(first)
        # Input identity differs; posterior numerics do not depend on the target.
        self.assertEqual(json.loads(a.read_text())['metadata'],json.loads(b.read_text())['metadata'])
        output=run_sequential_checks(dataset='example',model=LocalLevelJointVariance(evolution_variances=(0.,.1)),
            snapshot_id=self.snapshot.snapshot_id,store=self.store,experiment=self.spec,
            issued_at=self.history[-1].timestamp,output_directory=self.root/'sequential')
        self.assertEqual(self.read_json(output,'sequential_checks.json')['status'],'completed')

    def test_joint_cli(self):
        spec=self.root/'spec.json';spec.write_text(json.dumps(self.spec.to_dict()))
        result=subprocess.run([sys.executable,'scripts/run_joint_local_level.py','--dataset','example',
            '--snapshot-id',self.snapshot.snapshot_id,'--store',str(self.store),'--experiment',str(spec),
            '--interval-levels','.8','--output-directory',str(self.root/'cli')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.root/'cli/joint_variances.png').is_file())

    def test_controlled_study_has_matched_cases_and_shared_noise(self):
        output=run_dynamics_study(output_directory=self.root/'study',observation_count=12,development_count=9,
            minimum_history=3,evolution_variances=(0.,.1))
        summary=self.read_json(output,'summary.json')
        self.assertEqual(len(summary),6)
        self.assertTrue(all(row['cases']==6 and row['status']=='completed' for row in summary))
        self.assertTrue(plot_dynamics_study(output).is_file())
        import csv
        with (output/'source/stable.csv').open() as stream:stable=[float(row['value']) for row in csv.DictReader(stream)]
        with (output/'source/gradual.csv').open() as stream:gradual=[float(row['value']) for row in csv.DictReader(stream)]
        np.testing.assert_allclose(np.asarray(gradual)-np.arange(12)*.05,stable,atol=1e-12)

    def test_controlled_study_replay_and_invalid_settings(self):
        a=run_dynamics_study(output_directory=self.root/'a',observation_count=10,development_count=8,minimum_history=3,evolution_variances=(.1,))
        b=run_dynamics_study(output_directory=self.root/'b',observation_count=10,development_count=8,minimum_history=3,evolution_variances=(.1,))
        self.assertEqual((a/'summary.json').read_bytes(),(b/'summary.json').read_bytes())
        with self.assertRaises(ValueError):run_dynamics_study(output_directory=self.root/'bad',development_count=120)
