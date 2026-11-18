"""Independent CDF-integral losses and frozen, matched-cohort score workflows."""

import csv
from dataclasses import replace
from datetime import timedelta
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t
from scipy.special import betaln

from model_pipelining import backtest
from model_pipelining.distribution_scoring import score_distribution, validate_distribution, validate_distribution_forecast
from model_pipelining.experiment_review import build_experiment_review
from model_pipelining.experiments import EvaluationPlan, ExperimentSpec, ForecastTask
from model_pipelining.local_level import LocalLevel
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.mixtures import NormalMixture
from model_pipelining.models import Forecast, Quantile, TrainingMean
from model_pipelining.normal_mean import NormalMean
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.reporting import compare_runs
from test_normal_mean import FitPipelineFixtures, request


def student(df=5, location=0, scale=1):
    return dict(family='student_t',df=df,location=location,scale=scale)


def mixture(weights=(1.,), means=(0.,), variances=(1.,)):
    return dict(family='gaussian_mixture',weights=list(weights),means=list(means),variances=list(variances))


def integral(cdf, sf, y):
    # Independent definition, not the expectation/closed forms in implementation.
    return quad(lambda x:cdf(x)**2,-math.inf,y,epsabs=1e-9)[0]+quad(lambda x:sf(x)**2,y,math.inf,epsabs=1e-9)[0]


class DistributionLossTests(unittest.TestCase):
    def test_student_loss_matches_independent_cdf_integral_and_density(self):
        for df in (1.2,2,5,40):
            for y in (-3,0,2):
                with self.subTest(df=df,y=y):
                    law=t(df=df,loc=1,scale=2)
                    result=score_distribution(student(df,1,2),y)
                    self.assertAlmostEqual(result['crps'],integral(law.cdf,law.sf,y),places=7)
                    self.assertAlmostEqual(result['log_score'],-law.logpdf(y),places=10)

    def test_mixture_loss_matches_independent_cdf_integral(self):
        record=mixture((.7,.3),(-2,4),(.16,2))
        cdf=lambda x:.7*norm.cdf(x,-2,.4)+.3*norm.cdf(x,4,math.sqrt(2))
        sf=lambda x:.7*norm.sf(x,-2,.4)+.3*norm.sf(x,4,math.sqrt(2))
        for y in (-4,0,3,8):
            result=score_distribution(record,y)
            self.assertAlmostEqual(result['crps'],integral(cdf,sf,y),places=8)
            self.assertAlmostEqual(result['log_score'],-math.log(.7*norm.pdf(y,-2,.4)+.3*norm.pdf(y,4,math.sqrt(2))),places=10)

    def test_normal_reference_and_student_normal_limit(self):
        gaussian=score_distribution(mixture(),0)
        self.assertAlmostEqual(gaussian['crps'],(math.sqrt(2)-1)/math.sqrt(math.pi))
        self.assertAlmostEqual(gaussian['log_score'],.5*math.log(2*math.pi))
        for key,value in gaussian.items():
            self.assertAlmostEqual(score_distribution(student(1e6),0)[key],value,delta=1e-6)

    def test_mixture_is_not_scored_as_a_moment_matched_gaussian(self):
        result=score_distribution(mixture((.5,.5),(-5,5),(.1,.1)),0)
        matched=score_distribution(mixture(variances=(25.1,)),0)
        self.assertGreater(result['log_score'],matched['log_score']+100)
        self.assertGreater(abs(result['crps']-matched['crps']),.5)

    def test_units_affine_transformation_and_negative_log_loss(self):
        record=student(4,2,.03);first=score_distribution(record,2)
        self.assertLess(first['log_score'],0)  # Densities can exceed one.
        second=score_distribution(student(4,12,.3),12)
        self.assertAlmostEqual(second['crps'],10*first['crps'])
        self.assertAlmostEqual(second['log_score'],first['log_score']+math.log(10))

    def test_tail_log_density_does_not_underflow_or_apply_a_floor(self):
        self.assertAlmostEqual(score_distribution(mixture(),40)['log_score'],800+.5*math.log(2*math.pi))
        result=score_distribution(student(5,0,1e-100),1e100)
        expected=math.log(1e-100)+.5*math.log(5)+float(betaln(.5,2.5))+3*(400*math.log(10)-math.log(5))
        self.assertAlmostEqual(result['log_score'],expected,places=10)
        self.assertAlmostEqual(result['crps']/1e100,1)

    def test_undefined_first_moment_invalid_laws_and_actuals_fail(self):
        invalid=[student(1),student(.5),student(scale=0),student(df=True),student(location=math.nan),
                 mixture(variances=(0,)),mixture(weights=(0,)),mixture(weights=(True,)),{},None]
        for record in invalid:
            with self.subTest(record=record),self.assertRaises((ValueError,KeyError)):
                score_distribution(record,0)
        for y in (math.inf,math.nan,True):
            with self.assertRaises(ValueError):score_distribution(student(),y)
        with self.assertRaises(ValueError):score_distribution(mixture(variances=(1e-300,)),1e308)
        with self.assertRaisesRegex(ValueError,'precision'):score_distribution(student(1+1e-10),0)
        with self.assertRaisesRegex(ValueError,'unrepresentable'):score_distribution(mixture((1e-300,1e300),(0,0),(1,1)),0)

    def test_component_blocks_normalization_permutation_and_zero_weight(self):
        record=mixture((0,1,2),(100,0,0),(10,1,1))
        self.assertAlmostEqual(score_distribution(record,.3)['crps'],score_distribution(mixture(),.3)['crps'])
        many=mixture(tuple(np.ones(260)),tuple(np.zeros(260)),tuple(np.ones(260)))
        self.assertAlmostEqual(score_distribution(many,1)['crps'],score_distribution(mixture(),1)['crps'])
        self.assertEqual(score_distribution(record,1),score_distribution(mixture((2,1,0),(0,0,100),(1,1,10)),1))

    def test_serialized_law_is_independent_and_summary_mismatch_fails(self):
        record=mixture();saved=validate_distribution(record);record['means'][0]=9
        self.assertEqual(saved['means'],[0.])
        with self.assertRaisesRegex(ValueError,'mean differs'):
            validate_distribution_forecast(saved,Forecast(1))
        with self.assertRaisesRegex(ValueError,'mean differs'):
            validate_distribution_forecast(student(scale=1e-30),Forecast(1e-20))
        with self.assertRaisesRegex(ValueError,'quantiles'):
            validate_distribution_forecast(saved,Forecast(0,(Quantile(.9,0),)))


class DistributionWorkflowTests(FitPipelineFixtures,unittest.TestCase):
    def test_all_current_bayesian_models_supply_fitted_observation_laws(self):
        for i,model in enumerate((NormalMeanVariance(),NormalMean(),LocalLevel(),LocalLevelVariance(),
                                   LocalLevelJointVariance(evolution_variances=(0.,.1)))):
            with self.subTest(model=model.name):
                run=self.run_model(f'model-{i}',model=model)
                manifest=self.read_json(run,'manifest.json')
                self.assertEqual(manifest['status'],'completed')
                frozen=self.read_json(run,'predictive_distributions.json')
                evaluations=self.read_csv(run,'distribution_evaluation.csv')
                for law,row in zip(frozen,evaluations):
                    expected=score_distribution(law['distribution'],float(row['actual']))
                    self.assertAlmostEqual(float(row['crps']),expected['crps'])
                    self.assertAlmostEqual(float(row['log_score']),expected['log_score'])
                self.assertEqual(self.read_json(run,'scores.json')['distribution']['scored_count'],2)

    def test_laws_are_all_frozen_before_actual_selection_and_never_rewritten(self):
        selected=backtest.select_actuals;captured=[]
        def inspect(*args):
            path=self.root/'freeze/predictive_distributions.json'
            captured.append(path.read_bytes())
            self.assertEqual(len(json.loads(captured[-1])),2)
            self.assertFalse((self.root/'freeze/distribution_evaluation.csv').exists())
            return selected(*args)
        with patch.object(backtest,'select_actuals',side_effect=inspect):
            run=self.run_model('freeze',model=NormalMeanVariance())
        self.assertEqual((run/'predictive_distributions.json').read_bytes(),captured[0])

    def test_own_target_and_final_values_cannot_change_earlier_laws(self):
        model=NormalMeanVariance()
        first=self.run_model('first',self.register((1,3,5,7)),model)
        changed=self.run_model('changed',self.register((1,3,500,700)),model)
        self.assertEqual(self.read_json(first,'predictive_distributions.json')[0],self.read_json(changed,'predictive_distributions.json')[0])
        boundary=request((1,3,5,7)).history[3].timestamp
        spec=ExperimentSpec(task=ForecastTask(minimum_history=2),evaluation=EvaluationPlan(final_start=boundary,final_end=boundary+timedelta(days=1)))
        a=self.run_model('dev-a',self.register((1,3,5,7)),model,spec)
        b=self.run_model('dev-b',self.register((1,3,5,700)),model,spec)
        for name in ('predictive_distributions.json','distribution_evaluation.csv','scores.json'):
            self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())

    def test_replay_is_identical_and_scoring_does_not_need_a_model_or_input_store(self):
        snapshot=self.register();a=self.run_model('a',snapshot,NormalMeanVariance());b=self.run_model('b',snapshot,NormalMeanVariance())
        for name in ('predictive_distributions.json','distribution_evaluation.csv','scores.json'):
            self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
        with patch.object(NormalMeanVariance,'fit',side_effect=AssertionError('No refit')):
            saved=self.read_json(a,'predictive_distributions.json')[0]
            self.assertTrue(math.isfinite(score_distribution(saved['distribution'],5)['crps']))
            report=build_experiment_review(forecast_directory=a,output_directory=self.root/'review')
            self.assertEqual(self.read_json(report,'report.json')['status'],'ready_for_review')

    def test_missing_declared_distribution_fails_with_every_origin_retained(self):
        class Missing(TrainingMean):
            supports_distribution=True
        run=self.run_model('missing',model=Missing())
        self.assertEqual(self.read_json(run,'manifest.json')['status'],'failed')
        self.assertEqual(len(self.read_json(run,'predictive_distributions.json')),2)
        self.assertEqual(self.read_json(run,'scores.json')['distribution']['failed_count'],2)

    def test_point_only_and_legacy_runs_have_no_invented_distribution(self):
        snapshot=self.register();a=self.run_model('point',snapshot,TrainingMean());b=self.run_model('legacy',snapshot,TrainingMean())
        self.assertEqual(self.read_json(a,'scores.json')['distribution']['status'],'not_applicable')
        self.assertFalse((a/'predictive_distributions.json').exists())
        m=self.read_json(b,'manifest.json');m['model'].pop('supports_distribution');(b/'manifest.json').write_text(json.dumps(m))
        out=compare_runs(a,b,output_directory=self.root/'comparison')
        self.assertEqual(self.read_json(out,'comparison.json')['distribution_comparison']['status'],'not_applicable')

    def test_scoring_failure_is_visible_and_does_not_destroy_point_predictions(self):
        class Extreme(TrainingMean):
            supports_distribution=True
            def forecast(self,request):
                return Forecast(0,distribution=mixture(variances=(1e-320,)))
        run=self.run_model('failed',self.register((1,3,5,0)),Extreme())
        self.assertEqual(self.read_json(run,'manifest.json')['status'],'failed')
        self.assertTrue(self.read_json(run,'scores.json')['complete'])
        self.assertEqual(self.read_json(run,'scores.json')['distribution']['scored_count'],1)
        self.assertTrue(all(r['status']=='passed' for r in self.read_json(run,'predictive_distributions.json')))

    def test_matched_saved_comparison_preserves_sources_and_saves_signed_losses_and_chart(self):
        snapshot=self.register();a=self.run_model('first',snapshot,NormalMeanVariance());b=self.run_model('second',snapshot,NormalMean())
        before={p:p.read_bytes() for r in (a,b) for p in r.rglob('*') if p.is_file()}
        with patch.object(NormalMeanVariance,'fit',side_effect=AssertionError('No refit')):
            out=compare_runs(a,b,output_directory=self.root/'comparison')
        result=self.read_json(out,'comparison.json')['distribution_comparison']
        self.assertEqual(result['status'],'completed');self.assertEqual(len(result['case_differences']),2)
        self.assertAlmostEqual(result['mean_crps_first_minus_second'],result['runs'][0]['mean_crps']-result['runs'][1]['mean_crps'])
        self.assertTrue((out/'distribution_scores.png').read_bytes().startswith(b'\x89PNG'))
        self.assertTrue(all(p.read_bytes()==content for p,content in before.items()))

    def test_incomplete_or_inconsistent_saved_score_cohorts_cannot_be_compared(self):
        snapshot=self.register();a=self.run_model('first',snapshot,NormalMeanVariance());b=self.run_model('second',snapshot,NormalMean())
        path=b/'distribution_evaluation.csv';original=path.read_text();path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'incomplete'):compare_runs(a,b,output_directory=self.root/'bad')
        path.write_text(original.replace(',passed,',',failed,',1))
        with self.assertRaises(ValueError):compare_runs(a,b,output_directory=self.root/'bad')
        path.write_text(original)
        value=self.read_json(b,'scores.json');value['distribution']['mean_crps']+=1;(b/'scores.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'aggregate'):compare_runs(a,b,output_directory=self.root/'bad')

    def test_changed_methods_or_histories_do_not_enter_a_comparison(self):
        snapshot=self.register();a=self.run_model('first',snapshot,NormalMeanVariance());b=self.run_model('second',snapshot,NormalMean())
        value=self.read_json(b,'manifest.json');value['evaluation']['distribution_scoring']['version']=999;(b/'manifest.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'method'):compare_runs(a,b,output_directory=self.root/'bad')
        c=self.run_model('different',self.register((1,3,5,9)),NormalMean())
        with self.assertRaisesRegex(ValueError,'must match'):compare_runs(a,c,output_directory=self.root/'bad')

    def test_release_vintage_changes_scores_without_changing_frozen_predictions(self):
        import test_experiments as fixtures
        # Borrow fixture methods without inheriting (and rerunning) its tests.
        other=fixtures.ExperimentTests();other.setUp();self.addCleanup(other.doCleanups)
        snapshot,spec=other.vintage_fixture()
        a=other.run_experiment(snapshot,'first',experiment=spec,model=NormalMeanVariance())
        b=other.run_experiment(snapshot,'latest',experiment=replace(spec,task=replace(spec.task,actual_policy='latest_in_snapshot')),model=NormalMeanVariance())
        self.assertEqual((a/'predictive_distributions.json').read_bytes(),(b/'predictive_distributions.json').read_bytes())
        first=self.read_csv(a,'distribution_evaluation.csv')[0];latest=self.read_csv(b,'distribution_evaluation.csv')[0]
        self.assertEqual(float(first['actual']),3);self.assertEqual(float(latest['actual']),303)
        self.assertNotEqual(first['log_score'],latest['log_score'])
        with self.assertRaises(ValueError):compare_runs(a,b,output_directory=self.root/'bad')

    def test_tampered_distribution_actuals_and_histories_fail_saved_review(self):
        a=self.run_model('run',model=NormalMeanVariance())
        path=a/'distribution_evaluation.csv';rows=self.read_csv(a,'distribution_evaluation.csv')
        for key,value in (('actual','999'),('history_sha256','wrong')):
            changed=[dict(row) for row in rows];changed[0][key]=value
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(changed[0]));writer.writeheader();writer.writerows(changed)
            report=build_experiment_review(forecast_directory=a,output_directory=self.root/f'review-{key}')
            self.assertEqual(self.read_json(report,'report.json')['status'],'failed')
