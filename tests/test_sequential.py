"""Mixture-CDF sequential criticism, time boundaries and frozen prediction evidence."""
import csv
from dataclasses import replace
from datetime import timedelta
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from scipy.stats import t
from model_pipelining import sequential as seq
from model_pipelining.evidence_workflows import run_experiment_evidence
from model_pipelining.experiments import EvidencePlan, EvaluationPlan
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings
from model_pipelining.models import ForecastRequest
import test_predictive_checks as fixtures


class SequentialTests(unittest.TestCase):
    setUp = fixtures.PredictiveWorkflowTests.setUp
    register = fixtures.PredictiveWorkflowTests.register
    read_json = fixtures.PredictiveWorkflowTests.read_json

    def run_checks(self, name='run', **changes):
        values = dict(dataset='example', model=LocalLevelVariance(), snapshot_id=self.snapshot.snapshot_id,
                      store=self.store, experiment=self.spec, issued_at=self.history[-1].timestamp,
                      output_directory=self.root/name)
        values.update(changes)
        return seq.run_sequential_checks(**values)

    def rows(self, output):
        with (output/'sequential.csv').open() as stream:
            return list(csv.DictReader(stream))

    def test_pit_matches_independent_conditional_mixture_and_standardization(self):
        output = self.run_checks()
        row = self.rows(output)[0]
        model = LocalLevelVariance()
        request = ForecastRequest(self.history[:2], self.history[1].timestamp, self.history[2].timestamp)
        posterior = model.posterior(request)
        law = posterior.predictive(1)
        from scipy.special import ndtr
        expected = sum(w*ndtr((5-m)/(v**.5)) for w,m,v in zip(law.weights,law.means,law.variances))
        self.assertAlmostEqual(float(row['pit']), expected, places=12)
        self.assertAlmostEqual(float(row['standardized_innovation']), (5-law.mean)/law.variance**.5)
        self.assertEqual(len(self.rows(output)), 3)

    def test_student_t_provider_uses_t_cdf(self):
        output = self.run_checks(model=self.model)
        row = self.rows(output)[0]
        posterior = self.model.posterior(ForecastRequest(self.history[:2],self.history[1].timestamp,self.history[2].timestamp))
        self.assertAlmostEqual(float(row['pit']), t.cdf(5, df=posterior.degrees_of_freedom,
            loc=posterior.mean, scale=posterior.predictive_scale_squared**.5))

    def test_all_predictions_are_written_before_actual_selection(self):
        original = seq.select_actuals
        def select(versions, task):
            path = self.root/'run/predictive_distributions.json'
            self.assertEqual(len(json.loads(path.read_text())), 3)
            self.assertNotIn('actual', json.loads(path.read_text())[0])
            return original(versions, task)
        with patch.object(seq,'select_actuals',side_effect=select):
            self.run_checks()

    def test_future_and_final_values_do_not_change_development_results(self):
        boundary = self.history[4].timestamp
        spec = replace(self.spec, evaluation=EvaluationPlan(final_start=boundary,final_end=boundary+timedelta(days=2)))
        first = self.run_checks('first', experiment=spec, issued_at=boundary-timedelta(seconds=1))
        changed = self.register((1,3,5,7,900))
        second = self.run_checks('second', experiment=spec, issued_at=boundary-timedelta(seconds=1), snapshot_id=changed.snapshot_id)
        for filename in ('predictive_distributions.json','sequential.csv','summary.json','checks.json'):
            self.assertEqual((first/filename).read_bytes(),(second/filename).read_bytes())
        with self.assertRaises(ValueError):
            self.run_checks('invalid',experiment=spec,issued_at=boundary)

    def test_current_target_does_not_enter_its_own_prediction(self):
        first = self.run_checks('first')
        changed = self.register((1,3,500,7,9))
        second = self.run_checks('second',snapshot_id=changed.snapshot_id)
        a=self.read_json(first,'predictive_distributions.json')
        b=self.read_json(second,'predictive_distributions.json')
        self.assertEqual(a[0],b[0])
        self.assertNotEqual(self.rows(first)[0]['pit'],self.rows(second)[0]['pit'])

    def test_failed_precision_is_retained_and_plottable(self):
        model=LocalLevelVariance(quadrature=QuadratureSettings(minimum_nodes=8,maximum_nodes=16,maximum_expansions=1))
        output=self.run_checks(model=model)
        self.assertEqual(self.read_json(output,'sequential_checks.json')['status'],'failed')
        self.assertIsNotNone(self.read_json(output,'predictive_distributions.json')[0]['inference_failure'])
        with patch.object(LocalLevelVariance,'posterior',side_effect=AssertionError('No refit')):
            self.assertTrue(seq.plot_sequential_checks(output)[0].is_file())

    def test_replay_and_saved_only_plot(self):
        first=self.run_checks('first')
        second=self.run_checks('second')
        self.assertEqual((first/'sequential.csv').read_bytes(),(second/'sequential.csv').read_bytes())
        with patch.object(LocalLevelVariance,'posterior',side_effect=AssertionError('No refit')):
            seq.plot_sequential_checks(first)
        with self.assertRaises(FileExistsError):
            seq.plot_sequential_checks(first)

    def test_experiment_plan_can_require_sequential_checks(self):
        spec=replace(self.spec,evidence=EvidencePlan(required_workflows=('sequential_checks',)))
        output=run_experiment_evidence(dataset='example',model=LocalLevelVariance(),snapshot_id=self.snapshot.snapshot_id,
            store=self.store,experiment=spec,issued_at=self.history[-1].timestamp,output_directory=self.root/'bundle')
        self.assertEqual(self.read_json(output,'evidence.json')['status'],'completed')

    def test_cli_runs_both_existing_models(self):
        spec=self.root/'spec.json';spec.write_text(json.dumps(self.spec.to_dict()))
        for kind in ('local-level-variance','normal-mean-variance'):
            output=self.root/kind
            result=subprocess.run([sys.executable,'scripts/check_sequential.py','--model',kind,'--dataset','example',
                '--snapshot-id',self.snapshot.snapshot_id,'--store',str(self.store),'--as-of',self.history[-1].timestamp.isoformat(),
                '--experiment',str(spec),'--config',f'configs/{kind.replace("-","_")}_priors.json',
                '--output-directory',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((output/'sequential.png').is_file())
