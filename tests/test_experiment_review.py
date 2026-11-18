"""Saved-only experiment review identities, receipts, requirements and scopes."""
from dataclasses import replace
from datetime import timedelta
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import unittest
from unittest.mock import patch
from model_pipelining import experiment_review as review
from model_pipelining.backtest import run_backtest
from model_pipelining.calibration import run_calibration
from model_pipelining.evidence_workflows import run_experiment_evidence
from model_pipelining.experiments import EvidencePlan,EvaluationPlan
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.models import TrainingMean
import test_predictive_checks as fixtures


class ExperimentReviewTests(unittest.TestCase):
    setUp=fixtures.PredictiveWorkflowTests.setUp
    register=fixtures.PredictiveWorkflowTests.register
    read_json=fixtures.PredictiveWorkflowTests.read_json

    def artifacts(self,*,spec=None,model=None):
        self.spec=spec or replace(self.spec,evidence=EvidencePlan(required_workflows=('predictive_checks','sequential_checks'),replications=5))
        model=model or self.model
        forecast=run_backtest(dataset='example',model=model,snapshot_id=self.snapshot.snapshot_id,store=self.store,
            experiment=self.spec,interval_levels=(.8,),output_directory=self.root/'forecast')
        evidence=run_experiment_evidence(dataset='example',model=model,snapshot_id=self.snapshot.snapshot_id,store=self.store,
            experiment=replace(self.spec,evaluation=replace(self.spec.evaluation,role='development')),
            issued_at=self.history[-1].timestamp if self.spec.evaluation.final_start is None else self.spec.evaluation.final_start-timedelta(seconds=1),
            output_directory=self.root/'evidence')
        calibration=run_calibration(model=model,observation_count=4,trials=3,output_directory=self.root/'calibration')
        return forecast,evidence,calibration

    def report(self,forecast,evidence=None,calibration=None,name='review',**options):
        path=review.build_experiment_review(forecast_directory=forecast,evidence_directory=evidence,
            calibration_directories=(calibration,) if calibration else (),output_directory=self.root/name,**options)
        return self.read_json(path,'report.json')

    def mutate(self,path,change):
        value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))

    def test_complete_saved_report_is_read_only_and_keeps_human_review_pending(self):
        f,e,c=self.artifacts()
        before={path:path.read_bytes() for directory in (f,e,c) for path in directory.rglob('*') if path.is_file()}
        with patch.object(NormalMeanVariance,'posterior',side_effect=AssertionError('No fitting')), \
             patch('model_pipelining.experiments.retrieve_snapshot',side_effect=AssertionError('No source data')):
            result=self.report(f,e,c,require_calibration=True)
        self.assertEqual(result['status'],'ready_for_review')
        self.assertEqual(result['human_review_status'],'pending')
        self.assertTrue(all(path.read_bytes()==data for path,data in before.items()))
        self.assertTrue((self.root/'review/report.md').is_file())
        for source in result['sources']:
            for name,digest in source['artifact_sha256'].items():
                self.assertEqual(hashlib.sha256((Path(source['path'])/name).read_bytes()).hexdigest(),digest)

    def test_missing_required_workflows_and_calibration_are_incomplete(self):
        f,_,_=self.artifacts()
        result=self.report(f,require_calibration=True)
        self.assertEqual(result['status'],'incomplete')
        self.assertTrue(any(row['status']=='missing' for row in result['checks']))
        self.assertTrue(result['omitted_evidence']['calibration'])

    def test_optional_sources_are_explicitly_omitted(self):
        forecast=run_backtest(dataset='example',snapshot_id=self.snapshot.snapshot_id,store=self.store,
            experiment=self.spec,model=TrainingMean(),output_directory=self.root/'forecast')
        result=self.report(forecast)
        self.assertEqual(result['status'],'ready_for_review')
        self.assertEqual(result['omitted_evidence'],{'historical_evidence':True,'calibration':True})

    def test_receipt_mismatch_fails_instead_of_trusting_completed_manifest(self):
        f,e,c=self.artifacts()
        path=e/'predictive_checks/checks.json'
        path.write_text(path.read_text()+'\n')
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'failed')
        self.assertIn('receipt mismatch',result['sources'][1]['error'])

    def test_input_snapshot_and_model_mismatches_are_rejected(self):
        f,e,c=self.artifacts()
        self.mutate(e/'evidence.json',lambda v:v['context']['input'].update(snapshot_id='wrong'))
        self.mutate(c/'calibration.json',lambda v:v['model']['configuration'].update(prior_mean=999))
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'failed')
        self.assertIn('snapshot_id differs',result['sources'][1]['error'])
        self.assertIn('configuration differs',result['sources'][2]['error'])

    def test_selected_history_hash_and_final_cutoff_are_checked(self):
        f,e,c=self.artifacts()
        self.mutate(e/'evidence.json',lambda v:v['context']['selected_history'][0].update(value=999))
        result=self.report(f,e,c)
        self.assertIn('history hash mismatch',result['sources'][1]['error'])

    def test_evidence_cutoff_cannot_enter_final_period(self):
        boundary=self.history[4].timestamp
        spec=replace(self.spec,evidence=EvidencePlan(required_workflows=('predictive_checks',),replications=4),
            evaluation=EvaluationPlan(final_start=boundary,final_end=boundary+timedelta(days=1)))
        f,e,c=self.artifacts(spec=spec)
        self.mutate(e/'evidence.json',lambda v:v['context'].update(issued_at=boundary.isoformat()))
        result=self.report(f,e,c)
        self.assertIn('cutoff enters the final period',result['sources'][1]['error'])

    def test_nonfinite_saved_json_is_rejected(self):
        f,e,c=self.artifacts()
        self.mutate(c/'summary.json',lambda v:v.update(invalid=float('nan')))
        result=self.report(f,e,c)
        self.assertIn('Nonfinite JSON',result['sources'][2]['error'])

    def test_final_forecast_can_link_compatible_development_evidence(self):
        boundary=self.history[3].timestamp
        spec=replace(self.spec,evidence=EvidencePlan(required_workflows=('predictive_checks',),replications=4),
            evaluation=EvaluationPlan(role='final',final_start=boundary,final_end=boundary+timedelta(days=2)))
        f,e,c=self.artifacts(spec=spec)
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'ready_for_review')
        self.assertEqual(result['experiment']['evaluation']['role'],'final')
        self.assertLess(result['sources'][1]['details']['cutoff'],boundary.isoformat())

    def test_source_code_or_environment_conflicts_are_visible(self):
        f,e,c=self.artifacts()
        self.mutate(c/'calibration.json',lambda v:v['source_sha256'].update(normal_mean_variance_py='0'*64))
        # Use an existing overlapping filename, not a new implementation helper.
        value=self.read_json(c,'calibration.json');value['source_sha256']['normal_mean_variance.py']='0'*64
        value['environment']['python']='0.0';(c/'calibration.json').write_text(json.dumps(value))
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'failed')
        self.assertGreaterEqual(len(result['version_conflicts']),2)

    def test_failed_computation_remains_a_failure_with_compatible_identity(self):
        f,e,c=self.artifacts()
        self.mutate(c/'checks.json',lambda v:v[0].update(status='failed'))
        self.mutate(c/'calibration.json',lambda v:v.update(status='failed'))
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['sources'][2]['validation_status'],'passed')
        self.assertFalse(result['sources'][2]['details']['passed'])

    def test_missing_bundle_required_check_cannot_pass_through_children(self):
        f,e,c=self.artifacts()
        self.mutate(e/'checks.json',lambda v:v.pop())
        result=self.report(f,e,c)
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['sources'][1]['details']['passed'])

    def test_named_fit_requirements_are_rechecked(self):
        f,e,c=self.artifacts()
        value=self.read_json(f,'manifest.json')
        value['experiment']['evidence']['required_fit_checks']=['missing-check']
        value['experiment_sha256']=hashlib.sha256(json.dumps(value['experiment'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (f/'manifest.json').write_text(json.dumps(value))
        result=self.report(f)
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['sources'][0]['details']['passed'])

    def test_calibration_rows_cannot_omit_a_trial(self):
        f,e,c=self.artifacts()
        path=c/'calibration.csv';lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
        result=self.report(f,e,c)
        self.assertIn('omit or duplicate',result['sources'][2]['error'])

    def test_artifact_link_cannot_escape_source_tree(self):
        f,e,c=self.artifacts()
        self.mutate(e/'evidence.json',lambda v:v['workflows'][0].update(directory='../forecast'))
        result=self.report(f,e,c)
        self.assertIn('artifact link',result['sources'][1]['error'])

    def test_symlinks_nested_outputs_and_existing_outputs_are_rejected(self):
        f,e,c=self.artifacts()
        with self.assertRaises(ValueError):review.build_experiment_review(forecast_directory=f,output_directory=f/'review')
        result=self.report(f,e,c)
        with self.assertRaises(FileExistsError):self.report(f,e,c)
        (c/'link').symlink_to(f/'manifest.json')
        result=self.report(f,e,c,name='symlink')
        self.assertEqual(result['status'],'failed')
        self.assertIn('Symlink',result['sources'][2]['error'])

    def test_source_change_during_assembly_cannot_create_ready_report(self):
        f,e,c=self.artifacts();original=review._inventory;calls=0
        def changing(path):
            nonlocal calls
            calls+=1
            inventory=original(path)
            if calls==4:inventory={**inventory,'changed':'yes'}
            return inventory
        with patch.object(review,'_inventory',side_effect=changing),self.assertRaisesRegex(ValueError,'changed during'):
            self.report(f,e,c)
        self.assertFalse((self.root/'review').exists())

    def test_cli_returns_readiness_and_does_not_execute_models(self):
        f,e,c=self.artifacts()
        command=[sys.executable,'scripts/review_experiment.py','--forecast-directory',str(f),'--evidence-directory',str(e),
            '--calibration-directories',str(c),'--require-calibration','--output-directory',str(self.root/'cli')]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.root/'cli/report.md').is_file())
        result=subprocess.run([sys.executable,'scripts/review_experiment.py','--forecast-directory',str(f),
            '--output-directory',str(self.root/'missing')],capture_output=True,text=True)
        self.assertEqual(result.returncode,1,result.stderr)
