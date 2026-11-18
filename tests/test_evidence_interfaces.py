"""Experiment-owned requirements and evidence providers independent of model classes."""

import csv
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from model_pipelining import evidence_workflows as ew
from model_pipelining import predictive_checks as pc
from model_pipelining import prior_influence as pi
from model_pipelining.backtest import run_backtest
from model_pipelining.evidence import get_evidence_provider
from model_pipelining.evidence_normal import NormalMeanVarianceEvidence
from model_pipelining.experiments import EvidencePlan, EvaluationPlan, ExperimentSpec
from model_pipelining.inference import FitEvidence, FittedForecast, assess_checks
from model_pipelining.local_level_variance import LocalLevelVariance, QuadratureSettings
from model_pipelining.models import Forecast, TrainingMean
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.reporting import compare_runs
from model_pipelining.tags import assign_tag
import test_predictive_checks as existing_checks


class EvidencePlanTests(unittest.TestCase):
    def test_version_one_load_and_version_two_requirement_roundtrip(self):
        old = json.loads(Path('configs/daily_forecast_experiment.json').read_text())
        self.assertEqual(old['format_version'], 1)
        restored = ExperimentSpec.from_dict(old)
        self.assertEqual(restored.evidence, EvidencePlan())
        current = replace(restored, evidence=EvidencePlan(required_workflows=('predictive_checks', 'prior_influence'),
            required_fit_checks=('precision',), replications=17, seed=23, interval_level=.8))
        serialized = json.loads(json.dumps(current.to_dict()))
        self.assertEqual(serialized['format_version'], 2)
        self.assertEqual(ExperimentSpec.from_dict(serialized), current)
        for changed in ({**serialized, 'evidence': []}, {**serialized, 'format_version': 3},
                        {**serialized, 'unexpected': 1}, {**old, 'evidence': {}}):
            with self.assertRaises(ValueError):
                ExperimentSpec.from_dict(changed)

    def test_invalid_plans_fail_before_execution(self):
        for values in ({'required_workflows': ['predictive_checks']}, {'required_workflows': ('unknown',)},
                       {'required_workflows': ('predictive_checks', 'predictive_checks')},
                       {'required_fit_checks': ('',)}, {'required_fit_checks': ('a', 'a')},
                       {'replications': True}, {'replications': 0}, {'seed': -1}, {'seed': 2**32},
                       {'interval_level': float('nan')}, {'interval_level': True}, {'interval_level': 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EvidencePlan(**values)
        with self.assertRaises(ValueError):
            ExperimentSpec(evidence={})

    def test_required_names_cannot_be_bypassed_by_optional_checks(self):
        base = {'name': 'computed', 'required': True, 'status': 'passed'}
        for status in ('failed', 'not_applicable'):
            passed, requirements = assess_checks((base, {'name': 'caller', 'required': False, 'status': status}),
                                                required_names=('caller', 'missing'))
            self.assertFalse(passed)
            self.assertTrue(all(item['status'] == 'failed' for item in requirements))
        passed, requirements = assess_checks((base, {'name': 'caller', 'required': False, 'status': 'passed'}),
                                            required_names=('caller',))
        self.assertTrue(passed)
        self.assertEqual(requirements[0]['status'], 'passed')
        for checks in ([], (base, base), ({**base, 'status': 'unknown'},), ({**base, 'required': 'yes'},)):
            with self.assertRaises(ValueError):
                assess_checks(checks)


class ProxyModel:
    """A test adapter unknown to shared workflow code, delegating only the math."""
    name = 'proxy'

    def __init__(self, model):
        self.model = model

    @property
    def configuration(self):
        return self.model.configuration

    def evidence_provider(self):
        return ProxyProvider(self)


class ProxyProvider(NormalMeanVarianceEvidence):
    def __init__(self, proxy):
        super().__init__(proxy.model)
        self.metadata = replace(self.metadata, family='example_provider')

    def validate_scenarios(self, models):
        super().validate_scenarios(tuple(model.model for model in models))


class AuditModel:
    name = 'audit-example'
    supports_fit_evidence = True
    supports_quantiles = False
    supports_highest_density = False
    configuration = {}

    def __init__(self, checks):
        self.checks = checks

    def fit(self, request):
        return FittedForecast(Forecast(0), FitEvidence({}, self.checks))

    def forecast(self, request):
        raise AssertionError('Use the fit-evidence dispatch')


class EvidenceWorkflowTests(unittest.TestCase):
    setUp = existing_checks.PredictiveWorkflowTests.setUp
    register = existing_checks.PredictiveWorkflowTests.register
    read_json = existing_checks.PredictiveWorkflowTests.read_json

    def evidence_spec(self, **changes):
        values = dict(required_workflows=('predictive_checks', 'prior_influence'),
                      replications=8, seed=19, interval_level=.8)
        values.update(changes)
        return replace(self.spec, evidence=EvidencePlan(**values))

    def run_evidence(self, name='evidence', **changes):
        values = dict(dataset='example', model=self.model, issued_at=self.history[-1].timestamp,
                      snapshot_id=self.snapshot.snapshot_id, store=self.store, experiment=self.evidence_spec(),
                      scenarios={'shift': replace(self.model, prior_mean=10)}, output_directory=self.root/name)
        values.update(changes)
        return ew.run_experiment_evidence(**values)

    def test_both_existing_models_follow_experiment_settings_and_save_receipts(self):
        for model in (self.model, LocalLevelVariance()):
            scenarios = {'shift': replace(model, **({'prior_mean': 10} if isinstance(model, NormalMeanVariance)
                                                   else {'initial_mean': 10}))}
            output = self.run_evidence(name=model.name, model=model, scenarios=scenarios)
            manifest = self.read_json(output, 'evidence.json')
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(self.read_json(output, 'review.json')['status'], 'pending')
            predictive = self.read_json(output/'predictive_checks', 'predictive_checks.json')
            influence = self.read_json(output/'prior_influence', 'prior_influence.json')
            self.assertEqual((predictive['replications'], predictive['seed']), (8, 19))
            self.assertEqual(influence['interval_level'], .8)
            self.assertEqual(predictive['context']['history_sha256'], influence['context']['history_sha256'])
            for row in manifest['workflows']:
                self.assertEqual(row['status'], 'passed')
                self.assertTrue(all(check['status'] == 'passed' for check in row['check_requirements']))
                for filename, digest in row['artifact_sha256'].items():
                    self.assertEqual(hashlib.sha256((output/row['directory']/filename).read_bytes()).hexdigest(), digest)

    def test_unregistered_provider_runs_both_workflows_and_saved_only_plots(self):
        proxy = ProxyModel(self.model)
        output = self.run_evidence(model=proxy, scenarios={'shift': ProxyModel(replace(self.model, prior_mean=10))})
        self.assertEqual(self.read_json(output, 'evidence.json')['status'], 'completed')
        self.assertEqual(self.read_json(output/'predictive_checks', 'predictive_checks.json')['family'], 'example_provider')
        with patch.object(ProxyModel, 'evidence_provider', side_effect=AssertionError('Saved files only')):
            self.assertEqual(len(ew.plot_experiment_evidence(output)), 4)

    def test_provider_cannot_omit_or_duplicate_requested_replications(self):
        valid = pc.simulate_prior_predictive(self.model, observation_count=3, replications=2)
        for name, datasets in (('missing', ()), ('duplicate', (valid.datasets[0], valid.datasets[0]))):
            with patch.object(NormalMeanVarianceEvidence, 'simulate', return_value=replace(valid, datasets=datasets)):
                output = pc.run_prior_predictive_checks(model=self.model, observation_count=3,
                    replications=2, output_directory=self.root/name)
            self.assertEqual(self.read_json(output, 'predictive_checks.json')['status'], 'failed')
            self.assertIn('omitted or duplicated', self.read_json(output/'prior', 'summary.json')['error'])

    def test_provider_cannot_complete_prior_updates_with_empty_summaries(self):
        with patch.object(NormalMeanVarianceEvidence, 'summarize', return_value={}):
            output = self.run_evidence()
        self.assertEqual(self.read_json(output, 'evidence.json')['status'], 'failed')
        summaries = self.read_json(output/'prior_influence', 'configuration_summaries.json')
        self.assertTrue(all(item['completed_updates'] == 0 for item in summaries))

    def test_provider_metadata_and_absent_capabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            get_evidence_provider(TrainingMean(), 'predictive_checks')
        provider = self.model.evidence_provider()
        provider.metadata = replace(provider.metadata, parameter_fields=('mean',))
        with patch.object(NormalMeanVariance, 'evidence_provider', return_value=provider), self.assertRaises(ValueError):
            pc.run_prior_predictive_checks(model=self.model, observation_count=3,
                output_directory=self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_missing_scenarios_fails_requirement_and_later_workflow_continues(self):
        output = self.run_evidence(scenarios=None,
            experiment=self.evidence_spec(required_workflows=('prior_influence', 'predictive_checks')))
        manifest = self.read_json(output, 'evidence.json')
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual([row['status'] for row in manifest['workflows']], ['failed', 'passed'])
        self.assertIn('explicit named scenarios', manifest['workflows'][0]['error'])
        self.assertEqual(len(ew.plot_experiment_evidence(output)), 2)

    def test_unavailable_workflow_is_saved_as_failed_required_evidence(self):
        provider = self.model.evidence_provider()
        provider.metadata = replace(provider.metadata, workflows=('predictive_checks',))
        with patch.object(NormalMeanVariance, 'evidence_provider', return_value=provider):
            output = self.run_evidence()
        self.assertEqual([row['status'] for row in self.read_json(output, 'evidence.json')['workflows']],
                         ['passed', 'failed'])
        unsupported = self.run_evidence(name='unsupported', model=TrainingMean())
        self.assertEqual(self.read_json(unsupported, 'evidence.json')['status'], 'failed')
        self.assertFalse((unsupported/'predictive_checks').exists())

    def test_missing_required_check_cannot_pass_completed_child(self):
        original = ew.run_predictive_checks
        def omit(**kwargs):
            output = original(**kwargs)
            checks = self.read_json(output, 'checks.json')
            (output/'checks.json').write_text(json.dumps([row for row in checks if row['name'] != 'observed_statistics_computed']))
            return output
        with patch.object(ew, 'run_predictive_checks', side_effect=omit):
            output = self.run_evidence()
        manifest = self.read_json(output, 'evidence.json')
        self.assertEqual(manifest['status'], 'failed')
        missing = manifest['workflows'][0]['check_requirements'][-1]
        self.assertEqual(missing['reason'], 'Missing required check')
        self.assertEqual(manifest['workflows'][1]['status'], 'passed')

    def test_child_evidence_must_match_the_pinned_identity_and_model(self):
        original = ew.run_predictive_checks
        for field in ('history', 'model', 'settings'):
            def change(**kwargs):
                output = original(**kwargs)
                manifest = self.read_json(output, 'predictive_checks.json')
                if field == 'history':
                    manifest['context']['history_sha256'] = 'wrong'
                elif field == 'model':
                    manifest['configuration']['prior_mean'] = 999
                else:
                    manifest['seed'] = 999
                (output/'predictive_checks.json').write_text(json.dumps(manifest))
                return output
            with patch.object(ew, 'run_predictive_checks', side_effect=change):
                output = self.run_evidence(name=field)
            manifest = self.read_json(output, 'evidence.json')
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('experiment', manifest['workflows'][0]['error'])

    def test_tag_is_pinned_once_for_all_workflows(self):
        changed = self.register((100, 300, 500, 700, 900))
        assign_tag(dataset='example', tag='current', snapshot_id=self.snapshot.snapshot_id,
                   changed_by='test', store=self.store)
        original = ew.run_predictive_checks
        def move_tag(**kwargs):
            output = original(**kwargs)
            assign_tag(dataset='example', tag='current', snapshot_id=changed.snapshot_id,
                       changed_by='test', store=self.store)
            return output
        with patch.object(ew, 'run_predictive_checks', side_effect=move_tag):
            output = self.run_evidence(snapshot_id=None, tag='current')
        for workflow, filename in ew._MANIFESTS.items():
            self.assertEqual(self.read_json(output/workflow, filename)['context']['input']['snapshot_id'],
                             self.snapshot.snapshot_id)
        self.assertEqual(self.read_json(output, 'evidence.json')['context']['input']['selected_tag'], 'current')

    def test_replay_and_future_final_isolation(self):
        boundary = self.history[3].timestamp
        spec = replace(self.evidence_spec(), evaluation=EvaluationPlan(final_start=boundary,
                       final_end=self.history[-1].timestamp+timedelta(days=1)))
        first = self.run_evidence(name='first', experiment=spec, issued_at=boundary-timedelta(seconds=1))
        replay = self.run_evidence(name='replay', experiment=spec, issued_at=boundary-timedelta(seconds=1))
        changed = self.register((1, 3, 5, 700, 900))
        future = self.run_evidence(name='future', experiment=spec, issued_at=boundary-timedelta(seconds=1),
                                   snapshot_id=changed.snapshot_id)
        manifests = {'evidence.json', 'predictive_checks.json', 'prior_influence.json'}
        for path in first.rglob('*'):
            if path.is_file() and path.name not in manifests:
                relative = path.relative_to(first)
                self.assertEqual(path.read_bytes(), (replay/relative).read_bytes())
                self.assertEqual(path.read_bytes(), (future/relative).read_bytes())
        for changed_options in ({'issued_at': boundary}, {'experiment': replace(spec, evaluation=replace(spec.evaluation, role='final'))},
                                {'experiment': self.spec}):
            with self.assertRaises(ValueError):
                self.run_evidence(name='invalid', **({'experiment': spec} | changed_options))
        self.assertFalse((self.root/'invalid').exists())

    def test_numerical_failures_keep_prior_and_diagnostics_and_continue(self):
        model = LocalLevelVariance(quadrature=QuadratureSettings(minimum_nodes=8, maximum_nodes=16, maximum_expansions=1))
        output = self.run_evidence(model=model, scenarios={'shift': replace(model, initial_mean=10)})
        self.assertEqual(self.read_json(output, 'evidence.json')['status'], 'failed')
        self.assertEqual(self.read_json(output/'predictive_checks/prior', 'summary.json')['status'], 'completed')
        self.assertIsNotNone(self.read_json(output/'predictive_checks/posterior', 'summary.json')['inference_failure'])
        self.assertTrue((output/'prior_influence/integration_checks.json').is_file())

    def test_io_errors_propagate_and_outputs_are_never_reused(self):
        with patch.object(ew, 'run_predictive_checks', side_effect=OSError('disk failure')), self.assertRaises(OSError):
            self.run_evidence()
        self.assertEqual(self.read_json(self.root/'evidence', 'evidence.json')['status'], 'running')
        with self.assertRaises(FileExistsError):
            self.run_evidence()

    def test_legacy_predictive_and_prior_artifacts_plot_without_providers(self):
        for model in (self.model, LocalLevelVariance()):
            scenario = replace(model, **({'prior_mean': 10} if isinstance(model, NormalMeanVariance) else {'initial_mean': 10}))
            output = self.run_evidence(name=model.name, model=model, scenarios={'shift': scenario})
            for workflow, filename in ew._MANIFESTS.items():
                manifest = self.read_json(output/workflow, filename)
                manifest.pop('presentation')
                manifest.pop('family')
                (output/workflow/filename).write_text(json.dumps(manifest))
            with patch.object(type(model), 'evidence_provider', side_effect=AssertionError('No model')):
                self.assertEqual(len(ew.plot_experiment_evidence(output)), 4)

    def test_new_cli_executes_both_models_and_exit_code_reports_failure(self):
        for kind, model in (('normal-mean-variance', self.model), ('local-level-variance', LocalLevelVariance())):
            config = self.root/f'{kind}.json'
            config.write_text(json.dumps({'reference': model.configuration, 'scenarios': {}}))
            spec = self.root/'experiment.json'
            spec.write_text(json.dumps(self.evidence_spec(required_workflows=('predictive_checks',)).to_dict()))
            output = self.root/kind
            command = [sys.executable, 'scripts/run_evidence.py', '--model', kind, '--dataset', 'example',
                '--snapshot-id', self.snapshot.snapshot_id, '--store', str(self.store), '--as-of',
                self.history[-1].timestamp.isoformat(), '--experiment', str(spec), '--config', str(config),
                '--output-directory', str(output)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output/'predictive_checks/posterior_predictive.png').is_file())
        spec.write_text(json.dumps(self.evidence_spec().to_dict()))
        failed_command = command[:-1] + [str(self.root/'failed-cli')]
        result = subprocess.run(failed_command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.read_json(self.root/'failed-cli', 'evidence.json')['status'], 'failed')

    def test_required_fit_check_missing_or_optional_not_applicable_fails_saved_fit(self):
        base = {'name': 'computation', 'required': True, 'status': 'passed'}
        cases = ((), ({'name': 'caller', 'required': False, 'status': 'not_applicable'},),
                 ({'name': 'caller', 'required': False, 'status': 'passed'},))
        for index, extra in enumerate(cases):
            output = self.root/f'fit-{index}'
            run_backtest(dataset='example', snapshot_id=self.snapshot.snapshot_id, store=self.store,
                experiment=self.evidence_spec(required_fit_checks=('caller',)),
                model=AuditModel((base, *extra)), output_directory=output)
            expected = 'completed' if index == 2 else 'failed'
            self.assertEqual(self.read_json(output, 'manifest.json')['status'], expected)
            summaries = list((output/'fits').glob('*/fit.json'))
            self.assertTrue(summaries)
            self.assertTrue(all(json.loads(path.read_text())['status'] == expected for path in summaries))
            first = json.loads(summaries[0].read_text())
            self.assertEqual(first['experiment_requirements'][0]['name'], 'caller')

    def test_model_without_fit_evidence_cannot_bypass_experiment_requirement(self):
        output = self.root/'fit-required'
        run_backtest(dataset='example', snapshot_id=self.snapshot.snapshot_id, store=self.store,
            experiment=self.evidence_spec(required_fit_checks=('caller',)), model=TrainingMean(), output_directory=output)
        self.assertEqual(self.read_json(output, 'manifest.json')['status'], 'failed')
        self.assertTrue((output/'fit_index.csv').is_file())
        with (output/'fit_index.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row['status'] == 'failed' and not row['fit_directory'] for row in rows))

    def test_forecast_completion_keeps_separate_workflow_requirements_pending(self):
        output = self.root/'forecast'
        run_backtest(dataset='example', snapshot_id=self.snapshot.snapshot_id, store=self.store,
            experiment=self.evidence_spec(), model=TrainingMean(), output_directory=output)
        manifest = self.read_json(output, 'manifest.json')
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(manifest['separate_evidence']['status'], 'pending')
        self.assertFalse((output/'predictive_checks').exists())

    def test_saved_comparison_normalizes_v1_defaults_but_enforces_v2_requirements(self):
        for name in ('old', 'current'):
            run_backtest(dataset='example', snapshot_id=self.snapshot.snapshot_id, store=self.store,
                experiment=self.spec, model=TrainingMean(), output_directory=self.root/name)
        old = self.read_json(self.root/'old', 'manifest.json')
        old['experiment'].pop('evidence')
        old['experiment']['format_version'] = 1
        (self.root/'old/manifest.json').write_text(json.dumps(old))
        compare_runs(self.root/'old', self.root/'current', output_directory=self.root/'comparison')
        current = self.read_json(self.root/'current', 'manifest.json')
        current['experiment']['evidence']['required_workflows'] = ['predictive_checks']
        (self.root/'current/manifest.json').write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, 'must match'):
            compare_runs(self.root/'old', self.root/'current', output_directory=self.root/'mismatched')


if __name__ == '__main__':
    unittest.main()
