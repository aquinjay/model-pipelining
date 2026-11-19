"""Freeze and execute the bounded innovation-mixture reference exercise."""

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time

import numpy as np

from model_pipelining.innovation_reference import (InnovationMixtureReference, ReferenceSettings,
    MAX_REFERENCE_HISTORY, run_innovation_reference, _values, _points)
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.models import ForecastRequest, Observation


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def diagnostics(value):
    if isinstance(value, dict):
        return {key: diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostics(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_diagnostic': str(value)}
    return value


def gaussian_limits(model, settings):
    """Compare the independent reference to the existing checked scalar model."""
    values = (-.1, .2, 1.)
    start = datetime(2000, 1, 1, tzinfo=UTC)
    history = tuple(Observation(start + timedelta(days=i), y) for i, y in enumerate(values))
    request = ForecastRequest(history, history[-1].timestamp, history[-1].timestamp + timedelta(days=1))
    rows = []
    limits = (('zero-jump', replace(model, jump_probability=0), model.small_variance),
              ('always-jump', replace(model, jump_probability=1), model.large_variance),
              ('coincident-scales', replace(model, large_variance=model.small_variance), model.small_variance))
    for name, selected, w in limits:
        result = run_innovation_reference(values, model=selected, settings=settings)['batch_adaptive']
        scalar = LocalLevelVariance(initial_mean=model.initial_mean, initial_variance=model.initial_variance,
            evolution_variance=w, prior_shape=model.prior_shape, prior_scale=model.prior_scale).posterior(request)
        law = scalar.predictive()
        errors = [abs(result['log_evidence'] - scalar.log_evidence), abs(result['last_mean'] - scalar.level_mean),
                  abs(result['last_variance'] - scalar.level_variance),
                  abs(result['predictive_variance'] - law.variance),
                  abs(result['observation_variance_mean'] - scalar.observation_variance_mean)]
        errors += [abs(probability - law.cdf(point)) for point, probability in zip(result['predictive_points'], result['predictive_cdf'])]
        errors += [abs(density - law.pdf(point)) for point, density in zip(result['predictive_points'], result['predictive_density'])]
        error = max(errors)
        rows.append({'name': name, 'maximum_absolute_difference': error,
                     'status': 'passed' if error <= settings.agreement_tolerance else 'failed'})
    return rows


def plot_saved(directory):
    """Draw only the successful saved query grids and label probabilities."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    manifest = json.loads((directory / 'study.json').read_text())
    results = [(row['name'], json.loads((directory / row['file']).read_text()))
               for row in manifest['cases'] if row['status'] == 'completed']
    if not results:
        return None
    figure = Figure(figsize=(13, 2.6 * len(results)), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(results), 3, squeeze=False)
    for (name, result), row in zip(results, axes):
        for method, style, label in (('batch_adaptive', '-', 'Batch / adaptive'),
                                     ('sequential_quadrature', '.', 'Filter / quadrature')):
            saved = result[method]
            row[0].plot(saved['predictive_points'], saved['predictive_cdf'], style, label=label)
            row[1].plot(saved['predictive_points'], saved['predictive_density'], style)
        posterior = result['batch_adaptive']
        if posterior['jump_probabilities']:
            row[2].bar(range(len(result['values'])), posterior['jump_probabilities'])
        row[2].axhline(result['model']['jump_probability'], color='grey', linestyle='--', label='Fixed prior probability')
        row[0].set(title=f'{name}: next-observation CDF', ylim=(0, 1), xlabel='Observed value')
        row[1].set(title='Next-observation density', xlabel='Observed value')
        row[2].set(title='Past large-innovation probability', ylim=(0, 1), xlabel='Transition index (first is 0)')
    axes[0][0].legend()
    axes[0][2].legend()
    figure.suptitle('All-sequence short-history reference; descriptive cases, no model approval')
    destination = directory / 'reference.png'
    figure.savefig(destination, dpi=150)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/innovation_reference.json')
    parser.add_argument('--output-directory', type=Path, required=True)
    options = parser.parse_args()
    raw = options.config.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) != {'model', 'settings', 'predictive_points', 'state_points', 'variance_points', 'cases'}:
        raise ValueError('Config needs model, settings, query grids and cases')
    model, settings = InnovationMixtureReference(**config['model']), ReferenceSettings(**config['settings'])
    if not isinstance(config['cases'], dict) or not config['cases'] or len(config['cases']) > 64:
        raise ValueError('Choose 1 to 64 explicitly named cases')
    for name, values in config['cases'].items():
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
            raise ValueError('Case names must use letters, digits, hyphens or underscores')
        _values(values)
    common = dict(model=model, settings=settings,
        predictive_points=_points(config['predictive_points']), state_points=_points(config['state_points']),
        variance_points=_points(config['variance_points'], positive=True))
    output = options.output_directory.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    design = {**config, 'model': model.configuration, 'settings': asdict(settings)}
    write_json(output / 'design.json', design)  # Freeze all inputs before any numerical calculation.
    files = ('scripts/run_innovation_reference.py', 'src/model_pipelining/innovation_reference.py',
             'src/model_pipelining/local_level_variance.py', 'src/model_pipelining/mixtures.py',
             'src/model_pipelining/models.py', 'src/model_pipelining/inference.py', 'src/model_pipelining/uncertainty.py')
    manifest = {'format_version': 1, 'kind': 'innovation_reference_study', 'status': 'running',
        'started_at': datetime.now(UTC).isoformat(), 'config_sha256': hashlib.sha256(raw).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        'environment': {'python': platform.python_version(), **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')}},
        'cases': [], 'maximum_observations': MAX_REFERENCE_HISTORY,
        'scope': 'Bounded independent numerical reference; no adapter, backtest or long-history approximation'}
    write_json(output / 'study.json', manifest)
    runtimes, summary = [], []
    for name, values in config['cases'].items():
        print(f'Reference: {name} ({len(values)} observations)', flush=True)
        started = time.perf_counter()
        relative = f'cases/{name}.json'
        try:
            result = run_innovation_reference(values, **common)
            write_json(output / relative, result)
            manifest['cases'].append({'name': name, 'file': relative, 'status': 'completed'})
            fitted = result['batch_adaptive']
            summary.append({'name': name, 'observations': len(values), 'sequences': len(result['sequences']),
                'joint_nodes': result['sequential_quadrature']['numerics']['joint_nodes'],
                'maximum_scaled_difference': result['maximum_scaled_difference'],
                'past_jump_probabilities': fitted['jump_probabilities'], 'last_mean': fitted['last_mean'],
                'observation_variance_mean': fitted['observation_variance_mean'],
                'variance_jump_count_covariance': fitted['variance_jump_count_covariance']})
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            write_json(output / relative, {'status': 'failed', 'values': values,
                'error': f'{type(error).__name__}: {error}', 'evidence': diagnostics(getattr(error, 'evidence', None))})
            manifest['cases'].append({'name': name, 'file': relative, 'status': 'failed'})
        runtimes.append({'name': name, 'seconds': time.perf_counter() - started})
        write_json(output / 'study.json', manifest)
    try:
        limits = gaussian_limits(model, settings)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        limits = [{'name': 'gaussian_limits', 'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                   'evidence': diagnostics(getattr(error, 'evidence', None))}]
    write_json(output / 'gaussian_limits.json', limits)
    write_json(output / 'summary.json', summary)
    write_json(output / 'runtime.json', runtimes)
    write_json(output / 'feasibility.json', {'maximum_observations': MAX_REFERENCE_HISTORY,
        'maximum_positive_prior_sequences': 2**MAX_REFERENCE_HISTORY,
        'nondegenerate_sequences_at_160_observations': str(2**160),
        'conclusion': 'Enumeration is a validation oracle for short histories; a longer-history method requires a separate precision design'})
    passed = all(row['status'] == 'completed' for row in manifest['cases']) and all(row['status'] == 'passed' for row in limits)
    write_json(output / 'checks.json', [{'name': 'all_reference_cases_and_gaussian_limits', 'required': True,
                                       'status': 'passed' if passed else 'failed'}])
    write_json(output / 'review.json', {'status': 'pending', 'decision': None,
        'limitations': ['Numerical agreement is not proof for every input or empirical model adequacy',
            'Fixed daily transition probability/scales; shape > 1; eight observations maximum',
            'No long-history approximation, predictive-check provider or simulation-calibration claim',
            'Finite query grids check the full-law formula at those points, not a uniform bound over every value']})
    manifest.update(status='completed' if passed else 'failed', finished_at=datetime.now(UTC).isoformat())
    write_json(output / 'study.json', manifest)
    print(f"Saved {output}; computation: {manifest['status']}; chart: {plot_saved(output)}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
