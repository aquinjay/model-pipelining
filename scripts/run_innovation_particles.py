"""Freeze a numerical particle study; retain unresolved cases and saved oracles."""

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import re
import time

import numpy as np

from model_pipelining.innovation_particles import (ParticleSettings, ParticlePrecisionError,
    MAX_PARTICLE_HISTORY, run_innovation_particles, _flatten, _multipliers)
from model_pipelining.innovation_reference import (InnovationMixtureReference,
    run_innovation_reference, _points)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def expand_design(config):
    expected = {'model', 'settings', 'predictive_points', 'state_points', 'variance_points', 'short_cases', 'long_history'}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f'Config needs exactly {sorted(expected)}')
    model = InnovationMixtureReference(**config['model'])
    settings = ParticleSettings(**config['settings'])
    if model.prior_shape <= 2:
        raise ValueError('The particle study requires prior_shape > 2')
    for key in ('predictive_points', 'state_points', 'variance_points'):
        _points(config[key], positive=key == 'variance_points')
    if not isinstance(config['short_cases'], dict) or not 1 <= len(config['short_cases']) <= 32:
        raise ValueError('Choose 1..32 short numerical cases')
    cases = []
    for name, values in config['short_cases'].items():
        if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name)
                or not isinstance(values, list) or len(values) > 8
                or any(type(y) not in (int, float) or not np.isfinite(y) for y in values)):
            raise ValueError('Short cases need safe names and 0..8 finite observations')
        cases.append(dict(name='short-' + name, values=values, model=model.configuration,
            settings=asdict(settings), oracle=True, generator=None))
    for name, selected in (('zero-jump', replace(model, jump_probability=0)),
            ('always-jump', replace(model, jump_probability=1)),
            ('coincident-scales', replace(model, large_variance=model.small_variance)),
            ('state-atoms', replace(model, initial_variance=0, small_variance=0))):
        cases.append(dict(name=name, values=[0., .1, 2.], model=selected.configuration,
                          settings=asdict(settings), oracle=True, generator=None))
    long = config['long_history']
    if not isinstance(long, dict) or set(long) != {'count', 'onset', 'seeds', 'baseline', 'shift', 'slope', 'noise_sd'}:
        raise ValueError('long_history needs count, onset, seeds, baseline, shift, slope and noise_sd')
    if (type(long['count']) is not int or not 9 <= long['count'] <= MAX_PARTICLE_HISTORY
            or type(long['onset']) is not int or not 0 < long['onset'] < long['count']
            or not isinstance(long['seeds'], list) or not 1 <= len(long['seeds']) <= 8
            or any(type(seed) is not int or seed < 0 for seed in long['seeds'])
            or len(set(long['seeds'])) != len(long['seeds'])):
        raise ValueError('Invalid long-history count/onset/seeds')
    for key in ('baseline', 'shift', 'slope', 'noise_sd'):
        if type(long[key]) not in (int, float) or not np.isfinite(long[key]):
            raise ValueError(f'Invalid long-history {key}')
    if long['noise_sd'] <= 0:
        raise ValueError('noise_sd must be positive')
    for seed in long['seeds']:
        rng = random.Random(seed)
        noise = np.array([rng.gauss(0, long['noise_sd']) for _ in range(long['count'])])
        index = np.arange(long['count'])
        for scenario, change in (('stable', np.zeros(long['count'])),
                ('abrupt', long['shift'] * (index >= long['onset'])),
                ('gradual', long['slope'] * np.maximum(0, index - long['onset'] + 1))):
            cases.append(dict(name=f'{scenario}-{seed}', values=(long['baseline'] + noise + change).tolist(),
                model=replace(model, initial_mean=long['baseline']).configuration, settings=asdict(settings),
                oracle=False, generator=dict(**long, selected_seed=seed, scenario=scenario)))
    if any(not np.isfinite(case['values']).all() for case in cases):
        raise ValueError('Unrepresentable generated observations')
    return {**config, 'model': model.configuration, 'settings': asdict(settings), 'cases': cases}


def compare_oracle(result, oracle):
    summary = result.get('posterior')
    if summary is None:
        return dict(status='unavailable', reason='Approximation unresolved')
    target = oracle['batch_adaptive']
    expected = {key: sum(target['jump_probabilities']) if key == 'jump_count' else target[key] for key in summary}
    multipliers = _multipliers(summary, result['comparison_scale'])
    errors = np.abs(_flatten(summary) - _flatten(expected)) * multipliers
    if result['mode'] == 'gaussian_quadrature':
        widths = np.full(len(errors), 5e-6)
    else:
        level = result['levels'][-1]
        widths = level['critical_value'] * _flatten(result['mcse']) * multipliers + 5e-6
    return dict(status='passed' if np.all(errors <= widths) else 'failed',
        maximum_scaled_difference=float(errors.max()),
        outside_empirical_envelope=int(np.count_nonzero(errors > widths)),
        scope='Requested moments, every short path probability and CDF/density probes; empirical envelope, not coverage proof')


def plot_saved(directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    design = json.loads((directory / 'design.json').read_text())
    rows = json.loads((directory / 'summary.json').read_text())
    figure = Figure(figsize=(12, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    short = [row for row in rows if row.get('oracle_comparison')]
    axes[0, 0].barh([row['name'] for row in short],
        [max(row['oracle_comparison'].get('maximum_scaled_difference', 0), 1e-12) for row in short])
    axes[0, 0].set(xscale='log', title='Short-history discrepancy from independent oracle')
    axes[0, 0].set_xlabel('Maximum scaled difference at requested probes')
    for row in rows:
        if row['oracle']:
            continue
        result = json.loads((directory / 'cases' / f"{row['name']}.json").read_text())
        if not result['levels'] or 'posterior' not in result['levels'][-1]:
            continue
        levels = result['levels']
        label = f"{row['name']} ({row['status']})"
        axes[0, 1].plot([level['particles'] for level in levels],
            [level['maximum_scaled_halfwidth'] for level in levels], '.-', label=label)
        posterior = result.get('posterior', levels[-1]['posterior'])
        axes[1, 0].plot(posterior['jump_probabilities'], label=label)
        axes[1, 1].plot([step['distinct_variance_ancestors'] for step in levels[-1]['replicates'][0]['diagnostics']['steps']], label=label)
    axes[0, 1].axhline(design['settings']['tolerance'], color='black', linestyle='--')
    axes[0, 1].set(xscale='log', yscale='log', title='Independent-run empirical halfwidth', xlabel='Particles per replicate')
    axes[1, 0].set(title='Past large-innovation probabilities at final attempted level', xlabel='Transition', ylim=(0, 1))
    axes[1, 1].set(title='Distinct V ancestors in first replicate', xlabel='Transition', yscale='log')
    axes[1, 0].legend(fontsize=7)
    figure.suptitle('Particle feasibility study; unresolved outputs remain provisional')
    figure.savefig(directory / 'particles.png', dpi=150)
    return directory / 'particles.png'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/innovation_particles.json')
    parser.add_argument('--output-directory', type=Path, required=True)
    options = parser.parse_args(argv)
    raw_config = options.config.read_bytes()
    design = expand_design(json.loads(raw_config))
    output = options.output_directory.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    (output / 'oracles').mkdir()
    write_json(output / 'design.json', design)
    files = ('scripts/run_innovation_particles.py', 'src/model_pipelining/innovation_particles.py',
        'src/model_pipelining/innovation_reference.py', 'src/model_pipelining/local_level_variance.py',
        'src/model_pipelining/models.py', 'src/model_pipelining/mixtures.py',
        'src/model_pipelining/inference.py', 'src/model_pipelining/uncertainty.py')
    manifest = dict(format_version=1, kind='innovation_particle_study', status='running',
        started_at=datetime.now(UTC).isoformat(), config_sha256=hashlib.sha256(raw_config).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        environment=dict(python=platform.python_version(), **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')}), cases=[])
    write_json(output / 'study.json', manifest)
    summary = []
    for case in design['cases']:
        print(f"Particles: {case['name']} ({len(case['values'])} observations)", flush=True)
        start = time.perf_counter()
        common = dict(model=InnovationMixtureReference(**case['model']), predictive_points=design['predictive_points'],
                      state_points=design['state_points'], variance_points=design['variance_points'])
        try:
            result = run_innovation_particles(case['values'], settings=ParticleSettings(**case['settings']), **common)
        except ParticlePrecisionError as error:
            result = error.evidence
            result['error'] = str(error)
        write_json(output / 'cases' / f"{case['name']}.json", result)
        comparison = None
        if case['oracle']:
            try:
                oracle = run_innovation_reference(case['values'], **common)
                write_json(output / 'oracles' / f"{case['name']}.json", oracle)
                comparison = compare_oracle(result, oracle)
            except ValueError as error:
                comparison = dict(status='failed', error=str(error))
                write_json(output / 'oracles' / f"{case['name']}.json", dict(status='failed', error=str(error),
                    evidence=getattr(error, 'evidence', None)))
        final = result['levels'][-1] if result['levels'] else {}
        row = dict(name=case['name'], observations=len(case['values']), oracle=case['oracle'],
            status=result['status'], accepted_particles=result.get('accepted_particles'),
            attempted_particles=[level['particles'] for level in result['levels']],
            maximum_scaled_halfwidth=final.get('maximum_scaled_halfwidth'),
            refinement_envelope=final.get('refinement_envelope'),
            minimum_variance_ancestors=final.get('minimum_variance_ancestors'),
            replicate_evidence_ess=final.get('replicate_evidence_ess'), oracle_comparison=comparison)
        summary.append(row)
        manifest['cases'].append(dict(name=case['name'], status=result['status'], seconds=time.perf_counter() - start))
        write_json(output / 'summary.json', summary)
        write_json(output / 'study.json', manifest)
    passed = all(row['status'] == 'completed' and (row['oracle_comparison'] is None or
        row['oracle_comparison']['status'] == 'passed') for row in summary)
    manifest.update(status='completed' if passed else 'unresolved', finished_at=datetime.now(UTC).isoformat())
    write_json(output / 'study.json', manifest)
    write_json(output / 'checks.json', dict(status='passed' if passed else 'unresolved',
        all_cases_resolved_and_short_oracles_agree=passed, model_approval=False))
    write_json(output / 'review.json', dict(status='pending', decision=None, limitations=[
        'Independent-run jackknife/Student-t envelopes are empirical; finite-run bias and missed modes can remain',
        'No long-history oracle or uniform distribution error bound; no genealogy rejuvenation',
        'Fixed daily transition settings; no forecasting adapter, evidence provider or calibration claim']))
    print(f"Saved {output}; computation: {manifest['status']}; chart: {plot_saved(output)}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
