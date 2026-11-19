"""Freeze an independent fixed-budget probability-bound follow-up to a saved study."""

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from model_pipelining.innovation_probability_bounds import (
    ProbabilityBoundSettings, bound_innovation_probabilities, _ratio_bounds)
from model_pipelining.innovation_reference import InnovationMixtureReference


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def oracle_comparison(result, oracle):
    target = oracle['batch_adaptive']
    outside, total = 0, 0
    for key, interval in result['intervals'].items():
        expected = np.asarray(target[key])
        lower, upper = np.asarray(interval['lower']), np.asarray(interval['upper'])
        if expected.shape != lower.shape:
            raise ValueError(f'Oracle shape mismatch: {key}')
        # The finite-sample bounds concern exact expectations; the independent
        # numerical oracle has its separately declared integration tolerance.
        outside += int(np.count_nonzero((expected < lower - 5e-6) | (expected > upper + 5e-6)))
        total += expected.size
    return dict(status='passed' if outside == 0 else 'failed', outside_bounds=outside,
                checked_probabilities=total, numerical_allowance=5e-6)


def validate_statistics(result):
    stats, settings = result['sufficient_statistics'], result['settings']
    lower, upper, denominator, _ = _ratio_bounds(np.asarray(stats['means']),
        np.asarray(stats['sample_variances']), settings['samples'], settings['confidence'])
    if not np.array_equal(lower, np.concatenate([v['lower'] for v in result['intervals'].values()])):
        raise ValueError('Saved lower intervals do not replay')
    if not np.array_equal(upper, np.concatenate([v['upper'] for v in result['intervals'].values()])):
        raise ValueError('Saved upper intervals do not replay')
    expected_checks = dict(positive_normalizer=denominator[0] > 0,
        probability_width=float(max(upper - lower, default=0.)) <= 2 * settings['tolerance'])
    if result['checks'] != expected_checks:
        raise ValueError('Saved precision checks do not replay')


def plot_saved(directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    rows = json.loads((directory / 'summary.json').read_text())
    figure = Figure(figsize=(11, 6), layout='constrained')
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    good = [row for row in rows if 'maximum_probability_width' in row]
    left.barh([row['name'] for row in good], [row['maximum_probability_width'] for row in good])
    left.set(xlim=(0, 1), xlabel='Maximum simultaneous probability interval width',
             title='Fixed-budget follow-up; long bounds remain unresolved')
    labels, upper, truth = [], [], []
    for case, paths in (('short-quiet', ('011100', '101001')), ('short-gradual', ('000111',))):
        path = directory / 'cases' / f'{case}.json'
        oracle_path = directory / 'inputs' / 'oracles' / f'{case}.json'
        if not path.exists() or not oracle_path.exists():
            continue
        result, oracle = json.loads(path.read_text()), json.loads(oracle_path.read_text())
        if 'intervals' not in result:
            continue
        for sequence in paths:
            labels.append(f'{case}: {sequence}')
            upper.append(result['intervals']['sequence_probabilities']['upper'][int(sequence, 2)])
            truth.append(oracle['batch_adaptive']['sequence_probabilities'][int(sequence, 2)])
    right.barh(labels, upper, alpha=.5, label='Conservative upper endpoint (lower = 0)')
    right.scatter(truth, labels, color='black', marker='x', label='Independent oracle')
    right.set(xlabel='Posterior path probability', title='Previously failed rare paths')
    if labels:
        right.legend(fontsize=7)
    figure.savefig(directory / 'probability-bounds.png', dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.95)
    options = parser.parse_args(argv)
    source = options.source_directory.expanduser().absolute()
    design = json.loads((source / 'design.json').read_bytes())
    parent = json.loads((source / 'study.json').read_bytes())
    if parent['kind'] != 'innovation_particle_study' or not design['cases']:
        raise ValueError('Select a saved innovation particle study with cases')
    # Fix the entire family before seeing any new draws; no adaptive stopping.
    ProbabilityBoundSettings(samples=options.samples, seed=options.seed, confidence=options.confidence)
    case_confidence = 1 - (1 - options.confidence) / len(design['cases'])
    settings = ProbabilityBoundSettings(samples=options.samples, seed=options.seed, confidence=case_confidence)
    for name, digest in parent['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Parent source differs from current implementation: {name}')
    # Freeze all parent bytes needed for this exercise before computing. Retain
    # original particle failures without modifying or relabeling their artifacts.
    inputs = {name: (source / name).read_bytes() for name in ('design.json', 'study.json', 'summary.json')}
    for case in design['cases']:
        inputs[f"cases/{case['name']}.json"] = (source / 'cases' / f"{case['name']}.json").read_bytes()
        saved = json.loads(inputs[f"cases/{case['name']}.json"])
        if saved['values'] != case['values'] or saved['model'] != case['model']:
            raise ValueError('Parent case identity differs from frozen design')
        if case['oracle']:
            name = f"oracles/{case['name']}.json"
            inputs[name] = (source / name).read_bytes()
            oracle = json.loads(inputs[name])
            if (oracle['values'] != case['values'] or oracle['model'] != case['model'] or
                    any(oracle['batch_adaptive'][key] != design[key] for key in ('predictive_points', 'state_points', 'variance_points'))):
                raise ValueError('Parent oracle identity differs from frozen design')
    output = options.output_directory.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    for name, data in inputs.items():
        path = output / 'inputs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    sources = ('scripts/check_innovation_probability_bounds.py',
        'src/model_pipelining/innovation_probability_bounds.py', *parent['source_sha256'])
    for name in sources:
        path = output / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    manifest = dict(kind='innovation_probability_bound_study', status='running',
        started_at=datetime.now(UTC).isoformat(), source_directory=str(source),
        study_confidence=options.confidence, case_confidence=case_confidence,
        samples_per_case=options.samples, source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        input_sha256={name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
        environment=dict(python=platform.python_version(), **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')}))
    write_json(output / 'study.json', manifest)
    rows = []
    for case in design['cases']:
        print(f"Probability bounds: {case['name']}", flush=True)
        started = time.perf_counter()
        try:
            result = bound_innovation_probabilities(case['values'], model=InnovationMixtureReference(**case['model']),
                settings=settings, **{key: design[key] for key in ('predictive_points', 'state_points', 'variance_points')})
            validate_statistics(result)
            comparison = oracle_comparison(result, json.loads(inputs[f"oracles/{case['name']}.json"])) if case['oracle'] else None
            row = dict(name=case['name'], observations=len(case['values']), status=result['status'],
                maximum_probability_width=result['maximum_probability_width'],
                scaled_normalizer_lower=result['normalizer']['scaled_lower'], oracle_comparison=comparison)
        except (ValueError, FloatingPointError, OverflowError) as error:
            result = dict(status='failed', error=str(error))
            row = dict(name=case['name'], status='failed', error=str(error))
        write_json(output / 'cases' / f"{case['name']}.json", result)
        rows.append(dict(**row, elapsed_seconds=time.perf_counter() - started))
        write_json(output / 'summary.json', rows)
    computed = all(row['status'] != 'failed' for row in rows)
    passed = all(row['status'] == 'completed' and
        (not row.get('oracle_comparison') or row['oracle_comparison']['status'] == 'passed') for row in rows)
    manifest.update(status='completed' if computed else 'failed', precision_status='passed' if passed else 'unresolved',
        completed_at=datetime.now(UTC).isoformat())
    write_json(output / 'study.json', manifest)
    write_json(output / 'validation.json', dict(status='passed' if computed else 'failed',
        scope='Frozen parent/source hashes and all saved sufficient-statistic interval/check recalculations; no full Monte Carlo replay',
        precision_status=manifest['precision_status'], model_approval=False))
    plot_saved(output)
    print(f"Saved {output}; precision: {manifest['precision_status']}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
