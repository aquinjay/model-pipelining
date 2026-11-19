"""Run the lattice pilot or all frozen cases against their retained baselines."""

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from check_innovation_probability_bounds import oracle_comparison, write_json
from model_pipelining.innovation_lattice import bound_innovation_probabilities_lattice
from model_pipelining.innovation_particles import _flatten, _multipliers
from model_pipelining.innovation_probability_bounds import ProbabilityBoundSettings
from model_pipelining.innovation_reference import InnovationMixtureReference, _scale


ROOT = Path(__file__).resolve().parents[1]
PILOT = ('short-abrupt', 'short-dominant', 'abrupt-42')
QUERIES = ('predictive_points', 'state_points', 'variance_points')
SOURCES = ('scripts/study_innovation_lattice.py', 'src/model_pipelining/innovation_lattice.py')


def _archive(output, files):
    hashes = {}
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def _load_parent(source, all_cases):
    snapshots = {}

    def read(name):
        raw = (source / name).read_bytes()
        snapshots[name] = raw
        return json.loads(raw)

    parent = read('study.json')
    if parent['kind'] != 'innovation_probability_bound_study':
        raise ValueError('Use the fixed-budget probability bound study as the parent')
    design = read('inputs/design.json')
    old_summary = read('inputs/summary.json')
    by_name = {case['name']: case for case in design['cases']}
    if len(by_name) != len(design['cases']):
        raise ValueError('Duplicate case in frozen design')
    records = []
    for name in by_name if all_cases else PILOT:
        if not name or any(not (char.isascii() and (char.isalnum() or char in '_-')) for char in name):
            raise ValueError('Invalid frozen case name')
        case = by_name[name]
        baseline = read(f'cases/{name}.json')
        particle = read(f'inputs/cases/{name}.json')
        oracle = read(f'inputs/oracles/{name}.json') if case['oracle'] else None
        for record in (baseline, particle, *([oracle] if oracle else [])):
            if record['model'] != case['model'] or record['values'] != case['values']:
                raise ValueError(f'Design identity differs for {name}')
        if any(baseline[key] != design[key] or particle[key] != design[key] or
               (oracle and oracle['batch_adaptive'][key] != design[key]) for key in QUERIES):
            raise ValueError(f'Query grid differs for {name}')
        if (baseline['settings']['confidence'] != parent['case_confidence'] or
                baseline['settings']['samples'] != parent['samples_per_case']):
            raise ValueError(f'Fixed budget differs for {name}')
        records.append((case, baseline, particle, oracle))
    for name, data in snapshots.items():
        if name.startswith('inputs/') and hashlib.sha256(data).hexdigest() != parent['input_sha256'][name[7:]]:
            raise ValueError(f'Parent input hash differs for {name}')
    sources = {name: (ROOT / name).read_bytes() for name in (*SOURCES, *parent['source_sha256'])}
    for name, digest in parent['source_sha256'].items():
        if hashlib.sha256(sources[name]).hexdigest() != digest:
            raise ValueError(f'Parent source hash differs for {name}')
    return parent, design, records, snapshots, sources, old_summary


def _compare_estimates(actual, reference, scale):
    if actual is None or reference is None:
        return None
    matching = {key: value for key, value in actual.items() if key in reference}
    targets = {key: reference[key] for key in matching}
    return float(np.max(np.abs(_flatten(matching) - _flatten(targets)) * _multipliers(matching, scale)))


def _run_case(case, baseline, particle, oracle, design, samples):
    model = InnovationMixtureReference(**case['model'])
    settings = dict(baseline['settings'])
    if samples is not None:
        settings['samples'] = samples
    result = bound_innovation_probabilities_lattice(case['values'], model=model,
        settings=ProbabilityBoundSettings(**settings), **{key: design[key] for key in QUERIES})
    scale = _scale(model, tuple(case['values']))
    comparison = oracle_comparison(result, oracle) if oracle else None
    estimates = result.get('posterior_estimates')
    report = dict(name=case['name'], observations=len(case['values']), status=result['status'],
        baseline_maximum_width=baseline['maximum_probability_width'],
        maximum_width=result['maximum_probability_width'],
        normalizer_lower=result['normalizer']['scaled_lower'], oracle_comparison=comparison,
        maximum_scaled_oracle_estimate_difference=(
            _compare_estimates(estimates, oracle['batch_adaptive'], scale) if oracle else None),
        maximum_scaled_particle_estimate_difference=(
            _compare_estimates(estimates, particle.get('posterior'), scale)),
        comparison_scope='Moment and density differences are descriptive. Long particle comparisons are empirical.')
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--samples', type=int, help='Override the parent draw count for a separate run')
    options = parser.parse_args(argv)
    if options.samples is not None:
        ProbabilityBoundSettings(samples=options.samples)
    source = options.source_directory.expanduser().absolute()
    parent, design, records, inputs, sources, _ = _load_parent(source, options.all_cases)
    output = options.output_directory.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = _archive(output / 'inputs', inputs)
    source_hashes = _archive(output / 'sources', sources)
    (output / 'cases').mkdir()
    manifest = dict(kind='innovation_lattice_study', status='running',
        started_at=datetime.now(UTC).isoformat(), parent=str(source),
        selection='all' if options.all_cases else 'pilot', cases=[item[0]['name'] for item in records],
        samples_per_case=options.samples if options.samples is not None else parent['samples_per_case'],
        study_confidence=parent['study_confidence'], case_confidence=parent['case_confidence'],
        input_sha256=input_hashes, source_sha256=source_hashes,
        environment=dict(python=platform.python_version(), **{k: version(k) for k in ('numpy', 'scipy')}))
    write_json(output / 'study.json', manifest)
    rows = []
    for case, baseline, particle, oracle in records:
        print(f"Lattice proposal: {case['name']}", flush=True)
        started = time.perf_counter()
        try:
            result, row = _run_case(case, baseline, particle, oracle, design, options.samples)
        except (ValueError, FloatingPointError, OverflowError) as error:
            result = dict(status='failed', values=case['values'], model=case['model'], error=str(error))
            row = dict(name=case['name'], status='failed', error=str(error))
        write_json(output / 'cases' / f"{case['name']}.json", result)
        rows.append(dict(row, elapsed_seconds=time.perf_counter() - started))
        write_json(output / 'summary.json', rows)
    computed = all(row['status'] != 'failed' for row in rows)
    passes = computed and all(row['status'] == 'completed' and
        (row['oracle_comparison'] is None or row['oracle_comparison']['status'] == 'passed') for row in rows)
    manifest.update(status='completed' if computed else 'failed',
        precision_status='passed' if passes else 'unresolved', completed_at=datetime.now(UTC).isoformat())
    write_json(output / 'study.json', manifest)
    print(f"Saved {output} with precision {manifest['precision_status']}", flush=True)
    return 0 if passes else 1


if __name__ == '__main__':
    raise SystemExit(main())
