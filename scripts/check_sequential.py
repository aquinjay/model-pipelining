"""Save release-aware sequential predictive diagnostics and draw their chart."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance
from model_pipelining.local_level_variance import LocalLevelVariance
from model_pipelining.normal_mean_variance import NormalMeanVariance
from model_pipelining.sequential import run_sequential_checks, plot_sequential_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('local-level-variance', 'normal-mean-variance', 'local-level-joint-variance'))
    parser.add_argument('--dataset', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--snapshot-id')
    selection.add_argument('--tag')
    parser.add_argument('--store', default='data/versioned')
    parser.add_argument('--as-of', required=True, type=datetime.fromisoformat)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--config', type=Path, required=True, help='Existing prior configuration; reference settings are used')
    parser.add_argument('--output-directory', required=True)
    options = vars(parser.parse_args())
    config = json.loads(options.pop('config').read_text())['reference']
    kind=options.pop('model')
    model = (LocalLevelJointVariance.from_configuration(config) if kind=='local-level-joint-variance' else
             LocalLevelVariance.from_configuration(config) if kind=='local-level-variance' else NormalMeanVariance(**config))
    experiment = options.pop('experiment')
    options['issued_at'] = options.pop('as_of')
    output = run_sequential_checks(**options, model=model,
        experiment=ExperimentSpec.load(experiment) if experiment else None)
    print(f'Saved: {output}; charts: {plot_sequential_checks(output)}')
    return 0 if json.loads((output/'sequential_checks.json').read_text())['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
