"""Forecast with categorical evolution-variance and continuous noise learning."""
import argparse
import json
from pathlib import Path
from model_pipelining.backtest import run_backtest
from model_pipelining.experiments import ExperimentSpec
from model_pipelining.local_level_joint_variance import LocalLevelJointVariance,plot_joint_variances
from model_pipelining.reporting import plot_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--snapshot-id');selection.add_argument('--tag')
    parser.add_argument('--store',default='data/versioned')
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/local_level_joint_priors.json'))
    parser.add_argument('--interval-levels',nargs='+',type=float,default=(.8,.95))
    parser.add_argument('--output-directory',required=True)
    options=vars(parser.parse_args())
    config=json.loads(options.pop('config').read_text())['reference']
    model=LocalLevelJointVariance.from_configuration(config)
    spec=options.pop('experiment')
    output=run_backtest(**options,model=model,experiment=ExperimentSpec.load(spec) if spec else None)
    print(f'Saved: {output}; charts: {plot_run(output)}, {plot_joint_variances(output)}')
    return 0 if json.loads((output/'manifest.json').read_text())['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
