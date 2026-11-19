"""Run data-free posterior-CDF calibration and parameter recovery, then plot."""
import argparse
import json
from pathlib import Path
from model_pipelining.adapters import EVIDENCE_MODEL_NAMES, build_model
from model_pipelining.calibration import run_calibration, plot_calibration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=EVIDENCE_MODEL_NAMES)
    parser.add_argument('--config',type=Path,required=True,help='Existing prior configuration; reference settings used')
    parser.add_argument('--observation-count',type=int,default=20)
    parser.add_argument('--trials',type=int,default=200)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--interval-level',type=float,default=.95)
    parser.add_argument('--output-directory',required=True)
    options=vars(parser.parse_args())
    config=json.loads(options.pop('config').read_text())['reference']
    model=build_model(options.pop('model'),config)
    output=run_calibration(model=model,**options)
    print(f'Saved: {output}; chart: {plot_calibration(output)}')
    return 0 if json.loads((output/'calibration.json').read_text())['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
