"""Run explicit dynamics settings on controlled synthetic development fixtures."""
import argparse
import json
from model_pipelining.dynamics_study import run_dynamics_study,plot_dynamics_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evolution-variances',type=float,nargs='+',default=(0,.01,.1,1))
    parser.add_argument('--observation-count',type=int,default=120)
    parser.add_argument('--development-count',type=int,default=100)
    parser.add_argument('--minimum-history',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-directory',required=True)
    options=vars(parser.parse_args());options['evolution_variances']=tuple(options['evolution_variances'])
    output=run_dynamics_study(**options)
    print(f'Saved: {output}; chart: {plot_dynamics_study(output)}')
    return 0 if json.loads((output/'study.json').read_text())['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
