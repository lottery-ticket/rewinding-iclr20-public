#!/usr/bin/env python3

import argparse
import subprocess
import run
import os

def run_iterative(method, trial, version, retrain_epochs, base_dir):
    for i in range(1, 27):
        density = '{}'.format(float('{:.2f}'.format(100*0.8**i)))
        subprocess.check_call([
            './run.py', method,
            '--trial', trial,
            '--version', version,
            '--retrain-epochs', retrain_epochs,
            '--density', density,
            '--base-dir', base_dir,
            '--iteration-count', str(i),
        ])
        base_dir = os.path.dirname(run._get_path(method, trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--retrain-epochs', required=True)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--method', required=True)

    args = parser.parse_args()

    run_iterative(args.method, args.trial, args.version, args.retrain_epochs, args.base_dir)

if __name__ == '__main__':
    main()
