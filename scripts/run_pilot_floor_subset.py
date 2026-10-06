#!/usr/bin/env python3
"""Run the pilot-floor sweep for a chosen subset of estimator/policy pairs.

Example: --pairs lmmse:uniform mae_uniform:uniform --result-name pilot_floor
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import read_config  # noqa: E402
from experiments.pilot_sweeps import fixed_sweep  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='run directory with resolved_config.yaml and models')
    parser.add_argument('--cache', default=PROJECT_DIR / 'cache')
    parser.add_argument('--pairs', nargs='+', required=True, help='method:policy')
    parser.add_argument('--result-name', default='pilot_floor')
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    run = Path(args.output)
    cfg = read_config(run / 'resolved_config.yaml')
    pairs = [tuple(pair.split(':')) for pair in args.pairs]
    for seed in args.seeds or cfg.seeds:
        directory = run / f'seed_{seed}'
        print(fixed_sweep(cfg, seed, directory, args.cache, 'pilot-floor', pairs=pairs,
                          result_name=args.result_name), flush=True)
        (directory / 'pilot-floor.done').write_text('complete\n')


if __name__ == '__main__':
    main()
