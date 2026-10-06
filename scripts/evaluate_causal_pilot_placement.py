#!/usr/bin/env python3
"""Causal placement ablation on an existing run: spatial spread and seed geometry.

Evaluates, at the sparse densities, the causal policy with a spread constraint
(causal_spread) and the causal policy's fixed seed plus the uniform sequence
(seed_uniform). Results go to seed_<n>/causal_ablation.csv and use the same
channels, noise and masks as pilot_floor.csv, so they pair with it row by row.
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
    parser.add_argument('--output', required=True, help='existing run directory')
    parser.add_argument('--cache', default=PROJECT_DIR / 'cache')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--max-density', type=float, default=1 / 16)
    args = parser.parse_args()
    run = Path(args.output)
    cfg = read_config(run / 'resolved_config.yaml')
    densities = [d for d in cfg.densities if d <= args.max_density]
    pairs = [('mae_causal_spread', 'causal_spread'), ('mae_seed_uniform', 'seed_uniform')]
    for seed in args.seeds or cfg.seeds:
        directory = run / f'seed_{seed}'
        print(fixed_sweep(cfg, seed, directory, args.cache, 'pilot-floor', pairs=pairs,
                          densities=densities, result_name='causal_ablation'), flush=True)


if __name__ == '__main__':
    main()
