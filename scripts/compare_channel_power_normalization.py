#!/usr/bin/env python3
"""Does per-trajectory power normalisation make slow channels look easier?

Regenerates test trajectories without normalisation and compares coded BLER with
perfect CSI and with dense LS under (a) per-trajectory normalisation, as used in the
experiments, and (b) one global normalisation over all trajectories of a speed.
"""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import runtime as _runtime
import numpy as np
from config import read_config  # noqa: E402
from channels.generation import trajectory_with_metadata, seed_for, complex_noise  # noqa: E402
from link.baselines import interpolate  # noqa: E402
from link.simulation import measure_link_batch  # noqa: E402
from pilots.policies import select_mask, budget_count  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='run directory with resolved_config.yaml')
    parser.add_argument('--trajectories', type=int, default=50)
    parser.add_argument('--velocities', type=float, nargs='+', default=[3, 120])
    parser.add_argument('--snrs', type=float, nargs='+', default=[15, 20, 25, 30])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run = Path(args.output)
    cfg = read_config(run / 'resolved_config.yaml')
    tids = list(range(cfg.test_ids[0], cfg.test_ids[0] + args.trajectories))
    dense = select_mask(cfg, budget_count(.5, cfg), 'uniform')
    rows = []
    for profile in cfg.eval_profiles:
        for velocity in args.velocities:
            raw = np.asarray([trajectory_with_metadata(cfg, profile, velocity, tid, args.seed, normalize=False)[0]
                              for tid in tids])
            power = np.mean(np.abs(raw) ** 2, axis=(1, 2, 3))
            variants = {'per_trajectory': raw / np.sqrt(power)[:, None, None, None],
                        'global': raw / np.sqrt(power.mean())}
            for name, sequences in variants.items():
                channels = sequences.reshape(-1, cfg.nf, cfg.nt)
                for snr in args.snrs:
                    noise = np.asarray([complex_noise(np.random.default_rng(seed_for(args.seed, profile, velocity, i, snr, 'norm')),
                                                      cfg.shape, 10 ** (-snr / 10)) for i in range(len(channels))])
                    estimates = {'perfect_csi': channels,
                                 'ls_dense': np.asarray([interpolate((h + n) * dense, dense) for h, n in zip(channels, noise)])}
                    for method, hhat in estimates.items():
                        for modulation in cfg.modulations:
                            seeds = [seed_for(args.seed, profile, velocity, i, snr, modulation, 'norm_link')
                                     for i in range(len(channels))]
                            links = measure_link_batch(channels, hhat, dense, snr, modulation, seeds, cfg.code_interleaver)
                            errors = sum(r['block_errors'] for r in links)
                            rows.append(dict(profile=profile, velocity=velocity, normalisation=name, method=method,
                                             snr_db=snr, modulation=modulation, blocks=len(links), block_errors=errors,
                                             bler=errors / len(links),
                                             trajectory_power_spread_db=float(10 * np.log10(power.max() / power.min()))))
                            print(rows[-1], flush=True)
    out = run / 'normalisation_check.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', out)


if __name__ == '__main__':
    main()
