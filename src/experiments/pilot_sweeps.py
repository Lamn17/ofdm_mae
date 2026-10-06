import runtime as _runtime
import csv
from collections import deque
import json
import time
from pathlib import Path
import numpy as np
import torch
from channels.generation import trajectory, seed_for, complex_noise
from link.baselines import interpolate
from models import MaskedEstimator, parameter_count, as_tensor, analytical_flops
from link.simulation import measure_link, measure_link_batch, measure_link_grouped, calibration_bler
from pilots.policies import (causal_features, select_mask, budget_count, signaling, nr_masks,
                        expand_patch, patch_mean, BudgetController, condition_on_mask,
                        SelectionState, GphiTopKPolicy, seed_uniform_mask)
from models.training import (device_for, initialize, load_bundle, infer, infer_batch, predict, genie_mask,
                        sample_channels, train_estimator, save_model, rollout_training, train_predictor)
from features.predictability import both_indices
from channels.noise_estimation import leave_one_pilot_out
from pilots.control import aging_product
from experiments.io import StreamingRows

SWEEP_PAIRS = ([('ls', 'uniform'), ('lmmse', 'uniform'), ('cnn_uniform', 'uniform')]
               + [('mae_' + p, p) for p in ('uniform', 'random', 'static', 'causal', 'oracle')])
MODEL_ALIASES = {'mae_causal_spread': 'mae_causal', 'mae_seed_uniform': 'mae_uniform'}

def estimate(method, models, classical, ls, mask, noise_var, device):
    if method == 'ls':
        return interpolate(ls, mask)
    if method == 'lmmse':
        return classical(ls, mask, noise_var)
    return infer(models[method], ls, mask, device)


def reference_indices(cfg, sequence, snr, noises):
    history, values, sigma_history = [], [], []
    mask = select_mask(cfg, budget_count(.5, cfg), 'uniform')
    for t, h in enumerate(sequence):
        if t >= cfg.history + 1:
            true_hist = torch.as_tensor(sequence[t-cfg.history-1:t][None])
            est_hist = torch.as_tensor(np.asarray(history[-cfg.history-1:])[None])
            pi_h, pi_hat = both_indices(true_hist, est_hist, sigma_history[-1],
                                        W=cfg.history, B_win=8)
            _, _, fd_hat = causal_features(history[-cfg.history:], snr, cfg.symbol_duration)
            values.append((float(pi_h.item()), float(pi_hat.item()), float(fd_hat)))
        else:
            values.append((float('nan'), float('nan'), float('nan')))
        ls = (h + noises[t]) * mask
        history.append(interpolate(ls, mask))
        sigma_history.append(leave_one_pilot_out(
            lambda values, visible: interpolate(values, visible), ls, mask,
            np.random.default_rng(seed_for('Pi_hat_LOPO', t, snr))))
    return values


def feasibility(cfg, seed, directory, cache):
    """Cheap feasibility check: can the coded link meet the reliability target?"""
    initialize(cfg, seed)
    rows = StreamingRows(Path(directory) / 'feasibility.csv')
    start, stop = cfg.test_ids
    stop = min(stop, start + cfg.diagnostic_trajectories)
    for profile in cfg.eval_profiles:
        for velocity in cfg.velocities:
            trajectory_ids = list(range(start, stop))
            sequences = np.asarray([trajectory(cfg, profile, velocity, tid, seed, cache)
                                    for tid in trajectory_ids])
            channels = sequences.reshape(-1, cfg.nf, cfg.nt)
            block_ids = [(tid,t) for tid in trajectory_ids for t in range(cfg.frames)]
            for snr in cfg.diagnostic_snrs:
                noises = np.asarray([complex_noise(
                    np.random.default_rng(seed_for(seed, profile, velocity, tid, snr, 'F0_pilot_noise')),
                    sequence.shape, 10**(-snr/10)) for tid,sequence in zip(trajectory_ids,sequences)])
                noises = noises.reshape(channels.shape)
                for density in cfg.diagnostic_densities:
                    mask = select_mask(cfg, budget_count(density, cfg), 'uniform')
                    estimates = (
                        ('perfect_csi', channels),
                        ('ls_dense', np.asarray([interpolate((h+n) * mask, mask)
                                                for h,n in zip(channels,noises)])),
                    )
                    for method, hhat in estimates:
                        nmse = np.sum(np.abs(channels-hhat)**2,axis=(1,2)) / np.maximum(
                            np.sum(np.abs(channels)**2,axis=(1,2)), 1e-12)
                        for modulation in cfg.modulations:
                            seeds = [seed_for(seed, profile, velocity, tid, t, snr, density,
                                              modulation, 'F0_link') for tid,t in block_ids]
                            links = measure_link_batch(channels, hhat, mask, snr, modulation, seeds,
                                                       cfg.code_interleaver)
                            for i, ((tid,t), link) in enumerate(zip(block_ids,links)):
                                row = dict(experiment='feasibility', seed=seed, backend=cfg.backend,
                                           profile=profile, velocity=velocity, trajectory=tid,
                                           frame=t, snr_db=snr, modulation=modulation,
                                           method=method, policy='uniform',
                                           requested_density=density, predictability=0.,
                                           nmse=float(nmse[i]), selector_ms=0., selector_exceeds_frame=False)
                                row.update(link)
                                row.update(signaling(mask, modulation, 'S0', len(cfg.diagnostic_densities)))
                                rows.append(row)
            print(f'feasibility seed={seed} {profile} v={velocity}: {len(trajectory_ids)} trajectories', flush=True)
    return rows.finish()


def estimate_batch(method, models, classical, ls, masks, noise_var, device):
    """Batched estimate(): row i equals estimate(method, ..., ls[i], masks[i], ...)."""
    method = MODEL_ALIASES.get(method, method)
    if method == 'ls':
        return np.asarray([interpolate(x, m) for x, m in zip(ls, masks)])
    if method == 'lmmse':
        return np.asarray([classical(x, m, noise_var) for x, m in zip(ls, masks)])
    return infer_batch(models[method], ls, masks, device)


def _genie_from_pilot_floor(source, destination):
    """genie-bracket is the 12.5% slice of pilot-floor: same pairs, masks, noise and link seeds."""
    rows = StreamingRows(destination)
    with Path(source).open(newline='') as f:
        for row in csv.DictReader(f):
            if float(row['requested_density']) == .125:
                row['experiment'] = 'genie-bracket'
                rows.append(row)
    return rows.finish()


def fixed_sweep(cfg, seed, directory, cache, experiment='pilot-floor', pairs=None, densities=None,
                result_name=None):
    """Pilot-floor sweep; pairs/densities/result_name select a subset written to its own CSV."""
    if experiment == 'genie-bracket' and .125 in cfg.densities and (Path(directory) / 'pilot_floor.csv').exists():
        return _genie_from_pilot_floor(Path(directory) / 'pilot_floor.csv', Path(directory) / 'genie_bracket.csv')
    initialize(cfg, seed)
    device = device_for(cfg)
    models, predictor, static, classical = load_bundle(cfg, directory, device)
    pairs = pairs or SWEEP_PAIRS
    densities = densities or (cfg.densities if experiment == 'pilot-floor' else [.125])
    result_name = result_name or experiment.replace('-', '_')
    rows = StreamingRows(Path(directory) / f'{result_name}.csv')
    tids = list(range(*cfg.test_ids))
    warm = cfg.history + 1
    for profile in cfg.eval_profiles:
        for velocity in cfg.velocities:
            # Trajectories are independent, so all of them advance frame by frame together;
            # each one still only sees its own past estimates.
            sequences = np.asarray([trajectory(cfg, profile, velocity, tid, seed, cache) for tid in tids])
            for snr in cfg.snrs:
                noise_var = 10**(-snr/10)
                noises = np.asarray([complex_noise(np.random.default_rng(seed_for(seed, profile, velocity, tid, snr, 'noise')),
                                                   sequences.shape[1:], noise_var) for tid in tids])
                references = [reference_indices(cfg, sequence, snr, noise) for sequence, noise in zip(sequences, noises)]
                for density in densities:
                    count = budget_count(density, cfg)
                    for method, policy in pairs:
                        history = [[] for _ in tids]
                        for t in range(cfg.frames):
                            h = sequences[:, t]
                            masks = np.empty(h.shape, bool)
                            selector_ms = np.zeros(len(tids))
                            pis = np.full(len(tids), np.nan)
                            fds = np.full(len(tids), np.nan)
                            candidates = []
                            for i, tid in enumerate(tids):
                                begin = time.perf_counter()
                                rng_mask = np.random.default_rng(seed_for(seed, profile, velocity, tid, t, density, 'mask'))
                                if t < warm:
                                    masks[i] = select_mask(cfg, count, 'uniform')
                                else:
                                    features, pis[i], fds[i] = causal_features(history[i][-warm:], snr, cfg.symbol_duration)
                                    if policy in ('causal', 'causal_spread'):
                                        state = SelectionState(np.asarray(history[i][-warm:]),
                                            features[0], features[1], features[2], snr, fds[i])
                                        masks[i] = GphiTopKPolicy(cfg, lambda x: predict(predictor, x, device),
                                                                  spread=policy == 'causal_spread').select(state, count)
                                    elif policy == 'seed_uniform':
                                        masks[i] = seed_uniform_mask(cfg, count)
                                    elif policy == 'oracle':
                                        options = [select_mask(cfg, count, p, rng_mask, static_order=static)
                                                   for p in ('uniform', 'random', 'static')]
                                        options += [select_mask(cfg, count, 'random', rng_mask) for _ in range(cfg.oracle_candidates)]
                                        candidates.append(options)
                                    else:
                                        masks[i] = select_mask(cfg, count, policy, rng_mask, static_order=static)
                                selector_ms[i] = 1000 * (time.perf_counter() - begin)
                            if candidates:
                                # Finite-candidate genie: score every candidate with noiseless TRUE current CSI.
                                begin = time.perf_counter()
                                options = np.asarray(candidates)
                                per = options.shape[1]
                                truth = np.repeat(h, per, axis=0)
                                flat = options.reshape(-1, *cfg.shape)
                                losses = np.mean(np.abs(truth - estimate_batch(method, models, classical, truth * flat,
                                                                                 flat, noise_var, device))**2, axis=(1, 2))
                                masks[:] = options[np.arange(len(tids)), losses.reshape(len(tids), per).argmin(1)]
                                # Amortized per-trajectory share of the batched genie search.
                                selector_ms += 1000 * (time.perf_counter() - begin) / len(tids)
                            hhat = estimate_batch(method, models, classical, (h + noises[:, t]) * masks, masks, noise_var, device)
                            for i in range(len(tids)):
                                history[i].append(hhat[i])
                            if t < warm:
                                continue
                            nmse = np.sum(np.abs(h - hhat)**2, axis=(1, 2)) / np.maximum(np.sum(np.abs(h)**2, axis=(1, 2)), 1e-12)
                            for modulation in cfg.modulations:
                                links = measure_link_grouped(h, hhat, masks, snr, modulation,
                                    [seed_for(seed, profile, velocity, tid, t, snr, modulation, 'link') for tid in tids],
                                    cfg.code_interleaver)
                                for i, (tid, link) in enumerate(zip(tids, links)):
                                    reference = references[i][t]
                                    row = dict(experiment=experiment, seed=seed, backend=cfg.backend, profile=profile,
                                               velocity=velocity, trajectory=tid, frame=t, snr_db=snr,
                                               modulation=modulation, method=method, policy=policy,
                                               requested_density=density, predictability=reference[0],
                                               Pi_H=reference[0], Pi_hat=reference[1],
                                               Pi_hat_bias=reference[1]-reference[0],
                                               policy_predictability=float(pis[i]), doppler_hat=float(fds[i]),
                                               reference_doppler_hat=reference[2], nmse=float(nmse[i]),
                                               selector_ms=float(selector_ms[i]),
                                               selector_exceeds_frame=bool(selector_ms[i] > cfg.nt*cfg.symbol_duration*1000),
                                               oracle_kind='finite_candidate_genie' if policy == 'oracle' else 'none')
                                    row.update(link)
                                    adaptive = policy in ('causal', 'causal_spread', 'oracle')
                                    row.update(signaling(masks[i], modulation, 'S2' if adaptive else 'S0', len(cfg.densities)))
                                    rows.append(row)
                print(f'{experiment} seed={seed} {profile} v={velocity} snr={snr}: {len(tids)} trajectories', flush=True)
    return rows.finish()
