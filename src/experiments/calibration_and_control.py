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
from experiments.pilot_sweeps import reference_indices

def _calibration_snr_grid(cfg):
    low = int(np.floor(min(cfg.snrs + cfg.diagnostic_snrs)))
    high = int(np.ceil(max(cfg.snrs + cfg.diagnostic_snrs)))
    return sorted(set(range(low, high + 1, 2)) | {high})


def _cdl_perfect_csi_curve(cfg, seed, cache, profile, velocity, mask, modulation, snr_grid):
    """Validation-only physical-link BLER curve used after the NMSE-to-SNR mapping."""
    start, stop = cfg.val_ids
    stop = min(stop, start + cfg.calibration_trajectories)
    channels = [h for tid in range(start, stop)
                for h in trajectory(cfg, profile, velocity, tid, seed, cache)]
    if not channels:
        raise ValueError('No validation channels available for CDL calibration')
    curves = []
    for snr in snr_grid:
        batch = np.asarray([channels[block % len(channels)] for block in range(cfg.calibration_blocks)])
        seeds = [seed_for(seed, profile, velocity, block, snr, modulation,
                          int(mask.sum()), 'cdl_cal') for block in range(cfg.calibration_blocks)]
        results = measure_link_batch(batch, batch, mask, snr, modulation, seeds,
                                     cfg.code_interleaver)
        errors = sum(result['block_errors'] for result in results)
        curves.append((errors + .5) / (cfg.calibration_blocks + 1))
    return np.minimum.accumulate(curves).tolist()


def controller_seed_mask(cfg, options):
    """Sparse probe mask that conditions g_phi when the budget controller scores a frame."""
    return select_mask(cfg, min(4, int(options[-1][1].sum())), 'uniform')


def calibration_for(calibration, variant, modulation, profile, velocity):
    """Read new stratified calibration, with compatibility for completed v2 runs."""
    key = f'{variant}_{modulation}_{profile}_{velocity}'
    if key in calibration:
        return calibration[key]
    legacy = f'{variant}_{modulation}'
    if legacy in calibration:
        return calibration[legacy]
    raise KeyError(f'Missing calibration for {variant}, M={modulation}, {profile}, v={velocity}')


def calibrate(cfg, seed, directory, cache):
    """K-dependent NMSE scale plus AWGN or profile/velocity-stratified CDL FEC curves."""
    initialize(cfg, seed)
    device = device_for(cfg)
    models, predictor, static, classical = load_bundle(cfg, directory, device)
    rng = np.random.default_rng(seed_for(seed, 'calibrate'))
    output = {}
    # Reuse actual validation history from frozen estimator. Truth is ONLY a target.
    # 1024 validation frames give about 128 per training SNR for the per-SNR scales.
    data = rollout_training(cfg, models['mae_random'], seed, 'val', max(8, min(cfg.train_samples, 1024)), device, cache)
    predicted = np.array([predict(predictor, f, device) for f in data[0]])
    physical_curve_cache = {}
    for variant in cfg.controller_variants:
        options = nr_masks(cfg) if variant == 'NR' else [(str(d), select_mask(cfg, budget_count(d,cfg), 'uniform')) for d in sorted(cfg.budget_levels)]
        masks = [m for _, m in options]
        method = {'S1': 'mae_uniform', 'S2': 'mae_causal', 'NR': 'mae_nr'}[variant]
        # Scales must map the predictor output the controller actually sees, which is
        # conditioned on the sparse probe mask, not on the rollout's own mask; the latter
        # predicts ~8x less error and made the controller ~10 dB too pessimistic.
        probe = controller_seed_mask(cfg, options).astype(np.float32)
        controller_predicted = []
        for f in data[0]:
            probed = f.copy()
            probed[5] = probe
            controller_predicted.append(predict(predictor, probed, device).mean())
        scales = []
        snrs = np.asarray(data[4])
        scales_by_snr = {str(float(v)): [] for v in sorted(set(snrs))}
        for i, fixed in enumerate(masks):
            actual, predicted_mean = [], []
            for j, (h, snr) in enumerate(zip(data[3], data[4])):
                m = select_mask(cfg, int(fixed.sum()), 'causal', scores=predicted[j]) if variant == 'S2' else fixed
                noise = complex_noise(rng, cfg.shape, 10**(-snr/10))
                est = infer(models[method], (h+noise)*m, m, device)
                actual.append(np.mean(np.abs(h-est)**2))
                predicted_mean.append(controller_predicted[j])
            scales.append(float(np.mean(actual) / max(np.mean(predicted_mean), 1e-12)))
            # Pooling over -5..30 dB is dominated by low-SNR errors; fit each SNR separately too.
            actual, predicted_mean = np.asarray(actual), np.asarray(predicted_mean)
            for key in scales_by_snr:
                at = snrs == float(key)
                scales_by_snr[key].append(float(actual[at].mean() / max(predicted_mean[at].mean(), 1e-12)))
        # No enforced monotonicity for error scales: measure what the estimator actually does.
        for modulation in cfg.modulations:
            snr_grid = _calibration_snr_grid(cfg)
            if cfg.calibration_source == 'awgn':
                curves = []
                for i, m in enumerate(masks):
                    errors = []
                    for snr in snr_grid:
                        count = calibration_bler(int((~m).sum()), modulation, snr, cfg.calibration_blocks,
                                                 seed_for(seed, 'awgn_cal', modulation, i, snr))
                        errors.append((count + .5) / (cfg.calibration_blocks + 1))
                    curves.append(np.minimum.accumulate(errors).tolist())
                output[f'{variant}_{modulation}'] = dict(
                    nmse_scales=scales, nmse_scales_by_snr=scales_by_snr, snr_db=snr_grid, bler=curves,
                    source='validation_nmse_and_awgn_convolutional_bler',
                    blocks_per_point=cfg.calibration_blocks)
            else:
                for profile in cfg.eval_profiles:
                    for velocity in cfg.velocities:
                        curves = []
                        for m in masks:
                            curve_key = (profile, velocity, modulation, m.tobytes())
                            if curve_key not in physical_curve_cache:
                                physical_curve_cache[curve_key] = _cdl_perfect_csi_curve(
                                    cfg, seed, cache, profile, velocity, m, modulation, snr_grid)
                            curves.append(physical_curve_cache[curve_key])
                        output[f'{variant}_{modulation}_{profile}_{velocity}'] = dict(
                            nmse_scales=scales, nmse_scales_by_snr=scales_by_snr, snr_db=snr_grid, bler=curves,
                            source='validation_cdl_perfect_csi_convolutional_bler',
                            blocks_per_point=cfg.calibration_blocks,
                            profile=profile, velocity=velocity)
    path = Path(directory) / 'calibration.json'
    path.write_text(json.dumps(output, indent=2))
    return output


def closed_loop(cfg, seed, directory, cache):
    initialize(cfg, seed)
    device = device_for(cfg)
    models, predictor, static, classical = load_bundle(cfg, directory, device)
    calibration_path = Path(directory) / 'calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else calibrate(cfg, seed, directory, cache)
    rows = StreamingRows(Path(directory) / 'closed_loop.csv')
    tids = list(range(*cfg.test_ids))
    warm = cfg.history + 1
    for profile in cfg.eval_profiles:
        for velocity in cfg.velocities:
            sequences = np.asarray([trajectory(cfg, profile, velocity, tid, seed, cache) for tid in tids])
            fd_hz = velocity / 3.6 * cfg.carrier_frequency / 299792458.
            for snr in cfg.snrs:
                noises = np.asarray([complex_noise(np.random.default_rng(seed_for(seed, profile, velocity, tid, snr, 'noise')),
                                                   sequences.shape[1:], 10**(-snr/10)) for tid in tids])
                references = [reference_indices(cfg, sequence, snr, noise) for sequence, noise in zip(sequences, noises)]
                for modulation in cfg.modulations:
                    variants = list(cfg.controller_variants) + [f'S0:{d}' for d in cfg.densities]
                    for variant in variants:
                        is_static = variant.startswith('S0')
                        delays = [0] if is_static else cfg.loop_delays
                        # kappa only prices signaling; the controller never reads it, so one
                        # rollout per delay serves every kappa.
                        kappas = [3.] if is_static else cfg.kappa_values
                        options = nr_masks(cfg) if variant == 'NR' else [(str(d), select_mask(cfg, budget_count(d,cfg),'uniform')) for d in sorted(cfg.budget_levels)]
                        method = {'S2': 'mae_causal', 'NR': 'mae_nr'}.get(variant, 'mae_uniform')
                        fixed = (variant, select_mask(cfg, budget_count(float(variant.split(':')[1]), cfg), 'uniform')) if is_static else None
                        for delay in delays:
                            controllers = [None if is_static else BudgetController(
                                cfg.target_bler, cfg.olla_up,
                                calibration_for(calibration, variant, modulation, profile, velocity)) for _ in tids]
                            history = [[] for _ in tids]
                            queues = [deque([options[-1]] * delay) for _ in tids]
                            for t in range(cfg.frames):
                                h = sequences[:, t]
                                masks = np.empty(h.shape, bool)
                                chosen = [None] * len(tids)
                                selector_ms = np.zeros(len(tids))
                                for i, controller in enumerate(controllers):
                                    start = time.perf_counter()
                                    if is_static:
                                        fresh = fixed
                                    elif t < warm:
                                        fresh = options[-1]
                                    else:
                                        past = history[i][-warm:]
                                        features, _, fd_hat = causal_features(past, snr, cfg.symbol_duration)
                                        seed_mask = controller_seed_mask(cfg, options)
                                        scores = predict(predictor, condition_on_mask(features, seed_mask), device)
                                        state = SelectionState(np.asarray(past), features[0], features[1], features[2], snr, fd_hat)
                                        candidates = [GphiTopKPolicy(cfg, lambda x: predict(predictor, x, device)).select(
                                            state, int(m.sum())) for _,m in options] if variant == 'S2' else [m for _,m in options]
                                        index = controller.choose(scores, candidates, snr, np.mean(np.abs(history[i][-1])**2))
                                        fresh = (options[index][0], candidates[index])
                                    if delay:
                                        chosen[i], masks[i] = queues[i].popleft()
                                        queues[i].append(fresh)
                                    else:
                                        chosen[i], masks[i] = fresh
                                    selector_ms[i] = 1000 * (time.perf_counter() - start)
                                est = infer_batch(models[method], (h + noises[:, t]) * masks, masks, device)
                                for i in range(len(tids)):
                                    history[i].append(est[i])
                                links = measure_link_grouped(h, est, masks, snr, modulation,
                                    [seed_for(seed, profile, velocity, tid, t, snr, modulation, 'link') for tid in tids],
                                    cfg.code_interleaver)
                                deltas = [controller.delta if controller else 0. for controller in controllers]
                                for controller, link in zip(controllers, links):
                                    if controller:
                                        controller.update(link['block_errors'])
                                if t < cfg.closed_loop_warmup:
                                    continue
                                nmse = np.sum(np.abs(h - est)**2, axis=(1, 2)) / np.maximum(np.sum(np.abs(h)**2, axis=(1, 2)), 1e-12)
                                aging = float(aging_product(fd_hz, delay, cfg.frame_duration))
                                for kappa in kappas:
                                    for i, (tid, link) in enumerate(zip(tids, links)):
                                        reference = references[i][t]
                                        row = dict(experiment='closed_loop', seed=seed, backend=cfg.backend, profile=profile,
                                                   velocity=velocity, trajectory=tid, frame=t, snr_db=snr, modulation=modulation,
                                                   method=method, policy=variant, selected_action=chosen[i],
                                                   d_frames=delay, aging=aging,
                                                   kappa=kappa, predictability=reference[0], Pi_H=reference[0],
                                                   Pi_hat=reference[1], Pi_hat_bias=reference[1]-reference[0],
                                                   olla_delta_db=deltas[i], nmse=float(nmse[i]),
                                                   selector_ms=float(selector_ms[i]),
                                                   selector_exceeds_frame=bool(selector_ms[i] > cfg.frame_duration*1000))
                                        row.update(link)
                                        row.update(signaling(masks[i], modulation, 'S0' if is_static else ('S2' if variant=='S2' else 'S1'),
                                                             len(options), kappa=kappa, groups=cfg.signaling_groups,
                                                             levels_per_group=cfg.signaling_levels))
                                        rows.append(row)
                print(f'closed-loop seed={seed} {profile} v={velocity} snr={snr}: {len(tids)} trajectories', flush=True)
    return rows.finish()
