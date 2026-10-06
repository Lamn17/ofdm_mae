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
from experiments.io import write_csv

def transfer(cfg, seed, directory, cache):
    initialize(cfg, seed)
    device = device_for(cfg)
    h, snrs = sample_channels(cfg, seed, profiles=['A','B','D'], cache=cache)
    pretrain = MaskedEstimator(cfg).to(device)
    train_estimator(cfg, pretrain, h, snrs, seed, cfg.epochs, clean=True, device=device)
    initial = {k: v.detach().cpu().clone() for k,v in pretrain.state_dict().items()}
    target_h, target_snr = sample_channels(cfg, seed, profiles=['C'], cache=cache)
    test_h, test_snr = sample_channels(cfg, seed, split='test', profiles=['C'], count=max(32, cfg.train_samples), cache=cache)
    rows = []
    for fraction in cfg.transfer_fractions:
        size = max(1, int(np.ceil(len(target_h)*fraction)))
        for mode in ('pretrained', 'scratch'):
            torch.manual_seed(seed)
            model = MaskedEstimator(cfg).to(device)
            if mode == 'pretrained':
                model.load_state_dict(initial)
            # Match total optimizer EXAMPLES across arms, including pretraining cost.
            updates = cfg.epochs * len(h) + cfg.finetune_epochs * size
            if mode == 'pretrained':
                fine_h, fine_snr = target_h[:size], target_snr[:size]
                epochs = cfg.finetune_epochs
            else:
                ids = np.arange(updates) % size
                fine_h, fine_snr = target_h[ids], target_snr[ids]
                epochs = 1
            train_estimator(cfg, model, fine_h, fine_snr, seed+1, epochs, device=device, objective='masked' if mode=='pretrained' else 'supervised')
            save_model(model, Path(directory) / f'E4_{mode}_{fraction}.pt')
            for i, (channel, snr) in enumerate(zip(test_h, test_snr)):
                rng = np.random.default_rng(seed_for(seed, 'E4_eval', i))
                mask = select_mask(cfg, budget_count(.125,cfg), 'random', rng)
                noise = complex_noise(rng,cfg.shape,10**(-snr/10))
                est = infer(model,(channel+noise)*mask,mask,device)
                rows.append(dict(experiment='transfer',seed=seed,backend=cfg.backend,profile='C',fraction=fraction,mode=mode,
                                 sample=i,labeled_samples=size,optimizer_examples=updates,
                                 nmse=float(np.sum(np.abs(channel-est)**2)/np.sum(np.abs(channel)**2))))
    write_csv(Path(directory) / 'transfer.csv', rows)
    return rows
