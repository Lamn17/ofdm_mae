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

def ablation(cfg, seed, directory, cache):
    initialize(cfg, seed)
    device = device_for(cfg)
    models, _, static, classical = load_bundle(cfg, directory, device)
    train = rollout_training(cfg, models['mae_random'], seed, 'train', cfg.train_samples, device, cache)
    test = rollout_training(cfg, models['mae_random'], seed, 'test', max(32,cfg.train_samples), device, cache)
    rows = []
    for feature_set in ('temporal','frequency','innovation','plus_snr','full'):
        model = train_predictor(cfg, train, seed, device, feature_set)
        error, weight = 0., 0.
        for f,y,w in zip(*test[:3]):
            error += float(np.sum((predict(model,f,device)-y)**2*w))
            weight += float(w.sum())
        estimator=MaskedEstimator(cfg).to(device)
        estimator.load_state_dict(models['mae_random'].state_dict())
        policy_data=rollout_training(cfg,estimator,seed,'train',cfg.train_samples,device,cache,model)
        train_estimator(cfg,estimator,policy_data[3],policy_data[4],seed+1,cfg.finetune_epochs,
                        policy='causal',masks=policy_data[5],device=device)
        evaluation=rollout_training(cfg,estimator,seed,'test',max(32,cfg.train_samples),device,cache,model,return_estimates=True)
        for modulation in cfg.modulations:
            links=[measure_link(h,est,m,snr,modulation,seed_for(seed,'A1_link',i,modulation),
                                cfg.code_interleaver)
                   for i,(h,snr,m,est) in enumerate(zip(evaluation[3],evaluation[4],evaluation[5],evaluation[6]))]
            rows.append(dict(experiment='feature_ablation',seed=seed,feature_set=feature_set,modulation=modulation,
                             prediction_mse=error/max(weight,1),
                             nmse=float(np.mean(np.sum(np.abs(evaluation[3]-evaluation[6])**2,axis=(1,2))/
                                                 np.sum(np.abs(evaluation[3])**2,axis=(1,2)))),
                             ber=sum(x['bit_errors'] for x in links)/sum(x['bits'] for x in links),
                             bler=sum(x['block_errors'] for x in links)/len(links),blocks=len(links)))
        save_model(estimator, Path(directory) / f'ablation_estimator_{feature_set}.pt')
        save_model(model, Path(directory) / f'predictor_{feature_set}.pt')
    write_csv(Path(directory) / 'feature_ablation.csv', rows)
    return rows
