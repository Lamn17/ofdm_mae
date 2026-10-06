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

def complexity(cfg, seed, directory):
    device = device_for(cfg)
    models, predictor, _, _ = load_bundle(cfg, directory, device)
    mask = select_mask(cfg,budget_count(.125,cfg),'random',np.random.default_rng(seed))
    ls = as_tensor(np.ones((1,*cfg.shape),np.complex64)*mask,device)
    observed = torch.as_tensor(mask[None,None],dtype=torch.float32,device=device)
    rows=[]
    for name,model in list(models.items())+[('predictor',predictor)]:
        args = (torch.zeros(1,7,*cfg.shape,device=device),) if name=='predictor' else (ls,observed)
        with torch.no_grad():
            for _ in range(3):
                model(*args)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(10): model(*args)
            if device.type=='cuda': torch.cuda.synchronize()
            latency=(time.perf_counter()-start)*100
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
                model(*args)
        rows.append(dict(seed=seed,model=name,parameters=parameter_count(model),
                         profiler_flops=sum(e.flops for e in prof.key_averages()),
                         analytical_matmul_conv_flops=analytical_flops(model,observed),latency_ms=latency,
                         device=str(device),flops_note='partial_operator_count_attention_may_be_unaccounted'))
    write_csv(Path(directory)/'complexity.csv',rows)
    return rows
