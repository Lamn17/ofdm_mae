"""Offline training and validation; test trajectory IDs are never read here."""
import runtime as _runtime
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from channels.generation import trajectory, seed_for, complex_noise
from models import MaskedEstimator, ErrorPredictor, matched_cnn, masked_loss, as_tensor
from pilots.policies import (causal_features, select_mask, budget_count, patch_mean, expand_patch,
                       nr_masks, uniform_order, condition_on_mask)
from link.baselines import interpolate, LMMSE


def device_for(cfg):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else torch.device(cfg.device)


def initialize(cfg, seed):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    np.random.seed(seed)


def infer(model, ls, observed, device):
    with torch.no_grad():
        out = model(as_tensor(ls[None], device), torch.as_tensor(observed[None, None], dtype=torch.float32, device=device))[0]
    out = out.detach().cpu().numpy()
    return (out[0] + 1j * out[1]).astype(np.complex64)


def infer_batch(model, ls, observed, device, chunk=1024):
    """Batched infer(): row i equals infer(model, ls[i], observed[i], device)."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(ls), chunk):
            out = model(as_tensor(ls[start:start + chunk], device),
                        torch.as_tensor(observed[start:start + chunk, None], dtype=torch.float32, device=device))
            out = out.cpu().numpy()
            outputs.append((out[:, 0] + 1j * out[:, 1]).astype(np.complex64))
    return np.concatenate(outputs)


def predict(predictor, features, device):
    with torch.no_grad():
        return predictor(torch.from_numpy(features[None]).to(device))[0].cpu().numpy()


def sample_channels(cfg, seed, split='train', profiles=None, count=None, cache=None):
    ids = getattr(cfg, split + '_ids')
    rng = np.random.default_rng(seed_for(seed, split, 'samples', profiles))
    profiles = profiles or cfg.profiles
    channels, snrs = [], []
    for _ in range(count or cfg.train_samples):
        profile = str(rng.choice(profiles))
        velocity = int(rng.choice(cfg.velocities))
        tid = int(rng.integers(*ids))
        frames = trajectory(cfg, profile, velocity, tid, seed, cache)
        channels.append(frames[int(rng.integers(cfg.frames))])
        snrs.append(float(rng.choice(cfg.training_snrs)))
    return np.asarray(channels), np.asarray(snrs)


def train_estimator(cfg, model, channels, snrs, seed, epochs, policy='random', masks=None, clean=False,
                    static_order=None, device=None, objective='masked'):
    device = device or device_for(cfg)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=1e-5)
    log = []
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(channels), cfg.batch_size):
            # Re-permute once per epoch, not separately per batch.
            if start == 0:
                order = rng.permutation(len(channels))
            ids = order[start:start + cfg.batch_size]
            h = channels[ids]
            if masks is not None:
                observed = masks[ids]
            else:
                sampled_policies = rng.choice(['random', 'uniform', 'policy', 'genie'], len(ids),
                                              p=[.4, .3, .2, .1]) if policy == 'mixture' else [policy] * len(ids)
                # Round zero has no learned policy/genie cache, so those mixture arms fall back to random.
                observed = np.array([select_mask(cfg, budget_count(float(rng.choice(cfg.densities)), cfg),
                            'random' if p in ('policy', 'genie') else p, rng,
                            static_order=static_order) for p in sampled_policies])
            noise = np.array([complex_noise(rng, cfg.shape, 0. if clean else 10**(-snrs[i] / 10)) for i in ids])
            ls = as_tensor((h + noise) * observed, device)
            target = as_tensor(h, device)
            mask = torch.as_tensor(observed[:, None], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            prediction = model(ls, mask)
            loss = masked_loss(prediction, target, mask) if objective == 'masked' else (prediction - target).square().sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        log.append(float(np.mean(losses)))
        scheduler.step()
        print(f'estimator {policy} epoch {epoch + 1}/{epochs}: {log[-1]:.5g}', flush=True)
    model.eval()
    return log


def learn_static(cfg, model, channels, snrs, seed, device):
    """Train-only straight-through relaxed top-K scores, shared over all frames.

    Distinct RE selections (no duplicate-pilot artifact). Uses a frozen pretrained
    reconstruction network, then a separate static-policy estimator is fine-tuned.
    """
    rng = np.random.default_rng(seed)
    logits = torch.nn.Parameter(torch.zeros(cfg.nf * cfg.nt, device=device))
    optimizer = torch.optim.Adam([logits], lr=.03)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for epoch in range(cfg.finetune_epochs):
        temperature = max(.1, 1 - epoch / max(cfg.finetune_epochs, 1))
        for start in range(0, len(channels), cfg.batch_size):
            h = channels[start:start + cfg.batch_size]
            k = budget_count(float(rng.choice(cfg.densities)), cfg)
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            perturbed = logits - torch.log(-torch.log(u))
            hard = torch.zeros_like(logits).scatter(0, perturbed.topk(k).indices, 1.)
            soft = torch.sigmoid((perturbed - perturbed.topk(k).values[-1].detach()) / temperature)
            mask = (hard - soft.detach() + soft).reshape(1, 1, *cfg.shape).expand(len(h), -1, -1, -1)
            noise = np.array([complex_noise(rng, cfg.shape, 10**(-s / 10)) for s in snrs[start:start+len(h)]])
            ls = as_tensor(h + noise, device)
            optimizer.zero_grad()
            loss = (model(ls, mask) - as_tensor(h, device)).square().sum(1).mean()
            loss.backward()
            optimizer.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return logits.detach().cpu().numpy().argsort()[::-1].copy()


def rollout_training(cfg, estimator, seed, split, count, device, cache, predictor=None, return_estimates=False):
    rng = np.random.default_rng(seed_for(seed, split, 'rollout'))
    features_all, targets, valid, channels, snrs, masks = [], [], [], [], [], []
    estimates = []
    while len(channels) < count:
        profile, velocity = str(rng.choice(cfg.profiles)), int(rng.choice(cfg.velocities))
        tid, snr = int(rng.integers(*getattr(cfg, split + '_ids'))), float(rng.choice(cfg.training_snrs))
        sequence = trajectory(cfg, profile, velocity, tid, seed, cache)
        history = []
        for t, h in enumerate(sequence):
            noise = complex_noise(rng, cfg.shape, 10**(-snr / 10))
            k = budget_count(float(rng.choice(cfg.densities)), cfg)
            features = None
            if t >= cfg.history + 1:
                features, _, _ = causal_features(history[-cfg.history-1:], snr, cfg.symbol_duration)
            if features is not None and predictor is not None:
                initial = np.zeros(cfg.shape, bool)
                corners = [(cfg.nf//4,cfg.nt//4),(cfg.nf//4,3*cfg.nt//4),
                           (3*cfg.nf//4,cfg.nt//4),(3*cfg.nf//4,3*cfg.nt//4)]
                for f, n in corners: initial[min(f,cfg.nf-1),min(n,cfg.nt-1)] = True
                scores = predict(predictor, condition_on_mask(features, initial), device)
                mask = select_mask(cfg, k, 'causal', scores=scores)
            else:
                mask = select_mask(cfg, k, 'random', rng)
            estimate = infer(estimator, (h + noise) * mask, mask, device)
            if features is not None:
                unobserved = (~mask).astype(float)
                denominator = patch_mean(unobserved, cfg)
                target = patch_mean(np.abs(h - estimate)**2 * unobserved, cfg) / np.maximum(denominator, 1e-12)
                features_all.append(condition_on_mask(features, mask))
                targets.append(target)
                valid.append(denominator > 0)
                channels.append(h)
                snrs.append(snr)
                masks.append(mask)
                estimates.append(estimate)
            history.append(estimate)
            if len(channels) >= count:
                break
    result = tuple(np.asarray(x) for x in (features_all, targets, valid, channels, snrs, masks))
    return result + (np.asarray(estimates),) if return_estimates else result


def train_predictor(cfg, data, seed, device, feature_set='full'):
    torch.manual_seed(seed)
    model = ErrorPredictor(cfg, feature_set).to(device)
    features, targets, valid = data[:3]
    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(targets).float(), torch.from_numpy(valid).float())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(cfg.predictor_epochs):
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            predicted = model(x)
            # Counterfactual O'=O union one random candidate. Its target is not needed
            # for the monotonicity constraint, which regularizes total predicted error.
            x_added = x.clone()
            flat = x_added[:, 5].flatten(1)
            candidate = torch.randint(flat.shape[1], (len(x),), device=x.device)
            flat[torch.arange(len(x), device=x.device), candidate] = 1.
            predicted_added = model(x_added)
            # Ignore all-observed patches in the supervised term.
            supervised = ((torch.log(predicted + 1e-6) - torch.log(y + 1e-6)).square() * w).sum() / w.sum().clamp_min(1)
            mono = torch.relu(predicted_added.sum(1) - predicted.sum(1)).mean()
            loss = supervised + .1 * mono
            loss.backward()
            optimizer.step()
    return model.eval()


def save_model(model, path):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_bundle(cfg, directory, device):
    directory = Path(directory)
    models = {}
    for policy in ('random', 'uniform', 'static', 'causal', 'oracle', 'nr'):
        model = MaskedEstimator(cfg).to(device)
        model.load_state_dict(torch.load(directory / f'mae_{policy}.pt', map_location=device, weights_only=True))
        models['mae_' + policy] = model.eval()
    for policy in ('random', 'uniform', 'static', 'causal', 'oracle'):
        cnn = matched_cnn(cfg).to(device)
        cnn.load_state_dict(torch.load(directory / f'cnn_{policy}.pt', map_location=device, weights_only=True))
        models['cnn_' + policy] = cnn.eval()
    predictor = ErrorPredictor(cfg).to(device)
    predictor.load_state_dict(torch.load(directory / 'predictor.pt', map_location=device, weights_only=True))
    return models, predictor.eval(), np.load(directory / 'static_order.npy'), LMMSE(np.load(directory / 'covariance.npy'))


def genie_mask(cfg, h, count, estimator, rng, device, static_order=None):
    # Finite candidate search with TRUE current CSI loss: genie benchmark, NOT a global bound.
    candidates = [select_mask(cfg, count, 'uniform'), select_mask(cfg, count, 'random', rng)]
    if static_order is not None:
        candidates.append(select_mask(cfg, count, 'static', static_order=static_order))
    candidates += [select_mask(cfg, count, 'random', rng) for _ in range(cfg.oracle_candidates)]
    losses = [np.mean(np.abs(h - infer(estimator, h * m, m, device))**2) for m in candidates]
    return candidates[int(np.argmin(losses))]


def train_bundle(cfg, seed, directory, cache):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    initialize(cfg, seed)
    device = device_for(cfg)
    h, snrs = sample_channels(cfg, seed, cache=cache)
    pretrain = MaskedEstimator(cfg).to(device)
    log = {'mixture': train_estimator(cfg, pretrain, h, snrs, seed, cfg.epochs,
                                      policy='mixture', clean=False, device=device)}
    save_model(pretrain, directory / 'pretrained.pt')
    # v3 uses one mixture-trained f_theta for every placement policy. Keep the old
    # checkpoint names as aliases so completed CLI/report workflows remain readable.
    for policy in ('random', 'uniform', 'static', 'causal', 'oracle', 'nr'):
        save_model(pretrain, directory / f'mae_{policy}.pt')
    np.save(directory / 'covariance.npy', LMMSE.fit(h).covariance)
    static_order = learn_static(cfg, pretrain, h, snrs, seed, device)
    np.save(directory / 'static_order.npy', static_order)
    data = rollout_training(cfg, pretrain, seed, 'train', cfg.predictor_samples, device, cache)
    predictor = train_predictor(cfg, data, seed, device)
    save_model(predictor, directory / 'predictor.pt')
    cnn = matched_cnn(cfg).to(device)
    log['cnn_mixture'] = train_estimator(cfg, cnn, h, snrs, seed, cfg.epochs,
                                         policy='mixture', clean=False, device=device)
    for policy in ('random', 'uniform', 'static', 'causal', 'oracle'):
        save_model(cnn, directory / f'cnn_{policy}.pt')
    (directory / 'training.json').write_text(json.dumps(log, indent=2))
    cfg.save(directory / 'resolved_config.yaml')
    return data
