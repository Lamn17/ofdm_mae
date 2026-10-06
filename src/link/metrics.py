"""Linear-scale, paired metrics and sequential Monte-Carlo stopping."""
from __future__ import annotations
import runtime as _runtime

import numpy as np


def nmse(H, H_hat, axis=(-2, -1)):
    H, H_hat = np.asarray(H), np.asarray(H_hat)
    return np.sum(np.abs(H - H_hat) ** 2, axis=axis) / np.maximum(np.sum(np.abs(H) ** 2, axis=axis), 1e-12)


def nmse_db(H, H_hat, axis=(-2, -1)):
    return 10 * np.log10(np.maximum(nmse(H, H_hat, axis), 1e-30))


def causal_efficiency(mse_random, mse_causal, mse_genie):
    """CE on paired linear MSE samples. It is intentionally not clipped."""
    random, causal, genie = map(np.asarray, (mse_random, mse_causal, mse_genie))
    if random.shape != causal.shape or random.shape != genie.shape:
        raise ValueError("CE arms must contain paired realizations with equal shapes")
    denominator = random.mean() - genie.mean()
    return float((random.mean() - causal.mean()) / denominator) if abs(denominator) > 1e-12 else float("nan")


def bootstrap_ce(mse_random, mse_causal, mse_genie, seed=0, samples=2000, alpha=0.05):
    arrays = [np.asarray(x) for x in (mse_random, mse_causal, mse_genie)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("bootstrap inputs must be paired")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        ids = rng.integers(0, len(arrays[0]), len(arrays[0]))
        values.append(causal_efficiency(*(x[ids] for x in arrays)))
    return np.quantile(values, [alpha / 2, 1 - alpha / 2])


def run_until(sim_fn, max_blocks=200_000, target_errors=100, chunk=2048):
    errors = blocks = 0
    while blocks < max_blocks and errors < target_errors:
        requested = min(chunk, max_blocks - blocks)
        new_errors, new_blocks = sim_fn(requested)
        if not 0 < new_blocks <= requested or not 0 <= new_errors <= new_blocks:
            raise ValueError("sim_fn returned invalid counts")
        errors += int(new_errors)
        blocks += int(new_blocks)
    return {"bler": errors / blocks, "errors": errors, "blocks": blocks,
            "capped": errors < target_errors}
