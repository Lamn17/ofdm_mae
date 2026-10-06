"""Causal reconstruction-error variance estimators."""
from __future__ import annotations
import runtime as _runtime

import numpy as np
from scipy.ndimage import uniform_filter


def leave_one_pilot_out(estimator, H_ls: np.ndarray, observed: np.ndarray, rng=None, fraction=0.1) -> float:
    observed = np.asarray(observed, dtype=bool)
    ids = np.flatnonzero(observed)
    if not len(ids):
        raise ValueError("LOPO requires observed pilots")
    rng = rng or np.random.default_rng(0)
    count = max(1, int(np.ceil(fraction * len(ids))))
    held = rng.choice(ids, size=min(count, len(ids)), replace=False)
    kept = observed.copy()
    kept.flat[held] = False
    estimate = estimator(np.asarray(H_ls) * kept, kept)
    return float(np.mean(np.abs(np.asarray(estimate).flat[held] - np.asarray(H_ls).flat[held]) ** 2))


def smoothed_ls_residual(H_ls: np.ndarray, observed: np.ndarray, size=(5, 3)) -> float:
    """Cheap causal fallback; never consumes true CSI."""
    observed = np.asarray(observed, dtype=bool)
    if not observed.any():
        raise ValueError("residual estimator requires observed pilots")
    real = uniform_filter(np.asarray(H_ls).real, size=size, mode="nearest")
    imag = uniform_filter(np.asarray(H_ls).imag, size=size, mode="nearest")
    return float(np.mean(np.abs(np.asarray(H_ls)[observed] - (real + 1j * imag)[observed]) ** 2))
