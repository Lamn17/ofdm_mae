"""Signaling-aware delayed pilot-budget controller."""
from __future__ import annotations
import runtime as _runtime

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Decision:
    K: int
    observed: np.ndarray


class OLLA:
    def __init__(self, target=1e-2, mu_down=0.01, clip=(-6.0, 6.0)):
        if not 0 < target < 1:
            raise ValueError("target must lie in (0,1)")
        self.target, self.mu_down = target, mu_down
        self.mu_up = mu_down * (1 - target) / target
        self.clip = clip
        self.delta_db = 0.0

    def update(self, block_error: bool) -> float:
        step = self.mu_up if block_error else -self.mu_down
        self.delta_db = float(np.clip(self.delta_db + step, *self.clip))
        return self.delta_db

    def reset(self):
        self.delta_db = 0.0


class DelayedController:
    """Delay the complete decision; budget and pattern can never get out of sync."""

    def __init__(self, d: int, default: Decision, decide):
        if d < 0:
            raise ValueError("delay must be non-negative")
        self.d = d
        self.decide = decide
        # A decision made at t applies at t+d. d=0 is the labeled instantaneous reference.
        self.queue = deque([default] * d)

    def step(self, state) -> Decision:
        fresh = self.decide(state)
        if self.d == 0:
            return fresh
        self.queue.append(fresh)
        return self.queue.popleft()


def snr_effective(snr_linear, nmse_pred, delta_db=0.0):
    value = snr_linear / (1.0 + snr_linear * np.asarray(nmse_pred))
    return value * 10.0 ** (delta_db / 10.0)


def aging_product(fd_hz, d_frames, frame_dur_s=0.5e-3):
    return np.asarray(fd_hz) * (d_frames + 1) * frame_dur_s
