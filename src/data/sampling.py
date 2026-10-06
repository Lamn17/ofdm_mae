"""Trajectory-preserving data structures."""
from __future__ import annotations
import runtime as _runtime

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    H_hist: np.ndarray
    H_cur: np.ndarray
    meta: dict


class FrameWindowSampler:
    """Yield causal windows without ever crossing trajectory boundaries.

    H_hist contains frames ``t-W-1 .. t-1`` and therefore has ``W+1`` entries.
    The extra frame is required by the innovation feature.
    """

    def __init__(self, trajectories: np.ndarray, W: int, metadata: list[dict] | None = None):
        values = np.asarray(trajectories)
        if values.ndim != 4:
            raise ValueError("trajectories must have shape (trajectory, frame, Nf, Nt)")
        if W < 1 or values.shape[1] <= W + 1:
            raise ValueError("each trajectory needs more than W+1 frames")
        self.trajectories = values
        self.W = W
        self.metadata = metadata or [{} for _ in range(len(values))]
        if len(self.metadata) != len(values):
            raise ValueError("metadata length must equal trajectory count")

    def __len__(self) -> int:
        return len(self.trajectories) * (self.trajectories.shape[1] - self.W - 1)

    def __iter__(self) -> Iterator[FrameSample]:
        for trajectory_id, (frames, base_meta) in enumerate(zip(self.trajectories, self.metadata)):
            for t in range(self.W + 1, len(frames)):
                meta = dict(base_meta, trajectory_id=trajectory_id, frame=t)
                yield FrameSample(frames[t - self.W - 1:t], frames[t], meta)
