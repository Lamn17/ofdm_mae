"""Independent deterministic random streams used by paired experiments."""
from __future__ import annotations
import runtime as _runtime

from dataclasses import dataclass
import hashlib
import numpy as np


def derive_seed(seed: int, stream: str, *parts: object) -> int:
    payload = "|".join(map(str, (seed, stream, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class RNGStreams:
    """Changing a mask policy must not change channels or receiver noise."""

    seed: int

    def channel(self, *parts: object) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "channel", *parts))

    def noise(self, *parts: object) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "noise_estimation", *parts))

    def mask(self, *parts: object) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "mask", *parts))

    def data(self, *parts: object) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "data", *parts))
