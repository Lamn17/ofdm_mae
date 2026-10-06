import runtime as _runtime
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError
from scipy.linalg import cho_factor, cho_solve


def interpolate(ls, observed):
    if not observed.any():
        return np.zeros_like(ls)
    points = np.argwhere(observed)
    targets = np.indices(ls.shape).reshape(2, -1).T
    nearest = NearestNDInterpolator(points, ls[observed])(targets)
    try:
        values = LinearNDInterpolator(points, ls[observed])(targets)
        values = np.where(np.isfinite(values), values, nearest)
    except (QhullError, ValueError):
        values = nearest
    return values.reshape(ls.shape).astype(np.complex64)


class LMMSE:
    """Uncentered second moment fitted on independent training CSI only."""
    def __init__(self, covariance):
        self.covariance = covariance

    @classmethod
    def fit(cls, frames, shrinkage=.02):
        x = np.asarray(frames).reshape(len(frames), -1)
        covariance = x.T @ x.conj() / len(x)
        power = np.trace(covariance).real / len(covariance)
        covariance = (1 - shrinkage) * covariance + shrinkage * power * np.eye(len(covariance))
        return cls(covariance.astype(np.complex64))

    def __call__(self, ls, observed, noise_var):
        ids = np.flatnonzero(observed)
        r = self.covariance
        system = r[np.ix_(ids, ids)] + noise_var * np.eye(len(ids))
        estimate = r[:, ids] @ cho_solve(cho_factor(system, lower=True), ls.ravel()[ids])
        return estimate.reshape(ls.shape).astype(np.complex64)
