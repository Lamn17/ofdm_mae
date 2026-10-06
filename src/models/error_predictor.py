"""Mask-conditioned marginal reconstruction-error predictor ``g_phi``."""
from __future__ import annotations
import runtime as _runtime

import torch
from torch import nn
import torch.nn.functional as F


class MarginalGainPredictor(nn.Module):
    """Predict a non-negative 12x7 error field for a counterfactual mask.

    Input planes are rho_t, rho_f, innovation, Re(H[t-1]), Im(H[t-1]), O,
    normalized SNR, and normalized Doppler.  The mask is deliberately an input:
    greedy selection evaluates how total predicted error changes after adding a pilot.
    """

    def __init__(self, patch=(4, 2)):
        super().__init__()
        self.patch = tuple(patch)
        self.body = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Conv2d(32, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Softplus())

    def forward(self, features: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (B,C,Nf,Nt)")
        if observed is not None:
            if observed.ndim == 3:
                observed = observed[:, None]
            if features.shape[1] == 7:
                # Seven state planes exclude O; insert it before the SNR plane.
                features = torch.cat((features[:, :5], observed, features[:, 5:]), 1)
            else:
                features = features.clone()
                features[:, 5:6] = observed
        if features.shape[1] != 8:
            raise ValueError("g_phi requires eight planes including O and Doppler")
        return self.head(F.avg_pool2d(self.body(features), self.patch))[:, 0]


def predictor_loss(predicted, target, predicted_added=None, c=1e-6, lambda_mono=0.1):
    prediction = (torch.log(predicted + c) - torch.log(target + c)).square().mean()
    monotonicity = predicted.new_zeros(())
    if predicted_added is not None:
        monotonicity = torch.relu(predicted_added.sum((-2, -1)) - predicted.sum((-2, -1))).mean()
    return prediction + lambda_mono * monotonicity, {
        "log_mse": prediction.detach(), "monotonicity": monotonicity.detach()
    }


def patch_error_target(H: torch.Tensor, H_hat: torch.Tensor, observed: torch.Tensor, patch=(4, 2)):
    if observed.ndim == 3:
        observed = observed[:, None]
    error = (H - H_hat).abs().square()
    if error.ndim == 3:
        error = error[:, None]
    hidden = 1.0 - observed.float()
    numerator = F.avg_pool2d(error * hidden, patch, divisor_override=1)
    denominator = F.avg_pool2d(hidden, patch, divisor_override=1)
    return (numerator / denominator.clamp_min(1.0))[:, 0]
