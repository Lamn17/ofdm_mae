"""Causal predictability features from proposal equations (9)--(12) and (15)."""
from __future__ import annotations
import runtime as _runtime

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CorrelationResult:
    value: torch.Tensor
    clip_rate: float


def _sigma_for(x: torch.Tensor, sigma_e2: torch.Tensor | float) -> torch.Tensor:
    sigma = torch.as_tensor(sigma_e2, dtype=x.real.dtype, device=x.device)
    while sigma.ndim < x.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def rho_temporal(
    H_hist: torch.Tensor,
    W: int = 4,
    sigma_e2: torch.Tensor | float | None = None,
    delta_floor: float = 1e-3,
    return_diagnostics: bool = False,
) -> torch.Tensor | CorrelationResult:
    if not torch.is_complex(H_hist) or H_hist.ndim != 4:
        raise ValueError("H_hist must be complex with shape (B,W+1,Nf,Nt)")
    if H_hist.shape[1] < W + 1:
        raise ValueError("H_hist does not contain the requested causal window")
    h = H_hist[:, -(W + 1):]
    left, right = h[:, 1:], h[:, :-1]
    numerator = (left * right.conj()).sum(1).abs()
    if sigma_e2 is None:
        denominator = (left.abs() * right.abs()).sum(1)
    else:
        sigma = _sigma_for(left, sigma_e2)
        power = h.abs().square().mean(dim=(-3, -2, -1), keepdim=True)
        delta = delta_floor * power
        a = torch.maximum(left.abs().square() - sigma, delta)
        b = torch.maximum(right.abs().square() - sigma, delta)
        denominator = torch.sqrt(a * b).sum(1)
    raw = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)
    clipped = raw.clamp(0.0, 1.0)
    result = CorrelationResult(clipped, float((raw > 1).float().mean().item()))
    return result if return_diagnostics else result.value


def rho_frequency(
    H_prev: torch.Tensor,
    B_win: int = 8,
    sigma_e2: torch.Tensor | float | None = None,
    delta_floor: float = 1e-3,
    return_diagnostics: bool = False,
) -> torch.Tensor | CorrelationResult:
    if not torch.is_complex(H_prev) or H_prev.ndim != 3:
        raise ValueError("H_prev must be complex with shape (B,Nf,Nt)")
    if B_win < 1:
        raise ValueError("B_win must be positive")
    a, b = H_prev[:, :-1], H_prev[:, 1:]
    product = a * b.conj()
    kernel = min(B_win, product.shape[1])

    def rolling(x: torch.Tensor) -> torch.Tensor:
        # frequency is treated as the 1-D signal dimension; OFDM symbols are channels
        x = x.transpose(1, 2)
        left = (kernel - 1) // 2
        right = kernel - 1 - left
        return F.avg_pool1d(F.pad(x, (left, right), mode="replicate"), kernel, stride=1).transpose(1, 2) * kernel

    numerator = torch.complex(rolling(product.real), rolling(product.imag)).abs()
    if sigma_e2 is None:
        denominator = rolling(a.abs() * b.abs())
    else:
        sigma = _sigma_for(a, sigma_e2)
        power = H_prev.abs().square().mean(dim=(-2, -1), keepdim=True)
        delta = delta_floor * power
        denominator = rolling(torch.sqrt(torch.maximum(a.abs().square() - sigma, delta)
                                         * torch.maximum(b.abs().square() - sigma, delta)))
    raw = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)
    raw = torch.cat((raw, raw[:, -1:]), dim=1)
    clipped = raw.clamp(0.0, 1.0)
    result = CorrelationResult(clipped, float((raw > 1).float().mean().item()))
    return result if return_diagnostics else result.value


def innovation(H_hist: torch.Tensor) -> torch.Tensor:
    if H_hist.ndim != 4 or H_hist.shape[1] < 2:
        raise ValueError("innovation needs at least two past frames")
    previous, older = H_hist[:, -1], H_hist[:, -2]
    power = previous.abs().square().mean(dim=(-2, -1), keepdim=True)
    return (previous - older).abs().square() / power.clamp_min(torch.finfo(power.dtype).eps)


def predictability_index(rho_t: torch.Tensor, rho_f: torch.Tensor) -> torch.Tensor:
    if rho_t.shape != rho_f.shape:
        raise ValueError("temporal and frequency maps must have equal shapes")
    return (rho_t * rho_f).mean(dim=(-2, -1))


def both_indices(H_true_hist: torch.Tensor, H_est_hist: torch.Tensor, sigma_e2, W=4, B_win=8):
    """Always compute the design-time and deployable indices together."""
    pi_h = predictability_index(rho_temporal(H_true_hist, W), rho_frequency(H_true_hist[:, -1], B_win))
    pi_hat = predictability_index(rho_temporal(H_est_hist, W, sigma_e2),
                                  rho_frequency(H_est_hist[:, -1], B_win, sigma_e2))
    return pi_h, pi_hat
