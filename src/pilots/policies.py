"""Selectors take history-derived scores, never current CSI (except named genie)."""
import runtime as _runtime
import math
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class SelectionState:
    H_hist_est: object
    rho_t: object
    rho_f: object
    D: object
    snr_db: float
    fd_hz_est: float
    H_true: object | None = None


class Policy:
    """Base policy with an executable current-CSI leakage guard."""

    name = "base"
    genie = False

    def select(self, state: SelectionState, K: int):
        if not self.genie and state.H_true is not None:
            raise AssertionError(f"{self.name} policy must not receive current true CSI")
        return self._select(state, K)

    def _select(self, state, K):  # pragma: no cover - abstract convention
        raise NotImplementedError


def policy_state(state: SelectionState, *, genie: bool = False) -> SelectionState:
    """Strip truth at the orchestration boundary unless the policy is the genie."""
    if genie:
        if state.H_true is None:
            raise ValueError("genie requires current true CSI")
        return state
    return SelectionState(state.H_hist_est, state.rho_t, state.rho_f, state.D,
                          state.snr_db, state.fd_hz_est, None)


def causal_features(history, snr_db, symbol_duration, carrier_frequency=3.5e9):
    history = np.asarray(history)
    if history.ndim != 3 or len(history) < 3:
        raise ValueError('Need at least 3 PAST reconstructed frames')
    left, right = history[1:], history[:-1]
    rho_t = np.abs((left * right.conj()).sum(0)) / np.maximum((np.abs(left) * np.abs(right)).sum(0), 1e-12)
    last = history[-1]
    products = last[:-1] * last[1:].conj()
    denominator = np.abs(last[:-1]) * np.abs(last[1:])
    numerator = uniform_filter1d(products.real, 5, axis=0) + 1j * uniform_filter1d(products.imag, 5, axis=0)
    rho_f = np.abs(numerator) / np.maximum(uniform_filter1d(denominator, 5, axis=0), 1e-12)
    rho_f = np.concatenate([rho_f, rho_f[-1:]], 0)
    power = max(np.mean(np.abs(last)**2), 1e-12)
    innovation = np.abs(last - history[-2])**2 / power
    # Small-lag J0 approximation, estimated from previous within-frame CSI only.
    lag_corr = np.real(np.sum(last[:, 1:] * last[:, :-1].conj())) / max(
        np.sqrt(np.sum(np.abs(last[:, 1:])**2) * np.sum(np.abs(last[:, :-1])**2)), 1e-12)
    fd_hat = np.sqrt(max(0., 1 - np.clip(lag_corr, -1, 1))) / (np.pi * symbol_duration)
    # Seven state planes; insert the counterfactual O between Im(H) and SNR for g_phi.
    features = np.stack([rho_t.clip(0, 1), rho_f.clip(0, 1), innovation,
                         last.real / np.sqrt(power), last.imag / np.sqrt(power),
                         np.full(last.shape, snr_db / 20), np.full(last.shape, fd_hat / 400)]).astype(np.float32)
    return features, float(np.mean(rho_t.clip(0, 1) * rho_f.clip(0, 1))), float(fd_hat)


def condition_on_mask(features, observed):
    features = np.asarray(features)
    if features.shape[0] != 7:
        raise ValueError('expected seven causal state planes')
    return np.concatenate((features[:5], np.asarray(observed, np.float32)[None], features[5:]), axis=0)


def patch_mean(x, cfg):
    return np.asarray(x).reshape(cfg.nf // cfg.pf, cfg.pf, cfg.nt // cfg.pt, cfg.pt).mean((1, 3)).reshape(-1)


def expand_patch(values, cfg):
    return np.repeat(np.repeat(np.asarray(values).reshape(cfg.patch_grid), cfg.pf, 0), cfg.pt, 1)


def budget_count(density, cfg):
    # One pilot RE per selected patch first; additional REs in deterministic local order.
    return max(1, min(cfg.nf * cfg.nt - 1, int(np.floor(density * cfg.nf * cfg.nt + .5))))


@lru_cache(maxsize=16)
def uniform_order(shape):
    # Farthest-point sequence spreads observations over BOTH axes, with nested budgets.
    coords = np.indices(shape).reshape(2, -1).T / np.maximum(np.array(shape) - 1, 1)
    nearest = np.full(len(coords), np.inf)
    chosen = np.zeros(len(coords), bool)
    order, index = [], int(np.argmin(np.sum((coords - .5)**2, 1)))
    for _ in range(len(coords)):
        order.append(index)
        chosen[index] = True
        nearest = np.minimum(nearest, np.sum((coords - coords[index])**2, 1))
        nearest[chosen] = -np.inf
        index = int(nearest.argmax())
    return np.array(order)


def _spread_patch_rank(patch_score, cfg):
    """Visit order of patches: high score first, but skip neighbours of chosen patches.

    Adjacent patches often share a high predicted error, so a plain score order still
    clusters pilots at low density. A patch is taken in the first pass only if none of
    its 8 neighbours on the patch grid is already taken; skipped patches follow in
    later passes, again in descending score.
    """
    rows, cols = cfg.patch_grid
    remaining = list(np.argsort(-patch_score, kind='stable'))
    order = []
    while remaining:
        taken, skipped = set(), []
        for q in remaining:
            r, c = divmod(int(q), cols)
            if any((r + dr) * cols + (c + dc) in taken for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if 0 <= r + dr < rows and 0 <= c + dc < cols):
                skipped.append(q)
            else:
                taken.add(int(q)); order.append(int(q))
        remaining = skipped
    rank = np.empty(rows * cols, int)
    rank[order] = np.arange(len(order))
    return rank


def causal_order(scores, cfg, spread=False):
    """RE order for per-patch error scores: one pilot per patch before any second one.

    Plain score sorting fills the highest-error patches completely (8 tied REs each),
    so at 12.5% density only ~21 of 84 patches were observed versus 72 for uniform.
    Patches are visited in descending score; within a patch, REs follow the spatially
    balanced uniform sequence. With spread=True, the first pilot of each patch also
    avoids patches next to ones already chosen (see _spread_patch_rank).
    """
    base = uniform_order(cfg.shape)
    position = np.empty_like(base)
    position[base] = np.arange(len(base))
    f, t = np.divmod(np.arange(cfg.nf * cfg.nt), cfg.nt)
    patch = (f // cfg.pf) * (cfg.nt // cfg.pt) + t // cfg.pt
    rank = np.empty_like(base)
    for p in range(int(np.prod(cfg.patch_grid))):
        members = np.flatnonzero(patch == p)
        rank[members[np.argsort(position[members])]] = np.arange(len(members))
    scores = np.asarray(scores, dtype=float).ravel()
    if spread:
        return np.lexsort((position, _spread_patch_rank(scores, cfg)[patch], rank))
    return np.lexsort((position, -scores[patch], rank))


def seed_uniform_mask(cfg, count):
    """Causal policy's fixed four-pilot seed plus the uniform sequence; no adaptation.

    Separates the effect of the seed geometry from the effect of the predicted errors.
    """
    observed = _seed_pattern(cfg.shape)
    if count <= observed.sum():
        result = np.zeros_like(observed)
        result.flat[np.flatnonzero(observed)[:count]] = True
        return result
    for idx in uniform_order(cfg.shape):
        if not observed.flat[idx]:
            observed.flat[idx] = True
            if observed.sum() == count:
                break
    return observed


def select_mask(cfg, count, policy, rng=None, scores=None, static_order=None):
    if not 0 < count < cfg.nf * cfg.nt:
        raise ValueError('Pilot budget must leave at least one data RE')
    if policy == 'uniform':
        order = uniform_order(cfg.shape)
    elif policy == 'random':
        order = rng.permutation(cfg.nf * cfg.nt)
    elif policy == 'static':
        if static_order is None:
            raise ValueError('Static policy needs its TRAIN-learned ordering')
        order = np.asarray(static_order)
    elif policy == 'causal':
        if scores is None or len(scores) != np.prod(cfg.patch_grid):
            raise ValueError('Causal selection requires per-patch predicted error')
        order = causal_order(scores, cfg)
    else:
        raise ValueError(policy)
    observed = np.zeros(cfg.shape, bool)
    if policy != 'causal' or (cfg.spacing_f <= 1 and cfg.spacing_t <= 1):
        observed.flat[order[:count]] = True
        return observed
    # Greedy spacing can fall short of the budget at high densities (e.g. 50% with
    # 2x1 spacing); relax frequency spacing as GphiTopKPolicy does.
    for spacing_f in range(cfg.spacing_f, -1, -1):
        observed[:] = False
        chosen = []
        for idx in order:
            f, t = divmod(int(idx), cfg.nt)
            if all(abs(f - of) >= spacing_f or abs(t - ot) >= cfg.spacing_t for of, ot in chosen):
                chosen.append((f, t))
                observed[f, t] = True
                if len(chosen) == count:
                    return observed
    raise ValueError(f'Budget {count} infeasible with spacing {cfg.spacing_f}x{cfg.spacing_t}')


def nr_masks(cfg):
    """38.211 v17.6.0: mapping A, l0=2, duration 14, maxLength=1, port 1000.

    Single CDM group without data. Type 1: 6 RE/RB/symbol; type 2: 4.
    Other ports, mapping B, multiple CDM groups and double-symbol DM-RS are excluded.
    """
    if cfg.nt != 14 or cfg.nf % 12:
        raise ValueError('NR subset needs 14 symbols and complete 12-subcarrier RBs')
    options = []
    for typ, offsets in [(1, [0, 2, 4, 6, 8, 10]), (2, [0, 1, 6, 7])]:
        if typ != cfg.nr_config_type:
            continue
        for additional, times in enumerate(([2], [2, 11], [2, 7, 11], [2, 5, 8, 11])):
            mask = np.zeros(cfg.shape, bool)
            freq = np.array([rb + k for rb in range(0, cfg.nf, 12) for k in offsets])
            mask[np.ix_(freq, times)] = True
            options.append((f'type{typ}_pos{additional}', mask))
    return sorted(options, key=lambda x: int(x[1].sum()))


def signaling(observed, modulation, regime, q, kappa=3.0, code_rate=.5,
              groups=12, levels_per_group=4):
    if regime == 'S0':
        bits = 0
    elif regime == 'S1':
        bits = math.log2(q)
    elif regime == 'S2':
        bits = groups * math.log2(levels_per_group)
    else:
        raise ValueError(regime)
    eta_p = float(observed.mean())
    eta_sig = kappa * bits / (observed.size * math.log2(modulation) * code_rate)
    return dict(signaling_bits=bits, eta_p=eta_p, eta_sig=eta_sig, eta_eff=eta_p + eta_sig)


def causal_efficiency(random_nmse, causal_nmse, oracle_nmse):
    denominator = random_nmse - oracle_nmse
    # CE can be negative or >1 empirically. Never clip away a failed policy/oracle.
    return float((random_nmse - causal_nmse) / denominator) if denominator > 1e-12 else float('nan')


class BudgetController:
    def __init__(self, target, up, calibration):
        self.target, self.up = target, up
        self.down = up * target / (1 - target)
        self.delta = 0.
        self.calibration = calibration

    def choose(self, predicted_errors, masks, snr_db, power):
        snr = 10**(snr_db / 10)
        for i, mask in enumerate(masks):
            # g predicts error under random masking; K response is a validation-calibrated scale.
            predicted_nmse = self._scale(i, snr_db) * predicted_errors.mean() / max(power, 1e-12)
            effective = snr / (1 + snr * predicted_nmse)
            eff_db = 10 * np.log10(max(effective, 1e-12)) - self.delta
            bler = np.interp(eff_db, self.calibration['snr_db'], self.calibration['bler'][i], left=1., right=self.calibration['bler'][i][-1])
            if bler <= self.target:
                return i
        return len(masks) - 1

    def _scale(self, i, snr_db):
        # Per-SNR scales when calibrated that way; pooled scales for older calibrations.
        by_snr = self.calibration.get('nmse_scales_by_snr')
        if by_snr:
            nearest = min(by_snr, key=lambda key: abs(float(key) - snr_db))
            return by_snr[nearest][i]
        return self.calibration['nmse_scales'][i]

    def update(self, block_error):
        self.delta = float(np.clip(self.delta + (self.up if block_error else -self.down), -10, 30))


def _seed_pattern(shape):
    nf, nt = shape
    points = [(nf // 4, nt // 4), (nf // 4, 3 * nt // 4),
              (3 * nf // 4, nt // 4), (3 * nf // 4, 3 * nt // 4)]
    mask = np.zeros(shape, dtype=bool)
    for point in points:
        mask[min(point[0], nf - 1), min(point[1], nt - 1)] = True
    return mask


def gphi_features(state: SelectionState, observed):
    """Build the exact eight inference planes without current-frame truth."""
    if state.H_true is not None:
        raise AssertionError("causal g_phi features must not contain H_true")
    previous = np.asarray(state.H_hist_est)[-1]
    power = max(float(np.mean(np.abs(previous) ** 2)), 1e-12)
    return np.stack((np.asarray(state.rho_t), np.asarray(state.rho_f), np.asarray(state.D),
                     previous.real / np.sqrt(power), previous.imag / np.sqrt(power),
                     np.asarray(observed, dtype=np.float32),
                     np.full(previous.shape, state.snr_db / 20.0),
                     np.full(previous.shape, state.fd_hz_est / 400.0))).astype(np.float32)


class GphiTopKPolicy(Policy):
    name = "gphi_topk"

    def __init__(self, cfg, scorer, spacing_f=2, spacing_t=1, spread=False):
        self.cfg, self.scorer = cfg, scorer
        self.spacing_f, self.spacing_t = spacing_f, spacing_t
        self.spread = spread

    def _select(self, state, K):
        observed = _seed_pattern((self.cfg.nf, self.cfg.nt))
        if K <= observed.sum():
            result = np.zeros_like(observed)
            result.flat[np.flatnonzero(observed)[:K]] = True
            return result
        order = causal_order(self.scorer(gphi_features(state, observed)), self.cfg, self.spread)
        spacing = self.spacing_f
        while True:
            result = observed.copy()
            chosen = [tuple(x) for x in np.argwhere(result)]
            for idx in order:
                f, t = divmod(int(idx), self.cfg.nt)
                if result[f, t]:
                    continue
                if all(abs(f - of) >= spacing or abs(t - ot) >= self.spacing_t for of, ot in chosen):
                    result[f, t] = True
                    chosen.append((f, t))
                    if result.sum() == K:
                        return result
            if spacing == 0:
                raise RuntimeError("unable to satisfy pilot budget")
            spacing -= 1


class GphiGreedyPolicy(Policy):
    name = "gphi_greedy"

    def __init__(self, cfg, scorer, top_m=32):
        self.cfg, self.scorer, self.top_m = cfg, scorer, top_m

    def _select(self, state, K):
        observed = _seed_pattern((self.cfg.nf, self.cfg.nt))
        if K < observed.sum():
            result = np.zeros_like(observed)
            result.flat[np.flatnonzero(observed)[:K]] = True
            return result
        while observed.sum() < K:
            current = np.asarray(self.scorer(gphi_features(state, observed)))
            re_error = expand_patch(current, self.cfg).ravel()
            candidates = [i for i in np.argsort(-re_error, kind="stable") if not observed.flat[i]][:self.top_m]
            gains = []
            for idx in candidates:
                counterfactual = observed.copy()
                counterfactual.flat[idx] = True
                after = np.asarray(self.scorer(gphi_features(state, counterfactual)))
                gains.append(float(current.sum() - after.sum()))
            # Negative estimates are allowed at inference; selecting the best still guarantees termination.
            observed.flat[candidates[int(np.argmax(gains))]] = True
        return observed


class UniformPolicy(Policy):
    name = "uniform"

    def __init__(self, cfg):
        self.cfg = cfg

    def _select(self, state, K):
        return select_mask(self.cfg, K, "uniform")


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng

    def _select(self, state, K):
        return select_mask(self.cfg, K, "random", self.rng)


class NRDMRSPolicy(Policy):
    name = "nr_dmrs"

    def __init__(self, cfg):
        self.cfg = cfg

    def _select(self, state, K):
        options = nr_masks(self.cfg)
        feasible = [(name, mask) for name, mask in options if mask.sum() >= K]
        return (feasible[0] if feasible else options[-1])[1].copy()


class CoherenceThresholdPolicy(Policy):
    name = "coherence_threshold"

    def __init__(self, cfg, snr_threshold_db=-2.0):
        self.cfg, self.snr_threshold_db = cfg, snr_threshold_db

    def _select(self, state, K):
        if state.snr_db < self.snr_threshold_db:
            return select_mask(self.cfg, K, "uniform")
        rt = np.asarray(state.rho_t).mean(0)
        rf = np.asarray(state.rho_f).mean(1)
        tc = next((i + 1 for i, value in enumerate(rt) if value < .5), self.cfg.nt)
        bc = next((i + 1 for i, value in enumerate(rf) if value < .5), self.cfg.nf)
        df, dt = max(1, bc // 2), max(1, tc // 2)
        order = np.array([f * self.cfg.nt + t for t in range(0, self.cfg.nt, dt)
                          for f in range(0, self.cfg.nf, df)], dtype=int)
        remainder = np.setdiff1d(uniform_order(self.cfg.shape), order, assume_unique=False)
        result = np.zeros(self.cfg.shape, bool)
        result.flat[np.r_[order, remainder][:K]] = True
        return result


class GenieGreedyPolicy(Policy):
    """Greedy reference using exact current-frame reconstruction MSE."""

    name = "genie"
    genie = True

    def __init__(self, cfg, estimator, candidate_limit=None):
        self.cfg, self.estimator, self.candidate_limit = cfg, estimator, candidate_limit

    def _loss(self, truth, observed):
        estimate = self.estimator(truth * observed, observed)
        return float(np.mean(np.abs(truth - estimate) ** 2))

    def _select(self, state, K):
        truth = np.asarray(state.H_true)
        observed = _seed_pattern(self.cfg.shape)
        if K < observed.sum():
            result = np.zeros_like(observed)
            result.flat[np.flatnonzero(observed)[:K]] = True
            return result
        while observed.sum() < K:
            candidates = np.flatnonzero(~observed)
            if self.candidate_limit:
                candidates = candidates[:self.candidate_limit]
            best, best_loss = None, np.inf
            for idx in candidates:
                proposal = observed.copy()
                proposal.flat[idx] = True
                loss = self._loss(truth, proposal)
                if loss < best_loss:
                    best, best_loss = int(idx), loss
            observed.flat[best] = True
        return observed
