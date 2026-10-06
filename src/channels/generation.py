"""Continuous trajectories. The synthetic backend is NOT a 3GPP CDL model."""
import runtime as _runtime
import hashlib
import numpy as np
from pathlib import Path


# 3GPP TR 38.901 Tables 7.7.1-1..4 (normalized delays, cluster powers in dB).
_CDL_TAPS = {
    'A': ([0,.3819,.4025,.5868,.461,.5375,.6708,.575,.7618,1.5375,1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,4.081,4.4579,4.5695,4.7966,5.0066,5.3043,9.6586],
          [-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,-12.4,-15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7]),
    'B': ([0,.1072,.2155,.2095,.287,.2986,.3752,.5055,.3681,.3697,.57,.5283,1.1021,1.2756,1.5474,1.7842,2.0169,2.8294,3.0219,3.6187,4.1067,4.279,4.7834],
          [0,-2.2,-4,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3,-8.9,-9,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,-11.4,-14.9,-9.2,-11.3]),
    'C': ([0,.2099,.2219,.2329,.2176,.6366,.6448,.656,.6584,.7935,.8213,.9336,1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523],
          [-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16,-15.7,-21.6,-22.8]),
    'D': ([0,0,.035,.612,1.363,1.405,1.804,2.596,1.775,4.042,7.937,9.424,9.708,12.525],
          [-.2,-13.5,-18.8,-21,-22.8,-17.9,-20.1,-21.9,-22.9,-27.8,-23.6,-24.8,-30,-27.7]),
}


def cdl_taps(profile, delay_spread_s):
    """Return normalized-power TR 38.901 cluster delays and powers."""
    if profile not in _CDL_TAPS:
        raise ValueError('profile must be A, B, C, or D')
    delays, powers_db = _CDL_TAPS[profile]
    powers = 10 ** (np.asarray(powers_db, dtype=float) / 10)
    powers /= powers.sum()
    return np.asarray(delays, dtype=float) * delay_spread_s, powers


def rms_delay_spread(delays, powers):
    delays, powers = np.asarray(delays), np.asarray(powers)
    mean = np.sum(delays * powers) / np.sum(powers)
    return float(np.sqrt(np.sum((delays - mean) ** 2 * powers) / np.sum(powers)))


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def complex_noise(rng, shape, variance=1.):
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.sqrt(variance / 2)).astype(np.complex64)


def cfr_from_taps(alpha, tau, n_f, scs_hz):
    """Convert symbol-rate complex tap gains to an OFDM CFR."""
    alpha, tau = np.asarray(alpha), np.asarray(tau)
    if alpha.ndim != 2 or tau.shape != (alpha.shape[0],):
        raise ValueError('alpha must be (tap,symbol) and tau must contain one value per tap')
    frequencies = (np.arange(n_f) - n_f // 2) * scs_hz
    return (np.exp(-2j * np.pi * frequencies[:, None] * tau) @ alpha).astype(np.complex64)


def trajectory(cfg, profile, velocity, trajectory_id, seed, cache=None):
    return trajectory_with_metadata(cfg, profile, velocity, trajectory_id, seed, cache)[0]


def trajectory_with_metadata(cfg, profile, velocity, trajectory_id, seed, cache=None, normalize=None):
    """normalize=False returns the raw generator output (unit power on average over
    realisations, not per trajectory). By default it follows cfg.power_normalisation.
    The realisation is the same either way; only the power scaling differs."""
    if normalize is None:
        normalize = getattr(cfg, 'power_normalisation', 'trajectory') == 'trajectory'
    key = (cfg.backend, profile, velocity, trajectory_id, seed, cfg.nf, cfg.nt,
           cfg.frames, cfg.subcarrier_spacing, cfg.carrier_frequency, cfg.delay_spread,
           tuple(cfg.delay_spread_range_ns), cfg.velocity_jitter, cfg.velocity_segments,
           cfg.frame_duration, cfg.cp_fraction, 'proposal-v3')
    # The RNG seed comes from `key`; raw data only needs its own cache file.
    cache_key = key if normalize else key + ('raw',)
    path = Path(cache) / (hashlib.sha256(repr(cache_key).encode()).hexdigest() + '.npy') if cache else None
    if path and path.exists():
        h = np.load(path)
        meta_path = path.with_suffix('.meta.npz')
        if meta_path.exists():
            with np.load(meta_path) as saved:
                meta = {k: saved[k] for k in saved.files}
                meta['profile'] = str(meta['profile'])
            return h, meta
    rng_seed = seed_for(*key)
    rng = np.random.default_rng(rng_seed)
    ds_ns = float(np.exp(rng.uniform(np.log(cfg.delay_spread_range_ns[0]),
                                     np.log(cfg.delay_spread_range_ns[1]))))
    segment_values = velocity * rng.uniform(1 - cfg.velocity_jitter, 1 + cfg.velocity_jitter,
                                            cfg.velocity_segments)
    v_per_frame = np.repeat(segment_values, int(np.ceil(cfg.frames / cfg.velocity_segments)))[:cfg.frames]
    if cfg.backend == 'sionna':
        # Sionna's CDL object accepts one speed per call. The in-repo sum-of-sinusoids
        # backend is used when the within-trajectory velocity axis is required.
        h = _sionna(cfg, profile, float(np.mean(v_per_frame)), rng_seed, ds_ns * 1e-9)
    else:
        h = _synthetic(cfg, profile, v_per_frame, rng_seed, ds_ns * 1e-9)
    # Normalize over the complete trajectory, never independently per frame.
    if normalize:
        h = h / np.sqrt(max(float(np.mean(np.abs(h) ** 2)), 1e-12))
    if h.shape != (cfg.frames, cfg.nf, cfg.nt) or not np.isfinite(h).all():
        raise RuntimeError(f'Invalid channel shape/values: {h.shape}')
    fd = v_per_frame / 3.6 * cfg.carrier_frequency / 299792458.
    meta = {'v_kmh_per_frame': v_per_frame.astype(np.float32),
            'fd_hz_per_frame': fd.astype(np.float32), 'ds_ns': ds_ns,
            'profile': str(profile), 'trajectory_id': int(trajectory_id)}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, h)
        np.savez(path.with_suffix('.meta.npz'), **meta)
    return h.astype(np.complex64), meta


def _synthetic(cfg, profile, velocity_per_frame, seed, delay_spread):
    rng = np.random.default_rng(seed)
    delays, powers = cdl_taps(profile, delay_spread)
    taps, rays = len(delays), 20
    velocity_symbols = np.repeat(np.asarray(velocity_per_frame), cfg.nt)
    doppler = velocity_symbols / 3.6 * cfg.carrier_frequency / 299792458.
    angles = np.broadcast_to(2 * np.pi * (np.arange(rays) + .5) / rays, (taps, rays))
    phases = rng.uniform(0, 2 * np.pi, (taps, rays))
    # Integrate instantaneous Doppler phase so velocity changes remain continuous.
    doppler_phase = 2 * np.pi * np.cumsum(doppler) * cfg.symbol_duration
    gains = (np.exp(1j * (phases[..., None] + np.cos(angles)[..., None] * doppler_phase)).sum(1)
             * np.sqrt(powers[:, None] / rays))
    response = cfr_from_taps(gains, delays, cfg.nf, cfg.subcarrier_spacing)
    return response.reshape(cfg.nf, cfg.frames, cfg.nt).transpose(1, 0, 2).astype(np.complex64)


def _sionna_impl(cfg, profile, velocity, seed, delay_spread=None):
    try:
        import torch
        from sionna.phy import config
        from sionna.phy.channel.tr38901 import Antenna, CDL
        from sionna.phy.channel import cir_to_ofdm_channel
    except ImportError as exc:
        raise RuntimeError("CDL requires Sionna 2.x (install with `pip install -e '.[cdl]'` from the project root). "
                           'Use a development configuration with backend: synthetic only for synthetic development runs.') from exc
    config.seed = seed
    antenna = Antenna(polarization='single', polarization_type='V', antenna_pattern='omni',
                      carrier_frequency=cfg.carrier_frequency)
    cdl = CDL(model=profile, delay_spread=delay_spread or cfg.delay_spread, carrier_frequency=cfg.carrier_frequency,
              ut_array=antenna, bs_array=antenna, direction='downlink',
              min_speed=velocity / 3.6, max_speed=velocity / 3.6)
    # ONE call per entire trajectory: separate calls would redraw the channel each frame.
    a, tau = cdl(batch_size=1, num_time_steps=cfg.frames * cfg.nt,
                 sampling_frequency=1 / cfg.symbol_duration)
    frequencies = (torch.arange(cfg.nf, device=a.device) - cfg.nf // 2) * cfg.subcarrier_spacing
    h = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
    return h[0, 0, 0, 0, 0].detach().cpu().numpy().reshape(cfg.frames, cfg.nt, cfg.nf).transpose(0, 2, 1)


def _sionna(cfg, profile, velocity, seed, delay_spread=None):
    # Sionna config.seed also seeds global Torch generators. Channel caching must
    # not change model initialization, optimizer randomness or sample ordering.
    import torch
    import random
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return _sionna_impl(cfg, profile, velocity, seed, delay_spread)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
