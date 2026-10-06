import runtime as _runtime
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import hashlib
from numbers import Real


@dataclass(frozen=True)
class Config:
    backend: str = 'sionna'
    nf: int = 48
    nt: int = 14
    pf: int = 4
    pt: int = 2
    subcarrier_spacing: float = 30000.
    carrier_frequency: float = 3.5e9
    delay_spread: float = 100e-9
    delay_spread_range_ns: list = field(default_factory=lambda: [30.0, 300.0])
    velocity_jitter: float = .25
    velocity_segments: int = 3
    power_normalisation: str = 'trajectory'
    frame_duration: float = .5e-3
    cp_fraction: float = 1 / 14
    nr_config_type: int = 1
    frames: int = 16
    history: int = 4
    frequency_window: int = 8
    train_ids: list = field(default_factory=lambda: [0, 700])
    val_ids: list = field(default_factory=lambda: [700, 850])
    test_ids: list = field(default_factory=lambda: [850, 1000])
    profiles: list = field(default_factory=lambda: ['A', 'B', 'C', 'D'])
    eval_profiles: list = field(default_factory=lambda: ['A', 'C'])
    velocities: list = field(default_factory=lambda: [3, 30, 60, 120])
    snrs: list = field(default_factory=lambda: [15, 20, 25, 30])
    training_snrs: list = field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25, 30])
    densities: list = field(default_factory=lambda: [.03125, .0625, .125, .25, .5])
    budget_levels: list = field(default_factory=lambda: [.5, .375, .25, .1875, .125, .09375, .0625, .03125])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    modulations: list = field(default_factory=lambda: [4, 16])
    dim: int = 128
    encoder_depth: int = 4
    decoder_dim: int = 64
    decoder_depth: int = 2
    heads: int = 4
    epochs: int = 20
    finetune_epochs: int = 5
    predictor_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = .0003
    train_samples: int = 8192
    predictor_samples: int = 200000
    calibration_blocks: int = 500
    calibration_source: str = 'cdl'
    calibration_trajectories: int = 64
    code_interleaver: bool = True
    target_bler: float = .01
    target_ber: float = .001
    olla_up: float = .1
    olla_down: float = .01
    closed_loop_warmup: int = 20
    loop_delays: list = field(default_factory=lambda: [0, 1, 2])
    kappa_values: list = field(default_factory=lambda: [1., 3., 6.])
    signaling_groups: int = 12
    signaling_levels: int = 4
    spacing_f: int = 1
    spacing_t: int = 1
    oracle_candidates: int = 16
    pi_bins: int = 5
    pi_binning: str = 'uniform'
    diagnostic_snrs: list = field(default_factory=lambda: [15, 20, 25, 30])
    diagnostic_densities: list = field(default_factory=lambda: [.5, .75, .875])
    diagnostic_trajectories: int = 50
    controller_variants: list = field(default_factory=lambda: ['S1', 'NR'])
    run_transfer: bool = False
    experiment_name: str = 'all'
    transfer_fractions: list = field(default_factory=lambda: [.01, .05, .25, 1.])
    device: str = 'auto'
    threads: int = 4

    @property
    def shape(self):
        return self.nf, self.nt

    @property
    def patch_grid(self):
        return self.nf // self.pf, self.nt // self.pt

    @property
    def symbol_duration(self):
        return self.frame_duration / self.nt

    @property
    def config_hash(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def validate(self):
        numeric = {
            'carrier_frequency': self.carrier_frequency,
            'subcarrier_spacing': self.subcarrier_spacing,
            'frame_duration': self.frame_duration,
        }
        invalid = [name for name, value in numeric.items() if not isinstance(value, Real) or value <= 0]
        if invalid:
            raise ValueError(f'positive numeric values required for: {", ".join(invalid)}')
        if self.nr_config_type not in (1,2):
            raise ValueError('NR DM-RS configuration type must be 1 or 2')
        if self.backend not in ('sionna', 'synthetic'):
            raise ValueError('backend must be sionna or synthetic')
        if self.nf % self.pf or self.nt % self.pt:
            raise ValueError('Grid must be divisible by patch size')
        if self.history < 2 or self.frames <= self.history + 1:
            raise ValueError('Need history >= 2 and frames > history+1')
        if len(self.delay_spread_range_ns) != 2 or not 0 < self.delay_spread_range_ns[0] <= self.delay_spread_range_ns[1]:
            raise ValueError('delay_spread_range_ns must be a positive ordered pair')
        if self.power_normalisation not in ('trajectory', 'none'):
            raise ValueError('power_normalisation must be trajectory or none')
        if not 0 <= self.velocity_jitter < 1 or self.velocity_segments < 1:
            raise ValueError('invalid intra-trajectory velocity randomization')
        if self.closed_loop_warmup < self.history + 1:
            raise ValueError('closed_loop_warmup must cover the causal history window')
        if any(d not in (0, 1, 2) for d in self.loop_delays):
            raise ValueError('loop_delays must be drawn from 0,1,2')
        if any(k < 1 for k in self.kappa_values):
            raise ValueError('kappa must be at least one')
        if not self.densities or any(not 0 < d < 1 for d in self.densities):
            raise ValueError('Pilot densities must lie strictly between zero and one')
        if not self.diagnostic_densities or any(not 0 < d < 1 for d in self.diagnostic_densities):
            raise ValueError('Diagnostic pilot densities must lie strictly between zero and one')
        if not self.diagnostic_snrs:
            raise ValueError('Diagnostic SNR grid must not be empty')
        if not self.snrs or not self.training_snrs:
            raise ValueError('Evaluation/training SNR grids must not be empty')
        if not self.controller_variants or any(x not in ('S1','S2','NR') for x in self.controller_variants):
            raise ValueError('controller_variants must contain S1, S2, and/or NR')
        if self.diagnostic_trajectories < 1 or self.calibration_trajectories < 1:
            raise ValueError('Diagnostic/calibration trajectory counts must be positive')
        if self.calibration_source not in ('awgn', 'cdl'):
            raise ValueError('calibration_source must be awgn or cdl')
        if self.pi_binning not in ('uniform', 'quantile'):
            raise ValueError('pi_binning must be uniform or quantile')
        if self.pi_bins < 1:
            raise ValueError('pi_bins must be positive')
        splits = [set(range(*x)) for x in (self.train_ids, self.val_ids, self.test_ids)]
        if any(not s for s in splits) or any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError('Trajectory splits must be nonempty and disjoint')
        if not 0 < self.target_bler < 1 or not 0 < self.target_ber < 1:
            raise ValueError('Reliability targets must be in (0,1)')
        if self.dim % self.heads or self.decoder_dim % self.heads or self.dim % 4 or self.decoder_dim % 4:
            raise ValueError('Embedding dimensions must be divisible by heads and four')
        if self.backend == 'sionna' and any(p not in 'ABCD' for p in self.profiles):
            raise ValueError('CDL profiles must be A/B/C/D')
        return self

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError('YAML configuration requires PyYAML') from exc
            path.write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        else:
            path.write_text(json.dumps(asdict(self), indent=2) + '\n')


_ALIASES = {
    'n_f': 'nf', 'n_t': 'nt', 'patch_f': 'pf', 'patch_t': 'pt',
    'scs_hz': 'subcarrier_spacing', 'fc_hz': 'carrier_frequency',
    'frame_dur_s': 'frame_duration', 'cp_len_ratio': 'cp_fraction',
    'frames_per_traj': 'frames', 'W': 'history', 'B': 'frequency_window',
    'embed_dim': 'dim', 'encoder_layers': 'encoder_depth',
    'decoder_layers': 'decoder_depth', 'lr': 'learning_rate',
    'd_frames': 'loop_delays', 'kappa': 'kappa_values',
    'Q_levels': 'budget_levels', 'G_groups': 'signaling_groups',
    'levels_per_group': 'signaling_levels',
    'name': 'experiment_name',
}


def _merge(left, right):
    result = dict(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml_tree(path, seen=None):
    import yaml
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f'cyclic YAML include: {path}')
    seen.add(path)
    document = yaml.safe_load(path.read_text()) or {}
    includes = document.pop('includes', [])
    if isinstance(includes, str):
        includes = [includes]
    merged = {}
    for included in includes:
        merged = _merge(merged, _read_yaml_tree(path.parent / included, seen))
    seen.remove(path)
    return _merge(merged, _flatten_sections(document))


def _flatten_sections(values):
    flat = {}
    for section, content in values.items():
        if isinstance(content, dict) and section in {
            'numerology', 'channel', 'features', 'link', 'signaling', 'control',
            'model_ftheta', 'model_gphi', 'data', 'runtime', 'experiment'
        }:
            for key, value in content.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        flat[_ALIASES.get(f'{key}_{nested_key}', _ALIASES.get(nested_key, nested_key))] = nested_value
                elif key == 'patch' and isinstance(value, (list, tuple)) and len(value) == 2:
                    flat['pf'], flat['pt'] = value
                else:
                    flat[_ALIASES.get(key, key)] = value
        else:
            flat[_ALIASES.get(section, section)] = content
    if isinstance(flat.get('loop_delays'), int):
        flat['loop_delays'] = [flat['loop_delays']]
    if isinstance(flat.get('kappa_values'), (int, float)):
        flat['kappa_values'] = [flat['kappa_values']]
    return flat


def read_config(path):
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        try:
            values = _read_yaml_tree(path)
        except ImportError as exc:
            raise RuntimeError('YAML configuration requires PyYAML') from exc
    else:
        values = json.loads(path.read_text())
    return Config(**(values if path.suffix.lower() in ('.yaml', '.yml') else _flatten_sections(values))).validate()
