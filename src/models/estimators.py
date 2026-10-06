import runtime as _runtime
import numpy as np
import torch
from torch import nn


def patchify(x, pf, pt):
    b, c, f, t = x.shape
    return x.reshape(b, c, f // pf, pf, t // pt, pt).permute(0, 2, 4, 3, 5, 1).reshape(b, f // pf * (t // pt), pf * pt * c)


def unpatchify(x, shape, pf, pt, channels=2):
    b = x.shape[0]
    f, t = shape
    return x.reshape(b, f // pf, t // pt, pf, pt, channels).permute(0, 5, 1, 3, 2, 4).reshape(b, channels, f, t)


def positions(grid, dim):
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in grid], indexing='ij'), -1).reshape(-1, 2)
    freq = 1 / (10000 ** (np.arange(dim // 4) / (dim // 4)))
    return torch.tensor(np.concatenate([func(coords[:, i:i+1] * freq) for i in range(2) for func in (np.sin, np.cos)], -1), dtype=torch.float32)[None]


class MaskedEstimator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(3 * cfg.pf * cfg.pt, cfg.dim)
        rows, cols = cfg.patch_grid
        self.encoder_row_pos = nn.Parameter(torch.zeros(1, rows, cfg.dim))
        self.encoder_col_pos = nn.Parameter(torch.zeros(1, cols, cfg.dim))
        self.decoder_row_pos = nn.Parameter(torch.zeros(1, rows, cfg.decoder_dim))
        self.decoder_col_pos = nn.Parameter(torch.zeros(1, cols, cfg.decoder_dim))
        for value in (self.encoder_row_pos, self.encoder_col_pos,
                      self.decoder_row_pos, self.decoder_col_pos):
            nn.init.normal_(value, std=.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4 * cfg.dim,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True), cfg.encoder_depth,
                    norm=nn.LayerNorm(cfg.dim), enable_nested_tensor=False)
        self.project = nn.Linear(cfg.dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        nn.init.normal_(self.mask_token, std=.02)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cfg.decoder_dim, cfg.heads,
                    4 * cfg.decoder_dim, dropout=0., activation='gelu', batch_first=True, norm_first=True),
                    cfg.decoder_depth, norm=nn.LayerNorm(cfg.decoder_dim), enable_nested_tensor=False)
        self.output = nn.Linear(cfg.decoder_dim, 2 * cfg.pf * cfg.pt)

    @property
    def encoder_pos(self):
        return (self.encoder_row_pos[:, :, None] + self.encoder_col_pos[:, None, :]).flatten(1, 2)

    @property
    def decoder_pos(self):
        return (self.decoder_row_pos[:, :, None] + self.decoder_col_pos[:, None, :]).flatten(1, 2)

    def forward(self, ls, observed):
        # Support partial patches (including DM-RS): a token is visible if ANY RE is observed.
        # Explicitly zero hidden inputs so callers cannot leak ground truth by accident.
        x = torch.cat([ls * observed, observed], 1)
        patches = patchify(x, self.cfg.pf, self.cfg.pt)
        visible = patchify(observed, self.cfg.pf, self.cfg.pt).any(-1)
        embedded = self.embedding(patches) + self.encoder_pos
        counts = visible.sum(1)
        max_visible = int(counts.max())
        decoded = self.mask_token.expand(x.shape[0], patches.shape[1], -1).clone()
        if max_visible:
            # Padding supports minibatches with different pilot counts; no hidden token enters encoder.
            ids = torch.argsort(visible.int(), dim=1, descending=True, stable=True)[:, :max_visible]
            packed = embedded.gather(1, ids[..., None].expand(-1, -1, embedded.shape[-1]))
            padding = torch.arange(max_visible, device=x.device)[None] >= counts[:, None]
            # Avoid an all-masked attention row for zero-observation examples.
            padding[counts == 0, 0] = False
            encoded = self.project(self.encoder(packed, src_key_padding_mask=padding))
            valid = ~padding
            valid[counts == 0, 0] = False
            rows = torch.arange(x.shape[0], device=x.device)[:, None].expand_as(ids)
            decoded[rows[valid], ids[valid]] = encoded[valid]
        reconstruction = unpatchify(self.output(self.decoder(decoded + self.decoder_pos)), self.cfg.shape,
                                     self.cfg.pf, self.cfg.pt)
        # Observed LS is preserved; loss is only on unobserved resource elements.
        return reconstruction * (1 - observed) + ls * observed


class CNNEstimator(nn.Module):
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(width, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, ls, observed):
        return self.net(torch.cat([ls * observed, observed], 1)) * (1 - observed) + ls * observed


class ErrorPredictor(nn.Module):
    def __init__(self, cfg, feature_set='full'):
        super().__init__()
        self.cfg, self.feature_set = cfg, feature_set
        self.body = nn.Sequential(nn.Conv2d(8, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(8, 32), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GroupNorm(8, 32), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(32, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Softplus())

    def forward(self, features):
        if features.shape[1] == 7:  # compatibility: explicitly condition on an empty mask
            features = torch.cat((features[:, :5], torch.zeros_like(features[:, :1]), features[:, 5:]), 1)
        if features.shape[1] != 8:
            raise ValueError('ErrorPredictor requires eight planes including the observation mask')
        indices = {'temporal': [0], 'frequency': [1], 'innovation': [2],
                   'plus_snr': [0, 1, 2, 6], 'full': list(range(8))}[self.feature_set]
        active = torch.zeros_like(features)
        active[:, indices] = features[:, indices]
        return self.head(torch.nn.functional.avg_pool2d(self.body(active),
                         (self.cfg.pf, self.cfg.pt))).flatten(1)


def masked_loss(prediction, target, observed):
    weight = 1 - observed
    # Complex squared magnitude is the sum of the real and imaginary errors.
    error = ((prediction - target).square() * weight).sum(1)
    hidden = weight[:, 0].sum((-2, -1)).clamp_min(1)
    signal = target.square().sum(1).mean((-2, -1)).clamp_min(1e-12)
    return (error.sum((-2, -1)) / hidden / signal).mean()


def as_tensor(h, device):
    return torch.from_numpy(np.stack([h.real, h.imag], axis=-3).astype(np.float32)).to(device)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def matched_cnn(cfg):
    target = parameter_count(MaskedEstimator(cfg))
    # Solve the conv parameter count analytically, instantiate once.
    width = min(range(4, 1025), key=lambda w: abs((6 - 2) * (9*w*w+w) + 46*w+2 - target))
    model = CNNEstimator(width)
    if abs(parameter_count(model) / target - 1) > .1:
        raise ValueError('Cannot match CNN parameter budget within 10%')
    return model


def analytical_flops(model, observed=None):
    """Multiply/add FLOPs including attention matmuls; excludes activations/norm/softmax."""
    if isinstance(model, MaskedEstimator):
        c=model.cfg
        tokens=int(np.prod(c.patch_grid))
        visible=int(patchify(observed,c.pf,c.pt).any(-1).sum())
        def transformer(n,d,depth):
            return depth*(24*n*d*d+4*n*n*d)
        return int(2*tokens*(3*c.pf*c.pt)*c.dim + transformer(visible,c.dim,c.encoder_depth)
                   +2*visible*c.dim*c.decoder_dim + transformer(tokens,c.decoder_dim,c.decoder_depth)
                   +2*tokens*c.decoder_dim*(2*c.pf*c.pt))
    if isinstance(model,(CNNEstimator,ErrorPredictor)):
        if observed is None: raise ValueError('Grid required')
        area=observed.shape[-2]*observed.shape[-1]
        return int(sum(2*area*m.in_channels*m.out_channels*np.prod(m.kernel_size)/m.groups
                       for m in model.modules() if isinstance(m,nn.Conv2d)))
    raise TypeError(type(model))
