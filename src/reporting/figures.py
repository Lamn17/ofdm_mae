"""Paper figures built from the report CSVs (no access to the per-frame data needed).

Palette: the first three categorical slots of the validated reference palette pass
every pair in light mode; a fourth series is drawn as a neutral dashed reference.
Velocities reuse the same four series styles as the estimators, so every figure shares one look.
Every series also carries its own marker so identity survives grayscale print.
"""
import runtime as _runtime
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

INK, INK_SOFT, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ESTIMATORS = [  # (method, policy, label, color, marker, linestyle)
    ('lmmse', 'uniform', 'LMMSE', '#2a78d6', 'o', '-'),
    ('mae_uniform', 'uniform', 'MAE', '#eb6834', 's', '-'),
    ('ls', 'uniform', 'LS', '#1baf7a', '^', '-'),
    ('cnn_uniform', 'uniform', 'CNN', '#8a8984', 'D', '--'),
]
VELOCITY_STYLES = [(color, marker, ls) for *_, color, marker, ls in ESTIMATORS]  # slow -> fast
CONTROLLERS = [  # (key, label, color, marker)
    ('static', 'Static pilot floor', '#2a78d6', 'o'),
    ('S1', 'S1 budget controller', '#eb6834', 's'),
    ('NR', 'NR DM-RS controller', '#1baf7a', '^'),
]
DENSITY_TICKS = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
DENSITY_LABELS = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2']
ONE_COLUMN, TWO_COLUMN = 3.5, 7.16  # IEEE column widths in inches
LNCS_WIDTH = 4.8  # Springer LNCS text width (122 mm)
# Figure sizes (inches), legend columns, and closed-loop grid per venue layout.
LAYOUTS = {
    'ieee': dict(pcurve=(TWO_COLUMN, 3.6), headroom=(TWO_COLUMN, 3.6), estimators=(ONE_COLUMN, 1.9),
                 closed_loop=(TWO_COLUMN, 2.0), legend_ncol=None, closed_loop_grid=(1, 4), sparse_ticks=False),
    'lncs': dict(pcurve=(LNCS_WIDTH, 3.5), headroom=(LNCS_WIDTH, 3.5), estimators=(LNCS_WIDTH * .75, 2.1),
                 closed_loop=(LNCS_WIDTH, 3.3), legend_ncol=3, closed_loop_grid=(2, 2), sparse_ticks=True),
}
LAYOUT = LAYOUTS['ieee']


def _style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': INK_SOFT, 'axes.labelcolor': INK, 'xtick.color': INK_SOFT,
        'ytick.color': INK_SOFT, 'text.color': INK, 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5,
        'lines.linewidth': 1.4, 'lines.markersize': 4, 'legend.frameon': False,
        'savefig.dpi': 300, 'figure.dpi': 150, 'pdf.fonttype': 42,
    })


def _read(path):
    def value(text):
        try:
            return float(text)
        except ValueError:
            return text
    with Path(path).open(newline='') as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _pick(available, preferred, n):
    """Preferred SNRs that exist in this run; otherwise its highest n."""
    available = sorted(set(available))
    chosen = [x for x in preferred if x in available]
    return chosen or available[-n:]


def _density_axis(ax, axis='x'):
    getattr(ax, f'set_{axis}scale')('log', base=2)
    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_locator(FixedLocator(DENSITY_TICKS))
    target.set_major_formatter(FixedFormatter(DENSITY_LABELS))
    target.set_minor_locator(NullLocator())


def _save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(directory / f'{name}.{suffix}', bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def _velocity_legend(fig, velocities, extra=(), ncol=None):
    handles = [Line2D([], [], color=c, marker=m, ls=ls, label=f'{v:g} km/h')
               for v, (c, m, ls) in zip(velocities, VELOCITY_STYLES)]
    handles += list(extra)
    ncol = ncol or LAYOUT['legend_ncol'] or len(handles)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=ncol,
               handlelength=1.8, columnspacing=1.2)


def min_blocks_for_target(target, z=1.96):
    """Fewest codewords for which zero errors give a Wilson upper bound at or below target."""
    n = 1
    while z * z / (n + z * z) > target:
        n += 1
    return n


def pcurve_figure(rows, directory, profile='C', modulation=4., snrs=(20., 25., 30.), target=.01):
    """Pilot floor against unpredictability 1 - Pi; open triangles mark censored floors."""
    # Buckets with too few codewords cannot meet the target even without errors; hide them.
    enough = min_blocks_for_target(target)
    rows = [r for r in rows if r['metric'] == 'bler' and r['profile'] == profile
            and r['modulation'] == modulation and r['policy'] == 'uniform'
            and r.get('blocks', enough) >= enough]
    if not rows:
        return False
    velocities = sorted({r['velocity'] for r in rows})
    snrs = _pick((r['snr_db'] for r in rows), snrs, 3)
    methods = [e for e in ESTIMATORS if e[0] in ('lmmse', 'mae_uniform')]
    fig, axes = plt.subplots(len(methods), len(snrs), figsize=LAYOUT['pcurve'],
                             sharex=True, sharey=True, squeeze=False)
    floor, ceiling = DENSITY_TICKS[0], DENSITY_TICKS[-1]
    for i, (method, _, label, *_rest) in enumerate(methods):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            for v, (color, marker, ls) in zip(velocities, VELOCITY_STYLES):
                curve = sorted((r for r in rows if r['method'] == method and r['snr_db'] == snr
                                and r['velocity'] == v), key=lambda r: r['pi_gap'])
                xs, ys, censored, missing = [], [], [], []
                for r in curve:
                    status = r['wilson_status']
                    if status == 'below_grid_censored':
                        censored.append(r['pi_gap']); xs.append(r['pi_gap']); ys.append(floor)
                    elif status == 'target_not_reached' or not np.isfinite(r['eta_star_wilson']):
                        missing.append(r['pi_gap'])
                    else:
                        xs.append(r['pi_gap']); ys.append(r['eta_star_wilson'])
                # Open markers sit on the grid edges; nudge them per velocity so they do not overlap.
                nudge = 1.18 ** (velocities.index(v) - (len(velocities) - 1) / 2)
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, markersize=3.5, lw=1.2)
                ax.plot([x * nudge for x in censored], [floor] * len(censored), ls='none', marker='v',
                        markersize=4.5, mfc='white', mec=color, mew=1)
                ax.plot([x * nudge for x in missing], [ceiling * 1.6] * len(missing), ls='none', marker='^',
                        markersize=4.5, mfc='white', mec=color, mew=1, clip_on=False)
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(NullLocator())
            _density_axis(ax, 'y')
            ax.set_ylim(floor / 1.5, ceiling * 2)
            if i == 0:
                ax.set_title(f'SNR = {snr:g} dB')
            if j == 0:
                ax.set_ylabel(f'{label}\npilot floor $\\eta^\\star$')
            if i == len(methods) - 1 and j == len(snrs) // 2:
                ax.set_xlabel('Unpredictability $1-\\Pi$')
    open_marker = dict(ls='none', mfc='white', mec=INK_SOFT, mew=1, markersize=4.5)
    _velocity_legend(fig, velocities, extra=[
        Line2D([], [], marker='v', label='floor below 1/128', **open_marker),
        Line2D([], [], marker='^', label='not met at 1/2', **open_marker)])
    fig.tight_layout(h_pad=.6, w_pad=.6)
    _save(fig, directory, 'fig1_pcurve')
    return True


def _nmse_table(rows, modulation=4.):
    table = defaultdict(list)
    for r in rows:
        if r['modulation'] == modulation:
            table[(r['method'], r['snr_db'], r['velocity'], r['requested_density'])].append(r['nmse_db'])
    return {k: float(np.mean(v)) for k, v in table.items()}  # mean over profiles


def headroom_figure(rows, directory, snrs=(20., 25., 30.), causal_row=True):
    """Top: genie gain over the uniform grid. Bottom (optional): what the causal policy realises."""
    nmse = _nmse_table(rows)
    snrs = _pick((k[1] for k in nmse), snrs, 3)
    velocities = sorted({k[2] for k in nmse})
    densities = sorted({k[3] for k in nmse})
    panels = [('mae_oracle', 'Genie gain\nover uniform (dB)', 3., '3 dB threshold')]
    if causal_row:
        panels.append(('mae_causal', 'Causal gain\nover uniform (dB)', 0., None))
    width, height = LAYOUT['headroom']
    fig, axes = plt.subplots(len(panels), len(snrs), figsize=(width, height * (len(panels) + .25) / 2.25),
                             sharex=True, squeeze=False)
    for i, (method, ylabel, reference, note) in enumerate(panels):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            ax.axhline(reference, color=INK_SOFT, lw=.8, ls=':')
            if note and j == 0:
                ax.text(densities[-1], reference, note, color=INK_SOFT, fontsize=6, ha='right', va='bottom')
            for v, (color, marker, ls) in zip(velocities, VELOCITY_STYLES):
                gain = [nmse[('mae_uniform', snr, v, d)] - nmse[(method, snr, v, d)] for d in densities]
                ax.plot(densities, gain, color=color, marker=marker, ls=ls, markersize=3.5, lw=1.2)
            _density_axis(ax, 'x')
            if LAYOUT['sparse_ticks']:
                ax.set_xticks([1/128, 1/32, 1/8, 1/2], ['1/128', '1/32', '1/8', '1/2'])
            if i == 0:
                ax.set_title(f'SNR = {snr:g} dB')
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(panels) - 1:
                ax.set_xlabel('Pilot density $\\eta_p$')
        lows = [ax.get_ylim()[0] for ax in axes[i]]
        highs = [ax.get_ylim()[1] for ax in axes[i]]
        for ax in axes[i]:
            ax.set_ylim(min(lows), max(highs))
    _velocity_legend(fig, velocities)
    fig.tight_layout(h_pad=.6, w_pad=.6)
    _save(fig, directory, 'fig2_headroom')


CAUSAL_VARIANTS = [  # (method, label, color, marker, linestyle)
    ('mae_causal', 'Causal', '#eb6834', 's', '-'),
    ('mae_causal_spread', 'Causal with spread', '#1baf7a', '^', '-'),
    ('mae_seed_uniform', 'Seed and uniform (no adaptation)', '#2a78d6', 'o', '-'),
    ('mae_oracle', 'Genie', '#8a8984', 'D', '--'),
]


def causal_figure(rows, ablation, directory, snrs=(20., 25., 30.)):
    """NMSE gain over uniform of the causal variants at sparse densities, mean over profiles and speeds."""
    nmse = defaultdict(list)
    for r in list(rows) + list(ablation):
        if r['modulation'] == 4.:
            nmse[(r['method'], r['snr_db'], r['requested_density'])].append(r['nmse_db'])
    densities = sorted({r['requested_density'] for r in ablation})
    snrs = _pick((k[1] for k in nmse), snrs, 3)
    width, _ = LAYOUT['headroom']
    fig, axes = plt.subplots(1, len(snrs), figsize=(width, 2.0), sharey=True, squeeze=False)
    for ax, snr in zip(axes[0], snrs):
        ax.axhline(0, color=INK_SOFT, lw=.8, ls=':')
        for method, label, color, marker, ls in CAUSAL_VARIANTS:
            gain = [np.mean(nmse[('mae_uniform', snr, d)]) - np.mean(nmse[(method, snr, d)]) for d in densities]
            ax.plot(densities, gain, color=color, marker=marker, ls=ls, markersize=3.5, lw=1.2, label=label)
        _density_axis(ax, 'x')
        ax.set_xticks(densities, [DENSITY_LABELS[DENSITY_TICKS.index(d)] if d in DENSITY_TICKS else f'{d:g}'
                                  for d in densities])
        ax.set_title(f'SNR = {snr:g} dB')
        ax.set_xlabel('Pilot density $\\eta_p$')
    axes[0, 0].set_ylabel('Gain over\nuniform (dB)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2,
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=.4)
    _save(fig, directory, 'fig5_causal')


def estimator_figure(rows, directory, snrs=(20., 30.)):
    """NMSE against pilot density on the uniform grid, averaged over profiles and velocities."""
    table = defaultdict(list)
    for r in rows:
        if r['modulation'] == 4. and r['policy'] == 'uniform':
            table[(r['method'], r['snr_db'], r['requested_density'])].append(r['nmse_db'])
    densities = sorted({k[2] for k in table})
    snrs = _pick((k[1] for k in table), snrs, 2)
    fig, axes = plt.subplots(1, len(snrs), figsize=LAYOUT['estimators'], sharey=True, squeeze=False)
    for ax, snr in zip(axes[0], snrs):
        for method, _, label, color, marker, ls in ESTIMATORS:
            ax.plot(densities, [np.mean(table[(method, snr, d)]) for d in densities],
                    color=color, marker=marker, ls=ls, markersize=3.5, lw=1.2, label=label)
        _density_axis(ax, 'x')
        ax.set_xticks([1/128, 1/32, 1/8, 1/2], ['1/128', '1/32', '1/8', '1/2'])
        ax.set_title(f'SNR = {snr:g} dB')
        ax.set_xlabel('Pilot density $\\eta_p$')
    axes[0, 0].set_ylabel('NMSE (dB)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4,
               handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=.4)
    _save(fig, directory, 'fig3_estimators')


def closed_loop_figure(rows, directory, target_bler, profile='C', modulation=4., delay=1., kappa=3.):
    """Effective overhead against SNR; filled markers meet the BLER target, hollow ones miss it."""
    rows = [r for r in rows if r['profile'] == profile and r['modulation'] == modulation]
    velocities = sorted({r['velocity'] for r in rows})
    snrs = sorted({r['snr_db'] for r in rows})
    n_rows, n_cols = LAYOUT['closed_loop_grid']
    if n_rows * n_cols < len(velocities):
        n_rows, n_cols = 1, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=LAYOUT['closed_loop'], sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(velocities):]:
        ax.set_visible(False)
    for k, (ax, v) in enumerate(zip(axes.flat, velocities)):
        here = [r for r in rows if r['velocity'] == v]
        for key, label, color, marker in CONTROLLERS:
            xs, ys, met = [], [], []
            for snr in snrs:
                at = [r for r in here if r['snr_db'] == snr]
                if key == 'static':
                    passing = [r for r in at if str(r['policy']).startswith('S0')
                               and r['bler_wilson_upper_descriptive'] <= target_bler]
                    if not passing:
                        continue
                    best = min(passing, key=lambda r: r['eta_eff'])
                else:
                    match = [r for r in at if r['policy'] == key and r['d_frames'] == delay and r['kappa'] == kappa]
                    if not match:
                        continue
                    best = match[0]
                xs.append(snr); ys.append(best['eta_eff'])
                met.append(best['bler_wilson_upper_descriptive'] <= target_bler)
            ax.plot(xs, ys, color=color, lw=1.2)
            for x, y, ok in zip(xs, ys, met):
                ax.plot(x, y, ls='none', marker=marker, markersize=4.5, mec=color, mew=1,
                        mfc=color if ok else 'white')
        _density_axis(ax, 'y')
        ax.set_ylim(DENSITY_TICKS[0] / 1.5, .8)
        ax.set_xticks(snrs)
        ax.set_title(f'{v:g} km/h')
        if k // n_cols == n_rows - 1:
            ax.set_xlabel('SNR (dB)')
        if k % n_cols == 0:
            ax.set_ylabel('Effective overhead $\\eta_{\\mathrm{eff}}$')
    handles = [Line2D([], [], color=c, marker=m, label=l) for _, l, c, m in CONTROLLERS]
    handles.append(Line2D([], [], color=INK_SOFT, marker='o', ls='none', mfc='white', label='misses 1% BLER'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2 if n_cols < 4 else 4,
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=.4)
    _save(fig, directory, 'fig4_closed_loop')


def paper_figures(report_dir, target_bler=.01, layout='ieee', directory=None):
    """Write the paper figure set (default <report_dir>/figures) from the report CSVs."""
    global LAYOUT
    LAYOUT = LAYOUTS[layout]
    _style()
    report_dir = Path(report_dir)
    directory = Path(directory) if directory else report_dir / ('figures' if layout == 'ieee' else f'figures_{layout}')
    written = []
    if (report_dir / 'p_curve.csv').exists():
        if pcurve_figure(_read(report_dir / 'p_curve.csv'), directory, target=target_bler):
            written.append('fig1_pcurve')
    if (report_dir / 'pilot_floor_summary.csv').exists():
        summary = _read(report_dir / 'pilot_floor_summary.csv')
        methods = {r['method'] for r in summary}
        ablation_path = report_dir / 'causal_ablation_summary.csv'
        if {'mae_uniform', 'mae_oracle', 'mae_causal'} <= methods:
            headroom_figure(summary, directory, causal_row=not ablation_path.exists())
            written.append('fig2_headroom')
            if ablation_path.exists():
                causal_figure(summary, _read(ablation_path), directory); written.append('fig5_causal')
        estimator_figure(summary, directory); written.append('fig3_estimators')
    if (report_dir / 'closed_loop_summary.csv').exists():
        closed_loop_figure(_read(report_dir / 'closed_loop_summary.csv'), directory, target_bler)
        written.append('fig4_closed_loop')
    return written
