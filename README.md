# Channel Predictability and Pilot Floor Characterization in OFDM Systems

An implementation of **Channel Predictability and Pilot Floor Characterization in OFDM Systems**, built to measure how much pilot overhead a channel actually needs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.7%2B-ee4c2c.svg)](https://pytorch.org/)

> 🎯 **TL;DR**: a masked estimator `f_theta` reconstructs the full CFR from a
> sparse pilot set, a small predictor `g_phi` forecasts where it will fail, and
> the pipeline measures the minimum pilot density (P-curve) each channel needs
> as a function of its causal predictability `Pi`.

## 📖 About

This project implements a simulation and learning pipeline for measuring
pilot overhead against causal channel predictability. It is an experimental
instrument; its hypotheses may be rejected by the results.

The predictability index combines temporal and frequency correlation maps over
the past `W` frames and a `B`-subcarrier window:

$$
\Pi = \mathrm{mean}_{k,t}\left[\rho_t(k,t)\,\rho_f(k,t)\right]
$$

`Pi_H` uses the true past CSI (design-time bucketing); `Pi_hat` uses estimated
past CSI with noise-power subtraction and is what a receiver could compute.

Pilot cost always includes the signaling needed to tell the transmitter where
to place pilots:

$$
\eta_{\mathrm{eff}} = \eta_p + \eta_{\mathrm{sig}}, \qquad
\eta_{\mathrm{sig}} = \frac{\kappa\, B_{\mathrm{sig}}}{N_f N_t \log_2(M)\, R_c}
$$

Policies are scored by **Causal Efficiency** on paired linear-MSE samples:

$$
\mathrm{CE} = \frac{\overline{\mathrm{MSE}}_{\mathrm{random}} - \overline{\mathrm{MSE}}_{\mathrm{causal}}}
{\overline{\mathrm{MSE}}_{\mathrm{random}} - \overline{\mathrm{MSE}}_{\mathrm{genie}}}
$$

CE is not clipped; it may be negative or exceed one.

### Key Features

- 📡 **Continuous trajectories**: `(48, 14)` complex CFR frames, delay spread
  log-uniform over 30–300 ns, three velocity segments with ±25% jitter, and
  sampling that never crosses trajectory boundaries.
- 🧩 **Masked estimator `f_theta`**: 4×2 patches, variable visible-token count,
  learned 2-D positions and an explicit observation plane. One mixture-trained
  model is shared by every pilot policy.
- 🔮 **Error predictor `g_phi`**: eight input planes, dilated convolutions,
  under 50k parameters, trained with log-error loss and a monotonicity penalty.
- 🎛️ **Pilot policies**: uniform, random, NR-DMRS, coherence-threshold, TopK,
  greedy marginal-gain and an exact-current-error genie. Non-genie policies are
  guarded against seeing current true CSI.
- 🔁 **Closed loop**: decision delay `d ∈ {0,1,2}`, `kappa ∈ {1,3,6}`, OLLA reset
  per trajectory, and a 20-frame warm-up excluded from metrics.
- 🎲 **Paired comparisons**: channel, noise, mask and payload use independent
  deterministic RNG streams, so every policy sees the same realizations.
- 📶 **Sionna CDL backend**: TR 38.901 CDL channels via Sionna, with an in-repo
  sum-of-sinusoids backend for synthetic development runs.

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.11 or newer
- CUDA-compatible GPU (recommended for training and the full grid)
- `sionna==2.0.1` for the default CDL backend

### Installation

From the repository root:

```bash
python3 -m venv .venv-ofdm
source .venv-ofdm/bin/activate
pip install -e '.[cdl]'  # base dependencies + Sionna CDL backend
```

For the synthetic backend, install with `pip install -e .`. Writing Parquet
results also needs `pip install pyarrow`.

### Check the link (Gate 0)

```bash
python scripts/ofdm_mae_cli.py gate0 \
  --config configs/predictability_aware_ofdm.yaml \
  --output runs/predictability_aware_ofdm
```

Gate 0 checks simulated BER against theory for AWGN, Rayleigh and the
configured backend with QPSK and 16-QAM. It writes `gate0.json` and fails if
any check does not pass.

## 🚀 Running Experiments

All commands are run from the repository root. Every command accepts the same
options:

| option | meaning |
|---|---|
| `--config` | YAML configuration (default `configs/predictability_aware_ofdm.yaml`) |
| `--output` | run directory (default `runs/predictability_aware_ofdm`) |
| `--cache` | channel cache directory (default `cache`) |
| `--seeds` | run only these configured seeds, e.g. `--seeds 0` |
| `--resume` | skip stages whose `.done` marker exists |

### 1. Feasibility screening

Run a reduced link-feasibility analysis before committing to the full grid:

```bash
python scripts/ofdm_mae_cli.py feasibility \
  --config configs/feasibility.yaml \
  --output runs/feasibility
```

`feasibility.yaml` uses 16-frame trajectories, profile C only, three
velocities and fewer test trajectories.

### 2. Full experiment suite

```bash
python scripts/run_ofdm_experiment_suite.py
```

This runs `all` on `predictability_aware_ofdm.yaml` with `--resume`
(64-frame trajectories, 200 estimator epochs, 60 predictor epochs, 200k
predictor pairs, seeds 0–2). Options: `--output`, `--cache`, and `--no-resume`
to recompute every stage.

The `all` command runs, per seed:

```text
feasibility → train → pilot-floor → genie-bracket → closed-loop
            → [transfer] → feature-ablation → complexity → report
```

`transfer` runs only when `run_transfer: true` in the configuration.

### 3. Run stages one by one

The grid is expensive; running stages separately lets you inspect results as
they arrive:

```bash
CFG=configs/predictability_aware_ofdm.yaml
OUT=runs/predictability_aware_ofdm

python scripts/ofdm_mae_cli.py gate0         --config $CFG --output $OUT
python scripts/ofdm_mae_cli.py train         --config $CFG --output $OUT --resume
python scripts/ofdm_mae_cli.py pilot-floor   --config $CFG --output $OUT --resume
python scripts/ofdm_mae_cli.py genie-bracket --config $CFG --output $OUT --resume
python scripts/ofdm_mae_cli.py closed-loop   --config $CFG --output $OUT --resume
python scripts/ofdm_mae_cli.py report        --config $CFG --output $OUT
```

| command | purpose |
|---|---|
| `gate0` | BER sanity check of the link and backend |
| `feasibility` | can the coded link meet the reliability target at all? |
| `train` | train `f_theta` and `g_phi` |
| `pilot-floor` | fixed-density sweep per estimator/policy → P-curves |
| `genie-bracket` | sweep including the (expensive) genie policy |
| `closed-loop` | adaptive control with delay, signaling and OLLA |
| `transfer` | profile transfer with fractional fine-tuning data |
| `feature-ablation` | which predictor input planes matter |
| `calibrate` | NMSE scale and AWGN/CDL FEC calibration curves |
| `complexity` | parameters, FLOPs and latency |
| `report` | aggregate all seeds into `report/` |
| `figures` | redraw paper figures from an existing report |

### 4. Run seeds in parallel

`--seeds` selects which configured seeds a process runs, so several processes
can share one output directory. Run `report` once after all of them finish:

```bash
for s in 0 1 2; do
  python scripts/ofdm_mae_cli.py all --config $CFG --output $OUT \
    --seeds $s --resume > $OUT.seed$s.log 2>&1 &
done
wait
python scripts/ofdm_mae_cli.py report --config $CFG --output $OUT
```

### 5. Paper figures

Redraw figures from an existing report without recomputing anything:

```bash
python scripts/ofdm_mae_cli.py figures --output $OUT --layout ieee
python scripts/ofdm_mae_cli.py figures --output $OUT --layout lncs \
  --figure-dir paper/figures
```

The default destination is `<output>/report/figures` (or `figures_lncs`).

### 6. Ablations on an existing run

These scripts read `resolved_config.yaml` and models from a finished run:

```bash
# Pilot-floor sweep for selected estimator:policy pairs only
python scripts/run_pilot_floor_subset.py --output $OUT \
  --pairs lmmse:uniform mae_uniform:uniform --result-name pilot_floor

# Causal placement ablation (spread constraint, fixed seed + uniform)
python scripts/evaluate_causal_pilot_placement.py --output $OUT \
  --max-density 0.0625

# Per-trajectory vs global channel-power normalization
python scripts/compare_channel_power_normalization.py --output $OUT \
  --trajectories 50 --velocities 3 120 --snrs 15 20 25 30
```

To rerun the whole grid without per-trajectory power normalization, use
`predictability_aware_ofdm_global_power.yaml` with a new `--output`.

## ⚙️ Configuration

| file | purpose |
|---|---|
| `base.yaml` | numerology, data splits, SNRs, MCS, signaling, control, seeds, densities |
| `channel.yaml` | backend (`sionna` / `synthetic`), CDL profiles, delay spread, velocity jitter |
| `model_ftheta.yaml` | masked estimator architecture and training |
| `model_gphi.yaml` | error predictor architecture and training |
| `predictability_aware_ofdm.yaml` | full-scale configuration composing the four files above |
| `feasibility.yaml` | small diagnostic configuration |
| `predictability_aware_ofdm_global_power.yaml` | full scale with `power_normalisation: none` |

Configurations compose through `includes:`. Every run writes the fully merged
`resolved_config.yaml`; reusing an output directory with a different
configuration raises an error, so **change the config → use a new `--output`**.
`--resume` skips stages with a `.done` marker but does not resume a stage that
was interrupted halfway.

## 📊 Output Files

| path | content |
|---|---|
| `resolved_config.yaml`, `metadata.json` | merged config, package versions, link assumptions |
| `gate0.json` | Gate 0 BER checks |
| `seed_<n>/*.pt` | estimator, predictor and transfer checkpoints |
| `seed_<n>/*.csv` | raw per-frame/per-sample measurements of each stage |
| `seed_<n>/calibration.json` | validation-only calibration and coded AWGN BLER |
| `report/p_curve.csv` | pilot floor vs `Pi` per SNR, velocity, estimator, policy, MCS; NaN + reason when the target is not met |
| `report/causal_efficiency*.csv` | CE, headroom split (free vs adaptive), static efficiency, seed mean/std |
| `report/*_summary.csv` | NMSE, BER, BLER, overhead, seed spread, trajectory bootstrap CIs |
| `report/complexity.csv` | parameters, FLOPs, latency |
| `report/figures/` | paper figures (PNG + PDF) |
| `report/README.md` | interpretation notes generated with the report |

## 📁 Project Layout

| path | purpose |
|---|---|
| `src/models/` | estimators, error predictor, training routines |
| `src/data/` | trajectory sampling and deterministic random streams |
| `src/channels/` | channel and noise generation, noise estimation |
| `src/features/` | predictability features `rho_t`, `rho_f`, `Pi` |
| `src/pilots/` | pilot policies and closed-loop control |
| `src/link/` | link simulation, LS/LMMSE/CNN baselines, metrics |
| `src/experiments/` | sweeps, calibration/closed loop, transfer, ablation, complexity |
| `src/reporting/` | report, aggregation, figures, result schema |
| `src/config/` | configuration loading and composition |
| `src/experiment_cli.py` | command entry point |
| `scripts/` | command-line wrappers and ablation scripts |
| `configs/` | YAML presets |
| `runs/`, `cache/` | generated output; Git-ignored |

## 🧭 Scope and Interpretation

- The model is frequency-domain SISO OFDM with sufficient CP, perfect timing,
  and no path loss or shadowing.
- The in-repo sum-of-sinusoids backend uses TR 38.901 CDL delay/power tables
  but does not reproduce spatial/angular correlation. Paper results use the
  Sionna CDL backend.
- Coding is a terminated rate-1/2 convolutional code with soft Viterbi
  decoding — the proposal's fallback, not 5G LDPC.
- P-curves are empirical floors for the tested estimator, policy, density grid,
  MCS, SNR and velocity.
- The genie is a greedy reference, not a certified upper bound. `d=0` is an
  explicitly labeled unrealizable reference.
- Full-pattern adaptation is expected to lose once realistic signaling and
  selector latency are charged; that is a valid result, not something to hide.
- Monte Carlo runs stop at 100 errors or 200,000 blocks. Zero-error bootstrap
  intervals alone do not certify a 1% BLER target.
