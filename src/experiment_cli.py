import runtime as _runtime
import argparse
import importlib.metadata
import platform
from pathlib import Path
from dataclasses import asdict, replace
from config import read_config
from link.simulation import gate0, gate0_backend
from models.training import train_bundle
from experiments.pilot_sweeps import feasibility, fixed_sweep
from experiments.calibration_and_control import closed_loop, calibrate
from experiments.profile_transfer import transfer
from experiments.feature_ablation import ablation
from experiments.model_complexity import complexity
from reporting.generation import report


PROJECT_DIR = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Predictability-aware pilot efficiency for OFDM')
    parser.add_argument('command', choices=[
        'gate0', 'feasibility', 'train', 'pilot-floor', 'genie-bracket',
        'closed-loop', 'transfer', 'feature-ablation', 'calibrate',
        'complexity', 'report', 'figures', 'all'])
    parser.add_argument('--config',default=PROJECT_DIR / 'configs' / 'predictability_aware_ofdm.yaml')
    parser.add_argument('--output',default=PROJECT_DIR / 'runs' / 'predictability_aware_ofdm')
    parser.add_argument('--cache',default=PROJECT_DIR / 'cache')
    parser.add_argument('--seeds',type=int,nargs='+',help='Explicit override of configured seeds')
    parser.add_argument('--resume',action='store_true',help='Skip completed stages with matching config')
    parser.add_argument('--layout',choices=['ieee','lncs'],default='ieee',help='figures: venue page layout')
    parser.add_argument('--figure-dir',help='figures: output directory (default <output>/report/figures[_layout])')
    args=parser.parse_args()
    cfg=read_config(args.config)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    existing=output/'resolved_config.yaml'
    if args.command=='figures':
        # Redraw the paper figures from an existing report's CSVs; nothing is recomputed.
        from reporting.figures import paper_figures
        stored=read_config(existing) if existing.exists() else cfg
        written=paper_figures(output/'report', stored.target_bler, args.layout, args.figure_dir)
        print('Wrote', ', '.join(written), 'for layout', args.layout)
        return
    # --seeds only selects which configured seeds this process runs, so parallel
    # per-seed processes can share one output directory and one stored config.
    without_seeds=lambda c: {k:v for k,v in asdict(c).items() if k!='seeds'}
    if args.command=='report' and existing.exists():
        # A completed run's stored config is the source of truth. This also keeps
        # older runs reportable after new optional Config fields are introduced.
        cfg=read_config(existing)
    elif existing.exists() and without_seeds(read_config(existing))!=without_seeds(cfg):
        raise ValueError('Output directory belongs to a different configuration; choose a new --output')
    elif not existing.exists():
        cfg.save(existing)
    if args.seeds is not None and args.command!='report': cfg=replace(cfg, seeds=args.seeds)
    if args.command!='report':
        versions={}
        for package in ('torch','numpy','scipy','sionna'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package]=None
        import json
        (output/'metadata.json').write_text(json.dumps(dict(python=platform.python_version(),versions=versions,
                backend=cfg.backend,channel_assumptions='SISO frequency-domain OFDM; sufficient CP; no ICI; perfect timing; known long-term SNR',
                coding='rate-1/2 convolutional code with deterministic coded-bit interleaving' if cfg.code_interleaver else 'rate-1/2 convolutional code without interleaving',
                genie='greedy reference, not a certified upper bound'),indent=2))
    if args.command in ('gate0','all'):
        result=gate0()+gate0_backend(cfg)
        (output/'gate0.json').write_text(json.dumps(result,indent=2))
        if not all(r['passed'] for r in result): raise RuntimeError('Gate 0 failed: BER mismatch')
        print(f'Gate 0: {len(result)}/{len(result)} AWGN/Rayleigh/{cfg.backend} QPSK/16-QAM checks passed',flush=True)
    if args.command=='report': report(cfg,output);return
    for seed in cfg.seeds:
        directory=output/f'seed_{seed}';directory.mkdir(exist_ok=True)
        if args.command=='all':
            stages=['feasibility','train','pilot-floor','genie-bracket','closed-loop']
            if cfg.run_transfer: stages.append('transfer')
            stages += ['feature-ablation','complexity']
        else:
            stages=[args.command]
        for stage in stages:
            marker=directory/f'{stage}.done'
            if args.resume and marker.exists():
                print(f'Skip seed={seed} stage={stage}',flush=True);continue
            if stage=='feasibility': feasibility(cfg,seed,directory,args.cache)
            elif stage=='train': train_bundle(cfg,seed,directory,args.cache)
            elif stage in ('pilot-floor','genie-bracket'): fixed_sweep(cfg,seed,directory,args.cache,stage)
            elif stage=='closed-loop':
                if cfg.frames <= cfg.closed_loop_warmup:
                    # Every frame would fall inside the OLLA warm-up, producing an empty CSV.
                    print(f'Skip seed={seed} stage=closed-loop: frames={cfg.frames} <= '
                          f'closed_loop_warmup={cfg.closed_loop_warmup}',flush=True)
                    continue
                closed_loop(cfg,seed,directory,args.cache)
            elif stage=='transfer': transfer(cfg,seed,directory,args.cache)
            elif stage=='feature-ablation': ablation(cfg,seed,directory,args.cache)
            elif stage=='calibrate': calibrate(cfg,seed,directory,args.cache)
            elif stage=='complexity': complexity(cfg,seed,directory)
            else: continue
            marker.write_text('complete\n')
    # Parallel --seeds processes share one report directory; run `report` once after all finish.
    if args.command=='all' and args.seeds is None: report(cfg,output)


if __name__=='__main__': main()
