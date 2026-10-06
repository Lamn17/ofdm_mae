"""Empirical floors and raw (unclipped) CE; no extrapolation of failed targets."""
import runtime as _runtime
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.io import write_csv
from pilots.policies import causal_efficiency
from reporting.database import ReportDatabase
from reporting.figures import paper_figures


def read_rows(paths):
    rows=[]
    for path in paths:
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                for key,value in row.items():
                    try: row[key]=float(value)
                    except (ValueError,TypeError): pass
                rows.append(row)
    return rows


def group(rows, keys):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def wilson_upper(errors, total, z=1.96):
    if total <= 0: return float('nan')
    p=errors/total
    return float((p+z*z/(2*total)+z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total))


def pilot_floor(points, target):
    """Return tested feasible floor and optional LINEAR crossing, never extrapolate."""
    points=sorted(points)
    feasible=[i for i,(_,error) in enumerate(points) if error<=target]
    if not feasible:
        return float('nan'),float('nan'),'target_not_reached'
    i=feasible[0]
    tested=points[i][0]
    if i==0: return tested,tested,'below_grid_censored'
    x0,y0=points[i-1];x1,y1=points[i]
    interpolated=x0+(target-y0)*(x1-x0)/(y1-y0) if y1!=y0 else x1
    return tested,interpolated,'bracketed'


def aggregate(rows, keys):
    result=[]
    for values,items in group(rows,keys).items():
        row=dict(zip(keys,values))
        for key in ('nmse','eta_p','eta_sig','eta_eff','selector_ms'):
            row[key]=float(np.average([r[key] for r in items],weights=[r.get('weight',1) for r in items]))
        row['nmse_db']=float(10*np.log10(max(row['nmse'],1e-12)))
        for error,total,metric in [('bit_errors','bits','ber'),('block_errors','blocks','bler')]:
            e=sum(r[error] for r in items);n=sum(r[total] for r in items)
            row[error],row[total],row[metric]=e,n,e/n
        row['bler_wilson_upper_descriptive']=wilson_upper(row['block_errors'],row['blocks'])
        per_seed=[]
        for _,sr in group(items,['seed']).items():
            per_seed.append(sum(r['block_errors'] for r in sr)/sum(r['blocks'] for r in sr))
        row['seed_count']=len(per_seed)
        row['bler_seed_std']=float(np.std(per_seed,ddof=1)) if len(per_seed)>1 else float('nan')
        nmse_seeds=[np.average([r['nmse'] for r in sr],weights=[r.get('weight',1) for r in sr]) for sr in group(items,['seed']).values()]
        row['nmse_seed_std']=float(np.std(nmse_seeds,ddof=1)) if len(nmse_seeds)>1 else float('nan')
        # Paired trajectories are the resampling unit, never adjacent frames.
        clusters=list(group(items,['seed','trajectory']).values())
        rng=np.random.default_rng(42)
        ratios=[]
        if len(clusters)>1:
            counts=np.array([[sum(r['block_errors'] for r in c),sum(r['blocks'] for r in c)] for c in clusters])
            for _ in range(200):
                sample=counts[rng.integers(0,len(counts),len(counts))].sum(0)
                ratios.append(sample[0]/sample[1])
        row['bler_trajectory_bootstrap_lo']=float(np.quantile(ratios,.025)) if ratios else float('nan')
        row['bler_trajectory_bootstrap_hi']=float(np.quantile(ratios,.975)) if ratios else float('nan')
        result.append(row)
    return result


# Pi saturates near one, so the pilot-floor curve is plotted against the
# unpredictability gap 1 - Pi on a log axis. The floor only guards log(0) at
# Pi == 1; Pi is measured to far coarser precision than this.
GAP_FLOOR = 1e-6


def pi_gap(pi_center):
    return max(1. - pi_center, GAP_FLOOR)


def report(cfg, run):
    run=Path(run); destination=run/'report';destination.mkdir(parents=True,exist_ok=True)
    notes=['# Predictability-aware OFDM run',f'\nBackend: **{cfg.backend}**.',
           'Synthetic runs validate software only; they are not CDL paper results.' if cfg.backend=='synthetic' else 'CDL generated with Sionna; see metadata.json for versions.',
           f'Coded-bit interleaver: **{"enabled" if cfg.code_interleaver else "disabled"}**. '
           f'Closed-loop adaptive variants: **{", ".join(cfg.controller_variants)}**.',
           'All headline overhead uses eta_eff. The greedy genie reference is not a certified global upper bound.',
           'Causal Efficiency is reported without clipping. Pilot floors are estimator/policy/grid dependent.',
           'The genie candidate set contains the uniform grid, so oracle_headroom_db splits into free_headroom_db '
           '(random to uniform: no adaptivity, no signalling) and adaptive_headroom_db (uniform to oracle: the only '
           'part that can pay for eta_sig). causal_efficiency_vs_uniform scores the causal policy against that grid.',
           'Pilot floors are plotted against 1 - Pi on a log axis because Pi saturates near one; p_curve.csv carries '
           'pi_center, pi_gap and the realised bin edges pi_lo/pi_hi with pi_weight sample counts.',
           'Wilson bounds are descriptive (frames within a trajectory are correlated); trajectory bootstrap intervals and seed standard deviations are also provided. Zero-error bootstrap intervals alone do not certify a 1% BLER target.']
    table2=[]
    f0_paths=sorted(run.glob('seed_*/feasibility.csv'))
    if f0_paths:
        database=ReportDatabase(f0_paths,destination/'feasibility.sqlite',cfg.pi_bins,cfg.pi_binning)
        keys=['profile','velocity','snr_db','modulation','method','policy','requested_density']
        feasibility=database.summaries(keys,aggregate)
        for row in feasibility:
            row['target_bler']=cfg.target_bler
            row['target_met_observed']=row['bler'] <= cfg.target_bler
            row['target_met_wilson_descriptive']=row['bler_wilson_upper_descriptive'] <= cfg.target_bler
        write_csv(destination/'feasibility_summary.csv',feasibility)
        notes.append(f'\nFeasibility: {database.count} perfect-CSI/dense-LS observations. See `feasibility_summary.csv`.')
        database.close()
    for exp in ('pilot_floor','genie_bracket','closed_loop'):
        paths=sorted(run.glob(f'seed_*/{exp}.csv'))
        if not paths: continue
        database=ReportDatabase(paths,destination/f'{exp}.sqlite',cfg.pi_bins,cfg.pi_binning)
        keys=['profile','velocity','snr_db','modulation','method','policy']
        if exp!='closed_loop': keys+=['requested_density']
        else: keys+=['d_frames','kappa']
        summary=database.summaries(keys,aggregate)
        write_csv(destination/f'{exp}_summary.csv',summary)
        if exp=='pilot_floor':
            bucketed=database.summaries(keys+['pi_bucket'],aggregate)
            floor_keys=[k for k in keys if k!='requested_density']+['pi_bucket']
            floors=[]
            for values,items in group(bucketed,floor_keys).items():
                base=dict(zip(floor_keys,values))
                for metric,target in [('ber',cfg.target_ber),('bler',cfg.target_bler)]:
                    tested,crossing,status=pilot_floor([(r['eta_eff'],r[metric]) for r in items],target)
                    upper,_,upper_status=pilot_floor([(r['eta_eff'],r['bler_wilson_upper_descriptive']) for r in items],cfg.target_bler) if metric=='bler' else (float('nan'),None,'not_computed')
                    pi_lo,pi_hi=database.pi_range(base['pi_bucket'])
                    center=database.pi_center(base['pi_bucket'])
                    floors.append(dict(base,metric=metric,target=target,pi_center=center,pi_gap=pi_gap(center),
                                       pi_lo=pi_lo,pi_hi=pi_hi,pi_weight=database.pi_weight(base['pi_bucket']),
                                       eta_star_tested=tested,eta_star_interpolated=crossing,status=status,
                                       eta_star_wilson=upper,wilson_status=upper_status,
                                       blocks=int(min(r['blocks'] for r in items))))
            write_csv(destination/'p_curve.csv',floors)
        elif exp=='genie_bracket':
            table2=summary
            ce=[]
            rows=database.ce_rows()
            # Pair seed and operating point; all policies see identical channel trajectories.
            for values,items in group(rows,['seed','profile','velocity','snr_db','modulation']).items():
                means={p:np.mean([r['nmse'] for r in items if r['method']=='mae_'+p]) for p in ('random','causal','oracle','static','uniform')}
                gain=10*np.log10(means['random']/means['oracle'])
                # The genie candidate set contains the uniform grid, so part of the
                # headroom over random needs no adaptivity and carries no signalling.
                # Split it out: only the uniform->oracle part can justify paying eta_sig.
                ce.append(dict(zip(['seed','profile','velocity','snr_db','modulation'],values),
                               causal_efficiency=causal_efficiency(means['random'],means['causal'],means['oracle']),
                               causal_efficiency_vs_uniform=causal_efficiency(means['uniform'],means['causal'],means['oracle']),
                               oracle_headroom_db=gain,
                               free_headroom_db=10*np.log10(means['random']/means['uniform']),
                               adaptive_headroom_db=10*np.log10(means['uniform']/means['oracle']),
                               static_efficiency=causal_efficiency(means['random'],means['static'],means['oracle']),
                               oracle_kind='finite_candidate_genie',gate1='low_headroom' if gain<1 else ('high_headroom' if gain>3 else 'intermediate')))
            write_csv(destination/'causal_efficiency.csv',ce)
            ce_summary=[]
            for values,items in group(ce,['profile','velocity','snr_db','modulation']).items():
                entry=dict(zip(['profile','velocity','snr_db','modulation'],values))
                entry['seed_count']=len(items)
                for metric in ('causal_efficiency','causal_efficiency_vs_uniform','oracle_headroom_db',
                               'free_headroom_db','adaptive_headroom_db','static_efficiency'):
                    samples=np.array([r[metric] for r in items]);samples=samples[np.isfinite(samples)]
                    entry[metric+'_mean']=float(np.mean(samples)) if len(samples) else float('nan')
                    entry[metric+'_std']=float(np.std(samples,ddof=1)) if len(samples)>1 else float('nan')
                ce_summary.append(entry)
            write_csv(destination/'causal_efficiency_summary.csv',ce_summary)
        notes.append(f'\n{exp}: {database.count} frame/modulation observations. See `{exp}_summary.csv`.')
        database.close()
    e4=read_rows(sorted(run.glob('seed_*/transfer.csv')))
    if e4:
        fig,ax=plt.subplots(figsize=(6,4));table=[]
        for mode in ('pretrained','scratch'):
            data=[]
            for fraction in cfg.transfer_fractions:
                samples=[r for r in e4 if r['mode']==mode and r['fraction']==fraction]
                seeds=[np.mean([r['nmse'] for r in sr]) for sr in group(samples,['seed']).values()]
                value=float(np.mean(seeds));std=float(np.std(seeds,ddof=1)) if len(seeds)>1 else float('nan')
                table.append(dict(mode=mode,fraction=fraction,nmse=value,nmse_seed_std=std))
                data.append(10*np.log10(value))
            ax.plot(np.array(cfg.transfer_fractions)*100,data,'o-',label=mode)
        write_csv(destination/'transfer_summary.csv',table)
        ax.set(xlabel='Target CDL-C labeled data (%)',ylabel='NMSE (dB)',title=f'{cfg.backend}: transfer')
        ax.legend();fig.tight_layout();fig.savefig(destination/'transfer.png');plt.close(fig)
    for name in ('feature_ablation','complexity'):
        write_csv(destination/f'{name}.csv',read_rows(sorted(run.glob(f'seed_*/{name}.csv'))))
    complexity_rows=read_rows(sorted(run.glob('seed_*/complexity.csv')))
    for row in table2:
        measured=[r for r in complexity_rows if r['model']==row['method']]
        for key in ('parameters','analytical_matmul_conv_flops','latency_ms'):
            row[key]=float(np.mean([r[key] for r in measured])) if measured else float('nan')
    write_csv(destination/'table_II.csv',table2)
    notes += ['\nNo claim of meeting the reliability target is made by the plotting code. Inspect block counts, uncertainty, and target_not_reached entries.',
              '\nGate 2: inspect static_efficiency in causal_efficiency.csv. Gate 3: compare S1/S2 with d_frames and kappa; S2 uses the configured region/level signaling cost.']
    (destination/'README.md').write_text('\n\n'.join(notes)+'\n')
    paper_figures(destination, cfg.target_bler)
