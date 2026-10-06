"""Disk-backed aggregation keeps full paper sweeps out of RAM."""
import runtime as _runtime
import csv
import itertools
import sqlite3
from pathlib import Path

# Cap on the predictability subsample used to place quantile bin edges. Two
# million samples pin the edges far tighter than the bin width we report.
SUBSAMPLE_CAP = 2_000_000


def _predictability_subsample(paths, cap=SUBSAMPLE_CAP):
    """Deterministic uniform subsample of Pi, bounded in memory.

    Keeps every row until the cap, then repeatedly halves by dropping every
    second sample and doubling the stride, so the result stays evenly spread
    over the input regardless of how many rows arrive.
    """
    samples, stride, seen = [], 1, 0
    for path in paths:
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                value = float(row['predictability'])
                if value != value:  # NaN carries no bin information
                    continue
                if seen % stride == 0:
                    samples.append(value)
                    if len(samples) >= cap:
                        samples = samples[::2]
                        stride *= 2
                seen += 1
    samples.sort()
    return samples


def _quantile_edges(samples, pi_bins):
    """Interior bin edges placed at equal-count quantiles of Pi.

    Duplicate edges (a Pi value dense enough to span a whole quantile) are
    dropped, so the realised bin count can fall below pi_bins rather than
    emitting empty bins.
    """
    if not samples or pi_bins < 2:
        return []
    edges = [samples[min(len(samples) - 1, int(round(i * len(samples) / pi_bins)))]
             for i in range(1, pi_bins)]
    return sorted(set(edges))


class ReportDatabase:
    keys=['seed','profile','velocity','trajectory','snr_db','modulation','method','policy',
          'requested_density','d_frames','kappa','pi_bucket']
    metrics=['nmse','eta_p','eta_sig','eta_eff','selector_ms','bit_errors','bits',
             'block_errors','blocks','Pi_H','Pi_hat','Pi_hat_bias','aging']

    def __init__(self,paths,path,pi_bins,binning='uniform'):
        paths=list(paths)
        self.pi_bins=pi_bins
        self.edges=_quantile_edges(_predictability_subsample(paths),pi_bins) if binning=='quantile' else None
        self._pi_sum,self._pi_count,self._pi_min,self._pi_max={},{},{},{}
        self.db=sqlite3.connect(path)
        self.db.execute('DROP TABLE IF EXISTS observations')
        columns=[k+(' TEXT' if k in ('profile','method','policy') else ' REAL') for k in self.keys+self.metrics]
        self.db.execute('CREATE TABLE observations ('+','.join(columns)+')')
        statement='INSERT INTO observations VALUES ('+','.join('?' for _ in columns)+')'
        batch=[];self.count=0
        with self.db:
            for path in paths:
                with Path(path).open() as f:
                    for row in csv.DictReader(f):
                        pi=float(row['predictability'])
                        row['pi_bucket']=self.bucket_of(pi)
                        self._observe_pi(row['pi_bucket'],pi)
                        values=[]
                        for k in self.keys+self.metrics:
                            value=row.get(k,0)
                            values.append(value if k in ('profile','method','policy') else float(value))
                        batch.append(values);self.count+=1
                        if len(batch)>=5000:
                            self.db.executemany(statement,batch);batch=[]
            if batch: self.db.executemany(statement,batch)

    def bucket_of(self,pi):
        if self.edges is None:
            return min(self.pi_bins-1,max(0,int(pi*self.pi_bins)))
        # Right-open bins: edge value falls in the upper bin, matching the
        # uniform path's int() truncation.
        low,high=0,len(self.edges)
        while low<high:
            middle=(low+high)//2
            if pi<self.edges[middle]: high=middle
            else: low=middle+1
        return low

    def _observe_pi(self,bucket,pi):
        if pi!=pi: return
        self._pi_sum[bucket]=self._pi_sum.get(bucket,0.)+pi
        self._pi_count[bucket]=self._pi_count.get(bucket,0)+1
        self._pi_min[bucket]=min(self._pi_min.get(bucket,pi),pi)
        self._pi_max[bucket]=max(self._pi_max.get(bucket,pi),pi)

    def pi_center(self,bucket):
        """Where the bin actually sits on the Pi axis.

        Uniform bins keep their nominal centre so existing runs reproduce;
        quantile bins report the mean Pi observed inside them, because a bin
        spanning 0.95-1.0 is not represented by its midpoint.
        """
        if self.edges is None:
            return (bucket+.5)/self.pi_bins
        count=self._pi_count.get(bucket,0)
        return self._pi_sum[bucket]/count if count else float('nan')

    def pi_range(self,bucket):
        if self.edges is None:
            return bucket/self.pi_bins,(bucket+1)/self.pi_bins
        return self._pi_min.get(bucket,float('nan')),self._pi_max.get(bucket,float('nan'))

    def pi_weight(self,bucket):
        return self._pi_count.get(bucket,0)

    def summaries(self,keys,aggregate):
        # Each returned item is one trajectory cluster within a summary group.
        group_keys=list(dict.fromkeys(keys+['seed','trajectory']))
        select=group_keys+['COUNT(*) AS weight']
        select += [('SUM' if k in ('bit_errors','bits','block_errors','blocks') else 'AVG')+f'({k}) AS {k}' for k in self.metrics]
        query='SELECT '+','.join(select)+' FROM observations GROUP BY '+','.join(group_keys)+' ORDER BY '+','.join(keys)
        cursor=self.db.execute(query)
        names=[d[0] for d in cursor.description]
        records=(dict(zip(names,row)) for row in cursor)
        result=[]
        for _,items in itertools.groupby(records,key=lambda r:tuple(r[k] for k in keys)):
            result.extend(aggregate(list(items),keys))
        return result

    def ce_rows(self):
        keys=['seed','profile','velocity','snr_db','modulation','method','policy']
        query='SELECT '+','.join(keys)+',AVG(nmse) AS nmse FROM observations GROUP BY '+','.join(keys)
        return [dict(zip(keys+['nmse'],row)) for row in self.db.execute(query)]

    def close(self):
        self.db.close()
