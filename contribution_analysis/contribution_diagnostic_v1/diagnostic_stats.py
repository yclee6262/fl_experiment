"""Fixed diagnostic pairs, no estimator fitting or score changes."""
import numpy as np
from scipy.stats import rankdata

SCORES=['A','anchor_L','anchor_E','full_L','full_E']

def rank(a,b):
    a=rankdata(a);b=rankdata(b)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])

def pair(a,b,xname,yname):
    raw=rank(a,b)
    flat=[name for values,name in [(a,xname),(b,yname)] if name!='A' and np.ptp(values)<=1e-4]
    if flat:return dict(status='near_flat:'+','.join(flat),spearman=None,raw_spearman=raw)
    if raw is None:return dict(status='exact_tie',spearman=None,raw_spearman=None)
    return dict(status='available',spearman=raw,raw_spearman=raw)

def boot(values):
    v=np.array([np.nan if x is None else x for x in values]);rng=np.random.default_rng(20261020)
    sampled=v[rng.integers(0,len(v),(10000,len(v)))];counts=np.isfinite(sampled).sum(axis=1)
    means=np.nansum(sampled,axis=1)[counts>0]/counts[counts>0]
    return dict(mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
        exploratory_95pct_CI=np.quantile(means,[.025,.975]).tolist() if len(means) else None,
        undefined_bootstraps=int(np.sum(counts==0)))
