"""Float64 normalized common score with explicit undefined scale."""
import math
import numpy as np


def score(panel):
    w=np.asarray(panel['weights'],float);probes=np.asarray(panel['probes'],float);full=np.asarray(panel['full'],float)
    M=len(w)
    if w.ndim!=1 or M==0 or not np.isfinite(w).all()or(w<0).any()or w.sum()<=0:raise ValueError('invalid_weights')
    if probes.ndim!=2 or probes.shape[0]!=M or probes.shape[1]<2:raise ValueError('invalid_probes')
    if full.shape!=(M,):raise ValueError('invalid_endpoint_shape')
    if not np.isfinite(probes).all()or not np.isfinite(full).all()or(probes<0).any()or(full<0).any():raise ValueError('invalid_loss')
    w=w/w.sum();scales=np.quantile(probes,.9,axis=1,method='linear')-np.quantile(probes,.1,axis=1,method='linear')
    base=dict(scales=scales.tolist(),C=None,own=None,others=None)
    if panel['minus']is None:return dict(base,status='undefined_no_remaining_weight')
    minus=np.asarray(panel['minus'],float)
    if minus.shape!=(M,):raise ValueError('invalid_endpoint_shape')
    if not np.isfinite(minus).all()or(minus<0).any():raise ValueError('invalid_loss')
    if any(s<=1e-12 and ww>0 for s,ww in zip(scales,w)):return dict(base,status='undefined_scale')
    terms=[float(ww*(b-a)/s)if ww else 0. for ww,a,b,s in zip(w,full,minus,scales)]
    target=set(panel['target']);own=math.fsum(v for i,v in enumerate(terms)if i in target);others=math.fsum(v for i,v in enumerate(terms)if i not in target)
    return dict(scales=scales.tolist(),C=math.fsum(terms),own=own,others=others,status='available')
