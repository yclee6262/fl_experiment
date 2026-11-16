"""Finite response-bank participant proxies; no historical calibration or LOO solves."""
import numpy as np
NAMES=['mean_abs','median_abs','abs_median_signed','abs_mean_signed']
def bank(record):
    points=[];losses=[];seen=set()
    def add(x,y):
        if y is None:return
        k=np.asarray(x,dtype=np.float64).tobytes()
        if k in seen:return
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite_bank')
        seen.add(k);points.append(x);losses.append(y)
    for t in record['iterations']:
        add(t['x'],t['current_losses'])
        for d in t['directions']:add(d['point'],d['reply_losses'])
        for d in t['trials']:add(d['x'],d['reply_losses'])
    if not points:raise ValueError('empty_bank')
    return np.array(points,dtype=float),np.array(losses,dtype=float)
def select(points,losses,candidates,alpha,k):
    alpha=np.asarray(alpha,dtype=float);w=alpha.copy();w[k]=0;w/=w.sum()
    # Float64 identity matches solver input representation. Do not remove nearby unrelated trials.
    excluded=np.all(points==np.asarray(candidates[k],dtype=float),axis=1)
    eligible=np.flatnonzero(~excluded)
    if not len(eligible):raise ValueError('empty_after_anchor_exclusion')
    index=int(eligible[np.argmin(losses[eligible]@w)])
    return index,points[index].copy(),w

def quality(predictions):
    p=np.asarray(predictions,dtype=float)
    return dict(mean_abs=float(np.mean(abs(p))),median_abs=float(np.median(abs(p))),
                abs_median_signed=float(abs(np.median(p))),abs_mean_signed=float(abs(np.mean(p))))
