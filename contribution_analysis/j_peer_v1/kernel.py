"""Peer-only endpoint scoring. No search policy or alpha updates."""
import numpy as np
def score_peer(loss_minus,loss_full,weights,excluded):
    a=np.asarray(loss_minus,float);b=np.asarray(loss_full,float);w=np.asarray(weights,float)
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=w.shape or len(w)<2:raise ValueError('invalid_or_empty_panel')
    if not 0<=excluded<len(w):raise ValueError('invalid_excluded_index')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(w).all() or np.any(w<0):raise ValueError('invalid_values_or_weights')
    keep=np.arange(len(w))!=excluded;remaining=w[keep]
    if remaining.sum()<=0:raise ValueError('zero_remaining_weight')
    remaining=remaining/remaining.sum()
    return float(np.dot(remaining,a[keep])-np.dot(remaining,b[keep]))
