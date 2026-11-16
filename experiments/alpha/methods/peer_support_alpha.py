"""Candidate support is computed only from local predictions, never pooled scores."""
import numpy as np
from initial_alpha_methods import initial_alpha as legacy
METHODS=['A_uniform','B_stage0','quality_shrink','peer_support_shrink','target_peer_shrink','ensemble_shrink']
SCENARIOS=['reference_clean','low_quality_25pct','sample_imbalance_4x']
LABELS=dict(zip(METHODS,['A：均勻','B：品質反比','Q：品質收縮','S：支持度收縮','QS：品質乘支持後收縮','E：最佳組合向均勻收縮']))
CONTRASTS=[(m,b) for m in ['peer_support_shrink','target_peer_shrink'] for b in ['A_uniform','B_stage0','ensemble_shrink']]+[('target_peer_shrink','quality_shrink')]

def support_signal(predictions,target):
    p=np.asarray(predictions,float)
    if p.ndim!=2 or p.shape[0]!=p.shape[1] or len(p)<2:raise ValueError('Peer support requires at least two models')
    n=len(p);mask=~np.eye(n,dtype=bool)
    if not np.isfinite(p[mask]).all() or not np.isfinite(target):raise ValueError('Invalid peer predictions')
    errors=np.abs(p-target);e=np.array([np.median(errors[mask[:,i],i]) for i in range(n)])
    tau=float(np.median(errors[mask])+1e-5)
    logits=-e/tau;raw=np.exp(logits-logits.max());s=raw/raw.sum()
    return s,{'peer_target_errors':e.tolist(),'peer_error_scale':tau,'support':s.tolist(),'self_predictions_excluded':True}

def weights(method,mse,predictions,target):
    mse=np.asarray(mse,float);n=len(mse)
    if n<2 or not np.isfinite(mse).all() or np.any(mse<0):raise ValueError('Invalid quality evidence')
    s,a=support_signal(predictions,target);u=np.full(n,1/n);b=1/(mse+1e-5);b/=b.sum()
    if len(s)!=n:raise ValueError('Model/proposer alignment mismatch')
    if method=='peer_support_shrink':w=.5*u+.5*s
    elif method=='target_peer_shrink':v=b*s;v/=v.sum();w=.5*u+.5*v
    elif method=='A_uniform':w=u
    elif method=='B_stage0':w=b
    elif method=='quality_shrink':w=.5*u+.5*b
    else:raise ValueError(method)
    a.update(quality=b.tolist(),shrinkage=.5,signal_semantics='candidate support, not participant contribution')
    return w,a

def initializer(peer):
    def initial(method,agents,mse,x,y,proposals,target,**kwargs):
        if method=='ensemble_shrink':return legacy(method,agents,mse,x,y,proposals,target,**kwargs)
        w,a=weights(method,mse,peer,target);a['agent_ids']=[g.agent_id for g in agents];return w,a
    return initial
