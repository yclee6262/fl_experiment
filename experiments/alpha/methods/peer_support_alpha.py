"""Candidate support is computed only from local predictions, never pooled scores."""
import numpy as np
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
    elif method=='A_uniform':w=u
    elif method=='B_stage0':w=b
    elif method=='quality_shrink':w=.5*u+.5*b
    else:raise ValueError(method)
    a.update(quality=b.tolist(),shrinkage=.5,signal_semantics='candidate support, not participant contribution')
    return w,a
