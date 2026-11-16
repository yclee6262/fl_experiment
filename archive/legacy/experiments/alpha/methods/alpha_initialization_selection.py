"""Deployment-only C/P selectors. No evaluator or synthetic truth imports."""
import torch  # Existing runtime import order.
import numpy as np
from scipy.optimize import minimize
from peer_support_alpha import support_signal

METHODS = ['A_uniform','B_stage0','stacked_full','ensemble_shrink','B_stacked_half','pure_support','peer_support_shrink','support_quality_mix50']
DISPLAY = dict(zip(METHODS, ['A','Q','E','EU','EQ','S','SU','SQ']))
TIE_ORDER = ['ensemble_shrink','stacked_full','support_quality_mix50','A_uniform','B_stage0','B_stacked_half','pure_support','peer_support_shrink']
SCENARIOS = ['reference_clean','low_quality_25pct','sample_imbalance_4x']

def choose(scores):
    good = {m:float(v) for m,v in scores.items() if m in TIE_ORDER and v is not None and np.isfinite(v)}
    if not good: return None
    best = min(good.values())
    return next(m for m in TIE_ORDER if m in good and good[m] <= best+1e-12)

def weights_from_predictions(predictions, labels, peer, target):
    p,y=np.asarray(predictions,float),np.asarray(labels,float)
    if p.ndim!=2 or y.shape!=(len(p),) or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid paired blind predictions/labels')
    n=p.shape[1];u=np.full(n,1/n);mse=((p-y[:,None])**2).mean(axis=0)
    q=1/(mse+1e-5);q/=q.sum()
    fit=minimize(lambda w:float(np.mean((p@w-y)**2)),u,method='SLSQP',bounds=[(0.,1.)]*n,
                 constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':500})
    fallback = not fit.success or not np.isfinite(fit.x).all() or np.min(fit.x)<-1e-8
    if fallback:
        w=q.copy(); eu=u.copy()  # Preserve legacy EU fallback, pure E/EQ fallback Q.
    else:
        w=np.maximum(fit.x,0);w/=w.sum();eu=.5*w+.5*u
    s,sa=support_signal(peer,target)
    if len(s)!=n:raise ValueError('Model/support alignment')
    values=[u,q,w,eu,q if fallback else .5*w+.5*q,s,.5*s+.5*u,.5*s+.5*q]
    return dict(zip(METHODS,values)), {'mse':mse.tolist(),'ensemble_fallback':fallback,'support':sa}

def select_c(predictions, labels, peer, target):
    p,y=np.asarray(predictions,float),np.asarray(labels,float)
    if len(y)!=20:raise ValueError('C requires the registered 20 blind labels')
    permutation=np.random.default_rng(20261015).permutation(20)
    residuals={m:np.empty(20) for m in METHODS};folds=[]
    for held in permutation.reshape(5,4):
        train=np.setdiff1d(np.arange(20),held)
        w,a=weights_from_predictions(p[train],y[train],peer,target)
        for m in METHODS:residuals[m][held]=p[held]@w[m]-y[held]
        folds.append({'train':train.tolist(),'held_out':held.tolist(),'alpha':{m:x.tolist() for m,x in w.items()},'audit':a})
    scores={m:float(np.sqrt(np.mean(r*r))) for m,r in residuals.items()}
    selected=choose(scores)
    return {'strategy':'C','selected_method':selected or 'B_stage0','scores':scores,'folds':folds,
            'out_of_fold_residuals':{m:r.tolist() for m,r in residuals.items()},'fallback_reason':'' if selected else 'no_valid_scores_use_Q'}

def select_p(rows, predictions, quality, target):
    q=np.asarray(quality,float)
    if q.ndim!=1 or not np.isfinite(q).all() or np.any(q<0) or abs(q.sum()-1)>1e-8:
        return {'strategy':'P','selected_method':None,'scores':{},'fallback_reason':'invalid_quality'}
    scores={};evidence={}
    for m,row in rows.items():
        x=row.get('X')
        if row.get('status')!='valid' or x is None or not np.isfinite(x).all() or np.max(np.abs(x))>1+1e-8:continue
        p=np.asarray(predictions[m],float)
        if p.shape!=q.shape or not np.isfinite(p).all():continue
        scores[m]=float(q@np.abs(p-target));evidence[m]=p.tolist()
    selected=choose(scores)
    return {'strategy':'P','selected_method':selected,'scores':scores,'predictions':evidence,
            'quality':q.tolist(),'fallback_reason':'' if selected else 'no_valid_scored_candidates'}
