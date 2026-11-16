"""Fixed next-round methods; only local qualification data enter weights."""
import numpy as np
from scipy.optimize import minimize
from oob_shrinkage import tune_shrinkage
from initial_alpha_methods import initial_alpha as legacy_initial_alpha

METHODS = ['A_uniform','B_stage0','bootstrap_oob_quality_shrink',
           'bootstrap_oob_ensemble_shrink','bootstrap_conservative_quality']

def combination(p, y):
    k=p.shape[1]; u=np.full(k,1/k)
    result=minimize(lambda w: float(np.mean((p@w-y)**2)),u,
        jac=lambda w: 2*p.T@(p@w-y)/len(y),method='SLSQP',
        bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1,
        'jac':lambda w:np.ones(k)},options={'ftol':1e-12,'maxiter':500})
    if not result.success or not np.isfinite(result.x).all():
        return u,False
    w=np.maximum(result.x,0);w/=w.sum()
    return w,True

def next_weights(method,p,y,replicates=1000,seed=20261011):
    p=np.asarray(p,float);y=np.asarray(y,float).reshape(-1)
    if p.ndim!=2 or p.shape[0]!=len(y) or len(y)<2 or p.shape[1]<1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid qualification data')
    if not isinstance(replicates,(int,np.integer)) or replicates<2:
        raise ValueError('Invalid replicate count')
    rng=np.random.default_rng(seed); samples=rng.integers(0,len(y),(replicates,len(y)))
    audit={'bootstrap_seed':int(seed),'bootstrap_replicates':int(replicates),'fallback_reason':''}
    if method=='bootstrap_conservative_quality':
        losses=((p-y[:,None])**2)[samples].mean(axis=1)
        risk=np.quantile(losses,.9,axis=0); scores=1/(risk+1e-5);w=scores/scores.sum()
        return w,{**audit,'risk_quantile':.9,'risk_mse':risk.tolist(),'definition':'inverse bootstrap MSE 90th percentile'}
    if method!='bootstrap_oob_ensemble_shrink':raise ValueError(method)
    ws=[];lams=[];failed=0;skipped=0
    for sample in samples:
        hold=np.ones(len(y),bool);hold[np.unique(sample)]=False
        if hold.sum()<2:skipped+=1;continue
        w,ok=combination(p[sample],y[sample]);failed+=int(not ok)
        lam,adjusted,_=tune_shrinkage(p[hold],y[hold],w)
        ws.append(adjusted);lams.append(lam)
    audit.update(accepted_replicates=len(ws),skipped_small_oob=skipped,combination_solver_fallbacks=failed,
        lambda_mean=float(np.mean(lams)) if lams else None,
        definition='inbag simplex mixture MSE optimum, OOB analytic uniform shrink, average')
    if not ws:
        audit['fallback_reason']='no_eligible_oob_use_uniform';return np.full(p.shape[1],1/p.shape[1]),audit
    w=np.mean(ws,axis=0);return w/w.sum(),audit

def initial_alpha(method,agents,mse,blind_x,blind_y,proposals,target,**kwargs):
    if method not in METHODS[3:]:return legacy_initial_alpha(method,agents,mse,blind_x,blind_y,proposals,target,**kwargs)
    if not len(agents) or len(agents)!=len(mse) or len(agents)!=len(proposals):raise ValueError('Coalition mismatch')
    ids=[int(a.agent_id) for a in agents]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate agent IDs')
    p=kwargs.get('cached_blind_predictions')
    reused=p is not None
    if p is None:p=np.column_stack([a.api_predict(blind_x) for a in agents])
    if np.asarray(p).shape!=(len(blind_y),len(agents)):raise ValueError('Prediction shape mismatch')
    w,audit=next_weights(method,p,blind_y,kwargs.get('bootstrap_replicates',1000),kwargs.get('bootstrap_seed',20261011))
    audit.update(agent_ids=ids,blind_prediction_source='qualification_reuse' if reused else 'fresh_api',raw_scores=w.tolist())
    return w,audit
