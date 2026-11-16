"""Train-only bootstrap risk, heldout uniform shrink; no external evaluator input."""
import numpy as np
from oob_shrinkage import tune_shrinkage
from next_alpha_initialization import initial_alpha as previous_initial_alpha

METHODS=['A_uniform','B_stage0','bootstrap_conservative_quality',
         'crossfit_conservative_unshrunk','crossfit_conservative_shrink']
PRIMARY=METHODS[-1]

def risk_weights(prediction,labels,replicates,seed):
    p=np.asarray(prediction,float);y=np.asarray(labels,float).reshape(-1)
    error=(p-y[:,None])**2
    if not np.isfinite(error).all():raise FloatingPointError('Nonfinite squared error')
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(y),(replicates,len(y)))
    risk=np.quantile(error[idx].mean(axis=1),.9,axis=0)
    score=1/(risk+1e-5);return score/score.sum(),risk

def crossfit_weights(p,y,seed=20261011,replicates=1000,repeats=20,shrink=True):
    p=np.asarray(p,float);y=np.asarray(y,float).reshape(-1)
    if p.ndim!=2 or len(p)!=len(y) or p.shape[1]<1 or len(y)<10 or len(y)%5 or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid fivefold qualification data')
    if not isinstance(replicates,(int,np.integer)) or replicates<2 or not isinstance(repeats,(int,np.integer)) or repeats<1:raise ValueError('Invalid resampling settings')
    results=[];folds=[];uniform=np.full(p.shape[1],1/p.shape[1])
    for repeat in range(repeats):
        permutation=np.random.default_rng(np.random.SeedSequence([seed,20261010,repeat])).permutation(len(y))
        for fold,hold in enumerate(np.array_split(permutation,5)):
            train=np.setdiff1d(np.arange(len(y)),hold)
            risk_seed=np.random.SeedSequence([seed,20261010,repeat,fold,1])
            q,risk=risk_weights(p[train],y[train],replicates,risk_seed)
            if shrink:lam,w,loss=tune_shrinkage(p[hold],y[hold],q)
            else:lam,w,loss=0.,q,None
            results.append(w)
            folds.append({'repeat':repeat,'fold':fold,'train_indices':train.tolist(),'heldout_indices':hold.tolist(),
                          'risk_mse':risk.tolist(),'risk_weight':q.tolist(),'lambda':lam,'weight':w.tolist(),'heldout_losses':loss})
    alpha=np.mean(results,axis=0);alpha/=alpha.sum()
    return alpha,{'fallback_reason':'','bootstrap_seed':int(seed),'bootstrap_replicates':int(replicates),
        'risk_quantile':.9,'fold_count':5,'repeats':int(repeats),'fits':len(folds),
        'lambda_mean':float(np.mean([f['lambda'] for f in folds])),
        'fold_audits':folds,'shrink_enabled':bool(shrink),
        'tuning_loss':'heldout mixture MSE, not unbiased quality estimate',
        'definition':'train-only 90th percentile inverse bootstrap MSE, heldout uniform shrink, average'}

def initial_alpha(method,agents,mse,blind_x,blind_y,proposals,target,**kwargs):
    if method not in METHODS[-2:]:return previous_initial_alpha(method,agents,mse,blind_x,blind_y,proposals,target,**kwargs)
    if not len(agents) or len(agents)!=len(mse) or len(agents)!=len(proposals):raise ValueError('Coalition mismatch')
    ids=[int(a.agent_id) for a in agents]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate agent IDs')
    p=kwargs.get('cached_blind_predictions');reused=p is not None
    if p is None:p=np.column_stack([a.api_predict(blind_x) for a in agents])
    if np.asarray(p).shape!=(len(blind_y),len(agents)):raise ValueError('Prediction shape mismatch')
    alpha,audit=crossfit_weights(p,blind_y,kwargs.get('bootstrap_seed',20261011),kwargs.get('bootstrap_replicates',1000),20,method==PRIMARY)
    audit.update(agent_ids=ids,blind_prediction_source='qualification_reuse' if reused else 'fresh_api',raw_scores=alpha.tolist())
    return alpha,audit
