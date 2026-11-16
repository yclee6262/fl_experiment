"""Frozen 27 candidates plus two controls with matched additional labels."""
import numpy as np
from unified_alpha_methods import METHODS as CORE, DEFINITIONS as CORE_DEFINITIONS, initial_alpha as core_alpha
from explore_targeted_label_alpha import proposal_local_points, proposal_local_alpha
from research_evaluation import synthetic_truth

EXTRA = ['proposal_local_9','local_bias_risk_100','B_augmented29','B_augmented100']
METHODS = list(CORE)+EXTRA
CANDIDATES = list(CORE)+EXTRA[:2]
DEFINITIONS = dict(CORE_DEFINITIONS, **{
 'proposal_local_9':('候選點附近新增 9 筆','原方案：局部 9 筆 MSE 與原品質五筆保護混合後取倒數。'),
 'local_bias_risk_100':('局部偏差風險（100 筆）','原方案：20＋80 個標記估 ridge 局部偏差平方加剩餘誤差。'),
 'B_augmented29':('29 筆品質反比對照','使用與局部 9 筆法完全相同的新增點，與原 20 筆合併，個人全域 MSE 取倒數。'),
 'B_augmented100':('100 筆品質反比對照','使用與局部風險法完全相同的 100 筆點，個人全域 MSE 取倒數。')})

def labelled_inputs(method,server,seed):
    if method in ['proposal_local_9','B_augmented29']:
        extra=proposal_local_points(server.I_list,seed,-1,1)
        if method=='proposal_local_9':return extra,synthetic_truth(extra),9
    else:
        extra=np.random.default_rng(np.random.SeedSequence([20261011,seed,80])).uniform(-1,1,(80,2))
    return np.concatenate([server.test_X,extra]),np.concatenate([server.test_y,synthetic_truth(extra)]),len(extra)

def initial_alpha(method,agents,mse,x,y,proposals,target,**kwargs):
    if method in CORE:return core_alpha(method,agents,mse,x,y,proposals,target,**kwargs)
    p=kwargs.get('cached_blind_predictions')
    if p is None:p=np.column_stack([a.api_predict(x) for a in agents])
    if method=='proposal_local_9':
        w,a=proposal_local_alpha(mse,p,y);a.update(agent_ids=[g.agent_id for g in agents],fallback_reason='');return w,a
    if method=='local_bias_risk_100':return core_alpha('local_bias_risk',agents,mse,x,y,proposals,target,**kwargs)
    if method in ['B_augmented29','B_augmented100']:
        full_mse=((np.asarray(p)-np.asarray(y)[:,None])**2).mean(axis=0)
        w,a=core_alpha('B_stage0',agents,full_mse,x,y,proposals,target,**kwargs)
        a.update(augmented_mse=full_mse.tolist(),label_count=len(y));return w,a
    raise ValueError(method)
