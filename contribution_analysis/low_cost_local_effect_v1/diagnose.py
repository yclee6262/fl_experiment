"""Prespecified truncation decomposition; read-only after all scores are locked."""
import numpy as np
from common import *
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def diagnose():
    check();rows=[];zeros=[]
    for c in read(ROOT/'cases.json')['development']:
        r=read(ROOT/'development'/(c['key']+'.json'));o=read(ORACLE/'development'/(c['key']+'.json'))
        allreplies={str(v['i'])+':'+key(v['x']):v for v in r['normal_replies']+r['new_replies']}
        w=np.asarray(c['alphas']);w=w/w.sum()
        def L(x):return float(np.dot(w,[abs(allreplies[str(i)+':'+key(x)]['y']) for i in range(len(w))]))
        for p in r['predictions']:
            if p['method']=='N0':
                zeros.append(p['score']);continue
            g=next(g for g in o['groups'] if g['t']==p['t'] and g['i']==p['i']);branch=g['roles'][p['role']]
            exact_minus=float(np.dot(w,branch['common_losses']));exact_full=g['full_loss']
            de_minus=L(p['endpoint'])-exact_minus;de_full=L(p['full_one_endpoint'])-exact_full
            near(de_minus-de_full,p['score']-branch['score'])
            rows.append(dict(key=c['key'],scenario=c['scenario'],t=p['t'],agent_id=p['agent_id'],role=p['role'],
                minus_truncation_loss_change=de_minus,full_truncation_loss_change=de_full,
                prediction_error=de_minus-de_full,both_first_rejected=(not p['accepted'] and not (g['full']['trials'] and g['full']['trials'][0]['accepted']))))
    save(ROOT/'truncation_decomposition.json',rows)
    aggregate=[]
    for scenario in ['clean','controlled_poison']:
        for role in ['anchor','model','joint']:
            rr=[r for r in rows if r['scenario']==scenario and r['role']==role]
            aggregate.append(dict(scenario=scenario,role=role,n=len(rr),both_first_rejected=sum(r['both_first_rejected'] for r in rr),
                minus_abs_change_sum=sum(abs(r['minus_truncation_loss_change']) for r in rr),
                full_abs_change_sum=sum(abs(r['full_truncation_loss_change']) for r in rr)))
    save(ROOT/'mechanism_diagnostics.json',dict(N0_min_score=min(zeros),N0_max_score=max(zeros),N0_negative_count=sum(v< -1e-4 for v in zeros),
        N1_truncation=aggregate,method_changed=False,new_model_samples=0,pooled_loaded=False,truth_loaded=False))
if __name__=='__main__':diagnose()
