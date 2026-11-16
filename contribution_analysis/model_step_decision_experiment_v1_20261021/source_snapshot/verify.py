"""Independent decision interpreter. Does not import engine or branch scoring."""
import copy
import sys
import time
from pathlib import Path
import numpy as np
from study import ROOT,read,save,check,setup,load_server,key,Budget,sha,plain

def one_round(s,pre,objective):
    """Direct reconstruction of frozen custom control flow, independent of AST hooks."""
    out=copy.deepcopy(pre);x=np.asarray(pre['S_current'],float).copy()
    w=np.asarray(s.alphas,float);w=w/w.sum();g=np.zeros_like(x)
    for k,anchor in enumerate(s.I_list):
        direction=anchor-x;distance=np.linalg.norm(direction)
        if distance<1e-8:continue
        unit=direction/distance
        if pre['current_method']=='secant':derivative=(pre['anchor_losses'][k]-pre['best_loss'])/distance
        else:
            width=bounded(s,x,unit,pre['delta'])
            if width<=1e-12:continue
            derivative=(objective(x+width*unit)-pre['best_loss'])/width
        g+=w[k]*derivative*unit
    norm=np.linalg.norm(g);trials=[];accepted=False;halted=False
    if norm<=1e-8:halted=True
    else:
        direction=g/norm;eta=pre['eta']
        for attempt in range(10):
            z=x-bounded(s,x,-direction,eta)*direction;loss=float(objective(z))
            accept=bool(loss<pre['best_loss']);trials.append(dict(x=z,loss=loss,accepted=accept))
            if accept:
                out['S_current']=z;out['best_loss']=loss;out['eta']=min(.5,eta*1.5)
                accepted=True;break
            eta/=2.
        if not accepted:
            if pre['current_method']=='secant':out['current_method']='dynamic';out['eta']=.5
            else:halted=True
    out['halted']=halted
    return dict(post=out,trials=trials,accepted=accepted,g=g)

def bounded(s,x,d,requested):
    limit=float(requested)
    for value,delta in zip(x,d):
        if delta>0:limit=min(limit,(s.feasible_upper-value)/delta)
        elif delta<0:limit=min(limit,(s.feasible_lower-value)/delta)
    return max(0.,limit)

def verify():
    check();cfg=read(ROOT/'registration.json');lock=read(ROOT/'scores_and_points_locked.json')
    for n,h in lock['hashes'].items():assert sha(ROOT/n)==h
    budget=Budget(cfg['verifier_limit'],cfg['verifier_seconds']);max_error=0.;cases=[];points=0
    start=time.perf_counter()
    for phase in ['pilot','development']:
        for c in read(ROOT/'cases.json')[phase]:
            print('VERIFY',c['key'],flush=True);r=read(ROOT/phase/(c['key']+'.json'));s=load_server(c)
            table={(v['i'],key(v['x'])):v['y'] for v in r['replies']}
            for v in r['replies']:
                budget.before();t=time.perf_counter();y=float(s.trusted_agents[v['i']].api_predict(v['x'])[0]);budget.charge(time.perf_counter()-t)
                assert y==v['y'],'raw_reply_reload'
                assert np.asarray(v['x'],np.float32).tolist()==v['api_input_float32']
            w=np.asarray(c['alphas']);w=w/w.sum();n=len(w)
            def objective(ids):
                sw=w[ids];sw=sw/sw.sum()
                def f(x):return float(np.dot(sw,[abs(table[j,key(x)]-s.target_T) for j in ids]))
                return f
            def same(actual,expected):
                nonlocal max_error
                for name in ['S_current','best_loss','eta']:
                    err=float(np.max(np.abs(np.asarray(actual['post'][name])-np.asarray(expected['post'][name]))));max_error=max(max_error,err)
                    assert err<=1e-10*max(1.,float(np.max(np.abs(expected['post'][name])))),('state',name)
                assert actual['post']['current_method']==expected['post']['current_method']
                assert actual['post']['halted']==expected['post']['halted']
                assert actual['accepted']==expected['accepted']
                np.testing.assert_allclose(actual['g'],expected['g'],rtol=0,atol=1e-10)
                assert len(actual['trials'])==len(expected['trials'])
                for a,b in zip(actual['trials'],expected['trials']):
                    np.testing.assert_array_equal(a['x'],b['x']);assert a['loss']==b['loss'] and a['accepted']==b['accepted']
            # Complete branch reconstruction includes failures and zero-direction termination.
            for fe in r['full']:
                same(one_round(s,fe['pre'],objective(list(range(n)))),fe)
            for b in r['branches']:
                if b['status']!='available':continue
                ids=[j for j in range(n) if j!=b['i']];obj=objective(ids)
                pre=copy.deepcopy(r['full'][b['t']]['pre']);x=np.asarray(pre['S_current'])
                pre['best_loss']=obj(x);pre['anchor_losses']=[obj(a) for a in s.I_list]
                result=one_round(s,pre,obj);same(result,b['decision'])
                np.testing.assert_array_equal(b['direction_weights'],s.alphas)
                lf=[abs(table[j,key(b['x_full'])]-s.target_T) for j in range(n)]
                lm=[abs(table[j,key(b['x_minus'])]-s.target_T) for j in range(n)]
                m=sum(float(w[j])*float(lm[j]) for j in range(n))-sum(float(w[j])*float(lf[j]) for j in range(n))
                err=abs(m-b['m']);max_error=max(max_error,err);assert err<=1e-10
                own=float(w[b['i']])*(lm[b['i']]-lf[b['i']])
                other=sum(float(w[j])*(lm[j]-lf[j]) for j in ids)
                assert abs(own-b['own'])<=1e-10 and abs(other-b['others'])<=1e-10
            cases.append(dict(key=c['key'],rounds=len(r['full']),branches=len(r['branches']),accepted=True))
    save(ROOT/'verification.json',dict(accepted=True,cases=cases,max_independent_error=max_error,
        new_model_samples=budget.samples,model_seconds=budget.model_seconds,wall_seconds=time.perf_counter()-start,
        does_not_import_branch_or_score=True,raw_replies_reloaded=True))

if __name__=='__main__':
    setup();verify()
