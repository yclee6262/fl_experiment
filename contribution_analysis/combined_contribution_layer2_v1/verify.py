"""Independent original-control-flow and raw-score rebuild, no branch imports."""
import copy
import signal
import time
import numpy as np
from study import ROOT,read,save,setup,check,key,sha,plain

def verify():
    from support import load_server,Budget
    from independent_step import one_round
    check();lock=read(ROOT/'scores_and_points_locked.json')['hashes']
    for n,h in lock.items():assert sha(ROOT/n)==h
    budget=Budget(600000,1800);start=time.perf_counter();maximum=0.;groups=0;branches=0;maxraw=0.
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verify_wall_budget')));signal.alarm(1800)
    for phase,cs in read(ROOT/'cases.json').items():
        for c in cs:
            print('VERIFY',c['key'],flush=True);r=read(ROOT/phase/(c['key']+'.json'))
            assert sha(r['inherited_replies_file'])==r['inherited_file_sha256']
            prior=read(r['inherited_replies_file']);s=load_server(c);w=np.asarray(c['alphas']);w=w/w.sum()
            table={(v['i'],key(v['x'])):v['y'] for v in prior['replies']+r['new_replies']}
            for v in r['new_replies']:
                budget.before();t=time.perf_counter();y=float(s.trusted_agents[v['i']].api_predict(v['x'])[0]);budget.charge(time.perf_counter()-t)
                maxraw=max(maxraw,abs(y-v['y']));assert y==v['y']
                assert np.asarray(v['x'],np.float32).tolist()==v['api_input_float32']
            for g in r['groups']:
                vals={};i=g['i'];t=g['t']
                for role in ['anchor','model','joint']:
                    b=g['roles'][role];panel=[k for k in range(len(w)) if role not in ['model','joint'] or k!=i]
                    anchors=[k for k in range(len(w)) if role not in ['anchor','joint'] or k!=i]
                    sw=w[panel];sw=sw/sw.sum();sbranch=copy.copy(s)
                    sbranch.I_list=np.asarray(c['candidates'])[anchors];sbranch.alphas=w[anchors]/w[anchors].sum()
                    assert b['panel']==panel and b['anchor_indices']==anchors
                    np.testing.assert_array_equal(sbranch.alphas,b['direction_weights'])
                    def obj(x):return float(np.dot(sw,[abs(table[j,key(x)]) for j in panel]))
                    pre=copy.deepcopy(prior['full'][t]['pre']);pre['best_loss']=obj(pre['S_current'])
                    pre['anchor_losses']=[obj(a) for a in sbranch.I_list]
                    outcome=one_round(sbranch,pre,obj);expected=b['decision']
                    for name in ['S_current','best_loss','eta']:
                        err=float(np.max(np.abs(np.asarray(outcome['post'][name])-np.asarray(expected['post'][name]))));maximum=max(maximum,err);assert err<=1e-10
                    assert outcome['post']['current_method']==expected['post']['current_method']
                    assert outcome['post']['halted']==expected['post']['halted']
                    assert outcome['accepted']==expected['accepted']
                    np.testing.assert_allclose(outcome['g'],expected['g'],rtol=0,atol=1e-10)
                    assert len(outcome['trials'])==len(expected['trials'])
                    for a,z in zip(outcome['trials'],expected['trials']):
                        np.testing.assert_array_equal(a['x'],z['x']);assert a['loss']==z['loss'] and a['accepted']==z['accepted']
                    fullx=prior['full'][t]['post']['S_current']
                    lf=sum(float(w[j])*abs(table[j,key(fullx)]) for j in range(len(w)))
                    lx=sum(float(w[j])*abs(table[j,key(b['endpoint'])]) for j in range(len(w)))
                    vals[role]=lx-lf;maximum=max(maximum,abs(vals[role]-b['score']));assert abs(vals[role]-b['score'])<=1e-10
                    branches+=1
                assert abs(g['c2']-vals['anchor']-vals['model'])<=1e-10
                assert abs(g['residual']-vals['joint']+vals['anchor']+vals['model'])<=1e-10
                previous=next(b for b in prior['branches'] if b['t']==t and b['i']==i)
                assert plain(g['roles']['model']['decision'])==plain(previous['decision'])
                assert g['m2']==previous['m'];groups+=1
    signal.alarm(0);check()
    total=budget.samples+sum(read(ROOT/(p+'_completion.json'))['new_model_samples'] for p in ['pilot','development']);assert total<=1200000
    save(ROOT/'verification.json',dict(accepted=True,groups=groups,branches=branches,max_independent_error=maximum,
        max_raw_reply_reload_error=maxraw,new_model_samples=budget.samples,all_stage_new_samples=total,
        wall_seconds=time.perf_counter()-start,model_seconds=budget.model_seconds,M2_exact_reproduction=True))

if __name__=='__main__':setup();verify()
