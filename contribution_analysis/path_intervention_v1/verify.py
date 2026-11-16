"""Independent interpreter and billing audit, without core/engine/study imports."""
import copy,hashlib,time
import numpy as np
from common import *

def bound(s,x,d,h):
    v=float(h)
    for a,b in zip(x,d):
        if b>0:v=min(v,(s.feasible_upper-a)/b)
        elif b<0:v=min(v,(s.feasible_lower-a)/b)
    return max(0.,v)
def reset(pre,f,anchors):
    s=copy.deepcopy(pre);x=np.array(s['S_current'],float);loss=f(x)
    s.update(S_current=x,global_best_S=x.copy(),best_loss=loss,global_best_loss=loss,anchor_losses=[f(np.asarray(v)) for v in anchors],
        history=[loss],states=['Intervention Start'],halted=False)
    return s
def walk(server,f,pre,limit):
    state=copy.deepcopy(pre);steps=[]
    for _ in range(limit):
        if state['halted']:break
        before=copy.deepcopy(state);x=np.asarray(state['S_current']);g=np.zeros_like(x);ws=np.asarray(server.alphas);ws=ws/ws.sum()
        for k,a in enumerate(server.I_list):
            vector=a-x;distance=np.linalg.norm(vector)
            if distance<1e-8:continue
            unit=vector/distance
            if state['current_method']=='secant':slope=(state['anchor_losses'][k]-state['best_loss'])/distance
            else:
                h=bound(server,x,unit,state['delta'])
                if h<=1e-12:continue
                slope=(f(x+h*unit)-state['best_loss'])/h
            g+=ws[k]*slope*unit
        norm=np.linalg.norm(g);trials=[];reason=None;accepted=False;label=state['current_method']
        if norm<=1e-8:state['halted']=True;reason='zero_direction'
        else:
            direction=g/norm;eta=state['eta']
            for attempt in range(10):
                z=x-bound(server,x,-direction,eta)*direction;loss=float(f(z));ok=loss<state['best_loss']
                trials.append(dict(iteration=state['next_iteration'],x=z,loss=loss,accepted=bool(ok)))
                if ok:
                    accepted=True;state['S_current']=z;state['best_loss']=loss;state['eta']=min(.5,eta*1.5)
                    if loss<state['global_best_loss']:state['global_best_S']=z.copy();state['global_best_loss']=loss
                    if attempt:label='Annealing Triggered'
                    break
                eta/=2.
            if not accepted:
                if state['current_method']=='secant':state['current_method']='dynamic';state['eta']=.5
                else:state['halted']=True;reason='dynamic_no_improvement'
            if not state['halted']:
                state['history'].append(float(f(np.asarray(state['S_current']))));state['states'].append(label)
        state['next_iteration']+=1
        steps.append(dict(pre=before,post=copy.deepcopy(state),trials=trials,g=g,stop_reason=reason,accepted=accepted))
    result=dict(solution=state['global_best_S'],loss=float(f(np.asarray(state['global_best_S']))))
    return dict(result=result,final=state,steps=steps)
def compare(actual,expected):
    err=near(actual['result']['solution'],expected['result']['solution']);err=max(err,near(actual['result']['loss'],expected['result']['loss']))
    assert len(actual['steps'])==len(expected['steps'])
    for a,b in zip(actual['steps'],expected['steps']):
        assert a['stop_reason']==b['stop_reason'] and a['accepted']==b['accepted'];err=max(err,near(a['g'],b['g']))
        assert len(a['trials'])==len(b['trials'])
        for v,z in zip(a['trials'],b['trials']):
            assert v['accepted']==z['accepted'] and v['iteration']==z['iteration'];err=max(err,near(v['x'],z['x']),near(v['loss'],z['loss']))
        for field in ['S_current','global_best_S','best_loss','global_best_loss','eta','delta','anchor_losses','history']:
            err=max(err,near(a['post'][field],b['post'][field]))
        for field in ['halted','current_method','next_iteration','states']:assert a['post'][field]==b['post'][field]
    return err
def verify():
    from support import load_server
    check();lock=read(ROOT/'scores_and_points_locked.json')['hashes']
    for p,h in lock.items():assert sha(ROOT/p)==h
    budget=Budget(1000000,1200);maxerr=0.;groups=0;cases=[];ledgers=[]
    for phase,cs in read(ROOT/'cases.json').items():
        for c in cs:
            print('VERIFY',c['key'],flush=True);r=read(ROOT/phase/(c['key']+'.json'));s=load_server({k:v for k,v in c.items() if k!='evaluator'})
            raw={key(v['i'],v['x']):v for v in r['raw_replies']}
            for v in raw.values():
                budget.before();tic=time.perf_counter();y=float(s.trusted_agents[v['i']].api_predict(np.asarray(v['x']))[0]);budget.charge(time.perf_counter()-tic)
                assert y==v['y'],'raw_reply_changed'
            indices=list(range(len(c['ids'])));weights=np.asarray(c['alphas']);weights=weights/weights.sum()
            def objective(panel):
                ww=np.asarray(c['alphas'])[panel];ww=ww/ww.sum()
                return lambda x:float(np.dot(ww,[abs(raw[key(j,x)]['y']) for j in panel]))
            full=objective(indices)
            for t,control in r['controls'].items():
                pre=r['baseline_trace']['pre'][int(t)];maxerr=max(maxerr,compare(walk(s,full,pre,30-int(t)),control))
            for g in r['groups']:
                t=g['t'];i=g['i'];pre=r['baseline_trace']['pre'][t];panel=[j for j in indices if j!=i]
                b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];b.alphas=np.asarray(s.alphas)[panel]/sum(np.asarray(s.alphas)[panel])
                f=objective(panel);one=walk(b,f,reset(pre,f,b.I_list),1);maxerr=max(maxerr,compare(one,g['first']))
                remaining=30-one['final']['next_iteration'];p=walk(b,f,one['final'],remaining);maxerr=max(maxerr,compare(p,g['P']))
                if remaining:
                    tt=walk(s,full,reset(one['final'],full,s.I_list),remaining);maxerr=max(maxerr,compare(tt,g['T']))
                else:tt=one
                assert g['restart']==bool(one['final']['halted'] and remaining>0)
                lf=full(np.asarray(r['baseline']['solution']));lp=full(np.asarray(p['result']['solution']));lt=full(np.asarray(tt['result']['solution']))
                maxerr=max(maxerr,near(lp-lf,g['pi']),near(lt-lf,g['tau']),near(lp-lt,g['u']))
                # Independently locate h endpoints; no shared point_at helper.
                for h in [1,3,5]:
                    for role,seg in [('P',p),('T',tt)]:
                        allsteps=one['steps']+(seg['steps'] if seg is not one else [])
                        chosen=[v['post'] for v in allsteps if v['post']['next_iteration']<=t+h][-1]
                        maxerr=max(maxerr,near(chosen['global_best_S'],g['observations'][str(h)][role]['x']))
                groups+=1
            # Audit research physical count and every standalone-q cache without inherited oracle keys.
            seen=set();charged=0;digest=hashlib.sha256()
            for e in r['cost_events']:
                expected=e['stage']!='baseline' and e['key'] in seen;assert expected==e['hit']
                charged+=int(not expected);seen.add(e['key'])
                if e['stage']=='baseline':
                    j,hx=e['key'].split(':');digest.update(int(j).to_bytes(4,'little'));digest.update(bytes.fromhex(hx))
            assert charged==r['physical_samples'];assert digest.hexdigest()==read(OLD/'baselines'/(c['key']+'.json'))['query_order_sha256']
            for item in r['deployment_ledgers']:
                cache={e['key'] for e in r['cost_events'] if e['stage']=='baseline'};new=0
                for g in r['groups']:
                    if item['q'] not in g['q_labels']:continue
                    for name in ['first','P','P_common']:
                        a,z=g['spans'][name]
                        for e in r['cost_events'][a:z]:
                            if e['key'] not in cache:new+=1;cache.add(e['key'])
                assert new==item['new_samples'];ledgers.append(dict(key=c['key'],phase=phase,**item))
            cases.append(dict(key=c['key'],groups=len(r['groups']),physical_samples=charged,unique_raw=len(raw)))
    save(ROOT/'deployment_ledgers_verified.json',ledgers);save(ROOT/'verification.json',dict(accepted=True,groups=groups,cases=cases,
        independent_max_error=maxerr,raw_reload_error=0,verification_model_samples=budget.samples,wall_seconds=time.perf_counter()-budget.start,
        pooled_loaded=False,truth_loaded=False));check()
if __name__=='__main__':setup();verify()
