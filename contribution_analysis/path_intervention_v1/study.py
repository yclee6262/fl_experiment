import argparse,copy,math,shutil,signal,time
from datetime import datetime,timezone
import numpy as np
from common import *
from core import advance,run_pair,point_at

def prepare():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    paths=[PLAN,L2/'cases.json',L2/'verification.json',CA/'contribution_four_way_experiment_v2_20261022/comparisons.json']
    for p in (MOD/'source_snapshot').glob('*.py'):
        if p.name in ['engine.py','host_server.py','agent_client.py','models.py','dataset.py','run_experiments.py','alpha_weight_experiments.py','research_evaluation.py','p3_contribution_validation.py']:
            shutil.copy2(p,snap/p.name);paths.append(p)
    p=L2/'source_snapshot/support.py';shutil.copy2(p,snap/'support.py');paths.append(p)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name);paths.append(p)
    cases=read(L2/'cases.json')
    for phase,cs in cases.items():
        (ROOT/phase).mkdir()
        for c in cs:
            paths.extend([Path(c['models']),Path(c['evaluator']),OLD/'baselines'/(c['key']+'.json'),MOD/phase/(c['key']+'.json'),L2/phase/(c['key']+'.json')])
            paths.extend(OLD/'removals'/(c['key']+'_'+str(i)+'.json') for i in c['ids'])
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths});save(ROOT/'source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    save(ROOT/'cases.json',cases);shutil.copy2(PLAN,ROOT/PLAN.name)
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),authorization='根據實驗設計去執行實驗',
        old_data_exploration=True,new_confirmation=False,qs=[0,.25,.5,.75],epsilon=1e-4,bootstrap_seed=20261024,repeats=10000,
        ci_percentiles=[.1,99.9],pilot_cap=100000,development_cap=900000,verification_cap=1000000,pooled_cap=4000,
        restore_restarts=True,pooled_loaded=False,truth_loaded=False))
    print(ROOT)

def bill(events,groups,qvalue,baseline):
    own={e['key'] for e in events if e['stage']=='baseline'};base=len([e for e in events if e['stage']=='baseline']);extra=0;req=0
    for g in groups:
        if qvalue not in g['q_labels']:continue
        for name in ['first','P','P_common']:
            begin,end=g['spans'][name]
            for e in events[begin:end]:
                req+=1
                if e['key'] not in own:extra+=1;own.add(e['key'])
    assert base==baseline
    return dict(q=qvalue,baseline_samples=base,new_samples=extra,total_samples=base+extra,extra_requests=req)

def record(c,phase,budget):
    from support import load_server
    from engine import run
    s=load_server({k:v for k,v in c.items() if k!='evaluator'});q=Queries(s,budget);tic=time.perf_counter();n=len(c['ids'])
    base,tr=run(s,q.objective(list(range(n))),30,initial=c['fixed_start'])
    old=read(OLD/'baselines'/(c['key']+'.json'));source=read(MOD/phase/(c['key']+'.json'));layer=read(L2/phase/(c['key']+'.json'))
    assert plain(base)==source['baseline'];assert len(tr.pre)==len(source['full'])
    assert len(q.events)==old['counter']['samples'];baseline_events=len(q.events)
    import hashlib
    digest=hashlib.sha256()
    for e in q.events:
        i,h=e['key'].split(':');digest.update(int(i).to_bytes(4,'little'));digest.update(bytes.fromhex(h))
    assert digest.hexdigest()==old['query_order_sha256']
    T=len(tr.pre);manifest={}
    for value in [0,.25,.5,.75]:manifest.setdefault(math.floor(value*T),[]).append(value)
    q.stage='F';controls={};groups=[]
    # New baseline trace supplies F; no dependence on historical counterfactual replies.
    for t,labels in manifest.items():
        pre=tr.pre[t];near(pre['S_current'],pre['global_best_S']);assert pre['best_loss']==pre['global_best_loss']
        full=advance(s,q.objective(list(range(n))),pre,30-t);near(full['result']['solution'],base['solution'])
        assert full['result']['history']==base['history'] and full['result']['states']==base['states']
        expected=source['full'][t:]
        assert len(full['steps'])==len(expected)
        for a,b in zip(full['steps'],expected):
            near(a['post']['S_current'],b['post']['S_current']);assert plain(a['trials'])==b['trials']
        controls[str(t)]=full
        for i in range(n):
            q.context=dict(t=t,i=i);out=run_pair(s,q,pre,i)
            orig=next(g for g in layer['groups'] if g['t']==t and g['i']==i)
            near(out['first']['result']['solution'],orig['roles']['joint']['endpoint'])
            if t==0:
                rem=read(OLD/'removals'/(c['key']+'_'+str(c['ids'][i])+'.json'))
                near(out['P']['result']['solution'],rem['result']['solution'])
                # Segmented histories differ in labels only; accepted positions are checked independently.
            out.update(t=t,i=i,agent_id=c['ids'][i],q_labels=labels,full_accepted=source['full'][t]['accepted'],mode=pre['current_method'])
            out['observations']={str(h):{role:point_at([out['first'],out[role]],t+h) for role in ['T','P']} for h in [1,3,5]}
            groups.append(out)
    # All decisions fixed. These queries can never influence continuation states.
    q.stage='common';q.context={};L0=q.full(base['solution'])
    for g in groups:
        t=g['t'];q.context=dict(t=t,i=g['i']);q.stage='common'
        lf=q.full(source['full'][t]['post']['S_current']);one=q.full(g['first']['result']['solution'])
        g['j']=one-lf
        orig=next(v for v in layer['groups'] if v['t']==t and v['i']==g['i']);near(g['j'],orig['j'])
        start=len(q.events);q.stage='P_common';pv=q.full(g['P']['result']['solution']);fv=q.full(base['solution'])
        g['spans']['P_common']=[start,len(q.events)];q.stage='common';tv=q.full(g['T']['result']['solution']);av=q.full(g['T_abs_x'])
        g.update(tau=tv-L0,pi=pv-L0,u=pv-tv,tau_abs=av-L0,full_final_L=L0,
            endpoints_locked=True)
        for h,obs in g['observations'].items():
            fp=point_at([controls[str(t)]],t+int(h));ll=q.full(fp['x'])
            for role in ['T','P']:obs[role]['difference']=q.full(obs[role]['x'])-ll
            obs['F']=fp
        near(g['observations']['1']['T']['difference'],g['j']);near(g['observations']['1']['P']['difference'],g['j'])
    ledgers=[bill(q.events,groups,v,baseline_events) for v in [0,.25,.5,.75]]
    return dict(case=c,phase=phase,baseline=base,baseline_trace=dict(pre=tr.pre,post=tr.post,final=tr.final,trials=tr.trials),
        controls=controls,groups=groups,checkpoint_manifest=manifest,cost_events=q.events,raw_replies=list(q.raw.values()),deployment_ledgers=ledgers,
        physical_samples=sum(not e['hit'] for e in q.events),baseline_samples=baseline_events,wall_seconds=time.perf_counter()-tic,
        pooled_loaded=False,truth_loaded=False)

def phase_run(phase):
    check();assert read(ROOT/'technical_results.json')['accepted']
    cap=100000 if phase=='pilot' else read(ROOT/'development_budget.json')['cap'];secs=180 if phase=='pilot' else 900
    budget=Budget(cap,secs);start=time.perf_counter();done=[];counts=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('phase_wall_cap')));signal.alarm(secs)
    try:
        for c in read(ROOT/'cases.json')[phase]:
            print(phase,c['key'],flush=True);r=record(c,phase,budget);save(ROOT/phase/(c['key']+'.json'),r)
            counts.append(dict(key=c['key'],groups=len(r['groups']),samples=r['physical_samples']));done.append(c['key'])
        save(ROOT/(phase+'_completion.json'),dict(complete=True,keys=done,counts=counts,physical_samples=budget.samples,wall_seconds=time.perf_counter()-start))
        if phase=='pilot':
            raw=int(np.ceil(2*320*max(v['samples']/v['groups'] for v in counts)))
            save(ROOT/'development_budget.json',dict(cap=min(raw,900000),unclipped_estimate=raw,effect_used=False))
            assert raw<=900000,'pilot_forecast_exceeds_cap'
        else:save(ROOT/'scores_and_points_locked.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for ph in ['pilot','development'] for p in (ROOT/ph).glob('*.json')},pooled_loaded=False))
        check()
    except Exception as e:save(ROOT/(phase+'_failure.json'),dict(error=repr(e),completed=done,samples=budget.samples));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','pilot','development']);args=a.parse_args()
    if args.mode=='prepare':prepare()
    else:setup();phase_run(args.mode)
