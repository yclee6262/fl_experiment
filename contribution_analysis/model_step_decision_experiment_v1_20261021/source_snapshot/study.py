"""Single-round scoring interventions. Evaluator only in the evaluate command."""
import argparse
import copy
import hashlib
import json
import shutil
import signal
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/custom_direction_experiment_v2_20261020'
ROOT=WORK/'contribution_analysis/model_step_decision_experiment_v1_20261021'
DESIGN=WORK/'docs/model_step_decision_experiment_design_20261021.md'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def save(p,x):
    with Path(p).open('x') as f:json.dump(plain(x),f,indent=2,allow_nan=False);f.write('\n')
def key(x):return np.asarray(x,dtype=np.float64).tobytes().hex()
def check():
    for n,h in read(ROOT/'inputs.json').items():assert sha(n)==h,('input_changed',n)
    for n,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/n)==h,('source_changed',n)
def setup():
    sys.path.insert(0,str(ROOT/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
def prepare():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    prior=read(OLD/'registration.json');paths=[DESIGN,OLD/'cases.json']
    for n,h in prior['historical_source_sha256'].items():
        assert sha(WORK/n)==h and sha(OLD/'source_snapshot'/n)==h,('historical_drift',n)
        shutil.copy2(OLD/'source_snapshot'/n,snap/n);paths.append(WORK/n)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    cases=read(OLD/'cases.json')
    for rows in cases.values():
        for c in rows:paths.extend([Path(c['models']),Path(c['evaluator']),OLD/'baselines'/(c['key']+'.json')])
    shutil.copy2(DESIGN,ROOT/DESIGN.name);save(ROOT/'cases.json',cases)
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths})
    save(ROOT/'source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='依照實驗設計文件去做實作並執行，最終整理結果給我',
        new_confirmation=False,deadzone=1e-4,bootstrap_seed=20261021,repeats=10000,
        interval_percentiles=[.625,99.375],pilot_limit=100000,pilot_seconds=180,
        development_cap=500000,development_seconds_cap=900,verifier_limit=600000,verifier_seconds=1800,
        total_limit=1200000,total_seconds=2880,pooled_limit=7000,
        design_sha256=sha(DESIGN),technical_tests=read(WORK/'contribution_analysis/model_step_decision_v1/technical_results.json')))
    (ROOT/'pilot').mkdir();(ROOT/'development').mkdir()
    print(ROOT,flush=True)

def load_server(c):
    import torch
    from host_server import HostServer
    from agent_client import AgentNode
    # No constructor-generated synthetic labels: only frozen local models.
    s=HostServer.__new__(HostServer)
    s.target_T=0.;s.n_features=len(c['fixed_start']);s.constraint_mode='box_constrained'
    s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.asarray(c['candidates']);s.alphas=np.asarray(c['alphas']);s.trusted_agents=[]
    payload=torch.load(c['models'],map_location='cpu',weights_only=True)
    for i in c['ids']:
        a=AgentNode(i,None,n_features=s.n_features);a.model.load_state_dict(payload['agents'][i]);a.model.eval();s.trusted_agents.append(a)
    return s

class Budget:
    def __init__(self,limit,seconds):self.limit=limit;self.seconds=seconds;self.samples=0;self.model_seconds=0.
    def before(self):
        if self.samples+1>self.limit or self.model_seconds>=self.seconds:raise RuntimeError('pre_call_budget')
    def charge(self,dt):self.samples+=1;self.model_seconds+=dt

class Replies:
    def __init__(self,s,budget):
        self.s=s;self.budget=budget;self.cache={};self.events=[];self.cost={};self.stage='baseline';self.cache_enabled=False
        self.original=[]
        for i,a in enumerate(s.trusted_agents):
            old=a.api_predict;self.original.append((a,old))
            def predict(x,_i=i,_old=old):return np.asarray([self.reply(_i,x,_old)])
            a.api_predict=predict
    def reply(self,i,x,old=None):
        x=np.asarray(x,dtype=float);k=str(i)+':'+key(x)
        cost=self.cost.setdefault(self.stage,dict(requests=0,samples=0,seconds=0.,cache_hits=0))
        cost['requests']+=1;hit=self.cache_enabled and k in self.cache
        if hit:y=self.cache[k]['y'];cost['cache_hits']+=1
        else:
            self.budget.before();t=time.perf_counter();fn=old or self.original[i][1]
            y=float(fn(x)[0]);dt=time.perf_counter()-t;self.budget.charge(dt)
            cost['samples']+=1;cost['seconds']+=dt
            if k in self.cache:assert self.cache[k]['y']==y,'non_deterministic_reply'
            self.cache[k]=dict(i=i,x=x.copy(),api_input_float32=np.asarray(x,dtype=np.float32).tolist(),y=y)
        self.events.append(dict(stage=self.stage,i=i,x=x.copy(),y=y,cache_hit=hit))
        return y
    def objective(self,indices=None):
        ids=list(range(len(self.s.alphas))) if indices is None else list(indices)
        w=np.asarray(self.s.alphas)[ids];w=w/w.sum()
        def f(x):
            vals=[abs(self.s.trusted_agents[i].api_predict(x)[0]-self.s.target_T) for i in ids]
            return float(np.dot(w,np.asarray(vals,dtype=float)))
        return f
    def losses(self,x):return np.array([abs(a.api_predict(x)[0]-self.s.target_T) for a in self.s.trusted_agents])
    def close(self):
        for a,old in self.original:a.api_predict=old

def event(trace,t):
    pre=trace.pre[t];post=next((s for s in trace.post if s['next_iteration']==t+1),trace.final)
    trials=[v for v in trace.trials if v['iteration']==t]
    return dict(pre=pre,post=post,trials=trials,accepted=any(v['accepted'] for v in trials),
                g=next(v['g'] for v in trace.gradients if v['iteration']==t))

def case_record(c,phase,budget):
    from engine import run,branch_state
    from host_server import HostServer
    s=load_server(c);r=Replies(s,budget);start=time.perf_counter()
    try:
        obj=r.objective();base,tr=run(s,obj,30,initial=c['fixed_start'])
        digest=hashlib.sha256()
        for e in r.events:digest.update(e['i'].to_bytes(4,'little'));digest.update(np.asarray(e['x'],dtype=np.float64).tobytes())
        old=read(OLD/'baselines'/(c['key']+'.json'))
        assert digest.hexdigest()==old['query_order_sha256'],'baseline_query_order'
        assert r.cost['baseline']['samples']==old['counter']['samples'],'baseline_count'
        np.testing.assert_array_equal(base['solution'],old['result']['solution'])
        assert base['history']==old['result']['history'] and base['states']==old['result']['states'] and base['loss']==old['result']['loss']
        assert len(tr.pre)==len(old['iterations'])
        full=[event(tr,t) for t in range(len(tr.pre))]
        for e,o in zip(full,old['iterations']):
            np.testing.assert_array_equal(e['pre']['S_current'],o['x'])
            assert e['pre']['current_method']==o['mode'] and e['pre']['best_loss']==o['loss']
        # An uninstrumented replay proves no trace-induced API or solver change.
        r.stage='control_replay';begin=len(r.events)
        control=HostServer._solve_subspace_custom(s,s.I_list,obj,num_iterations=30,
            direction_weights=s.alphas,initial_solution=np.asarray(c['fixed_start']))
        assert plain(base)==plain(control),'uninstrumented_result'
        assert [(e['i'],key(e['x'])) for e in r.events[:begin]]==[(e['i'],key(e['x'])) for e in r.events[begin:]]
        r.cache_enabled=True;branches=[];n=len(c['ids']);w=np.asarray(s.alphas);w=w/w.sum()
        # Freeze each decision endpoint before any common-panel supplemental query.
        for t,fe in enumerate(full):
            for i in range(n):
                ids=[j for j in range(n) if j!=i]
                if w[ids].sum()<=0:
                    branches.append(dict(t=t,i=i,status='undefined_remaining_weight'));continue
                r.stage='branch_decision';f=r.objective(ids);a=len(r.events)
                state=branch_state(fe['pre'],f,s.I_list)
                _,bt=run(s,f,1,resume=state)
                be=event(bt,0) if t==0 else dict(pre=bt.pre[0],post=bt.final,
                    trials=bt.trials,accepted=any(v['accepted'] for v in bt.trials),g=bt.gradients[0]['g'])
                assert all(e['i']!=i for e in r.events[a:]),'excluded_model_in_decision'
                branches.append(dict(t=t,i=i,status='available',decision=be,
                    x_full=fe['post']['S_current'],x_minus=be['post']['S_current'],
                    panel=ids,direction_weights=s.alphas.copy(),decision_locked=True))
        r.stage='common_scoring'
        for b in branches:
            if b['status']!='available':continue
            lf=r.losses(b['x_full']);lm=r.losses(b['x_minus']);i=b['i']
            delta=lm-lf;m=float(np.dot(w,lm)-np.dot(w,lf))
            own=float(w[i]*delta[i]);others=float(np.dot(np.delete(w,i),np.delete(delta,i)))
            assert abs(m-own-others)<1e-10
            b.update(losses_full=lf,losses_minus=lm,m=m,own=own,others=others,
                full_accepted=full[b['t']]['accepted'],minus_accepted=b['decision']['accepted'],
                same_position=bool(np.array_equal(b['x_full'],b['x_minus'])))
        return dict(case=c,phase=phase,baseline=base,full=full,branches=branches,
            baseline_query_hash=digest.hexdigest(),baseline_replay_exact=True,cost=r.cost,
            replies=list(r.cache.values()),events=r.events,seconds=time.perf_counter()-start,
            evaluator_loaded=False,truth_loaded=False,model_artifact_sha256=sha(c['models']))
    finally:r.close()

def phase_run(phase):
    check();cases=read(ROOT/'cases.json')[phase];cfg=read(ROOT/'registration.json')
    if phase=='pilot':limit=cfg['pilot_limit'];seconds=cfg['pilot_seconds']
    else:
        b=read(ROOT/'development_budget.json');limit=b['samples'];seconds=b['model_seconds']
    budget=Budget(limit,seconds);start=time.perf_counter();completed=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('stage_timeout')))
    signal.alarm(int(seconds)+60)
    try:
        for c in cases:
            print(phase,c['key'],flush=True);r=case_record(c,phase,budget)
            save(ROOT/phase/(c['key']+'.json'),r);completed.append(c['key'])
        signal.alarm(0)
        save(ROOT/(phase+'_completion.json'),dict(complete=True,cases=completed,samples=budget.samples,
            model_seconds=budget.model_seconds,wall_seconds=time.perf_counter()-start,limit=limit))
    except Exception as e:
        signal.alarm(0);save(ROOT/(phase+'_failure.json'),dict(error=repr(e),completed=completed,
            samples=budget.samples,model_seconds=budget.model_seconds));raise
    if phase=='pilot':
        records=[read(ROOT/'pilot'/(c['key']+'.json')) for c in cases]
        max_samples=max(sum(v['samples'] for v in x['cost'].values())/len(x['case']['ids']) for x in records)
        max_seconds=max(sum(v['seconds'] for v in x['cost'].values())/len(x['case']['ids']) for x in records)
        save(ROOT/'development_budget.json',dict(samples=min(500000,int(np.ceil(max_samples*80*4))),
            model_seconds=min(900,max(1.,max_seconds*80*4)),pilot_max_samples_per_role=max_samples,
            pilot_max_model_seconds_per_role=max_seconds,effects_not_used=True))
    else:
        files=list((ROOT/'pilot').glob('*.json'))+list((ROOT/'development').glob('*.json'))
        save(ROOT/'scores_and_points_locked.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            hashes={str(p.relative_to(ROOT)):sha(p) for p in files},evaluator_loaded=False,truth_loaded=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','development']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:setup();phase_run(a.mode)
