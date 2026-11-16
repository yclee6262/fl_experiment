"""Bounded old-data anchor LOO validation. Pooled evaluation is a later command."""
import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/custom_direction_experiment_v2_20261020'
LOCAL=WORK/'contribution_analysis/local_anchor_step_experiment_v1_20261020'
ROOT=WORK/'contribution_analysis/anchor_step_validation_experiment_v1_20261020'

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
def key(x):return np.asarray(x,dtype=np.float32).tobytes()
def lock(folder):return {p.name:sha(p) for p in folder.glob('*.json')}
def check():
    for name,digest in read(ROOT/'inputs.json').items():
        if sha(name)!=digest:raise ValueError('input_changed:'+name)
    for name,digest in read(ROOT/'source_lock.json').items():
        if sha(ROOT/'source_snapshot'/name)!=digest:raise ValueError('source_changed:'+name)
def setup():
    sys.path.insert(0,str(ROOT/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def prepare():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    paths=list(OLD.glob('*.json'))+list(OLD.glob('*.csv'))+list(LOCAL.glob('*.json'))+list(LOCAL.glob('*.csv'))
    cases=read(OLD/'cases.json')
    for phase,items in cases.items():
        for c in items:
            paths.extend([Path(c['models']),Path(c['evaluator']),OLD/'baselines'/(c['key']+'.json'),LOCAL/'records'/(c['key']+'.json')])
    prior=read(OLD/'registration.json')
    for n,h in prior['historical_source_sha256'].items():
        if sha(WORK/n)!=h or sha(OLD/'source_snapshot'/n)!=h:raise ValueError('historical_drift:'+n)
        paths.append(WORK/n)
        shutil.copy2(OLD/'source_snapshot'/n,snap/n)
    for n,h in prior['source_sha256'].items():
        p=OLD/'source_snapshot'/n
        if sha(p)!=h:raise ValueError('previous_source_drift:'+n)
        paths.append(p)
    shutil.copy2(OLD/'source_snapshot/common.py',snap/'common.py')
    for n in ['observer.py','experiment.py']:
        shutil.copy2(OLD/'source_snapshot'/n,snap/('legacy_'+n))
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    for name in ['anchor_step_loo_current_design_20261020.md','anchor_step_validation_preregistration_20261020.md']:
        p=WORK/'docs'/name;paths.append(p);shutil.copy2(p,ROOT/name)
    save(ROOT/'cases.json',cases)
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths})
    save(ROOT/'source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='old-data exploration: fixed-trajectory single-step anchor LOO validation',
        horizons=[0,1,3,5],step_scales=[1.,.5],deadzone=1e-4,
        python=sys.executable,new_confirmation=False,pilot_hard_samples=100000,pilot_seconds=60,
        development_hard_samples=300000,development_seconds=120,
        user_authorization='根據定義的驗證實驗去做執行並整理結果'))
    (ROOT/'pilot').mkdir();(ROOT/'development').mkdir()
    print(ROOT)

class Meter:
    def __init__(self,server,limit,seconds):
        self.server=server;self.limit=limit;self.seconds=seconds
        self.totals={};self.section='unassigned';self.calls=[];self.old=[]
        self.begin=time.perf_counter()
        for index,agent in enumerate(server.trusted_agents):
            old=agent.api_predict;self.old.append((agent,old))
            def pred(x,_old=old,_index=index):
                n=len(x) if np.asarray(x).ndim>1 else 1
                if sum(self.totals.values())+n>self.limit or time.perf_counter()-self.begin>self.seconds:raise RuntimeError('case_budget')
                self.totals[self.section]=self.totals.get(self.section,0)+n
                y=_old(x);self.calls.append(dict(section=self.section,agent_index=_index,x=np.asarray(x).copy(),response=np.asarray(y).copy()))
                return y
            agent.api_predict=pred
    def objective(self,x):return self.server._consensus_loss(np.asarray(x),mode='optimization')
    def close(self):
        for agent,old in self.old:agent.api_predict=old

def record_for(case,phase,limit,seconds):
    from legacy_experiment import load_server
    from state_engine import run,relocate
    server=load_server(case);meter=Meter(server,limit,seconds);started=time.perf_counter()
    old=read(OLD/'baselines'/(case['key']+'.json'));local=read(LOCAL/'records'/(case['key']+'.json'))
    result=dict(key=case['key'],phase=phase,scenario=case['scenario'],seed=case['seed'],ids=case['ids'])
    try:
        meter.section='full_state_replay'
        baseline,trace=run(server,meter.objective,30,initial=case['fixed_start'],count=lambda:len(meter.calls))
        replay_calls=copy.deepcopy(meter.calls);digest=hashlib.sha256()
        for row in replay_calls:
            digest.update(row['agent_index'].to_bytes(4,'little'));digest.update(np.asarray(row['x'],dtype=np.float64).tobytes())
        if digest.hexdigest()!=old['query_order_sha256']:raise ValueError('replay_query_order')
        if meter.totals['full_state_replay']!=old['counter']['samples']:raise ValueError('replay_query_count')
        if abs(baseline['loss']-old['result']['loss'])>1e-10 or not np.array_equal(baseline['solution'],old['result']['solution']):raise ValueError('replay_endpoint')
        if baseline['history']!=old['result']['history'] or baseline['states']!=old['result']['states']:raise ValueError('replay_history')
        if len(trace.pre)!=len(old['iterations']):raise ValueError('replay_iterations')
        for state,it in zip(trace.pre,old['iterations']):
            if not np.array_equal(state['S_current'],it['x']) or state['current_method']!=it['mode'] or state['best_loss']!=it['loss']:raise ValueError('replay_iteration_state')
        result['replay']=dict(query_hash=digest.hexdigest(),result=baseline,pre=trace.pre,post=trace.post,final=trace.final)
        # All half-distance trials; original-scale data remain unchanged.
        meter.section='half_step_sensitivity';half=[]
        for it in old['iterations']:
            accepted=[t for t in it['trials'] if t['accepted']]
            if not accepted:continue
            origin=np.array(it['x']);point=np.array(accepted[0]['x']);before=it['loss']
            full=origin+.5*(point-origin);fullL=meter.objective(full);effective_full=min(before,fullL)
            oldrows=[r for r in local['comparisons'] if r['iteration']==it['iteration']]
            for r in oldrows:
                if r['J'] is None:
                    half.append(dict(iteration=it['iteration'],anchor_index=r['anchor_index'],status='missing',c=None));continue
                alternative=origin+.5*(np.array(r['alternative_x'])-origin)
                if np.max(abs(alternative))>1+1e-8:
                    half.append(dict(iteration=it['iteration'],anchor_index=r['anchor_index'],status='infeasible',c=None));continue
                altL=meter.objective(alternative)
                half.append(dict(iteration=it['iteration'],anchor_index=r['anchor_index'],status='available',
                    origin=origin,before=before,full_point=full,full_trial_loss=fullL,
                    alternative_point=alternative,alternative_trial_loss=altL,
                    effective_full_loss=effective_full,effective_alternative_loss=min(before,altL),
                    c=min(before,altL)-effective_full,c_original=r['J']))
        result['half_step']=half
        first=next(it for it in old['iterations'] if any(t['accepted'] for t in it['trials']))
        firstindex=first['iteration'];base_state=copy.deepcopy(next(s for s in trace.post if s['next_iteration']==firstindex+1))
        meter.section='complete_continuation';start=len(meter.calls)
        complete,ct=run(server,meter.objective,5,resume=base_state,count=lambda:len(meter.calls))
        # Compare every continuation objective call to the original segment,
        # plus the original output objective at this truncated segment's best point.
        n_iters=len(ct.pre)
        expected_pre=trace.pre[firstindex+1:firstindex+1+n_iters]
        for a,b in zip(ct.pre,expected_pre):
            if a['current_method']!=b['current_method'] or a['eta']!=b['eta'] or not np.array_equal(a['S_current'],b['S_current']):raise ValueError('continuation_control_state')
        actual_calls=meter.calls[start:-len(case['ids'])]
        expected_calls=replay_calls[expected_pre[0]['call_index']:expected_pre[0]['call_index']+len(actual_calls)] if expected_pre else []
        if len(actual_calls)!=len(expected_calls):raise ValueError('continuation_call_count')
        for a,b in zip(actual_calls,expected_calls):
            if a['agent_index']!=b['agent_index'] or not np.array_equal(a['x'],b['x']) or not np.array_equal(a['response'],b['response']):raise ValueError('continuation_query_suffix')
        result['complete_short']=dict(result=complete,pre=ct.pre,post=ct.post,final=ct.final,
            start_state=base_state)
        # At end-of-iteration, history records metric(), exactly as original.
        def at_horizon(startstate,tr,h):
            if h==0:return startstate
            if h<=len(tr.post):return tr.post[h-1]
            return tr.final
        branches=[]
        meter.section='anchor_continuations'
        firstrows=[r for r in local['comparisons'] if r['iteration']==firstindex]
        for r in firstrows:
            if r['J'] is None:
                branches.append(dict(anchor_index=r['anchor_index'],status='unavailable'));continue
            point=r['alternative_x'] if r['alternative_accepted'] else first['x']
            state=relocate(base_state,point,r['effective_loss'])
            # Explicitly recompute position-dependent loss; retained anchor cache
            # is valid because models and target have not changed.
            current=meter.objective(state['S_current'])
            if abs(current-state['best_loss'])>1e-10:raise ValueError('relocation_loss')
            br,bt=run(server,meter.objective,5,resume=state)
            values=[]
            for h in [0,1,3,5]:
                bs=at_horizon(base_state,ct,h);alter=at_horizon(state,bt,h)
                values.append(dict(h=h,complete_x=bs['S_current'],alternative_x=alter['S_current'],
                    complete_loss=bs['best_loss'],alternative_loss=alter['best_loss'],
                    c=alter['best_loss']-bs['best_loss'],complete_halted=bs['halted'],alternative_halted=alter['halted']))
            if abs(values[0]['c']-r['J'])>1e-10:raise ValueError('initial_c_changed')
            branches.append(dict(anchor_index=r['anchor_index'],agent_id=r['agent_id'],status='available',
                initial_state=state,result=br,pre=bt.pre,post=bt.post,final=bt.final,values=values))
        result.update(first_iteration=firstindex,branches=branches,query_log=meter.calls,counters=meter.totals,
            evaluator_loaded=False,truth_loaded=False,wall_seconds=time.perf_counter()-started)
        save(ROOT/phase/(case['key']+'.json'),result)
        print(phase,case['key'],meter.totals,flush=True)
        return result
    finally:meter.close()

def phase_run(phase):
    check();cfg=read(ROOT/'registration.json');cases=read(ROOT/'cases.json')[phase]
    if phase=='development':budget=read(ROOT/'development_budget.json');limit=budget['samples'];seconds=budget['seconds']
    else:limit=cfg['pilot_hard_samples'];seconds=cfg['pilot_seconds']
    started=time.perf_counter();total=0;records=[]
    for case in cases:
        r=record_for(case,phase,limit-total,max(.001,seconds-(time.perf_counter()-started)))
        total+=sum(r['counters'].values());records.append(r)
    save(ROOT/(phase+'_lock.json'),dict(record_sha256=lock(ROOT/phase),samples=total,wall_seconds=time.perf_counter()-started,
        evaluator_loaded=False,truth_loaded=False))
    if phase=='pilot':
        # Cost-only rule fixed before pilot, no effect statistics used.
        max_normalized=max(sum(r['counters'].values())/len(r['ids']) for r in records)
        people=sum(len(c['ids']) for c in read(ROOT/'cases.json')['development'])
        estimate=math.ceil(4*max_normalized*people)
        save(ROOT/'development_budget.json',dict(samples=min(300000,estimate),seconds=120,
            formula='4 * maximum pilot total samples per model * development participant count',
            pilot_max_samples_per_model=max_normalized,development_participants=people,
            frozen_utc=datetime.now(timezone.utc).isoformat(),no_development_read=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','development']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:setup();phase_run(a.mode)
