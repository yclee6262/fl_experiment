"""Frozen-model four-way replay; no training, pooled inference or policy edits."""
import copy, hashlib, json, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parents[2]
CA=WORK/'contribution_analysis'
ROOT=CA/'contribution_four_way_experiment_v2_20261022'
L2=CA/'combined_contribution_layer2_experiment_v1_20261021'
MOD=CA/'model_step_decision_experiment_v1_20261021'
OLD=CA/'custom_direction_experiment_v2_20261020'
ANC=CA/'local_anchor_step_experiment_v1_20261020'
PLAN=WORK/'docs/contribution_four_way_comparison_plan_20261022.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def save(p,v):
    with Path(p).open('x') as f:json.dump(plain(v),f,indent=2,ensure_ascii=False,allow_nan=False)
def key(i,x):return str(i)+':'+np.asarray(x,np.float64).tobytes().hex()
def near(a,b):
    e=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.asarray(a).size else 0.
    assert e<1e-10,('mismatch',e);return e

class Queries:
    total=0
    def __init__(self,s):
        self.s=s;self.fns=[a.api_predict for a in s.trusted_agents]
        self.cache={};self.events=[];self.stage='baseline';self.model_seconds=0.
    def reply(self,i,x):
        k=key(i,x);hit=self.stage!='baseline' and k in self.cache
        if hit:y=self.cache[k]
        else:
            if Queries.total>=1200000:raise RuntimeError('research_sample_cap')
            tic=time.perf_counter();y=float(self.fns[i](np.asarray(x))[0]);self.model_seconds+=time.perf_counter()-tic
            Queries.total+=1
            if k in self.cache:assert self.cache[k]==y
            self.cache[k]=y
        self.events.append(dict(stage=self.stage,key=k,hit=hit))
        return abs(y)
    def objective(self,panel):
        w=np.asarray(self.s.alphas)[panel];w=w/w.sum()
        return lambda x:float(np.dot(w,[self.reply(i,x) for i in panel]))
    def losses(self,x):return np.asarray([self.reply(i,x) for i in range(len(self.s.alphas))])

def local(s,q,pre,i,role):
    from engine import run,branch_state
    n=len(s.alphas);panel=[j for j in range(n) if role=='anchor' or j!=i]
    ks=[j for j in range(n) if role=='model' or j!=i]
    b=copy.copy(s);b.I_list=np.asarray(s.I_list)[ks];a=np.asarray(s.alphas)[ks];b.alphas=a/a.sum()
    q.stage=role+'_decision';start=len(q.events);f=q.objective(panel)
    state=branch_state(pre,f,b.I_list);_,tr=run(b,f,1,resume=state)
    assert all(int(e['key'].split(':')[0]) in panel for e in q.events[start:])
    return np.asarray(tr.final['S_current'])

def replay(c,s,data,scheme):
    from host_server import HostServer
    from kernel import alternative
    q=Queries(s);n=len(c['ids']);a=np.asarray(c['alphas']);a=a/a.sum();ids=list(range(n))
    tic=time.perf_counter()
    base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=30,
        direction_weights=s.alphas,initial_solution=np.asarray(c['fixed_start']))
    # Historical runner appended delivery metadata outside the custom solver.
    assert all(plain(v)==data['base']['result'][k] for k,v in base.items()),'baseline_changed'
    assert data['base']['result']['box_feasible'] and data['base']['result']['valid_delivery']
    assert max(abs(np.asarray(base['solution'])))<=1+1e-8
    baseline_seconds=time.perf_counter()-tic
    scores={i:dict(A=0.,A2=0.,M=0.,J=0.,D_L=0.) for i in ids};details=[];missing={i:0 for i in ids}
    if scheme=='S4':
        for i in ids:
            panel=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/sum(a[panel])
            q.stage='full_decision';f=q.objective(panel)
            r=HostServer._solve_subspace_custom(b,b.I_list,f,num_iterations=30,direction_weights=b.alphas,
                initial_solution=np.asarray(c['fixed_start']))
            near(r['solution'],data['removals'][i]['result']['solution']);assert r['history']==data['removals'][i]['result']['history']
            q.stage='common';v=float(np.dot(a,q.losses(r['solution']))-np.dot(a,q.losses(base['solution'])))
            scores[i]['D_L']=v;details.append(dict(i=i,role='full',endpoint=r['solution'],score=v))
    else:
        for t,full in enumerate(data['model']['full']):
            it=data['base']['iterations'][t]
            for i in ids:
                if scheme=='S1':
                    accepted=[v for v in it['trials'] if v['accepted']]
                    if accepted:
                        components={d['k']:d['weight']*d['derivative']*np.asarray(d['u']) for d in it['directions']}
                        alt=alternative(it['x'],accepted[0]['x'],it['g'],components.get(i,np.zeros(len(it['g']))),a[i])
                        q.stage='equal_anchor'
                        if alt['status'] not in ['trial','no_motion']:missing[i]+=1
                        else:
                            value=min(it['loss'],float(np.dot(a,q.losses(alt['x']))))-accepted[0]['loss']
                            old=next(v for v in data['anchor']['comparisons'] if v['iteration']==it['iteration'] and v['anchor_index']==i)
                            near(value,old['J']);scores[i]['A']+=value
                            details.append(dict(t=t,i=i,role='equal_anchor',endpoint=alt['x'],score=value))
                roles=['joint'] if scheme=='S3' else ['model'] if scheme=='S1' else ['anchor','model']
                group=data['groups'][t,i]
                for role in roles:
                    x=local(s,q,full['pre'],i,role);expected=group['roles'][role]
                    near(x,expected['endpoint']);q.stage='common'
                    value=float(np.dot(a,q.losses(x))-np.dot(a,q.losses(full['post']['S_current'])))
                    near(value,expected['score']);name={'anchor':'A2','model':'M','joint':'J'}[role]
                    scores[i][name]+=value;details.append(dict(t=t,i=i,role=role,endpoint=x,score=value))
    wall=time.perf_counter()-tic
    rows=[]
    for i in ids:
        names={'S1':['A','M'],'S2':['A2','M'],'S3':['J'],'S4':['D_L']}[scheme]
        v={k:scores[i][k] for k in names}
        if scheme=='S1':
            near(v['A'],data['anchor']['A'][i]);v['C']=v['A']+v['M']
        if scheme=='S2':v['C2']=v['A2']+v['M']
        rows.append(dict(agent_id=c['ids'][i],values=v,missing=missing[i]))
    return dict(scheme=scheme,rows=rows,details=details,events=q.events,baseline_seconds=baseline_seconds,
        wall_seconds=wall,model_seconds=q.model_seconds,baseline_samples=sum(e['stage']=='baseline' for e in q.events),
        extra_samples=sum(not e['hit'] and e['stage']!='baseline' for e in q.events),
        samples=sum(not e['hit'] for e in q.events),requests=len(q.events),cache_hits=sum(e['hit'] for e in q.events))

def main():
    ROOT.mkdir(exist_ok=False);(ROOT/'records').mkdir();(ROOT/'source_snapshot').mkdir()
    cases=read(L2/'cases.json')['development'];paths=[PLAN,L2/'cases.json',L2/'verification.json',
        CA/'combined_contribution_layer1_experiment_v1_20261021/comparisons.json']
    for c in cases:
        for root,sub in [(OLD,'baselines'),(MOD,'development'),(L2,'development'),(ANC,'records')]:paths.append(root/sub/(c['key']+'.json'))
        paths.extend([Path(c['models'])]+[OLD/'removals'/(c['key']+'_'+str(i)+'.json') for i in c['ids']])
    import shutil
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,ROOT/'source_snapshot'/p.name);paths.append(p)
    # Load frozen historical implementation, never workspace solver edits.
    for p in (MOD/'source_snapshot').glob('*.py'):paths.append(p)
    paths.append(CA/'local_anchor_step_v1/kernel.py')
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths});save(ROOT/'cases.json',cases)
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),
        old_data_exploration=True,new_confirmation=False,schemes=['S1','S2','S3','S4'],warmups=1,repeats=3,
        key_dtype='float64',baseline_cache='full completed baseline; exact model/input; no other scheme',
        inference_cap=1200000,wall_cap_seconds=900,pooled_inferences=0,truth_loaded=False))
    sys.path.insert(0,str(MOD/'source_snapshot'));sys.path.insert(0,str(CA/'local_anchor_step_v1'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from study import load_server
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall_cap')));signal.alarm(900)
    started=time.perf_counter();records=[];timing=[]
    try:
        for ci,c in enumerate(cases):
            print(c['key'],flush=True);s=load_server({k:v for k,v in c.items() if k!='evaluator'})
            layer=read(L2/'development'/(c['key']+'.json'))
            data=dict(base=read(OLD/'baselines'/(c['key']+'.json')),model=read(MOD/'development'/(c['key']+'.json')),
                anchor=read(ANC/'records'/(c['key']+'.json')),groups={(g['t'],g['i']):g for g in layer['groups']},
                removals={i:read(OLD/'removals'/(c['key']+'_'+str(v)+'.json')) for i,v in enumerate(c['ids'])})
            saved={};schemes=['S1','S2','S3','S4']
            for rep in range(4):
                order=schemes[(ci+rep)%4:]+schemes[:(ci+rep)%4]
                for scheme in order:
                    out=replay(c,s,data,scheme)
                    if rep==0:
                        saved[scheme]=out;save(ROOT/'records'/(c['key']+'_'+scheme+'.json'),out)
                    else:
                        assert out['rows']==saved[scheme]['rows']
                        assert out['samples']==saved[scheme]['samples'] and out['events']==saved[scheme]['events']
                        timing.append(dict(key=c['key'],scenario=c['scenario'],scheme=scheme,repeat=rep,
                            wall_seconds=out['wall_seconds'],baseline_seconds=out['baseline_seconds'],model_seconds=out['model_seconds']))
            records.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],schemes={k:{x:v for x,v in o.items() if x not in ['events','details']} for k,o in saved.items()}))
        save(ROOT/'scores_and_costs.json',records);save(ROOT/'timing.json',timing)
        save(ROOT/'scores_locked.json',dict(sha256=sha(ROOT/'scores_and_costs.json'),new_pooled_loaded=False,
            reference_values_read_by_analysis=False,old_reference_already_known=True))
        for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h
        save(ROOT/'replay_completion.json',dict(complete=True,teams=len(records),physical_model_samples=Queries.total,
            wall_seconds=time.perf_counter()-started,pooled_inferences=0,truth_loaded=False,
            all_historical_endpoints_scores_and_baseline_histories_matched=True))
    except Exception as e:
        save(ROOT/'failure.json',dict(error=repr(e),completed_cases=len(records),samples=Queries.total));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
