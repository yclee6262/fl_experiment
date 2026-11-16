"""Aligned local anchor/model/joint removal; original trajectory stays frozen."""
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
ROOT=WORK/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021'
OLD=WORK/'contribution_analysis/custom_direction_experiment_v2_20261020'
PREV=WORK/'contribution_analysis/model_step_decision_experiment_v1_20261021'
DIAG=WORK/'contribution_analysis/contribution_diagnostic_experiment_v1_20261020'
DESIGN=WORK/'docs/combined_contribution_layer2_design_20261021.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def save(p,v):
    with Path(p).open('x') as f:json.dump(plain(v),f,indent=2,allow_nan=False);f.write('\n')
def key(x):return np.asarray(x,dtype=np.float64).tobytes().hex()
def check():
    for n,h in read(ROOT/'inputs.json').items():assert sha(n)==h,('input_drift',n)
    for n,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/n)==h,('source_drift',n)
def setup():
    sys.path.insert(0,str(ROOT/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def prepare():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    cases=read(PREV/'cases.json');inputs=[DESIGN,PREV/'cases.json',PREV/'scores_and_points_locked.json',
        PREV/'completion_receipt.json',OLD/'cases.json',OLD/'participant_comparison.csv',
        DIAG/'full_revaluation_lock.json',
        WORK/'contribution_analysis/combined_contribution_layer1_experiment_v1_20261021/comparisons.json',
        WORK/'contribution_analysis/combined_contribution_layer1_experiment_v1_20261021/completion_receipt.json']
    assert cases==read(OLD/'cases.json')
    for n,h in read(OLD/'registration.json')['historical_source_sha256'].items():
        assert sha(WORK/n)==h==sha(OLD/'source_snapshot'/n)
        shutil.copy2(OLD/'source_snapshot'/n,snap/n);inputs.append(WORK/n)
    plock=read(PREV/'scores_and_points_locked.json')['hashes']
    for phase,cs in cases.items():
        for c in cs:
            p=PREV/phase/(c['key']+'.json');assert sha(p)==plock[phase+'/'+p.name]
            inputs.extend([p,Path(c['models']),Path(c['evaluator']),OLD/'baselines'/p.name])
            if phase=='development':
                inputs.append(DIAG/'full_revaluation'/p.name)
                inputs.extend(OLD/'removals'/(c['key']+'_'+str(i)+'.json') for i in c['ids'])
    for name,target in [('study.py','support.py'),('engine.py','engine.py')]:
        p=PREV/'source_snapshot'/name
        assert sha(p)==read(PREV/'source_lock.json')[name]
        shutil.copy2(p,snap/target);inputs.append(p)
    # Independent interpreter copied verbatim from the already verified predecessor.
    p=PREV/'source_snapshot/verify.py';inputs.append(p)
    text=p.read_text();excerpt=text[text.index('def one_round'):text.index('def verify():')]
    (snap/'independent_step.py').write_text('import copy\nimport numpy as np\n\n'+excerpt)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name);inputs.append(p)
    shutil.copy2(DESIGN,ROOT/DESIGN.name);save(ROOT/'cases.json',cases)
    save(ROOT/'inputs.json',{str(p):sha(p) for p in inputs})
    save(ROOT/'source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='那依照剛剛對第二層的設計實作並實驗看看結果如何',new_confirmation=False,
        bootstrap_seed=20261021,bootstrap_repeats=10000,ci_percentiles=[.1,99.9],deadzone=1e-4,
        pilot_samples=100000,development_cap=500000,verification_samples=600000,total_samples=1200000,
        pilot_wall_seconds=180,development_wall_seconds=900,verification_wall_seconds=1800,pooled_total_limit=12000,
        roles=['anchor','model','joint'],tests=read(Path(__file__).parent/'technical_results.json')))
    for phase in ['pilot','development']:(ROOT/phase).mkdir()
    print(ROOT,flush=True)

def geometry(server,i,role):
    branch=copy.copy(server);n=len(server.alphas)
    indices=[j for j in range(n) if role not in ['model','joint'] or j!=i]
    anchors=[j for j in range(n) if role not in ['anchor','joint'] or j!=i]
    branch.I_list=np.asarray(server.I_list)[anchors].copy()
    a=np.asarray(server.alphas)[anchors];branch.alphas=a/a.sum()
    return branch,indices,anchors

def execute(server,replies,pre,i,role):
    from engine import run,branch_state
    b,panel,anchors=geometry(server,i,role)
    if len(anchors)==0 or np.asarray(server.alphas)[panel].sum()<=0:raise ValueError('undefined_role')
    f=replies.objective(panel);begin=len(replies.events)
    state=branch_state(pre,f,b.I_list);_,tr=run(b,f,1,resume=state)
    assert len(tr.pre)==1
    decision=dict(pre=tr.pre[0],post=tr.final,trials=tr.trials,
        accepted=any(v['accepted'] for v in tr.trials),g=tr.gradients[0]['g'])
    assert all(e['i'] in panel for e in replies.events[begin:]),'excluded_scoring_model_called'
    return dict(decision=decision,panel=panel,anchor_indices=anchors,direction_weights=b.alphas,
                endpoint=tr.final['S_current'],decision_locked=True)

def record(c,phase,budget):
    from support import load_server,Replies
    old=read(PREV/phase/(c['key']+'.json'));s=load_server(c);rep=Replies(s,budget)
    rep.cache={str(v['i'])+':'+key(v['x']):copy.deepcopy(v) for v in old['replies']}
    rep.cache_enabled=True;start=time.perf_counter();groups=[]
    try:
        for t,full in enumerate(old['full']):
            for i in range(len(c['ids'])):
                roles={}
                for role in ['model','anchor','joint']:
                    rep.stage=role+'_decision';roles[role]=execute(s,rep,full['pre'],i,role)
                previous=next(b for b in old['branches'] if b['t']==t and b['i']==i)
                assert plain(roles['model']['decision'])==plain(previous['decision']),'M2_decision_changed'
                groups.append(dict(t=t,i=i,agent_id=c['ids'][i],full=full,roles=roles))
        # All role decisions fixed; supplemental common scores cannot change them.
        w=np.asarray(c['alphas']);w=w/w.sum();rep.stage='common_scoring'
        for g in groups:
            lf=rep.losses(g['full']['post']['S_current']);L0=float(np.dot(w,lf));values={}
            for role,b in g['roles'].items():
                losses=rep.losses(b['endpoint']);score=float(np.dot(w,losses)-L0)
                b.update(common_losses=losses,score=score);values[role]=score
            previous=next(b for b in old['branches'] if b['t']==g['t'] and b['i']==g['i'])
            assert values['model']==previous['m'],'M2_score_changed'
            g.update(a2=values['anchor'],m2=values['model'],j=values['joint'],
                c2=values['anchor']+values['model'],residual=values['joint']-values['anchor']-values['model'],
                full_losses=lf,full_loss=L0)
        reused={str(v['i'])+':'+key(v['x']) for v in old['replies']}
        new=[v for k,v in rep.cache.items() if k not in reused]
        return dict(case=c,phase=phase,groups=groups,cost=rep.cost,new_replies=new,
            inherited_replies_file=str(PREV/phase/(c['key']+'.json')),inherited_file_sha256=sha(PREV/phase/(c['key']+'.json')),
            seconds=time.perf_counter()-start,evaluator_loaded=False,truth_loaded=False,
            model_score_exactly_reproduced=True)
    finally:rep.close()

def run_phase(phase):
    from support import Budget
    check();limit=100000 if phase=='pilot' else read(ROOT/'development_budget.json')['samples']
    wall=180 if phase=='pilot' else 900;budget=Budget(limit,wall);start=time.perf_counter();done=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('phase_wall_budget')));signal.alarm(wall)
    try:
        for c in read(ROOT/'cases.json')[phase]:
            print(phase,c['key'],flush=True);r=record(c,phase,budget);save(ROOT/phase/(c['key']+'.json'),r);done.append(c['key'])
        save(ROOT/(phase+'_completion.json'),dict(complete=True,cases=done,new_model_samples=budget.samples,
            model_seconds=budget.model_seconds,wall_seconds=time.perf_counter()-start,sample_limit=limit))
    except Exception as e:
        save(ROOT/(phase+'_failure.json'),dict(error=repr(e),completed=done,samples=budget.samples));raise
    finally:signal.alarm(0)
    if phase=='pilot':
        rr=[read(p) for p in (ROOT/'pilot').glob('*.json')]
        peak=max(sum(v['samples'] for v in r['cost'].values())/len(r['case']['ids']) for r in rr)
        save(ROOT/'development_budget.json',dict(samples=max(1,min(500000,int(np.ceil(peak*80*4)))),
            pilot_max_new_samples_per_role=peak,effect_not_used=True))
    else:
        save(ROOT/'scores_and_points_locked.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            evaluator_loaded=False,truth_loaded=False,hashes={str(p.relative_to(ROOT)):sha(p) for phase in ['pilot','development'] for p in (ROOT/phase).glob('*.json')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','development']);arg=ap.parse_args()
    if arg.mode=='prepare':prepare()
    else:setup();run_phase(arg.mode)
