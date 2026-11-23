"""Audit actual greedy recruitment against historical coalition outcomes."""
import argparse
from collections import defaultdict
import contextlib
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import spearmanr
from experiments.coalition.recruitment_selection import coalition_value,exact_select

ROOT=Path(__file__).resolve().parents[3]
BUDGETS=[2.5,3.5,8.]
ETAS=[0.,.5]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def write(p,data):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)


def sequential(candidates,budget,eta,quality_only=False,max_size=3):
    selected=[];spent=0.
    while len(selected)<max_size:
        best=None;score=-np.inf
        for i,c in enumerate(candidates):
            if i in selected or spent+c['bid']>budget:continue
            if not selected:div=1.
            else:
                sims=[abs(float(np.dot(c['residual'],candidates[j]['residual'])/(np.linalg.norm(c['residual'])*np.linalg.norm(candidates[j]['residual'])+1e-8))) for j in selected]
                div=max(0.,1-max(sims))
            value=c['quality_share'] if quality_only else c['quality_share']/max(c['bid'],1e-8)*(1+eta*div)
            if value>score:best,score=i,value
        if best is None or score<0:break
        selected.append(best);spent+=candidates[best]['bid']
    return tuple(sorted(selected))


def core_select(candidates,budget,eta):
    from core.host_server import HostServer
    class Agent:
        def __init__(self,c):self.agent_id=c['agent_id'];self.c=c
        def api_predict(self,x):return np.asarray(self.c['residual'])
        def get_minimum_bid(self):return self.c['bid']
    host=HostServer(.5,2,total_budget=budget,n_test=len(candidates[0]['residual']))
    host.test_y=np.zeros(host.n_test)
    with contextlib.redirect_stdout(io.StringIO()):
        host.phase1_filter_agents([Agent(c) for c in candidates],mse_threshold=1.,budget_fraction=1.,diversity_eta=eta,enable_inverse_check=False)
    lookup={c['agent_id']:i for i,c in enumerate(candidates)}
    return tuple(sorted(lookup[a.agent_id] for a in host.trusted_agents))


def percentile(value,values):
    v=np.asarray(values)
    return float((np.sum(v<value)+.5*np.sum(v==value))/len(v))


def correlation(values,errors):
    if len(values)<2 or np.ptp(values)==0 or np.ptp(errors)==0:return None
    return float(spearmanr(values,-np.asarray(errors)).statistic)


def summarize(data,keys,value):
    grouped=defaultdict(list)
    for r in data:grouped[tuple(r[k] for k in keys)].append(r)
    summary=[]
    for key,rr in grouped.items():
        x=np.array([r[value] for r in rr if r[value] is not None])
        if len(x):
            n=len(x);rng=np.random.default_rng(20261123);w=rng.multinomial(n,np.ones(n)/n,size=200000);bs=w@x/n
            lo,hi=np.quantile(bs,[.025,.975]);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else None
        else:mean=sd=lo=hi=None
        summary.append({**dict(zip(keys,key)),'metric':value,'n':len(x),'missing':len(rr)-len(x),'mean':mean,'sd':sd,'ci95_low':None if lo is None else float(lo),'ci95_high':None if hi is None else float(hi)})
    return summary


def analyze(source,output):
    output.mkdir(parents=True,exist_ok=False)
    cp=rows(source/'candidate_pool.csv');cr=rows(source/'coalitions.csv');manifest=read(source/'run_manifest.json')
    assert manifest['status']=='complete' and len(cp)==50 and len(cr)==875
    input_paths=[source/'candidate_pool.csv',source/'coalitions.csv',source/'run_manifest.json',source/'objective_validity.csv']
    rawsource={str(p.resolve()):sha(p) for p in input_paths}
    candidates=defaultdict(list);coalitions=defaultdict(dict)
    for r in cp:
        c={k:float(r[k]) for k in ['mse','bid','quality_share']};c.update(agent_id=int(r['agent_id']),residual=np.array(json.loads(r['residual'])),legacy_selected=r['legacy_selected']=='True')
        assert abs(np.mean(c['residual']**2)-c['mse'])<1e-12
        candidates[int(r['seed'])].append(c)
    for r in cr:
        seed=int(r['seed']);key=tuple(json.loads(r['coalition_indices']));x=np.array(json.loads(r['final_solution']))
        assert key not in coalitions[seed] and r['delivery_status']=='valid'
        truth=float(abs(x.sum()+np.sum(x[:-1]*x[1:])-.5));assert abs(truth-float(r['truth_error']))<1e-12
        coalitions[seed][key]=dict(pooled_error=float(r['pooled_error']),truth_error=truth,bid_sum=float(r['bid_sum']),size=int(r['coalition_size']))
    out=[];corr=[];checked=0;legacy=[]
    for seed,cs in sorted(candidates.items()):
        assert len(cs)==10 and abs(sum(c['quality_share'] for c in cs)-1)<1e-10
        expected=set(itertools.chain.from_iterable(itertools.combinations(range(10),k) for k in [1,2,3]));assert set(coalitions[seed])==expected
        for budget,eta in itertools.product(BUDGETS,ETAS):
            selected=sequential(cs,budget,eta);actual=core_select(cs,budget,eta);assert selected==actual,(seed,budget,eta,selected,actual);checked+=1
            choices={'core_greedy':selected,'quality_only':sequential(cs,budget,eta,True),'set_objective':tuple(exact_select(cs,budget,3,eta)['indices'])}
            feasible={k:v for k,v in coalitions[seed].items() if v['bid_sum']<=budget+1e-12}
            for policy in ['core_greedy','quality_only','set_objective','random_feasible','random_core_size']:
                if policy in choices:eligible=[choices[policy]]
                else:eligible=[k for k in feasible if policy=='random_feasible' or len(k)==len(selected)]
                assert eligible and all(k in feasible for k in eligible)
                for endpoint in ['pooled_error','truth_error']:
                    allerrors=[v[endpoint] for v in feasible.values()];es=[feasible[k][endpoint] for k in eligible]
                    out.append(dict(seed=seed,budget=budget,eta=eta,policy=policy,endpoint=endpoint,coalition=str(choices.get(policy,'uniform_expectation')),support=len(eligible),affordable_coalitions=len(feasible),mean_size=float(np.mean([len(k) for k in eligible])),mean_bid=float(np.mean([feasible[k]['bid_sum'] for k in eligible])),error=float(np.mean(es)),gap_to_best=float(np.mean(es)-min(allerrors)),error_percentile=float(np.mean([percentile(e,allerrors) for e in es]))))
            for size in [0,1,2,3]:
                kk=[k for k in feasible if size==0 or len(k)==size]
                scores=[coalition_value(k,cs,eta,3) for k in kk]
                for endpoint in ['pooled_error','truth_error']:
                    corr.append(dict(seed=seed,budget=budget,eta=eta,size='mixed' if size==0 else str(size),endpoint=endpoint,coalitions=len(kk),spearman=correlation(scores,[feasible[k][endpoint] for k in kk])))
            if budget==8 and eta==.5:
                marked=tuple(i for i,c in enumerate(cs) if c['legacy_selected']);assert marked==selected,(seed,marked,selected)
                legacy.append(dict(seed=seed,budget=budget,eta=eta,legacy=marked,core=selected,matched=True))
    summaries=[]
    for value in ['error','gap_to_best','error_percentile','mean_size','mean_bid']:
        summaries+=summarize(out,['budget','eta','policy','endpoint'],value)
    corrsummary=summarize(corr,['budget','eta','size','endpoint'],'spearman')
    by={(v['seed'],v['budget'],v['eta'],v['policy'],v['endpoint']):v for v in out}
    changes=[]
    for seed,budget,eta,endpoint,other in itertools.product(range(5),BUDGETS,ETAS,['pooled_error','truth_error'],['quality_only','set_objective','random_feasible','random_core_size']):
        core=by[seed,budget,eta,'core_greedy',endpoint];ref=by[seed,budget,eta,other,endpoint]
        changes.append(dict(seed=seed,budget=budget,eta=eta,endpoint=endpoint,contrast='core_minus_'+other,
                            error_delta=core['error']-ref['error'],percentile_delta=core['error_percentile']-ref['error_percentile']))
    cs=[]
    for metric in ['error_delta','percentile_delta']:
        cs+=summarize(changes,['budget','eta','endpoint','contrast'],metric)
    write(output/'policy_changes.csv',cs)
    for name,data in [('policy_seed_results',out),('policy_summary',summaries),('fixed_size_correlations',corr),('correlation_summary',corrsummary)]:write(output/(name+'.csv'),data)
    artifact_checks={}
    for name,h in manifest.get('artifact_sha256',{}).items():
        p=source/name
        if not p.exists():p=Path(name)
        assert p.exists() and sha(p)==h,str(p)
        artifact_checks[str(p.resolve())]=h
    audit=dict(passed=True,seeds=5,candidates=50,coalitions=875,actual_core_checks=checked,legacy_checks=legacy,input_sha256=rawsource,artifact_sha256=artifact_checks,analysis_sha256=sha(Path(__file__)),design_sha256=sha(Path(__file__).with_name('design.md')),core_sha256=sha(ROOT/'core/host_server.py'),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_model_queries=0,new_solver_calls=0,scope='Historical original-solver outcomes; descriptive five-seed reanalysis; corrected solver rerun remains task8.')
    dump(output/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if not k.endswith('sha256')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
