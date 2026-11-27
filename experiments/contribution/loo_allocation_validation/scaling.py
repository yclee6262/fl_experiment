"""New clean nested teams: participant count, dimension and interaction strength."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .methods import module_at,observed_solver,execute
from .verify import audit_events,independent_scores
from .run import read,save,plain,sha
from .scaling_data import build_pool

HERE=Path(__file__).resolve().parent
CONDITIONS=[('d2_low',2,.3),('d5_low',5,.3),('d10_low',10,.3),('d5_high',5,1.2)]
N_VALUES=[3,5,10,20]
METHODS=['FULL_LOO','LAST_J','CUM_A']


class ScalingBudget:
    def __init__(self):self.samples=0;self.trained=0;self.start=time.perf_counter()
    def check(self):
        if time.perf_counter()-self.start>1800 or self.trained>800:raise RuntimeError('Scaling wall/training budget')
    def charge(self):
        self.check()
        if self.samples>=25000000:raise RuntimeError('Scaling query budget')
        self.samples+=1


def unused_seeds(root):
    # rg includes ignored local histories; files are scanned as text, never executed.
    trials=[]
    for offset in range(10):
        seeds=[2026112001+100*offset+i for i in range(5)]
        cmd=['rg','--hidden','--no-ignore','-l','-F','--glob','*.json','--glob','*.jsonl','--glob','!.git/**']
        for seed in seeds:cmd+=['-e',str(seed)]
        result=subprocess.run(cmd+[str(root)],capture_output=True,text=True)
        if result.returncode not in (0,1):raise RuntimeError(result.stderr)
        trials.append(dict(seeds=seeds,matching_files=result.stdout.splitlines()))
        if result.returncode==1:return seeds,trials
    raise RuntimeError('No unused seed block found')


def make_server(host, agents, anchors, dimension, n):
    s=host.HostServer.__new__(host.HostServer)
    s.target_T=0.;s.n_features=dimension;s.constraint_mode='box_constrained'
    s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=anchors[:n].copy();s.alphas=np.ones(n)/n;s.trusted_agents=[a for _,a in agents[:n]]
    return s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    root=HERE.parents[2]
    seeds,seed_audit=unused_seeds(root)
    out.mkdir(parents=True);(out/'models').mkdir();(out/'records').mkdir()
    budget=ScalingBudget();save(out/'seed_audit.json',seed_audit)
    save(out/'registration.json',dict(seeds=seeds,conditions=CONDITIONS,participants=N_VALUES,repeats_seed=seeds[0],
      reference_stability_seed=seeds[0],main_iterations=30,stability_iterations=60,new_data_exploration=True,
      local_query_cap=25000000,candidate_cap=800,wall_cap=1800,pooled=0,protocol_sha256=sha(HERE/'scaling_design.md')))
    frozen=root/'contribution_analysis/model_step_decision_experiment_v1_20261021/source_snapshot'
    host=module_at('scaling_frozen_host',frozen/'host_server.py');engine=module_at('scaling_frozen_engine',frozen/'engine.py')
    solver=host.HostServer._solve_subspace_custom;observed=observed_solver(solver);engine.compile_engine(solver)
    sources=list(HERE.glob('*.py'))+[HERE/'scaling_design.md',root/'core/models.py',frozen/'host_server.py',frozen/'engine.py']
    hashes={str(p):sha(p) for p in sources};save(out/'source_lock.json',hashes)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    import shutil
    for p in sources:shutil.copy2(p,snapshot/(p.name if p.parent==HERE else p.parent.name+'__'+p.name))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    records=[];timing=[];stability=[];qualification=[];seen_events=0;gate=0
    try:
        for condition,d,gamma in CONDITIONS:
            for si,seed in enumerate(seeds):
                directory=out/'models'/f'{condition}_{seed}';directory.mkdir()
                print('TRAIN',condition,seed,flush=True)
                agents,anchors,audit=build_pool(seed,d,gamma,directory,budget)
                qualification.append(dict(condition=condition,seed=seed,accepted=len(agents),attempted=len(audit),agents=audit))
                if len(agents)<20:
                    save(out/'records'/f'{condition}_{seed}_unavailable.json',dict(status='insufficient_qualified',accepted=len(agents)))
                    continue
                for n in N_VALUES:
                    key=f'{condition}_{seed}_n{n}';print('SOLVE',key,flush=True)
                    s=make_server(host,agents,anchors,d,n)
                    c=dict(key=key,scenario=condition,seed=seed,dimension=d,gamma=gamma,ids=[i for i,_ in agents[:n]],
                           candidates=anchors[:n],alphas=s.alphas,fixed_start=np.full(d,.35/np.sqrt(d)))
                    version=hashlib.sha256(''.join(sha(directory/f'candidate_{i}.pt') for i in c['ids']).encode()).hexdigest()
                    first={};base=None
                    for rep in range(4 if si==0 else 1):
                        order=METHODS[(rep+si+N_VALUES.index(n))%3:]+METHODS[:(rep+si+N_VALUES.index(n))%3]
                        for method in order:
                            v=execute(s,c,method,budget,version,solver,observed,engine)
                            if base is None:base=v['base']
                            else:
                                np.testing.assert_array_equal(v['base']['solution'],base['solution'])
                                assert v['base']['history']==base['history']
                            counts=audit_events(v['events']);independent_scores(v,s.alphas,0.);seen_events+=len(v['events'])
                            cost=dict(primary=sum(z['samples'] for k,z in counts.items() if k!='full_supplement'),
                              baseline=counts['baseline']['samples'],supplement=counts.get('full_supplement',{}).get('samples',0),counts=counts)
                            if rep==0:
                                event_path=out/'records'/f'{key}_{method}_events.json.gz'
                                with gzip.open(event_path,'wt') as f:json.dump(v['events'],f,separators=(',',':'))
                                v.pop('events');first[method]=dict(case=c,result=v,cost=cost)
                                save(out/'records'/f'{key}_{method}.json',first[method]);records.append(first[method])
                            else:
                                if v['x'] is None:
                                    assert first[method]['result']['x'] is None
                                else:
                                    np.testing.assert_allclose(v['x'],first[method]['result']['x'],atol=0,rtol=0)
                                assert cost==first[method]['cost']
                                timing.append(dict(key=key,condition=condition,n=n,method=method,repeat=rep,seconds=v['seconds']))
                    # Fixed first-seed subset, regardless of allocation quality.
                    if si==0:
                        v60=execute(s,c,'FULL_LOO',budget,version,solver,observed,engine,iterations=60)
                        independent_scores(v60,s.alphas,0.);audit_events(v60['events']);seen_events+=len(v60['events'])
                        with gzip.open(out/'records'/f'{key}_LOO60_events.json.gz','wt') as f:json.dump(v60['events'],f,separators=(',',':'))
                        v60.pop('events');save(out/'records'/f'{key}_LOO60.json',v60)
                        stability.append(dict(case=c,result=v60))
                    if condition=='d2_low' and si==0 and n in (3,20):
                        gate+=1
                        save(out/f'gate_{n}.json',dict(technical_pass=True,samples=budget.samples,wall=time.perf_counter()-budget.start))
                    save(out/'records'/f'{key}_budget.json',dict(samples=budget.samples,candidates=budget.trained,wall=time.perf_counter()-budget.start))
        save(out/'scores.json',records);save(out/'timing.json',timing);save(out/'stability.json',stability);save(out/'qualification.json',qualification)
        for p,h in hashes.items():assert sha(p)==h,p
        save(out/'verification.json',dict(accepted=True,teams=len(records)//3,model_query_events_audited=seen_events,
          repeated_baselines_endpoints_counts_identical=True,peer_full_scores_independently_rebuilt=True,sources_unchanged=True,pooled=0))
        from .scaling_analysis import analyze
        analyze(out)
        save(out/'completion.json',dict(status='complete',teams=len(records)//3,qualified_pools=len(qualification),
          trained_models=budget.trained,local_queries=budget.samples,wall_seconds=time.perf_counter()-budget.start,pooled=0,
          new_data_exploration=True,independent_confirmation=False))
    except BaseException as e:
        save(out/'failure.json',dict(status='failed',error=repr(e),completed_method_records=len(records),
          trained_models=budget.trained,local_queries=budget.samples,wall_seconds=time.perf_counter()-budget.start))
        raise


if __name__=='__main__':main()
