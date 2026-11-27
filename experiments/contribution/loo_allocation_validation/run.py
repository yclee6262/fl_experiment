"""Execute the locked old-cohort experiment without any pooled inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
from experiments.contribution.loo_allocation_validation.methods import module_at,observed_solver,execute
from experiments.contribution.loo_allocation_validation.verify import audit_events,compare_saved,independent_scores
from experiments.contribution.loo_allocation_validation.analyze import analyze

HERE=Path(__file__).resolve().parent
MAP={'FULL_LOO':'LOO','LAST_J':'LAST_JOINT_REPLAY','CUM_A':'CUM'}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def save(p,x):
    with Path(p).open('x') as f:json.dump(plain(x),f,ensure_ascii=False,indent=2,allow_nan=False)


class Budget:
    def __init__(self, prior_samples=0, prior_seconds=0.):
        self.samples=0;self.prior_samples=prior_samples;self.prior_seconds=prior_seconds;self.started=time.perf_counter()
    def charge(self):
        if self.samples+self.prior_samples>=400000 or time.perf_counter()-self.started+self.prior_seconds>=900:raise RuntimeError('Research budget exhausted')
        self.samples+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--data-root',type=Path,default=Path.cwd())
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.source_dir.resolve();root=args.data_root.resolve();ca=root/'contribution_analysis'
    # Previous attempts in the same experiment output parent remain chargeable.
    prior_samples=sum(read(p).get('local_samples',0) for p in out.parent.glob('*/completion.json'))
    prior_seconds=sum(read(p).get('wall_seconds',0.) for p in out.parent.glob('*/completion.json'))
    budget=Budget(prior_samples,prior_seconds);manifest=dict(status='running',new_confirmation=False,pooled_samples=0,training=0)
    save(out/'protocol.json',read(HERE/'protocol.json'));save(out/'manifest_started.json',manifest)
    rows=[];timings=[];locks={};load_seconds=0.;audit_requests=0
    try:
        source_file=source/'scored.json';old=read(source_file);locks[str(source_file)]=sha(source_file)
        # Keep only declared fields; historical pooled scores are never used or reported.
        archived=[]
        for r in old:
            archived.append(dict(case={k:v for k,v in r['case'].items() if k!='evaluator'},checks=r['checks'],
              methods={k:{f:r['methods'][v][f] for f in ['x','raw','peer','full']} for k,v in MAP.items()}))
        save(out/'reused_endpoints_scores.json',archived)
        frozen=ca/'model_step_decision_experiment_v1_20261021/source_snapshot'
        sys.path.insert(0,str(frozen))
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        host=module_at('loo_frozen_host',frozen/'host_server.py')
        agent=module_at('loo_frozen_agent',frozen/'agent_client.py')
        engine=module_at('loo_frozen_engine',frozen/'engine.py')
        solver=host.HostServer._solve_subspace_custom;observed=observed_solver(solver)
        # Compile before timing.
        engine.compile_engine(solver)
        for p in list(frozen.glob('*.py'))+list(HERE.glob('*.py'))+[HERE/'protocol.json',HERE/'design.md']:
            locks[str(p.resolve())]=sha(p)
        historical_hashes=read(source/'source_lock.json')
        cases=[]
        for r in archived:
            c=r['case'];mp=Path(c['models'])
            if not mp.exists():
                suffix=mp.parts[mp.parts.index('contribution_analysis'):];mp=root.joinpath(*suffix)
            if not mp.exists():raise FileNotFoundError(mp)
            expected=historical_hashes.get(c['models']);assert expected and sha(mp)==expected
            c['models']=str(mp);locks[str(mp)]=sha(mp);cases.append(c)
        # Gate cases: smallest seed in each scenario, then remaining cases in source order.
        gates=[next(i for i,r in enumerate(archived) if r['case']['scenario']==s) for s in ['clean','controlled_poison']]
        order=gates+[i for i in range(len(cases)) if i not in gates]
        for ci in order:
            r=archived[ci];c=cases[ci];print(c['key'],flush=True)
            p=ca/'custom_direction_experiment_v2_20261020/baselines'/(c['key']+'.json')
            baseline=read(p);locks[str(p)]=sha(p)
            tic=time.perf_counter();s=host.HostServer.__new__(host.HostServer)
            s.target_T=0.;s.n_features=len(c['fixed_start']);s.constraint_mode='box_constrained'
            s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
            s.I_list=np.asarray(c['candidates']);s.alphas=np.asarray(c['alphas']);s.trusted_agents=[]
            payload=torch.load(c['models'],map_location='cpu',weights_only=True)
            for aid in c['ids']:
                a=agent.AgentNode(aid,None,n_features=s.n_features);a.model.load_state_dict(payload['agents'][aid]);a.model.eval();s.trusted_agents.append(a)
            load_seconds+=time.perf_counter()-tic
            methods=list(MAP);first={}
            for rep in range(4):
                for method in methods[(ci+rep)%3:]+methods[:(ci+rep)%3]:
                    v=execute(s,c,method,budget,sha(c['models']),solver,observed,engine)
                    compare_saved(v,r['methods'][method],baseline);independent_scores(v,c['alphas'],0.)
                    counts=audit_events(v['events']);audit_requests+=len(v['events'])
                    primary=sum(z['samples'] for k,z in counts.items() if k!='full_supplement')
                    extra=primary-counts['baseline']['samples']
                    item=dict(key=c['key'],scenario=c['scenario'],method=method,repeat=rep,
                      primary_samples=primary,extra_samples=extra,supplement_samples=counts.get('full_supplement',{}).get('samples',0),
                      seconds=v['seconds'],supplementary_seconds=v['supplementary_seconds'],counts=counts)
                    if rep==0:
                        first[method]=item
                        save(out/(c['key']+'_'+method+'.json'),v)
                        rows.append(dict(case=c,method=method,peer=v['peer'],full=v['full'],x=v['x'],raw=v['raw'],
                          cost=item,checks=r['checks'],observer_array_bytes=v['observer_array_bytes'],last_iteration=v['last_iteration']))
                    else:
                        assert primary==first[method]['primary_samples']
                        assert counts==first[method]['counts'];timings.append(item)
            save(out/(c['key']+'_budget.json'),dict(cumulative_local_samples=budget.samples,core_wall_seconds=time.perf_counter()-budget.started))
            if ci==gates[-1]:
                estimate=budget.samples/2*len(cases)
                save(out/'pilot_gate.json',dict(teams=2,physical_samples=budget.samples,estimated_all_teams=estimate,passed=estimate+prior_samples<=400000))
                if estimate+prior_samples>400000:raise RuntimeError('Pilot budget estimate too high')
        save(out/'source_lock.json',locks);save(out/'scores_and_costs.json',rows);save(out/'timing.json',timings)
        for p,h in locks.items():assert sha(p)==h,p
        save(out/'verification.json',dict(accepted=True,teams=len(cases),audited_requests=audit_requests,
          independent_peer_full_scores=True,all_endpoints_match=True,caches_audited=True,sources_unchanged=True,pooled_samples=0))
        analyze(out)
        manifest.update(status='complete',teams=len(cases),local_samples=budget.samples,
          wall_seconds=time.perf_counter()-budget.started,model_load_seconds=load_seconds)
    except BaseException as e:
        manifest.update(status='failed',error=repr(e),local_samples=budget.samples)
        raise
    finally:
        manifest.update(prior_attempt_samples=prior_samples,cumulative_research_samples=prior_samples+budget.samples,
                        wall_seconds=time.perf_counter()-budget.started,prior_attempt_seconds=prior_seconds)
        save(out/'completion.json',manifest)


if __name__=='__main__':main()
