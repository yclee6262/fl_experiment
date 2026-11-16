"""Independent saved-state audit and preregistered seed-level analysis."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    with p.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvsave(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status'])
        w.writeheader();w.writerows(rows)
def direction(c,d): return int(c>d)-int(c < -d)
def ci(values,seed):
    v=np.array(values)
    rng=np.random.default_rng(seed)
    samples=np.mean(v[rng.integers(0,len(v),(10000,len(v)))],axis=1)
    return list(map(float,np.quantile(samples,[0.00625,0.99375])))


def analyze(root):
    sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1)
    output=root/'analysis'
    output.mkdir(exist_ok=False)
    config=read(root/'registration.json')
    errors=[];effects=[];seed_metrics=[];solve_rows=[];replays=[];reloads=[];fingerprints=[]
    def check(ok,reason):
        if not ok: errors.append(reason)
    for name,h in config['source_sha256'].items():
        check(sha(root/'source_snapshot'/name)==h,f'source_hash:{name}')
    for phase in ['pilot','confirmation']:
        for scenario in ['clean','controlled_poison']:
            size=3 if scenario=='clean' else 5
            for seed in config[phase+'_seeds']:
                key=f'{phase}/{scenario}/{seed}'
                d=root/key
                if not (d/'preparation.json').exists() or read(d/'preparation.json').get('status')!='complete':
                    errors.append(f'{key}:missing_or_failed_preparation'); continue
                state=read(d/'preparation.json');artifact=d/state['artifact']
                fingerprints.append(dict(key=key,semantic_data_sha256=state['semantic_data_sha256']))
                for name,h in state['artifact_sha256'].items():
                    check((d/name).is_file() and sha(d/name)==h,f'{key}:artifact_hash:{name}')
                if not (d/'scores.json').exists():
                    errors.append(f'{key}:missing_scores');continue
                scores={r['file']:r for r in read(d/'scores.json')['rows']}
                payload=torch.load(artifact/'local_models.pt',map_location='cpu',weights_only=True)
                models={}
                for i in state['ids']:
                    model=PretrainedModel(in_features=2);model.load_state_dict(payload['agents'][i]);model.eval();models[i]=model
                ep=torch.load(artifact/'global_evaluator.pt',map_location='cpu',weights_only=True)
                evaluator=PretrainedModel(in_features=2);evaluator.load_state_dict(ep['state_dict']);evaluator.eval()
                solutions={}
                labels=['A','B']+(['replay_A'] if phase=='pilot' and seed==config['pilot_seeds'][0] else [])
                for label in labels:
                    for excluded in [-1,*range(size)]:
                        name=f'{label}_{excluded}.json';p=d/'solves'/name
                        skey=f'{key}/{name}'
                        if not p.exists() or read(p).get('status')!='valid_delivery':
                            errors.append(f'{skey}:missing_or_failed_solve');continue
                        r=read(p);solutions[label,excluded]=r
                        x=np.array(r['result']['solution'])
                        check(np.isfinite(x).all() and np.max(np.abs(x))<=1+1e-8,f'{skey}:infeasible')
                        ids=[i for j,i in enumerate(state['ids']) if j!=excluded]
                        check(r['remaining_ids']==ids,f'{skey}:remaining_ids')
                        check(np.allclose(r['alphas'],np.ones(len(ids))/len(ids),atol=1e-12,rtol=0),f'{skey}:alphas')
                        candidates=[v for j,v in enumerate(state['candidates']) if j!=excluded]
                        check(r['candidates']==candidates and r['fixed_start']==state['fixed_start'],f'{skey}:candidate_or_start')
                        check(r['geometry']['projection_residual']<=1e-8 and not r['evaluator_loaded'],f'{skey}:semantics')
                        check(r['local_models_sha256']==sha(artifact/'local_models.pt'),f'{skey}:model_pairing')
                        check(r['coalition_sha256']==sha(d/'coalition.json'),f'{skey}:coalition_pairing')
                        with torch.no_grad():
                            tx=torch.tensor(x,dtype=torch.float32)[None,:]
                            error=abs(float(evaluator(tx).item()))
                            loss=float(np.mean([abs(float(models[i](tx).item())) for i in ids]))
                        s=scores.get(name)
                        check(s is not None and s['status']=='scored',f'{skey}:score_missing')
                        if s is None: continue
                        delta=abs(error-s['pooled_error'])
                        check(delta<=1e-7,f'{skey}:pooled_reload')
                        check(abs(loss-r['result']['loss'])<=1e-7,f'{skey}:internal_reload')
                        reloads.append(dict(key=skey,pooled_delta=delta,internal_delta=abs(loss-r['result']['loss'])))
                        chosen=r['result']['chosen_engine']
                        check(abs(r['result']['loss']-min(r['engines'][e]['loss'] for e in ['bfgs','custom']))<=1e-12,f'{skey}:engine_selection')
                        check(r['calls']==sum(r['engines'][e]['calls'] for e in ['bfgs','custom']),f'{skey}:call_accounting')
                        solve_rows.append(dict(phase=phase,scenario=scenario,seed=seed,label=label,excluded_index=excluded,
                            chosen_engine=chosen,chosen_success=r['engines'][chosen]['success'],
                            scipy_success=r['engines']['scipy']['success'],scipy_status=r['engines']['scipy']['status'],
                            scipy_iterations=r['engines']['scipy']['nit'],custom_iterations=r['engines']['custom']['iterations_attempted'],
                            custom_stop=r['engines']['custom']['stop_reason'],calls=r['calls'],input_samples=r['input_samples'],seconds=r['seconds'],
                            pooled_error=error,internal_loss=loss))
                        if label=='replay_A':
                            original=solutions.get(('A',excluded))
                            if original:
                                dx=float(np.max(np.abs(x-original['result']['solution'])))
                                de=abs(s['pooled_error']-scores[f'A_{excluded}.json']['pooled_error'])
                                dl=abs(r['result']['loss']-original['result']['loss'])
                                replays.append(dict(key=skey,X_delta=dx,error_delta=de,loss_delta=dl))
                                check(max(dx,de,dl)<=1e-7,f'{skey}:replay')
                current=[]
                for i,agent_id in enumerate(state['ids']):
                    if not all((label,j) in solutions for label in ['A','B'] for j in [-1,i]):continue
                    ca=scores[f'A_{i}.json']['pooled_error']-scores['A_-1.json']['pooled_error']
                    cb=scores[f'B_{i}.json']['pooled_error']-scores['B_-1.json']['pooled_error']
                    ta=scores[f'A_{i}.json']['truth_error']-scores['A_-1.json']['truth_error']
                    tb=scores[f'B_{i}.json']['truth_error']-scores['B_-1.json']['truth_error']
                    xa=np.array(solutions['A',i]['result']['solution']);xb=np.array(solutions['B',i]['result']['solution'])
                    ba=np.array(solutions['A',-1]['result']['solution']);bb=np.array(solutions['B',-1]['result']['solution'])
                    row=dict(phase=phase,scenario=scenario,seed=seed,agent_id=agent_id,C_A=ca,C_B=cb,absolute_C_change=abs(ca-cb),
                             direction_A=direction(ca,1e-4),direction_B=direction(cb,1e-4),truth_C_A=ta,truth_C_B=tb,
                             baseline_error_A=scores['A_-1.json']['pooled_error'],baseline_error_B=scores['B_-1.json']['pooled_error'],
                             removal_error_A=scores[f'A_{i}.json']['pooled_error'],removal_error_B=scores[f'B_{i}.json']['pooled_error'],
                             baseline_X_change=float(np.linalg.norm(ba-bb)),removal_X_change=float(np.linalg.norm(xa-xb)))
                    effects.append(row);current.append(row)
                if len(current)==size:
                    for delta in [1e-4,1e-7,1e-3]:
                        nonzero=[r for r in current if direction(r['C_A'],delta)!=0 or direction(r['C_B'],delta)!=0]
                        seed_metrics.append(dict(phase=phase,scenario=scenario,seed=seed,delta=delta,
                            direction_agreement=float(np.mean([direction(r['C_A'],delta)==direction(r['C_B'],delta) for r in current])),
                            magnitude_agreement=float(np.mean([r['absolute_C_change']<=1e-3 for r in current])),
                            nonneutral_n=len(nonzero),nonneutral_matches=sum(direction(r['C_A'],delta)==direction(r['C_B'],delta) for r in nonzero),
                            strict_reversals=sum(direction(r['C_A'],delta)*direction(r['C_B'],delta)==-1 for r in current)))
    aggregate=[]
    for scenario in ['clean','controlled_poison']:
        rows=[r for r in seed_metrics if r['phase']=='confirmation' and r['scenario']==scenario and r['delta']==1e-4]
        es=[r for r in effects if r['phase']=='confirmation' and r['scenario']==scenario]
        if not rows:
            aggregate.append(dict(scenario=scenario,n_seeds=0,n_effects=0,pass_all=False));continue
        a=float(np.mean([r['direction_agreement'] for r in rows]));b=float(np.mean([r['magnitude_agreement'] for r in rows]))
        aci=ci([r['direction_agreement'] for r in rows],20261016)
        bci=ci([r['magnitude_agreement'] for r in rows],20261016)
        nn=sum(r['nonneutral_n'] for r in rows);nm=sum(r['nonneutral_matches'] for r in rows)
        sufficient=sum(r['nonneutral_n']>0 for r in rows)>=5 and nn>0 and nm/nn>=0.9
        aggregate.append(dict(scenario=scenario,n_seeds=len(rows),n_effects=len(es),direction_agreement=a,direction_CI=aci,
            magnitude_agreement=b,magnitude_CI=bci,nonneutral_seeds=sum(r['nonneutral_n']>0 for r in rows),
            nonneutral_effects=nn,nonneutral_agreement=None if not nn else nm/nn,
            strict_reversals=sum(r['strict_reversals'] for r in rows),median_absolute_C_change=float(np.median([r['absolute_C_change'] for r in es])),
            max_absolute_C_change=max(r['absolute_C_change'] for r in es),
            C_A_sign_counts=dict(Counter(r['direction_A'] for r in es)),C_B_sign_counts=dict(Counter(r['direction_B'] for r in es)),
            pooled_truth_conflicts_A=sum(direction(r['C_A'],1e-4)!=direction(r['truth_C_A'],1e-4) for r in es),
            pooled_truth_conflicts_B=sum(direction(r['C_B'],1e-4)!=direction(r['truth_C_B'],1e-4) for r in es),
            pass_all=len(rows)==10 and a>=.9 and b>=.9 and min(aci)>=.8 and min(bci)>=.8 and sufficient))
    execution=read(root/'execution.json')
    complete=execution['status']=='complete' and not errors and len(effects)==96 and len(replays)==10
    # Ensure unique datasets across distinct seeds; within-seed scenario labels are paired.
    check(len({r['semantic_data_sha256'] for r in fingerprints})==len(fingerprints),'dataset_hash_duplicate')
    complete=complete and not errors
    conclusion=('limited_budget_stability_established' if complete and all(r['pass_all'] for r in aggregate)
                else 'budget_stability_not_established' if complete else 'incomplete_or_failed_acceptance')
    summaries=[]
    for scenario in ['clean','controlled_poison']:
        for label in ['A','B']:
            rows=[r for r in solve_rows if r['phase']=='confirmation' and r['scenario']==scenario and r['label']==label]
            summaries.append(dict(scenario=scenario,label=label,n_solves=len(rows),calls=sum(r['calls'] for r in rows),
                input_samples=sum(r['input_samples'] for r in rows),seconds=sum(r['seconds'] for r in rows),
                chosen_engines=dict(Counter(r['chosen_engine'] for r in rows)),scipy_unsuccessful=sum(not r['scipy_success'] for r in rows),
                chosen_unsuccessful=sum(not r['chosen_success'] for r in rows),custom_stops=dict(Counter(r['custom_stop'] for r in rows))))
    csvsave(output/'participant_effects.csv',effects)
    csvsave(output/'seed_metrics.csv',seed_metrics)
    csvsave(output/'solver_records.csv',solve_rows)
    csvsave(output/'reload_verification.csv',reloads)
    csvsave(output/'replay_verification.csv',replays)
    formal=[r for r in effects if r['phase']=='confirmation']
    examples=sorted(formal,key=lambda r:(-r['absolute_C_change'],r['scenario'],r['seed'],r['agent_id']))[:6]
    save(output/'representative_cases.json',examples)
    save(output/'acceptance.json',dict(accepted=complete,errors=errors,n_effects=len(effects),n_replays=len(replays),
            n_reloads=len(reloads),max_pooled_reload_delta=max((r['pooled_delta'] for r in reloads),default=None),
            max_internal_reload_delta=max((r['internal_delta'] for r in reloads),default=None)))
    save(output/'results.json',dict(status=conclusion,aggregate=aggregate,costs=summaries,execution=execution,
         interpretation='Custom 30 vs 120 only; fixed models/start; no global optimality claim; pooled offline, truth diagnostic.'))
    print(json.dumps(dict(status=conclusion,accepted=complete,aggregate=aggregate,costs=summaries,errors=errors),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();analyze(args.root.resolve())
