"""Save every attempted condition, preserving qualification/solver failures."""
import argparse,contextlib,json,time,itertools
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from repeated_alpha_candidates import METHODS,EXTRA,DEFINITIONS,labelled_inputs,initial_alpha
from alpha_weight_experiments import attach_query_counter
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth
from research_analysis import audit_coverage

def run(reg,batch_index):
    reg=json.loads(Path(reg).read_text());batch=reg['batches'][batch_index];root=Path(batch['results'])
    root.mkdir(exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    for name,h in reg['source_sha256'].items():
        if shared.digest(name)!=h:raise RuntimeError('registered source changed '+name)
        (snapshot/name).write_bytes(Path(name).read_bytes())
    manifest={**batch,'methods':METHODS,'definitions':DEFINITIONS,'scenarios':reg['scenarios'],'status':'running',
              'expected':len(batch['seeds'])*len(reg['scenarios'])*len(METHODS),'source_sha256':reg['source_sha256']}
    shared.dump(root/'manifest.json',manifest);shared.initial_alpha=initial_alpha
    try:
        with (root/'solver.log').open('w') as log:
            for scenario in reg['scenarios']:
                source=Path(batch['source'])/scenario
                items={x['seed']:x for x in [json.loads(l) for l in (source/'shared_costs.jsonl').read_text().splitlines()]}
                for seed in batch['seeds']:
                    item=items[seed]
                    if item.get('missing_fixed_ids') or not item.get('artifact_path'):
                        shared.append(root/'shared.jsonl',{'scenario':scenario,'seed':seed,'preparation':item,'artifact':None})
                        for method in METHODS:shared.append(root/'results.jsonl',{'scenario':scenario,'seed':seed,'method':method,
                            'status':'qualification_failure','reason':item.get('failure_reason') or item.get('missing_fixed_ids'),
                            'X':None,'alpha':None,'pooled_error':None,'truth_error':None,
                            'initialization_seconds':0,'initialization_requests':0,'stage3_seconds':0,'stage3_requests':0,
                            'shared_preparation_cost_status':item.get('preparation_costs_status','saved_source_record')})
                        continue
                    server,_,artifact=shared.load_base(batch['source'],scenario,seed)
                    evaluator,_=load_evaluator_artifact(artifact)
                    shared.append(root/'shared.jsonl',{'scenario':scenario,'seed':seed,'preparation':item,
                        'artifact':str(artifact.resolve()),'sha256':{p.name:shared.digest(p) for p in artifact.iterdir() if p.is_file()}})
                    counter=attach_query_counter(server.trusted_agents);counter['stage']='qualification_reload';t=time.perf_counter()
                    p=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((p-server.test_y[:,None])**2).mean(axis=0)
                    shared.append(root/'calibration.jsonl',{'scenario':scenario,'seed':seed,'method_group':'core_shared',
                        'X':server.test_X.tolist(),'y':server.test_y.tolist(),'prediction':p.tolist(),
                        'seconds':time.perf_counter()-t,'requests':counter['requests']['qualification_reload'],
                        'input_samples':counter['input_samples']['qualification_reload']})
                    counter['stage']='finished'
                    for method in METHODS:
                        identity={'scenario':scenario,'seed':seed,'method':method}
                        x,y,cache=server.test_X,server.test_y,p
                        labels=0;acquisition=None
                        if method in EXTRA:
                            x,y,labels=labelled_inputs(method,server,seed)
                            c=attach_query_counter(server.trusted_agents);c['stage']='initialization';t=time.perf_counter()
                            cache=np.column_stack([a.api_predict(x) for a in server.trusted_agents])
                            acquisition={'seconds':time.perf_counter()-t,'requests':c['requests']['initialization'],
                                'input_samples':c['input_samples']['initialization']};c['stage']='finished'
                            shared.append(root/'calibration.jsonl',{**identity,'X':x.tolist(),'y':y.tolist(),'prediction':cache.tolist(),**acquisition})
                        try:
                            with contextlib.redirect_stdout(log):row=shared.solve(server,method,mse,x,y,cache,20261011+seed*100,1000)
                            row.update(identity,additional_labels=labels,total_trusted_labels=20+labels,bootstrap_seed=20261011+seed*100)
                            if acquisition:
                                for key,field in [('seconds','initialization_seconds'),('requests','initialization_requests'),('input_samples','initialization_input_samples')]:row[field]+=acquisition[key]
                            if row['status']=='valid':
                                t=time.perf_counter();_,score=score_solution(evaluator,np.asarray(row['X']),0)
                                row.update(pooled_error=score,truth_error=abs(float(synthetic_truth(np.asarray(row['X'])[None,:])[0])),scoring_seconds=time.perf_counter()-t)
                            shared.append(root/'results.jsonl',row)
                        except Exception as exc:
                            shared.append(root/'results.jsonl',{**identity,**getattr(exc,'partial_row',{}),'status':'interrupted','reason':repr(exc)})
                            raise
                    print(scenario+' completed '+str(seed),flush=True)
        manifest['status']='complete'
    except Exception as exc:manifest.update(status='interrupted',reason=repr(exc));raise
    finally:shared.dump(root/'manifest.json',manifest)

def verify(path):
    root=Path(path);m=json.loads((root/'manifest.json').read_text());rows=[json.loads(l) for l in (root/'results.jsonl').read_text().splitlines()]
    expected=list(itertools.product(m['scenarios'],m['seeds'],m['methods']));coverage=audit_coverage(rows,expected,['scenario','seed','method']);errors=[]
    if not coverage['complete']:errors.append('condition coverage incomplete')
    for name,h in m['source_sha256'].items():
        if shared.digest(root/'source_snapshot'/name)!=h:errors.append('source snapshot changed '+name)
    saved=[json.loads(l) for l in (root/'shared.jsonl').read_text().splitlines()]
    cal=[json.loads(l) for l in (root/'calibration.jsonl').read_text().splitlines()] if (root/'calibration.jsonl').exists() else []
    alpha_diff=score_diff=0.;valid_count=0
    index={(r['scenario'],r['seed'],r['method']):r for r in rows}
    for item in saved:
        if item['artifact'] is None:continue
        for name,h in item['sha256'].items():
            if shared.digest(Path(item['artifact'])/name)!=h:errors.append('model/data hash changed')
        server,_,artifact=shared.load_base(m['source'],item['scenario'],item['seed']);evaluator,_=load_evaluator_artifact(artifact)
        cached={c.get('method','core_shared'):c for c in cal if c['scenario']==item['scenario'] and c['seed']==item['seed']}
        base=cached['core_shared'];basep=np.column_stack([a.api_predict(np.asarray(base['X'])) for a in server.trusted_agents])
        mse=((basep-np.asarray(base['y'])[:,None])**2).mean(axis=0)
        for method in m['methods']:
            row=index.get((item['scenario'],item['seed'],method))
            if row is None or row['status']=='interrupted':errors.append('missing/interrupted method');continue
            c=cached[method if method in EXTRA else 'core_shared'];x=np.asarray(c['X']);y=np.asarray(c['y'])
            if method in EXTRA:
                expected_x,expected_y,_=labelled_inputs(method,server,item['seed'])
                if not np.array_equal(x,expected_x) or not np.array_equal(y,expected_y):errors.append('extra label regeneration changed')
            p=np.column_stack([a.api_predict(x) for a in server.trusted_agents])
            if not np.array_equal(p,np.asarray(c['prediction'])):errors.append('qualification reload changed')
            w,_=initial_alpha(method,server.trusted_agents,mse,x,y,server.I_list,0,cached_blind_predictions=p,bootstrap_seed=row['bootstrap_seed'],bootstrap_replicates=1000)
            diff=float(np.max(np.abs(w-np.asarray(row['alpha']))));alpha_diff=max(alpha_diff,diff)
            if diff>1e-7:errors.append('alpha reload changed')
            if row['status']=='valid':
                x_final=np.asarray(row['X']);v=server.box_violation(x_final)
                if not np.isfinite(x_final).all() or v>1e-8:errors.append('invalid delivery')
                _,score=score_solution(evaluator,x_final,0);diff=abs(score-row['pooled_error']);score_diff=max(score_diff,diff)
                if diff>1e-7:errors.append('pooled reload changed')
                valid_count+=1
    result={'passed':not errors and m['status']=='complete','errors':errors,'coverage':coverage,'expected':len(expected),
            'observed':len(rows),'valid':valid_count,'max_alpha_reload_diff':alpha_diff,'max_pooled_reload_diff':score_diff}
    shared.dump(root/'verification.json',result);print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration');p.add_argument('--batch',type=int);p.add_argument('--verify');args=p.parse_args()
    if args.verify:
        if not verify(args.verify)['passed']:raise SystemExit(1)
    else:run(args.registration,args.batch)
