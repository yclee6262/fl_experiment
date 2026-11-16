"""Prospectively specified extra-label alpha supplement to the unified screen."""
import sys,json,time,contextlib,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/yclee/Documents/GitHub/fl_experiment')
import oob_shrinkage_research as study
from unified_alpha_methods import initial_alpha
from explore_targeted_label_alpha import proposal_local_points,proposal_local_alpha
from alpha_weight_experiments import attach_query_counter
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth
from research_analysis import paired_bootstrap_mean_ci,audit_coverage

def extra_initial(method,agents,mse,x,y,proposals,target,**kwargs):
 p=kwargs['cached_blind_predictions']
 if method=='proposal_local_9':
  w,a=proposal_local_alpha(mse,p,y);a.update(fallback_reason='',agent_ids=[g.agent_id for g in agents]);return w,a
 if method=='local_bias_risk_100':return initial_alpha('local_bias_risk',agents,mse,x,y,proposals,target,**kwargs)
 raise ValueError(method)

def run():
 r=json.loads(Path('docs/unified_alpha_extra_labels_registration_20261011.json').read_text())
 root=Path(r['output']);root.mkdir(exist_ok=False)
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==r['runner_sha256'], 'runner changed after registration'
 root.joinpath('runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
 snapshot=root/'source_snapshot';snapshot.mkdir()
 for name,h in r['dependency_sha256'].items():
  assert study.digest(name)==h, 'dependency changed '+name
  (snapshot/name).write_bytes(Path(name).read_bytes())
 root.joinpath('registration.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 rows=[];errors=[];groups=[];references=[];max_alpha=0.;max_score=0.
 study.initial_alpha=extra_initial
 with (root/'solver.log').open('w') as log:
  for scenario in r['scenarios']:
   for seed in r['seeds']:
    server,item,artifact=study.load_base(r['source'],scenario,seed)
    evaluator,_=load_evaluator_artifact(artifact)
    refs={'qualification_reload_research_requests':3,'qualification_reload_research_inputs':60,'scenario':scenario,'seed':seed,'artifact':str(artifact.resolve()),'hashes':{p.name:study.digest(p) for p in artifact.iterdir() if p.is_file()}}
    references.append(refs)
    qualification=np.column_stack([g.api_predict(server.test_X) for g in server.trusted_agents])
    mse=((qualification-server.test_y[:,None])**2).mean(axis=0)
    for method in r['methods']:
     if method=='proposal_local_9':x=proposal_local_points(server.I_list,seed,-1,1);y=synthetic_truth(x);labels=9
     else:
      extra=np.random.default_rng(np.random.SeedSequence([20261011,seed,80])).uniform(-1,1,(80,2))
      x=np.concatenate([server.test_X,extra]);y=np.concatenate([server.test_y,synthetic_truth(extra)]);labels=80
     counter=attach_query_counter(server.trusted_agents);counter['stage']='initialization';start=time.perf_counter()
     prediction=np.column_stack([g.api_predict(x) for g in server.trusted_agents]);acq=time.perf_counter()-start
     calls=counter['requests']['initialization'];samples=counter['input_samples']['initialization'];counter['stage']='finished'
     with contextlib.redirect_stdout(log):row=study.solve(server,method,mse,x,y,prediction,20261011+seed*100,1000)
     row.update(scenario=scenario,seed=seed,method=method,additional_labels=labels,total_trusted_labels=20+labels,
       calibration_X=x.tolist(),calibration_y=y.tolist(),calibration_prediction=prediction.tolist(),
       initialization_seconds=row['initialization_seconds']+acq,
       initialization_requests=row['initialization_requests']+calls,
       initialization_input_samples=row['initialization_input_samples']+samples)
     if row['status']=='valid':
      _,score=score_solution(evaluator,np.asarray(row['X']),0)
      row.update(pooled_error=score,truth_error=abs(float(synthetic_truth(np.asarray(row['X'])[None,:])[0])))
     study.append(root/'results.jsonl',row);rows.append(row)
    print(scenario+' completed replication '+str(seed),flush=True)
 # Reload models and evaluator independently; preserve different label budgets.
 for refs in references:
  for name,h in refs['hashes'].items():
   if study.digest(Path(refs['artifact'])/name)!=h:errors.append('artifact hash mismatch')
  server,_,artifact=study.load_base(r['source'],refs['scenario'],refs['seed']);evaluator,_=load_evaluator_artifact(artifact)
  for row in [x for x in rows if x['scenario']==refs['scenario'] and x['seed']==refs['seed']]:
   x=np.asarray(row['calibration_X']);y=np.asarray(row['calibration_y']);p=np.column_stack([g.api_predict(x) for g in server.trusted_agents])
   if not np.array_equal(p,np.asarray(row['calibration_prediction'])):errors.append('prediction reload mismatch')
   w,_=extra_initial(row['method'],server.trusted_agents,np.asarray(row['mse']),x,y,server.I_list,0,cached_blind_predictions=p)
   delta=float(np.max(np.abs(w-np.asarray(row['alpha']))));max_alpha=max(max_alpha,delta)
   if delta>1e-7:errors.append('alpha reload mismatch')
   if row['status']=='valid':
    if server.box_violation(np.asarray(row['X']))>1e-8:errors.append('infeasible solution')
    _,score=score_solution(evaluator,np.asarray(row['X']),0);delta=abs(score-row['pooled_error']);max_score=max(max_score,delta)
    if delta>1e-7:errors.append('score reload mismatch')
 expected=[(s,k,m) for s in r['scenarios'] for k in r['seeds'] for m in r['methods']]
 coverage=audit_coverage(rows,expected,['scenario','seed','method'])
 if not coverage['complete']:errors.append('coverage incomplete')
 for s in r['scenarios']:
  for m in r['methods']:
   chosen=[x for x in rows if x['scenario']==s and x['method']==m];v=[x['pooled_error'] for x in chosen if x['status']=='valid']
   groups.append({'scenario':s,'method':m,'additional_labels':9 if m=='proposal_local_9' else 80,'mean':float(np.mean(v)) if v else None,
     'median':float(np.median(v)) if v else None,'valid':len(v),'failures':len(chosen)-len(v)})
 base=[json.loads(l) for l in Path(r['base_results']).read_text().splitlines()]
 index={(x['scenario'],x['seed'],x['method']):x for x in base};contrasts=[]
 for s in r['scenarios']:
  for m in r['methods']:
   for b in ['A_uniform','B_stage0']:
    ds=[x['pooled_error']-index[s,x['seed'],b]['pooled_error'] for x in rows if x['scenario']==s and x['method']==m and x['status']=='valid' and index[s,x['seed'],b]['status']=='valid']
    contrasts.append({'scenario':s,'method':m,'baseline':b,'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,'CI95':paired_bootstrap_mean_ci(ds),'interpretation':'exploratory extra-information comparison, not equal information budget'})
 verification={'passed':not errors,'errors':errors,'coverage':coverage,'expected':len(expected),'valid':sum(x['status']=='valid' for x in rows),'max_alpha_reload_diff':max_alpha,'max_pooled_reload_diff':max_score}
 study.dump(root/'artifact_references.json',references)
 study.dump(root/'analysis.json',{'verification':verification,'groups':groups,'comparisons':contrasts,'qualification_reload_research_overhead_requests':3*len(references),'qualification_reload_research_overhead_inputs':60*len(references)})
 if errors:raise RuntimeError(errors)
 print(json.dumps(verification))

if __name__=='__main__':run()
