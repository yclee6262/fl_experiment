"""Six paired methods, all failure rows saved, common preparation cached once."""
import argparse,contextlib,itertools,json,time
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from pure_es_initialization import METHODS,SCENARIOS,initializer
from peer_support_alpha import support_signal
from alpha_weight_experiments import attach_query_counter
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth
from research_analysis import audit_coverage

def snapshot(root,reg):
 d=root/'source_snapshot';d.mkdir()
 for name,h in reg['source_sha256'].items():
  if shared.digest(name)!=h:raise RuntimeError('Frozen source changed '+name)
  (d/name).write_bytes(Path(name).read_bytes())

def common_failure(root,identity,status,reason):
 for method in METHODS:shared.append(root/'results.jsonl',{**identity,'method':method,'status':status,'reason':reason,'X':None,'alpha':None,'pooled_error':None,'truth_error':None,'cost_status':'see shared/source preparation; no method optimization attempted','additional_labels':0})

def run(registration,batch):
 reg=json.loads(Path(registration).read_text());b=reg['batches'][batch];root=Path(b['results']);root.mkdir(exist_ok=False);snapshot(root,reg)
 manifest={**b,'methods':METHODS,'scenarios':SCENARIOS,'source_sha256':reg['source_sha256'],'status':'running','expected':len(b['seeds'])*3*len(METHODS),'registration':registration};shared.dump(root/'manifest.json',manifest)
 try:
  with (root/'solver.log').open('w') as log:
   for scenario in SCENARIOS:
    items={x['seed']:x for x in map(json.loads,(Path(b['source'])/scenario/'shared_costs.jsonl').read_text().splitlines())}
    for seed in b['seeds']:
     identity={'scenario':scenario,'seed':seed};item=items[seed]
     if item.get('missing_fixed_ids') or not item.get('artifact_path'):
      shared.append(root/'shared.jsonl',{**identity,'preparation':item,'artifact':None,'eligibility':'qualification_failure'});common_failure(root,identity,'qualification_failure',item.get('failure_reason') or item.get('missing_fixed_ids'));continue
     server,_,artifact=shared.load_base(b['source'],scenario,seed);proposals=np.asarray(server.I_list);n=len(server.trusted_agents)
     if n!=3 or proposals.shape!=(3,2) or not np.isfinite(proposals).all() or np.any(abs(proposals)>1+1e-8):
      shared.append(root/'shared.jsonl',{**identity,'preparation':item,'artifact':str(artifact),'eligibility':'invalid_proposal'});common_failure(root,identity,'preparation_failure','illegal_fixed_proposal');continue
     c=attach_query_counter(server.trusted_agents);c['stage']='qualification_reload';start=time.perf_counter();blind=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((blind-server.test_y[:,None])**2).mean(axis=0)
     blind_cost={'seconds':time.perf_counter()-start,'requests':c['requests']['qualification_reload'],'input_samples':c['input_samples']['qualification_reload']}
     p=np.empty((n,n));c['stage']='support';start=time.perf_counter()
     for j,a in enumerate(server.trusted_agents):
      for i,x in enumerate(proposals):
       if j!=i:p[j,i]=float(np.asarray(a.api_predict(x)).reshape(-1)[0])
     support_cost={'seconds':time.perf_counter()-start,'requests':c['requests']['support'],'input_samples':c['input_samples']['support']}
     c['stage']='diagonal_diagnostic';start=time.perf_counter()
     for i,a in enumerate(server.trusted_agents):p[i,i]=float(np.asarray(a.api_predict(proposals[i])).reshape(-1)[0])
     diagonal_cost={'seconds':time.perf_counter()-start,'requests':c['requests']['diagonal_diagnostic'],'input_samples':c['input_samples']['diagonal_diagnostic']};c['stage']='finished'
     try:s,audit=support_signal(p,0)
     except ValueError as exc:
      shared.append(root/'shared.jsonl',{**identity,'preparation':item,'artifact':str(artifact),'eligibility':'invalid_peer','support_cost':support_cost});common_failure(root,identity,'preparation_failure',str(exc));continue
     peer_record={**identity,'agent_ids':[a.agent_id for a in server.trusted_agents],'proposals':proposals.tolist(),'prediction_matrix':p.tolist(),'matrix_rows':'evaluating_model','matrix_columns':'candidate_proposer','support_audit':audit,'blind_prediction':blind.tolist(),'stage0_mse':mse.tolist(),'blind_cost':blind_cost,'support_cost':support_cost,'diagonal_diagnostic_cost':diagonal_cost}
     shared.append(root/'peer_evidence.jsonl',peer_record)
     shared.append(root/'shared.jsonl',{**identity,'preparation':item,'artifact':str(artifact.resolve()),'sha256':{q.name:shared.digest(q) for q in artifact.iterdir() if q.is_file()},'eligibility':'eligible','blind_cost':blind_cost,'support_cost':support_cost,'diagonal_diagnostic_cost':diagonal_cost})
     shared.initial_alpha=initializer(p)
     # Only pass local evidence to solve; external evaluator is loaded afterwards.
     evaluator=None
     for method in METHODS:
      try:
       with contextlib.redirect_stdout(log):row=shared.solve(server,method,mse,server.test_X,server.test_y,blind,seed,1000)
       row.update(identity,method=method,additional_labels=0,agent_ids=peer_record['agent_ids'],support_evidence_key=[scenario,seed],support_independent_cost=support_cost if method in ['pure_support'] else {'seconds':0,'requests':0,'input_samples':0})
       if row['status']=='valid':
        if evaluator is None:evaluator,_=load_evaluator_artifact(artifact)
        start=time.perf_counter();_,score=score_solution(evaluator,np.asarray(row['X']),0);row.update(pooled_error=score,truth_error=abs(float(synthetic_truth(np.asarray(row['X'])[None,:])[0])),external_scoring_seconds=time.perf_counter()-start)
       shared.append(root/'results.jsonl',row)
      except Exception as exc:
       shared.append(root/'results.jsonl',{**identity,'method':method,**getattr(exc,'partial_row',{}),'status':'interrupted','reason':repr(exc)});raise
     print(scenario+' completed '+str(seed),flush=True)
  manifest['status']='complete'
 except Exception as exc:manifest.update(status='interrupted',reason=repr(exc));raise
 finally:shared.dump(root/'manifest.json',manifest)

def verify(path):
 root=Path(path);m=json.loads((root/'manifest.json').read_text());rows=list(map(json.loads,(root/'results.jsonl').read_text().splitlines()));index={(r['scenario'],r['seed'],r['method']):r for r in rows};errors=[]
 coverage=audit_coverage(rows,list(itertools.product(SCENARIOS,m['seeds'],METHODS)),['scenario','seed','method'])
 if not coverage['complete'] or m['status']!='complete':errors.append('incomplete execution')
 for name,h in m['source_sha256'].items():
  if shared.digest(root/'source_snapshot'/name)!=h:errors.append('source changed '+name)
 peers={(r['scenario'],r['seed']):r for r in map(json.loads,(root/'peer_evidence.jsonl').read_text().splitlines())} if (root/'peer_evidence.jsonl').exists() else {}
 maxima={'alpha':0.,'pooled':0.,'matrix':0.};valid=0
 for rec in map(json.loads,(root/'shared.jsonl').read_text().splitlines()):
  key=(rec['scenario'],rec['seed'])
  if rec['eligibility']!='eligible':
   if any(index[key+(method,)]['status'] not in ['qualification_failure','preparation_failure'] for method in METHODS):errors.append('common failure not preserved')
   continue
  for name,h in rec['sha256'].items():
   if shared.digest(Path(rec['artifact'])/name)!=h:errors.append('artifact changed')
  server,item,artifact=shared.load_base(m['source'],*key);saved=peers[key];p=np.array([[float(np.asarray(agent.api_predict(x)).reshape(-1)[0]) for x in server.I_list] for agent in server.trusted_agents]);diff=float(np.max(abs(p-np.array(saved['prediction_matrix']))));maxima['matrix']=max(maxima['matrix'],diff)
  if diff>1e-7:errors.append('matrix reload differs')
  if saved['agent_ids']!=[a.agent_id for a in server.trusted_agents] or not np.array_equal(np.asarray(saved['proposals']),np.asarray(server.I_list)):errors.append('agent/proposal alignment differs')
  blind=np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents]);mse=((blind-server.test_y[:,None])**2).mean(axis=0);init=initializer(p);evaluator,_=load_evaluator_artifact(artifact)
  for method in METHODS:
   row=index[key+(method,)];w,_=init(method,server.trusted_agents,mse,server.test_X,server.test_y,server.I_list,0)
   diff=float(np.max(abs(w-np.array(row['alpha']))));maxima['alpha']=max(maxima['alpha'],diff)
   if diff>1e-7 or min(w)<0 or abs(sum(w)-1)>1e-8:errors.append('invalid/reload alpha')
   if row['status']=='valid':
    x=np.asarray(row['X']);violation=server.box_violation(x)
    if not np.isfinite(x).all() or violation>1e-8:errors.append('invalid delivery')
    _,score=score_solution(evaluator,x,0);diff=abs(score-row['pooled_error']);maxima['pooled']=max(maxima['pooled'],diff);valid+=1
    if diff>1e-7:errors.append('pooled reload differs')
   elif row['status'] not in ['solver_failure','invalid_output']:errors.append('unexpected method failure')
   if 'stage3_requests' not in row or row['stage3_requests']<0 or row['stage3_seconds']<0:errors.append('method costs missing')
 result={'passed':not errors,'errors':errors,'coverage':coverage,'expected':m['expected'],'saved':len(rows),'valid':valid,'max_reload_difference':maxima};shared.dump(root/'verification.json',result);print(json.dumps(result));return result

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--registration');a.add_argument('--batch',type=int);a.add_argument('--verify');x=a.parse_args()
 if x.verify:
  if not verify(x.verify)['passed']:raise SystemExit(1)
 else:run(x.registration,x.batch)
