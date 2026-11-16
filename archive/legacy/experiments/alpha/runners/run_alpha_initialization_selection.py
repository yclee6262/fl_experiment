"""Registered C/P runner. Decisions are sealed before external scoring."""
import torch
import argparse,contextlib,json,itertools,time,random
from collections import Counter
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from alpha_weight_experiments import attach_query_counter
from four_alpha_initialization import initializer as six_initializer
from pure_es_initialization import initializer as pure_initializer
from alpha_initialization_selection import METHODS,SCENARIOS,DISPLAY,select_c,select_p,weights_from_predictions
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth
from research_analysis import audit_coverage

def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def route(peer):
 six,pure=six_initializer(peer),pure_initializer(peer)
 def initial(method,*args,**kw):return (pure if method in ['stacked_full','pure_support'] else six)(method,*args,**kw)
 return initial

def evidence(base):
 c=attach_query_counter(base.trusted_agents);c['stage']='blind';t=time.perf_counter()
 p=np.column_stack([a.api_predict(base.test_X) for a in base.trusted_agents]);blind=time.perf_counter()-t
 c['stage']='peer';t=time.perf_counter();peer=np.full((3,3),np.nan)
 for j,a in enumerate(base.trusted_agents):
  for i,x in enumerate(base.I_list):
   if j!=i:peer[j,i]=float(a.api_predict(x)[0])
 peertime=time.perf_counter()-t;c['stage']='finished'
 return p,peer,{'blind_seconds':blind,'blind_requests':c['requests']['blind'],'blind_input_samples':c['input_samples']['blind'],
                'support_seconds':peertime,'support_requests':c['requests']['peer'],'support_input_samples':c['input_samples']['peer']}

def solve_one(source,sc,seed,method,peer,p):
 base,_,_=shared.load_base(source,sc,seed);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 old=shared.initial_alpha;shared.initial_alpha=route(peer)
 try:return shared.solve(base,method,((p-base.test_y[:,None])**2).mean(axis=0),base.test_X,base.test_y,p,seed,1000)
 finally:shared.initial_alpha=old

def main_run(regpath,stage):
 reg=json.load(open(regpath));conf=reg[stage];root=Path(conf['results']);root.mkdir(parents=True,exist_ok=False)
 for name,h in reg['source_sha256'].items():
  if shared.digest(name)!=h:raise RuntimeError('Source changed '+name)
 snap=root/'source_snapshot';snap.mkdir()
 for name in reg['source_sha256']:(snap/name).write_bytes(Path(name).read_bytes())
 manifest=dict(conf,status='running',methods=METHODS+['C','P'],scenarios=SCENARIOS,source_sha256=reg['source_sha256'],expected=len(conf['seeds'])*30,decisions={})
 shared.dump(root/'manifest.json',manifest)
 try:
  with (root/'solver.log').open('w') as log:
   for sc,seed in itertools.product(SCENARIOS,conf['seeds']):
    item=next(x for x in lines(Path(conf['source'])/sc/'shared_costs.jsonl') if x['seed']==seed)
    identity={'scenario':sc,'seed':seed};rows={}
    if item.get('missing_fixed_ids') or not item.get('artifact_path'):
     shared.append(root/'shared.jsonl',dict(identity,eligibility='qualification_failure',preparation=item))
     for m in METHODS+['C','P']:shared.append(root/'results.jsonl',dict(identity,method=m,status='qualification_failure',reason=item.get('failure_reason',item.get('missing_fixed_ids')),X=None,alpha=None,pooled_error=None,additional_labels=0))
     continue
    base,_,artifact=shared.load_base(conf['source'],sc,seed)
    p,peer,prep=evidence(base)
    if np.asarray(base.I_list).shape!=(3,2) or not np.isfinite(base.I_list).all() or np.max(np.abs(base.I_list))>1+1e-8:raise ValueError('Invalid common proposal')
    t=time.perf_counter();ce=select_c(p,base.test_y,peer,0);ctime=time.perf_counter()-t
    common=dict(identity,eligibility='eligible',artifact=str(artifact.resolve()),artifact_sha256={f.name:shared.digest(f) for f in artifact.iterdir() if f.is_file()},
                preparation=item,blind_predictions=p.tolist(),peer_predictions=[[None if not np.isfinite(v) else float(v) for v in r] for r in peer],signal_cost=prep)
    shared.append(root/'shared.jsonl',common)
    for m in METHODS:
     try:
      with contextlib.redirect_stdout(log):row=solve_one(conf['source'],sc,seed,m,peer,p)
     except Exception as exc:
      shared.append(root/'partial_failures.jsonl',dict(identity,method=m,**getattr(exc,'partial_row',{}),reason=repr(exc)));raise
     rows[m]=dict(row,method=m,additional_labels=0,**identity)
    # Always execute C independently: one solve, no implicit research cache credit.
    with contextlib.redirect_stdout(log):cr=solve_one(conf['source'],sc,seed,ce['selected_method'],peer,p)
    reuse_max=0.;reuse_equal=cr['status']==rows[ce['selected_method']]['status']
    if cr['status']=='valid':
     reuse_max=float(np.max(np.abs(np.array(cr['X'])-rows[ce['selected_method']]['X'])))
     reuse_equal=reuse_equal and reuse_max<=1e-7 and cr['solver']['chosen_engine']==rows[ce['selected_method']]['solver']['chosen_engine']
    if not reuse_equal:raise RuntimeError('Independent C differs from its fixed method')
    rows['C']=dict(cr,method='C',selected_method=ce['selected_method'],selector_seconds=ctime,additional_labels=0,**identity)
    q=np.asarray(rows['B_stage0']['alpha']);t=time.perf_counter();pred={};pcalls=0;psamples=0
    for m,row in rows.items():
     if m=='C' or row['status']!='valid':continue
     pred[m]=[float(a.api_predict(np.asarray(row['X']))[0]) for a in base.trusted_agents];pcalls+=3;psamples+=3
    pe=select_p({m:rows[m] for m in METHODS},pred,q,0);ptime=time.perf_counter()-t;selected=pe['selected_method']
    rows['P']=dict(rows[selected] if selected else {'status':'selector_failure','X':None,'alpha':None,'pooled_error':None},method='P',selected_method=selected,additional_labels=0,**identity)
    # Seal local decision artifacts, exclusively create each file.
    decision={'identity':identity,'C':ce,'P':pe,'C_independent_reuse_check':{'passed':reuse_equal,'X_difference':reuse_max},'deliveries':rows}
    decisions=root/'decisions';decisions.mkdir(exist_ok=True);dp=decisions/f'{sc}_{seed}.json'
    with dp.open('x') as f:json.dump(decision,f,ensure_ascii=False,allow_nan=False,default=shared.json_numpy)
    manifest['decisions'][str(dp)]=shared.digest(dp);shared.dump(root/'manifest.json',manifest)
    shared.append(root/'selector_evidence.jsonl',dict(identity,C=ce,P=pe))
    costs={}
    for m in METHODS:
     r=rows[m];support=m in ['pure_support','peer_support_shrink','support_quality_mix50']
     costs[m]={'stage3_runs':1,'seconds':r['stage3_seconds']+r['initialization_seconds']+(prep['support_seconds'] if support else 0),
               'requests':r['stage3_requests']+r['initialization_requests']+(6 if support else 0),
               'input_samples':r['stage3_input_samples']+r['initialization_input_samples']+(6 if support else 0)}
    chosen=costs[ce['selected_method']]
    # All eight CV candidates require six peer predictions, irrespective of winner.
    costs['C']={'stage3_runs':1,'seconds':rows['C']['stage3_seconds']+rows['C']['initialization_seconds']+ctime+prep['support_seconds'],
                'requests':rows['C']['stage3_requests']+rows['C']['initialization_requests']+6,
                'input_samples':rows['C']['stage3_input_samples']+rows['C']['initialization_input_samples']+6}
    costs['P']={'stage3_runs':8,'seconds':sum(rows[m]['stage3_seconds']+rows[m]['initialization_seconds'] for m in METHODS)+prep['support_seconds']+ptime,
                'requests':sum(rows[m]['stage3_requests']+rows[m]['initialization_requests'] for m in METHODS)+6+pcalls,
                'input_samples':sum(rows[m]['stage3_input_samples']+rows[m]['initialization_input_samples'] for m in METHODS)+6+psamples}
    shared.append(root/'deployment_costs.jsonl',dict(identity,method_costs=costs,shared_signal_cost=prep,shared_preparation=item['preparation'],fixed_proposal_seconds=item.get('fixed_proposal_seconds'),
             fixed_proposal_requests=item.get('fixed_proposal_requests'),fixed_proposal_input_samples=item.get('fixed_proposal_input_samples'),
             external_evaluator_seconds=item.get('evaluator_seconds'),P_scoring_requests=pcalls,P_scoring_input_samples=psamples,physical_stage3_runs=9,
             cost_scope='method costs exclude common blind inference/training/qualification/proposals; full shared components separately recorded'))
    # External evaluator is first loaded after sealed decisions for this task.
    evaluator,_=load_evaluator_artifact(artifact)
    for m,row in rows.items():
     if row['status']=='valid':
      t=time.perf_counter();_,score=score_solution(evaluator,np.asarray(row['X']),0)
      row.update(pooled_error=score,truth_error=abs(float(synthetic_truth(np.array(row['X'])[None,:])[0])),external_scoring_seconds=time.perf_counter()-t)
     shared.append(root/'results.jsonl',row)
    print(f'{stage}: {sc} seed {seed} complete',flush=True)
  manifest['status']='complete'
 except Exception as exc:manifest.update(status='interrupted',reason=repr(exc));raise
 finally:shared.dump(root/'manifest.json',manifest)
 verify(regpath,stage);analyze(regpath,stage)

def verify(regpath,stage):
 reg=json.load(open(regpath));conf=reg[stage];root=Path(conf['results']);manifest=json.load(open(root/'manifest.json'));rows=lines(root/'results.jsonl');errors=[];maxima={'alpha':0.,'scores':0.,'pooled':0.,'predictions':0.};valid=0
 coverage=audit_coverage(rows,list(itertools.product(SCENARIOS,conf['seeds'],METHODS+['C','P'])),['scenario','seed','method'])
 if not coverage['complete'] or manifest['status']!='complete':errors.append('coverage/execution incomplete')
 for name,h in manifest['source_sha256'].items():
  if shared.digest(name)!=h or shared.digest(root/'source_snapshot'/name)!=h:errors.append('source hash '+name)
 idx={(r['scenario'],r['seed'],r['method']):r for r in rows};costs={(r['scenario'],r['seed']):r for r in lines(root/'deployment_costs.jsonl')} if (root/'deployment_costs.jsonl').exists() else {}
 for common in lines(root/'shared.jsonl'):
  sc,seed=common['scenario'],common['seed']
  if common['eligibility']!='eligible':
   if any(idx[sc,seed,m]['status']!='qualification_failure' for m in METHODS+['C','P']):errors.append('common failure lost')
   continue
  for name,h in common['artifact_sha256'].items():
   if shared.digest(Path(common['artifact'])/name)!=h:errors.append('artifact changed')
  base,_,artifact=shared.load_base(conf['source'],sc,seed);p,peer,_=evidence(base)
  maxima['predictions']=max(maxima['predictions'],float(np.max(np.abs(p-common['blind_predictions']))))
  decisionpath=root/'decisions'/f'{sc}_{seed}.json'
  if shared.digest(decisionpath)!=manifest['decisions'][str(decisionpath)]:errors.append('decision hash changed')
  d=json.load(open(decisionpath));ce=select_c(p,base.test_y,peer,0)
  if ce!=d['C']:errors.append('C discrete/evidence mismatch')
  init=route(peer);mse=((p-base.test_y[:,None])**2).mean(axis=0)
  full,_=weights_from_predictions(p,base.test_y,peer,0)
  for m in METHODS:
   w,_=init(m,base.trusted_agents,mse,base.test_X,base.test_y,base.I_list,0,cached_blind_predictions=p)
   maxima['alpha']=max(maxima['alpha'],float(np.max(np.abs(w-idx[sc,seed,m]['alpha']))),float(np.max(np.abs(w-full[m]))))
  pred={m:[float(a.api_predict(np.array(idx[sc,seed,m]['X']))[0]) for a in base.trusted_agents] for m in METHODS if idx[sc,seed,m]['status']=='valid'}
  pe=select_p({m:idx[sc,seed,m] for m in METHODS},pred,idx[sc,seed,'B_stage0']['alpha'],0)
  if pe['selected_method']!=d['P']['selected_method']:errors.append('P discrete mismatch')
  for m,v in pe['scores'].items():maxima['scores']=max(maxima['scores'],abs(v-d['P']['scores'][m]))
  c=costs[sc,seed]
  if c['method_costs']['C']['stage3_runs']!=1 or c['method_costs']['P']['stage3_runs']!=8 or c['physical_stage3_runs']!=9:errors.append('solve costs missing')
  for m in METHODS+['C','P']:
   row=idx[sc,seed,m]
   if row['additional_labels']!=0:errors.append('new labels')
   if row['status']=='valid':
    x=np.asarray(row['X']);w=np.asarray(row['alpha'])
    if x.shape!=(2,) or not np.isfinite(x).all() or base.box_violation(x)>1e-8 or w.min()<0 or abs(w.sum()-1)>1e-8:errors.append('invalid delivery')
    evaluator,_=load_evaluator_artifact(artifact);_,score=score_solution(evaluator,x,0)
    maxima['pooled']=max(maxima['pooled'],abs(score-row['pooled_error']));valid+=1
    sealed=d['deliveries'][m]
    if row['X']!=sealed['X'] or row['alpha']!=sealed['alpha'] or sealed.get('pooled_error') is not None:errors.append('unsealed decision')
 if max(maxima.values())>1e-7:errors.append('reload numerical mismatch')
 result={'passed':not errors,'errors':errors,'coverage':coverage,'saved':len(rows),'valid':valid,'max_reload_difference':maxima,'decision_before_external_scoring':True,'full_training_replay':'separate replay verification'}
 shared.dump(root/'verification.json',result)
 if errors:raise RuntimeError(str(errors))
 print(json.dumps(result),flush=True)
 return result

def analyze(regpath,stage):
 reg=json.load(open(regpath));conf=reg[stage];root=Path(conf['results']);rows=lines(root/'results.jsonl');idx={(r['scenario'],r['seed'],r['method']):r for r in rows};groups=[];contrasts=[]
 for m in METHODS+['C','P']:
  vals=[np.mean([idx[sc,seed,m]['pooled_error'] for sc in SCENARIOS]) for seed in conf['seeds'] if all(idx[sc,seed,m]['status']=='valid' for sc in SCENARIOS)]
  groups.append({'method':m,'display':DISPLAY.get(m,m),'complete_seed_n':len(vals),'mean':float(np.mean(vals)) if vals else None,'median':float(np.median(vals)) if vals else None})
 for a,b in reg['contrasts']:
  vals=[np.mean([idx[sc,seed,a]['pooled_error']-idx[sc,seed,b]['pooled_error'] for sc in SCENARIOS]) for seed in conf['seeds'] if all(idx[sc,seed,a]['status']=='valid' and idx[sc,seed,b]['status']=='valid' for sc in SCENARIOS)]
  contrasts.append({'method':a,'baseline':b,'n':len(vals),'mean_delta':float(np.mean(vals)) if vals else None,'seed_deltas':vals,'inference':'pilot descriptive only'})
 ev=lines(root/'selector_evidence.jsonl') if (root/'selector_evidence.jsonl').exists() else [];cost=lines(root/'deployment_costs.jsonl') if (root/'deployment_costs.jsonl').exists() else [];ratios={}
 for m in ['C','P']:
  ratios[m]={k:float(np.median([c['method_costs'][m][k]/c['method_costs']['ensemble_shrink'][k] for c in cost])) for k in ['seconds','requests','input_samples']} if cost else {}
 scenario_means={sc:{DISPLAY.get(m,m):float(np.mean([idx[sc,s,m]['pooled_error'] for s in conf['seeds'] if idx[sc,s,m]['status']=='valid'])) if any(idx[sc,s,m]['status']=='valid' for s in conf['seeds']) else None for m in METHODS+['C','P']} for sc in SCENARIOS}
 selected={m:dict(Counter(DISPLAY.get(e[m]['selected_method'],e[m]['selected_method']) for e in ev)) for m in ['C','P']}
 result={'stage':stage,'inference':'pilot/smoke; no stable superiority claim','groups':groups,'contrasts':contrasts,'scenario_means':scenario_means,'selected_frequency':selected,
         'statuses':dict(Counter(r['status'] for r in rows)),'median_method_cost_ratios_to_EU':ratios,'P_Q_local_deltas':[e['P']['scores'].get(e['P']['selected_method'],0)-e['P']['scores'].get('B_stage0',0) for e in ev]}
 shared.dump(root/'analysis.json',result)
 v=json.load(open(root/'verification.json'));budgets={m:all(ratios[m].get(k,float('inf'))<=limit for k in ['seconds','requests','input_samples']) for m,limit in [('C',1.5),('P',10)]}
 shared.dump(root/'acceptance.json',{'flow_passed':v['passed'],'cost_budget_passed':budgets,'quality_status':'not established by pilot','full_training_replay_passed':None})
 print(json.dumps({'groups':groups,'selected':selected,'ratios':ratios}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('registration');a.add_argument('stage');a.add_argument('--verify',action='store_true');x=a.parse_args()
 if x.verify:verify(x.registration,x.stage);analyze(x.registration,x.stage)
 else:main_run(x.registration,x.stage)
