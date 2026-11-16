"""Read saved costs only. Unknown is None; shared work is charged once."""
import math
from peer_support_alpha import METHODS

def known_sum(values):
 values=list(values)
 if any(v is None or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):return None
 return sum(values)

def case_costs(record):
 source=record['preparation'];prep=source.get('preparation',{});eligible=record['eligibility']=='eligible'
 if prep:
  seconds=known_sum([prep.get(k) for k in ['training_seconds','qualification_seconds','proposal_seconds']]+[source.get('fixed_proposal_seconds',0) if source.get('missing_fixed_ids') else source.get('fixed_proposal_seconds')])
  requests=known_sum([prep.get('qualification_requests'),prep.get('proposal_requests'),source.get('fixed_proposal_requests',0) if source.get('missing_fixed_ids') else source.get('fixed_proposal_requests')])
  inputs=known_sum([prep.get('qualification_input_samples'),prep.get('proposal_input_samples'),source.get('fixed_proposal_input_samples',0) if source.get('missing_fixed_ids') else source.get('fixed_proposal_input_samples')])
 else:seconds=requests=inputs=None
 def stage(name):
  if name in record:return record[name]
  if record['eligibility'] in ['qualification_failure','invalid_proposal']:return {'seconds':0,'requests':0,'input_samples':0,'status':'not_run'}
  return {'seconds':None,'requests':None,'input_samples':None,'status':'not_saved'}
 return {'scenario':record['scenario'],'seed':record['seed'],'eligibility':record['eligibility'],
  'common_application_preparation':{'seconds':seconds,'requests':requests,'input_samples':inputs,'scope':'training, qualification, source proposal plus fixed proposal'},
  'support_once':stage('support_cost'),'blind_reload_research':stage('blind_cost'),'diagonal_research':stage('diagonal_diagnostic_cost'),
  'evaluator_training_research_seconds':source.get('evaluator_seconds',0) if record['eligibility']=='qualification_failure' else source.get('evaluator_seconds')}

def summarize_costs(rows,shared_records):
 shared={(r['scenario'],r['seed']):case_costs(r) for r in shared_records}
 metrics=[]
 for method in METHODS:
  selected=[r for r in rows if r['method']==method];attempted=[r for r in selected if r['status'] not in ['qualification_failure','preparation_failure']];samples=[]
  for r in selected:
   key=(r['scenario'],r['seed']);common=shared[key]['common_application_preparation']
   if r in attempted:
    s=r['support_independent_cost'];specific={f:known_sum([r.get('initialization_'+f),r.get('stage3_'+f),s.get(f)]) for f in ['seconds','requests','input_samples']}
   elif r['status']=='preparation_failure' and method in ['peer_support_shrink','target_peer_shrink']:
    # The attempted shared support query may have failed before optimization.
    specific={f:shared[key]['support_once'].get(f) for f in ['seconds','requests','input_samples']}
   else:specific={f:0 for f in ['seconds','requests','input_samples']}
   samples.append({'scenario':r['scenario'],'seed':r['seed'],'status':r['status'],'method_specific':specific,'independent_application_total':{f:known_sum([common[f],specific[f]]) for f in specific},'external_scoring_research_seconds':r.get('external_scoring_seconds',0) if r['status']!='valid' else r.get('external_scoring_seconds')})
  metric={'method':method,'planned_conditions':len(selected),'attempted_optimization_conditions':len(attempted),'all_statuses':{s:sum(r['status']==s for r in selected) for s in set(r['status'] for r in selected)},'per_condition':samples}
  for scope in ['method_specific','independent_application_total']:
   metric[scope+'_aggregate']={f:{'sum':known_sum(x[scope][f] for x in samples),'known_conditions':sum(x[scope][f] is not None for x in samples),'unknown_conditions':sum(x[scope][f] is None for x in samples)} for f in ['seconds','requests','input_samples']}
  metrics.append(metric)
 attempted=[r for r in rows if r['status'] not in ['qualification_failure','preparation_failure']]
 batch={}
 for f in ['seconds','requests','input_samples']:
  # Cached S/QS evidence contributes one support preparation per case.
  specific=known_sum(v for r in attempted for v in [r.get('initialization_'+f),r.get('stage3_'+f)])
  common=known_sum(r['common_application_preparation'][f] for r in shared.values())
  support=known_sum(r['support_once'].get(f) for r in shared.values())
  batch[f]={'application_total':known_sum([specific,common,support]),'method_init_and_stage3':specific,'common_preparation_once':common,'support_once':support}
 return {'method_independent_execution':metrics,'batch_shared_execution':batch,'common_once_per_case':list(shared.values()),
  'scope':'Application API calls and input samples are separate from direct local training/inversion operations. Direct operations have saved time but no inferred API count. Source baseline cache-generation runs, evaluator training, reloads, diagonal and external scoring are research overhead, reported separately.',
  'unknown_policy':'An aggregate with any unknown component is None; failures do not silently disappear from method cost aggregation.'}
