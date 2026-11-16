from peer_support_costs import summarize_costs,known_sum
from peer_support_alpha import METHODS

def records(fail=False,unknown=False):
 preparation={'preparation':{'training_seconds':1,'qualification_seconds':2,'proposal_seconds':3,'qualification_requests':4,'proposal_requests':0,'qualification_input_samples':10,'proposal_input_samples':0},'fixed_proposal_seconds':1,'fixed_proposal_requests':0,'fixed_proposal_input_samples':0,'evaluator_seconds':2}
 if unknown:preparation.pop('preparation')
 shared=[{'scenario':'x','seed':1,'preparation':preparation,'eligibility':'eligible','support_cost':{'requests':6,'input_samples':6,'seconds':.5},'blind_cost':{'requests':3,'input_samples':60,'seconds':.1},'diagonal_diagnostic_cost':{'requests':3,'input_samples':3,'seconds':.1}}]
 rows=[]
 for m in METHODS:
  support={'requests':6,'input_samples':6,'seconds':.5} if m in ['peer_support_shrink','target_peer_shrink'] else {'requests':0,'input_samples':0,'seconds':0}
  rows.append({'scenario':'x','seed':1,'method':m,'status':'solver_failure' if fail else 'valid','initialization_requests':0,'initialization_input_samples':0,'initialization_seconds':.1,'stage3_requests':2,'stage3_input_samples':2,'stage3_seconds':.2,'support_independent_cost':support,'external_scoring_seconds':.01})
 return rows,shared

def test_shared_support_once_and_independent_cost():
 rows,common=records();r=summarize_costs(rows,common)
 assert r['batch_shared_execution']['requests']['application_total']==4+6+6*2
 s=next(x for x in r['method_independent_execution'] if x['method']=='peer_support_shrink')
 assert s['independent_application_total_aggregate']['requests']['sum']==4+6+2

def test_failed_solver_keeps_consumed_cost():
 rows,common=records(fail=True);r=summarize_costs(rows,common)
 assert r['batch_shared_execution']['requests']['method_init_and_stage3']==12
 assert all(x['attempted_optimization_conditions']==1 for x in r['method_independent_execution'])
def test_unknown_preparation_not_zero():
 rows,common=records(unknown=True);r=summarize_costs(rows,common)
 assert r['batch_shared_execution']['requests']['application_total'] is None
 assert all(x['independent_application_total_aggregate']['requests']['unknown_conditions']==1 for x in r['method_independent_execution'])
def test_research_overhead_excluded_from_application_requests():
 rows,common=records();r=summarize_costs(rows,common)
 assert r['common_once_per_case'][0]['blind_reload_research']['requests']==3
 assert r['batch_shared_execution']['requests']['application_total']==22
def test_no_nan_as_known_cost():
 assert known_sum([1,float('nan')]) is None
 assert known_sum([1,None]) is None

def test_failed_support_query_cost_preserved_before_solver():
 rows,common=records();common[0]['eligibility']='invalid_peer'
 for r in rows:r['status']='preparation_failure'
 result=summarize_costs(rows,common)
 s=next(r for r in result['method_independent_execution'] if r['method']=='peer_support_shrink')
 assert s['method_specific_aggregate']['requests']['sum']==6
 assert result['batch_shared_execution']['requests']['support_once']==6
 assert result['batch_shared_execution']['requests']['method_init_and_stage3']==0
