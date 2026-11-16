import pytest
from analyze_peer_support import valid,eligible_seeds
from peer_support_alpha import SCENARIOS

def test_empty_nan_and_bad_delivery():
 for x,e in [(None,0),([0,float('nan')],0),([0,2],0),([0,0],float('nan'))]:assert not valid({'status':'valid','X':x,'pooled_error':e})
 assert valid({'status':'valid','X':[1+5e-9,-1],'pooled_error':0})
 assert not valid({'status':'valid','X':[1+2e-8,-1],'pooled_error':0})
def test_joint_failure_not_method_failure():
 reg={'batches':[{'seeds':[1,2]}]};elig={(s,k):True for s in SCENARIOS for k in [1,2]};elig[SCENARIOS[1],2]=False
 assert eligible_seeds(reg,elig,'equal_scenario_mean')==[1]
 assert eligible_seeds(reg,elig,SCENARIOS[0])==[1,2]

def fixture_batch(tmp_path,duplicate=False,missing=False,all_failed=False):
 import json,itertools
 from peer_support_alpha import METHODS
 directory=tmp_path/'batch';directory.mkdir()
 records=[{'scenario':s,'seed':1,'method':m,'status':'solver_failure' if all_failed else 'valid','X':None if all_failed else [0,0],'pooled_error':None if all_failed else .1} for s,m in itertools.product(SCENARIOS,METHODS)]
 if duplicate:records.append(records[0])
 if missing:records=[r for r in records if r['scenario']!=SCENARIOS[0]]
 (directory/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
 (directory/'shared.jsonl').write_text('\n'.join(json.dumps({'scenario':s,'seed':1,'eligibility':'eligible'}) for s in SCENARIOS))
 (directory/'manifest.json').write_text(json.dumps({'status':'complete'}));(directory/'verification.json').write_text(json.dumps({'passed':True}))
 return {'batches':[{'results':str(directory),'seeds':[1]}]}

def test_both_methods_missing_same_seed_rejected(tmp_path):
 from analyze_peer_support import indexed_rows
 with pytest.raises(ValueError):indexed_rows(fixture_batch(tmp_path,missing=True))
def test_duplicate_rejected(tmp_path):
 from analyze_peer_support import indexed_rows
 with pytest.raises(ValueError):indexed_rows(fixture_batch(tmp_path,duplicate=True))
def test_all_solver_failures_keep_analysis(tmp_path):
 from analyze_peer_support import indexed_rows,statistics
 reg=fixture_batch(tmp_path,all_failed=True);idx,elig=indexed_rows(reg);groups,comparisons=statistics(reg,idx,elig)
 assert all(g['mean'] is None and g['method_failure_n']==1 for g in groups)
 assert all(c['n']==0 and tuple(c['CI95'])==(None,None) and not c['stable_advantage'] for c in comparisons)
