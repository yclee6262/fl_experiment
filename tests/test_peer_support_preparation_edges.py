"""Exercise the actual runner's preparation gate without training models."""
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import run_peer_support_batch as runner
from peer_support_alpha import SCENARIOS,METHODS

class Agent:
 def __init__(self,ident):self.agent_id=ident
 def api_predict(self,x):return np.asarray(x).sum(axis=-1)+self.agent_id*.001

def execute(tmp_path,monkeypatch,proposals):
 source=tmp_path/'source';artifact=tmp_path/'artifact';artifact.mkdir();output=tmp_path/'results';calls=[]
 for scenario in SCENARIOS:
  d=source/scenario;d.mkdir(parents=True);(d/'shared_costs.jsonl').write_text(json.dumps({'seed':1,'artifact_path':'saved','coalition_ids':[1,2,20],'missing_fixed_ids':[]})+'\n')
 reg=tmp_path/'reg.json';reg.write_text(json.dumps({'source_sha256':{},'batches':[{'seeds':[1],'source':str(source),'results':str(output)}]}))
 server=SimpleNamespace(I_list=np.array(proposals).copy(),trusted_agents=[Agent(i) for i in [1,2,20]],test_X=np.zeros((20,2)),test_y=np.zeros(20),target_T=0)
 monkeypatch.setattr(runner.shared,'load_base',lambda *a:(server,{},artifact))
 monkeypatch.setattr(runner.shared,'initial_alpha',runner.shared.initial_alpha)
 def solve(server,method,*args):
  calls.append(method)
  return {'status':'valid','X':[0,0],'alpha':[1/3]*3,'initialization_seconds':0,'initialization_requests':0,'initialization_input_samples':0,'stage3_seconds':0,'stage3_requests':0,'stage3_input_samples':0}
 monkeypatch.setattr(runner.shared,'solve',solve);monkeypatch.setattr(runner,'load_evaluator_artifact',lambda *a:(None,None));monkeypatch.setattr(runner,'score_solution',lambda *a:(0,0));runner.run(str(reg),0)
 return output,calls

@pytest.mark.parametrize('proposals',[[[1+2e-8,0],[0,0],[0,0]],[[float('nan'),0],[0,0],[0,0]],[[0,0],[0,0]]])
def test_invalid_common_proposal_never_runs_a_method(tmp_path,monkeypatch,proposals):
 root,calls=execute(tmp_path,monkeypatch,proposals);rows=list(map(json.loads,(root/'results.jsonl').read_text().splitlines()))
 assert not calls and len(rows)==3*len(METHODS)
 assert all(r['status']=='preparation_failure' and r['X'] is None and r['pooled_error'] is None for r in rows)

@pytest.mark.parametrize('proposals',[[[-1,1],[1,-1],[0,0]],[[1+5e-9,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]])
def test_boundary_or_duplicate_candidates_are_preserved(tmp_path,monkeypatch,proposals):
 root,calls=execute(tmp_path,monkeypatch,proposals);saved=list(map(json.loads,(root/'peer_evidence.jsonl').read_text().splitlines()))
 assert len(calls)==3*len(METHODS) and len(saved)==3
 assert all(np.array_equal(r['proposals'],proposals) and r['agent_ids']==[1,2,20] for r in saved)
 assert all(np.asarray(r['prediction_matrix']).shape==(3,3) for r in saved)
