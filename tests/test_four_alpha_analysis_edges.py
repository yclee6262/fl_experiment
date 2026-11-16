import json,itertools
import numpy as np
import pytest
from pathlib import Path
from analyze_four_alpha import indexed_rows,statistics,valid
from four_alpha_initialization import METHODS,SCENARIOS

def fake_registration(tmp_path):
 batches=[]
 for i in range(3):
  root=tmp_path/str(i);root.mkdir();seeds=[i*2,i*2+1];batches.append(dict(results=str(root),seeds=seeds))
  (root/'manifest.json').write_text(json.dumps(dict(status='complete')))
  (root/'verification.json').write_text(json.dumps(dict(passed=True)))
  rows=[dict(scenario=s,seed=k,method=m,status='valid',X=[0.,0.],pooled_error=.1,truth_error=.1) for s,k,m in itertools.product(SCENARIOS,seeds,METHODS)]
  (root/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
  (root/'shared.jsonl').write_text('\n'.join(json.dumps(dict(scenario=s,seed=k,eligibility='eligible')) for s,k in itertools.product(SCENARIOS,seeds))+'\n')
 return dict(batches=batches)

def test_two_methods_missing_same_case_is_not_invisible(tmp_path):
 reg=fake_registration(tmp_path);p=Path(reg['batches'][0]['results'])/'results.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
 rows=[r for r in rows if not (r['seed']==0 and r['scenario']==SCENARIOS[0] and r['method'] in METHODS[:2])];p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 with pytest.raises(ValueError,match='missing/duplicate'):indexed_rows(reg)

def test_duplicate_key_rejected(tmp_path):
 reg=fake_registration(tmp_path);p=Path(reg['batches'][0]['results'])/'results.jsonl';rows=p.read_text().splitlines();p.write_text('\n'.join(rows+[rows[0]])+'\n')
 with pytest.raises(ValueError):indexed_rows(reg)

def test_all_methods_fail_preserves_counts_and_undefined_intervals(tmp_path):
 reg=fake_registration(tmp_path);index,elig=indexed_rows(reg)
 for r in index.values():r.update(status='solver_failure',X=None,pooled_error=None,truth_error=None)
 groups,comparisons=statistics(reg,index,elig)
 assert all(g['valid_n']==0 and g['method_failure_n']==6 and g['mean'] is None for g in groups)
 assert all(c['n']==0 and c['CI95']==(None,None) and not c['stable_advantage'] for c in comparisons)

@pytest.mark.parametrize('error',[None,np.nan,np.inf])
def test_nonfinite_quality_is_not_valid(error):
 assert not valid(dict(status='valid',X=[0.,0.],pooled_error=error))
