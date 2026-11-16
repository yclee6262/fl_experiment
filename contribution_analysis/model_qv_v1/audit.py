"""Independent saved-record checks; does not invoke model or solver."""
import json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[2];CA=W/'contribution_analysis'
O=CA/'model_qv_experiment_20261104_attempt_02'
def read(p):return json.loads(p.read_text())
reg=read(O/'registration.json');counts=0
for c in reg['cases']:
    records={a:read(O/'records'/f"{c['key']}_{a}.json") for a in ['base','AV','QV','LOO']}
    for a,r in records.items():
        assert r['base']==records['base']['base']
        assert sum(z['samples'] for z in r['stages'].values())==r['samples']
    q=records['QV']; assert q['stages']['V']['samples']==0
    for r in q['rows']:
        assert abs(r['Q']-r['U']/(10+r['U'])*r['Cdir'])<1e-14
    if c['phase']=='historical':
        old=read(CA/'model_pool_observed_vd_cost_20261103_attempt_07/records'/f"{c['key']}_loo.json")
        assert records['LOO']['samples']==old['samples']
        assert all(r['result']==z['result'] for r,z in zip(records['LOO']['removals'],old['rows']))
    counts+=1
for path,h in read(O/'source_lock.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
with (O/'audit.json').open('w') as f:json.dump(dict(passed=True,teams=counts,physical_cost_stage_sums=True,
    V_zero_calls=True,Q_formula=True,historical_LOO_exact=True,source_lock=True),f,indent=2)
print('Independent saved-record audit passed:',counts,'teams')
