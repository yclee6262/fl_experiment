"""Independent read-only audit of saved paired costs and score sums."""
import json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[2];O=W/'contribution_analysis/anchor_ca_model_v_cost_20261104_attempt_01'
def read(p):return json.loads(p.read_text())
def main():
    done=read(O/'completion.json')
    progress=read(O/'progress.json');reg=read(O/'registration.json')
    assert len(reg['jobs'])==300 and len(progress)==done['completed_jobs']
    for job in progress:
        rr={a:read(O/'records'/f"{job['key']}_{job['iterations']}_{a}.json") for a in ['B','A','AV','L']}
        assert all(r['base']==rr['B']['base'] for r in rr.values())
        assert len({r['baseline_digest'] for r in rr.values()})==1
        assert rr['A']['query_digest']==rr['AV']['query_digest']
        assert rr['A']['samples']==rr['AV']['samples'] and rr['AV']['V_new_calls']==0
        for a,r in rr.items():
            assert r['samples']==job['cost'][a]
            assert r['samples']==sum(v['samples'] for v in r['stages'].values())
            assert r['requests']==r['samples']+r['hits']
        for i,s in rr['AV']['scores'].items():
            rows=[r for r in rr['AV']['rows'] if r['i']==int(i)]
            assert s['SA']==sum(r['CA'] for r in rows) and s['SV']==sum(r['V'] for r in rows)
    for p,h in read(O/'source_lock.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    result=dict(passed=True,scope='completed jobs only',jobs=len(progress),planned_jobs=300,score_sums=True,source_lock=True,A_AV_same_queries=True,V_zero_new_calls=True)
    (O/'audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
