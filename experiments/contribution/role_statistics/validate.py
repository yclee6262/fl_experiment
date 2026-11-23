"""Independent participant-to-team rank and allocation audit; no solver calls."""
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr,kendalltau


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shares(x,tol):
    w=np.where(x>tol,x,0.);s=w/w.sum() if w.sum() else w
    return np.r_[s,1-s.sum()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--historical',type=Path,required=True)
    a=p.parse_args();out=a.source.parent/'replay_acceptance.json'
    if out.exists():raise FileExistsError(out)
    filenames=['participant_metrics.json','rank_metrics.json','judge_comparisons.json','interactions.json','payments.json','analysis.json','pooled_values.json']
    for f in filenames:assert read(a.source/f)==read(a.historical/f),f
    locks=read(a.source/'source_lock.json')
    for f,h in locks.items():assert sha(Path(f))==h,f
    pp=read(a.source/'participant_metrics.json');teams=defaultdict(list)
    assert len({(x['key'],x['id']) for x in pp})==len(pp)==80
    for x in pp:teams[x['key']].append(x)
    assert len(teams)==20
    rankrows=read(a.source/'rank_metrics.json')
    for r in rankrows:
        sub=sorted(teams[r['key']],key=lambda x:x['i']);tol=r['tol']
        x=np.array([v['scores'][r['judge']][r['score']] for v in sub]);y=np.array([v['D'][r['reference']] for v in sub])
        valid=len(x)>=3 and np.ptp(x)>tol and np.ptp(y)>tol
        for field,func in [('spearman',spearmanr),('kendall',kendalltau)]:
            if valid:assert abs(func(x,y).statistic-r[field])<1e-12
            else:assert r[field] is None
        sx=np.where(abs(x)>tol,np.sign(x),0);sy=np.where(abs(y)>tol,np.sign(y),0)
        assert abs(np.mean(sx==sy)-r['sign_agreement'])<1e-12
        for k in [1,2]:
            if not valid:assert r[f'top{k}'] is None;continue
            ix=set(np.flatnonzero(x>=np.sort(x)[-min(k,len(x))]-tol));iy=set(np.flatnonzero(y>=np.sort(y)[-min(k,len(y))]-tol))
            assert abs(len(ix&iy)/len(ix|iy)-r[f'top{k}'])<1e-12
    for r in read(a.source/'judge_comparisons.json'):
        sub=teams[r['key']];x=np.array([v['scores']['F'][r['role']] for v in sub]);y=np.array([v['scores'][r['other']][r['role']] for v in sub])
        tol=np.array([v['tauA'] for v in sub]) if r['role']=='A' else 1e-10
        assert sum(x>1e-10)==r['positive_F']
        assert sum((x>1e-10)&(y< -1e-10))==r['positive_F_other_negative']
        assert abs(np.abs(shares(x,tol)-shares(y,tol)).sum()/2-r['TV'])<1e-12
    for key,sub in teams.items():
        allocation=read(a.source/'records'/f'{key}.json')['allocation']
        for field,reserve,budget in [('basic','basic_reserved',100*len(sub)),('anchor','anchor_reserved',500),('model','model_reserved',500)]:
            assert sum(v[field] for v in sub)+allocation[reserve]==budget
    result=dict(passed=True,teams=20,owners=80,rank_rows_recomputed=len(rankrows),judge_rows_recomputed=160,
                source_locks=len(locks),historical_json_matches=len(filenames),budget_conserved=True,
                validation_model_queries=0,validation_solver_calls=0,validator_sha256=sha(Path(__file__)),
                replay_hashes={f:sha(a.source/f) for f in filenames})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
