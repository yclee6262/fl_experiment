"""Independent ledger, source-total, reference and statistics audit; no replay imports."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis'
ROOT=CA/'contribution_four_way_experiment_v2_20261022'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tied_rank(v):
    order=sorted(range(len(v)),key=lambda i:v[i]);r=np.zeros(len(v));p=0
    while p<len(v):
        end=p+1
        while end<len(v) and v[order[end]]==v[order[p]]:end+=1
        for i in order[p:end]:r[i]=(p+1+end)/2
        p=end
    return r
def sgn(x):return (1 if x>1e-4 else -1 if x< -1e-4 else 0)
def main():
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h,('input_changed',p)
    for p,h in read(ROOT/'analysis_source_lock.json').items():assert sha(ROOT/'analysis_source_snapshot'/p)==h
    assert read(ROOT/'scores_locked.json')['sha256']==sha(ROOT/'scores_and_costs.json')
    rows=read(ROOT/'comparisons.json');recs=read(ROOT/'scores_and_costs.json');summary=read(ROOT/'summary.json')
    requests=0;physical=0;maxerr=0.;branches=0
    for c in recs:
        key=c['key'];layer=read(CA/'combined_contribution_layer2_experiment_v1_20261021/development'/(key+'.json'))
        anchor=read(CA/'local_anchor_step_experiment_v1_20261020/records'/(key+'.json'))
        baseline=read(CA/'custom_direction_experiment_v2_20261020/baselines'/(key+'.json'))
        ids=layer['case']['ids']
        for scheme in ['S1','S2','S3','S4']:
            raw=read(ROOT/'records'/(key+'_'+scheme+'.json'));seen=set();charge=0;digest=hashlib.sha256();basecount=0
            for e in raw['events']:
                k=e['key'];idx,hx=k.split(':');isbase=e['stage']=='baseline'
                hit=(not isbase and k in seen);assert hit==e['hit']
                if not hit:charge+=1
                seen.add(k)
                if isbase:
                    basecount+=1;digest.update(int(idx).to_bytes(4,'little'));digest.update(bytes.fromhex(hx))
            assert digest.hexdigest()==baseline['query_order_sha256']
            assert basecount==baseline['counter']['samples']==raw['baseline_samples']
            assert charge==raw['samples'] and charge-basecount==raw['extra_samples']
            requests+=len(raw['events']);physical+=charge*4
            assert raw['rows']==c['schemes'][scheme]['rows']
            branches+=len(raw['details'])
        for i,agent in enumerate(ids):
            rr=next(r for r in rows if r['key']==key and r['agent_id']==agent)
            sums=dict(A=sum(r['J'] for r in anchor['comparisons'] if r['agent_id']==agent),
                A2=sum(g['a2'] for g in layer['groups'] if g['i']==i),M=sum(g['m2'] for g in layer['groups'] if g['i']==i),
                J=sum(g['j'] for g in layer['groups'] if g['i']==i))
            sums.update(C=sums['A']+sums['M'],C2=sums['A2']+sums['M'])
            for field,v in sums.items():maxerr=max(maxerr,abs(v-rr[field]));assert abs(v-rr[field])<1e-10
    assert physical==read(ROOT/'replay_completion.json')['physical_model_samples']
    teams=read(ROOT/'team_metrics.json')
    for t in teams:
        rr=[r for r in rows if r['key']==t['key']];x=[r[t['score']] for r in rr];y=[r[t['reference']] for r in rr]
        if t['rho'] is not None:
            xx=tied_rank(x);yy=tied_rank(y);xx-=xx.mean();yy-=yy.mean()
            rank=float(np.dot(xx,yy)/np.sqrt(np.dot(xx,xx)*np.dot(yy,yy)))
            assert abs(rank-spearmanr(x,y).statistic)<1e-12
            maxerr=max(maxerr,abs(rank-t['rho']));assert abs(rank-t['rho'])<1e-12
        decisive=[r for r in rr if sgn(r[t['reference']])]
        if decisive:assert abs(sum(sgn(r[t['score']])==sgn(r[t['reference']]) for r in decisive)/len(decisive)-t['sign'])<1e-12
    for m in summary['metrics']:
        ts=[t for t in teams if all(t[k]==m[k] for k in ['scenario','score','reference'])]
        for field in ['rho','sign','positive_recall','negative_recall','near_zero','normalized_gap']:
            v=[t[field] for t in ts if t[field] is not None];stat=m[field];assert stat['n']==len(v)
            if not v:continue
            rng=np.random.default_rng(20261022);indices=rng.integers(0,len(v),(10000,len(v)))
            means=np.asarray([sum(v[int(i)] for i in pick)/len(v) for pick in indices]);ci=np.quantile(means,[.001,.999])
            maxerr=max(maxerr,float(max(abs(ci-np.asarray(stat['interval'])))));assert np.max(abs(ci-np.asarray(stat['interval'])))<1e-10
    for c in summary['costs']:
        rs=[r for r in recs if r['scenario']==c['scenario']];scheme=c['scheme']
        assert c['total_samples']==sum(r['schemes'][scheme]['samples'] for r in rs)
        assert abs(c['samples_ratio_to_full']-c['total_samples']/sum(r['schemes']['S4']['samples'] for r in rs))<1e-12
        assert c['cost_pass']==(c['samples_ratio_to_full']<=.7)
    for p in summary['paired']:
        first,second=p['comparison'].split('-');ts=[t for t in teams if t['scenario']==p['scenario'] and t['reference']==p['reference']]
        aa={t['key']:t['rho'] for t in ts if t['score']==first};bb={t['key']:t['rho'] for t in ts if t['score']==second}
        values=[aa[k]-bb[k] for k in aa if aa[k] is not None and bb[k] is not None]
        assert len(values)==p['rank_delta']['n']
        assert abs(np.mean(values)-p['rank_delta']['mean'])<1e-12
        rng=np.random.default_rng(20261022);idx=rng.integers(0,len(values),(10000,len(values)))
        means=np.asarray([sum(values[int(i)] for i in pick)/len(values) for pick in idx])
        assert np.max(abs(np.quantile(means,[.001,.999])-p['rank_delta']['interval']))<1e-10
    result=dict(accepted=True,teams=len(recs),participants=len(rows),first_pass_requests_audited=requests,
        endpoint_score_details=branches,max_independent_numeric_error=maxerr,physical_replay_samples=physical,
        failed_run_and_diagnostic_samples=564,total_research_model_samples=physical+564,
        model_training=0,pooled_inference=0,truth_loaded=False,all_source_hashes_unchanged=True,
        independent_data_confirmation=False)
    with (ROOT/'verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
