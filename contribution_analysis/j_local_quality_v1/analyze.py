"""Read-only diagnostics of frozen J endpoints; no model imports or calls."""
import hashlib,itertools,json,shutil,time,signal
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis'
SOURCE=CA/'j_loo_start_stability_experiment_v1_20261027'
ROOT=CA/'j_local_quality_analysis_v1_20261027'
PLAN=WORK/'docs/j_local_quality_analysis_plan_20261027.md'
EPS=1e-4
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def sig(x):return int(x>EPS)-int(x< -EPS)
def avg(v):return float(np.mean(v)) if v else None
def near(a,b):
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
    return float(np.max(abs(np.array(a)-b))) if np.array(a).size else 0.
def rho(a,b):
    if len(a)<3 or np.ptp(a)<=EPS or np.ptp(b)<=EPS:return None
    return float(spearmanr(a,b).statistic)
def boot(v):
    if not v:return dict(mean=None,ci=None,n=0)
    a=np.array(v);indices=np.random.default_rng(20261027).integers(0,len(a),(10000,len(a)))
    return dict(mean=float(a.mean()),ci=np.percentile(a[indices].mean(axis=1),[2.5,97.5]).tolist(),n=len(a))
def quality(rows):
    nz=[r for r in rows if sig(r['j'])];both=[r for r in nz if sig(r['j_E'])];large=[r for r in rows if abs(r['j'])>.001]
    rs=[];rounds=sorted(set(r['t'] for r in rows))
    for t in rounds:
        rr=[r for r in rows if r['t']==t];v=rho([r['j'] for r in rr],[r['j_E'] for r in rr])
        if v is not None:rs.append(v)
    def rate(selected,fn):return avg([fn(r) for r in selected])
    pos=[r for r in nz if sig(r['j'])==1];neg=[r for r in nz if sig(r['j'])==-1];zero=[r for r in rows if not sig(r['j'])]
    return dict(n=len(rows),nonzero=len(nz),large=len(large),rank_rounds=len(rs),rounds=len(rounds),
        sign=rate(nz,lambda r:sig(r['j'])==sig(r['j_E'])),reverse=rate(nz,lambda r:sig(r['j'])==-sig(r['j_E'])),
        external_zero=rate(nz,lambda r:sig(r['j_E'])==0),both_nonzero_sign=rate(both,lambda r:sig(r['j'])==sig(r['j_E'])),
        positive_sign=rate(pos,lambda r:sig(r['j_E'])==1),negative_sign=rate(neg,lambda r:sig(r['j_E'])==-1),
        zero_external_nonzero=rate(zero,lambda r:sig(r['j_E'])!=0),
        severe_reverse=rate(large,lambda r:abs(r['j_E'])>.001 and r['j']*r['j_E']<0),
        rank=avg(rs),rank_coverage=len(rs)/len(rounds) if rounds else None)
QFIELDS=['sign','reverse','external_zero','both_nonzero_sign','positive_sign','negative_sign','zero_external_nonzero','severe_reverse','rank','rank_coverage']
AFIELDS=['P','N','J','G','cancellation','top1','top3','positive_fraction','negative_fraction','zero_fraction']
def main():
    tic=time.perf_counter();ROOT.mkdir(exist_ok=False);(ROOT/'source_snapshot').mkdir()
    receipt=read(SOURCE/'completion_receipt.json');assert receipt['complete'] and receipt['independent_verification_accepted']
    for p,h in receipt['hashes'].items():assert sha(SOURCE/p)==h,('source_hash',p)
    assert sha(receipt['report_path'])==receipt['report_sha256']
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    shutil.copy2(PLAN,ROOT/PLAN.name)
    inputs={str(SOURCE/'completion_receipt.json'):sha(SOURCE/'completion_receipt.json'),str(PLAN):sha(PLAN)}
    inputs.update({str(SOURCE/p):h for p,h in receipt['hashes'].items()});save(ROOT/'inputs.json',inputs)
    save(ROOT/'source_lock.json',{p.name:sha(p) for p in (ROOT/'source_snapshot').glob('*.py')})
    save(ROOT/'registration.json',dict(old_data_exploration=True,independent_confirmation=False,new_participant_inferences=0,new_pooled_inferences=0,epsilon=EPS,severe_threshold=.001,
        existing_results_already_known=True,analysis_rules_locked_before_execution=True))
    pools={r['key']:{key(v['x']):v['value'] for v in r['points']} for r in read(SOURCE/'pooled_points.json')}
    oldext={(r['key'],r['start']):r for r in read(SOURCE/'external_rows.json')};rows=[];agents=[];error=0.;caseinfo=[]
    for c in read(SOURCE/'cases.json')['development']:
        for si in range(5):
            path=SOURCE/'development'/(c['key']+f'_s{si}_J.json');r=read(path);raw={str(v['i'])+':'+key(v['x']):v['y'] for v in r['raw']};weights=np.array(c['alphas']);weights/=weights.sum()
            def L(x):return float(np.dot(weights,[abs(raw[str(i)+':'+key(x)]) for i in range(len(c['ids']))]))
            current=[]
            for b in r['branches']:
                j=L(b['x_minus'])-L(b['x_full']);error=max(error,near(j,b['score']))
                je=pools[c['key']][key(b['x_minus'])]-pools[c['key']][key(b['x_full'])]
                f=r['baseline_trace']['steps'][b['t']]
                row=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],start=si,i=b['i'],agent_id=b['agent_id'],t=b['t'],j=j,j_E=je,
                    full_accepted=f['accepted'],minus_accepted=b['trace']['steps'][0]['accepted'],mode=f['pre']['current_method'])
                current.append(row);rows.append(row)
            for i,aid in enumerate(c['ids']):
                rr=[v for v in current if v['i']==i];vs=[v['j'] for v in rr];P=sum(max(v,0.) for v in vs);N=sum(min(v,0.) for v in vs);J=sum(vs);G=P-N;ordered=sorted([abs(v) for v in vs],reverse=True)
                error=max(error,near(J,r['values'][i]),near(P+N,J),near(sum(v['j_E'] for v in rr),oldext[c['key'],si]['J_E'][i]))
                agents.append(dict(key=c['key'],scenario=c['scenario'],start=si,i=i,agent_id=aid,rounds=len(rr),P=P,N=N,J=J,G=G,
                    cancellation=None if G<=EPS else 1-abs(J)/G,top1=None if G<=EPS else ordered[0]/G,top3=None if G<=EPS else sum(ordered[:3])/G,
                    positive_rounds=sum(sig(v)==1 for v in vs),negative_rounds=sum(sig(v)==-1 for v in vs),zero_rounds=sum(sig(v)==0 for v in vs),
                    positive_fraction=avg([sig(v)==1 for v in vs]),negative_fraction=avg([sig(v)==-1 for v in vs]),zero_fraction=avg([sig(v)==0 for v in vs])))
            caseinfo.append(dict(key=c['key'],scenario=c['scenario'],start=si,**quality(current)))
    save(ROOT/'round_rows.json',rows);save(ROOT/'agent_decomposition.json',agents);save(ROOT/'start_quality.json',caseinfo)
    team=[];strata=[];matrix=[]
    for c in read(SOURCE/'cases.json')['development']:
        rr=[r for r in caseinfo if r['key']==c['key']];out=dict(key=c['key'],scenario=c['scenario'])
        for f in QFIELDS:out[f]=avg([r[f] for r in rr if r[f] is not None])
        out['valid_sign_starts']=sum(r['nonzero']>0 for r in rr);out['nonzero']=sum(r['nonzero'] for r in rr);out['large']=sum(r['large'] for r in rr)
        for f in AFIELDS:
            values=[]
            for si in range(5):
                v=[a[f] for a in agents if a['key']==c['key'] and a['start']==si and a[f] is not None]
                if v:values.append(avg(v))
            out['agent_'+f]=avg(values)
        team.append(out)
        for accepted in [True,False]:
            v=[]
            for si in range(5):
                selected=[r for r in rows if r['key']==c['key'] and r['start']==si and r['full_accepted']==accepted]
                if selected:v.append(quality(selected))
            strata.append(dict(key=c['key'],scenario=c['scenario'],full_accepted=accepted,**{f:avg([r[f] for r in v if r[f] is not None]) for f in QFIELDS}))
        for a,b in itertools.product([-1,0,1],repeat=2):
            v=[];count=0
            for si in range(5):
                rr=[r for r in rows if r['key']==c['key'] and r['start']==si];n=sum(sig(r['j'])==a and sig(r['j_E'])==b for r in rr);v.append(n/len(rr));count+=n
            matrix.append(dict(key=c['key'],scenario=c['scenario'],internal=a,external=b,count=count,fraction=avg(v)))
    save(ROOT/'team_metrics.json',team);save(ROOT/'team_acceptance_strata.json',strata);save(ROOT/'sign_matrix.json',matrix)
    summaries=[];startsummary=[];stratasummary=[];examples=[]
    for sc in ['clean','controlled_poison']:
        tt=[r for r in team if r['scenario']==sc]
        metrics={f:boot([r[f] for r in tt if r[f] is not None]) for f in QFIELDS+['agent_'+f for f in AFIELDS]}
        cov=metrics['sign']['n']>=8 and sum(r['valid_sign_starts']>=4 for r in tt)>=8
        enoughlarge=metrics['severe_reverse']['n']>=8 and sum(r['large'] for r in tt)>=20
        signpass=cov and metrics['sign']['mean']>=.85 and metrics['sign']['ci'][0]>.7
        reversepass=enoughlarge and metrics['severe_reverse']['mean']<=.1 and metrics['severe_reverse']['ci'][1]<.2
        rankpass=metrics['rank']['n']>=8 and metrics['rank_coverage']['mean']>=.8 and metrics['rank']['mean']>=.8 and metrics['rank']['ci'][0]>.5
        rr=[r for r in rows if r['scenario']==sc]
        summaries.append(dict(scenario=sc,branches=len(rr),nonzero=sum(r['nonzero'] for r in tt),large=sum(r['large'] for r in tt),
            severe_count=sum(abs(r['j'])>.001 and abs(r['j_E'])>.001 and r['j']*r['j_E']<0 for r in rr),metrics=metrics,
            coverage_pass=cov,sign_pass=signpass,severe_reverse_pass=reversepass,rank_pass=rankpass,complete_pass=signpass and reversepass and rankpass))
        for si in range(5):
            rr=[r for r in caseinfo if r['scenario']==sc and r['start']==si]
            startsummary.append(dict(scenario=sc,start=si,metrics={f:boot([r[f] for r in rr if r[f] is not None]) for f in QFIELDS}))
        for accepted in [True,False]:
            rr=[r for r in strata if r['scenario']==sc and r['full_accepted']==accepted]
            stratasummary.append(dict(scenario=sc,full_accepted=accepted,metrics={f:boot([r[f] for r in rr if r[f] is not None]) for f in QFIELDS}))
        aa=sorted([r for r in agents if r['scenario']==sc and r['G']>EPS],key=lambda r:(r['key'],r['start'],r['agent_id']))
        for kind,f in [('max_cancellation','cancellation'),('max_gross','G')]:
            chosen=max(aa,key=lambda r:r[f]);examples.append(dict(selection=kind,**chosen))
        opposite=sorted([r for r in rows if r['scenario']==sc and sig(r['j'])*sig(r['j_E'])==-1],key=lambda r:(r['key'],r['start'],r['agent_id'],r['t']))
        if opposite:examples.append(dict(selection='max_opposite',**max(opposite,key=lambda r:min(abs(r['j']),abs(r['j_E'])))))
    save(ROOT/'summary.json',summaries);save(ROOT/'start_summary.json',startsummary);save(ROOT/'acceptance_summary.json',stratasummary);save(ROOT/'examples.json',examples)
    for p,h in inputs.items():assert sha(p)==h
    save(ROOT/'analysis_completion.json',dict(complete=True,branches=len(rows),agent_start_records=len(agents),teams=len(team),starts=len(caseinfo),reconstruction_max_error=error,
        new_participant_model_inferences=0,new_pooled_inferences=0,new_training=0,truth_loaded=False,wall_seconds=time.perf_counter()-tic))
    for s in summaries:print(s['scenario'],{f:s['metrics'][f] for f in ['sign','reverse','external_zero','rank','rank_coverage','severe_reverse','agent_cancellation','agent_top1','agent_top3']},'pass',s['complete_pass'])
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('analysis_wall_cap')));signal.alarm(300)
    try:main()
    except Exception as e:
        if ROOT.exists():save(ROOT/'failure.json',dict(error=repr(e)))
        raise
