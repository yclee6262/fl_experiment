"""Fixed endpoint self/other scoring decomposition. No inference calls."""
import hashlib,json,shutil,signal,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis'
SRC=CA/'j_loo_start_stability_experiment_v1_20261027';PRIOR=CA/'j_local_quality_analysis_v1_20261027'
ROOT=CA/'j_self_other_diagnosis_v2_20261027';PLAN=WORK/'docs/j_self_other_diagnosis_plan_20261027.md'
EPS=1e-4;SCORES=['j','own','other','other_mean']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def sign(x):return int(x>EPS)-int(x< -EPS)
def avg(v):return float(np.mean(v)) if v else None
def near(a,b):
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10);return float(np.max(abs(np.asarray(a)-b)))
def rho(a,b):
    if len(a)<3 or np.ptp(a)<=EPS or np.ptp(b)<=EPS:return None
    return float(spearmanr(a,b).statistic)
def boot(v):
    if not v:return dict(mean=None,ci=None,n=0)
    x=np.array(v);ids=np.random.default_rng(20261027).integers(0,len(x),(10000,len(x)))
    return dict(mean=float(x.mean()),ci=np.percentile(x[ids].mean(axis=1),[2.5,97.5]).tolist(),n=len(x))
def metrics(rows):
    opposite=[r for r in rows if sign(r['j'])*sign(r['j_E'])==-1];external=[r for r in rows if sign(r['j_E'])];old=[r for r in rows if sign(r['j'])]
    out=dict(branches=len(rows),opposite=len(opposite),external_nonzero=len(external),old_nonzero=len(old))
    def rate(v,fn):return avg([fn(r) for r in v])
    out.update(self_direction_flip=rate(opposite,lambda r:sign(r['other'])==sign(r['j_E'])),
        peers_still_opposite=rate(opposite,lambda r:sign(r['other'])==sign(r['j'])),peers_nearzero=rate(opposite,lambda r:sign(r['other'])==0),
        own_peer_conflict=rate(rows,lambda r:sign(r['own'])*sign(r['other'])==-1),
        own_abs_share=avg([abs(r['own'])/(abs(r['own'])+abs(r['other'])) for r in rows if abs(r['own'])+abs(r['other'])>EPS]),
        own_gap_share=avg([abs(r['own_gap'])/(abs(r['own_gap'])+abs(r['other_gap'])) for r in rows if abs(r['own_gap'])+abs(r['other_gap'])>EPS]),
        model_disagreement=rate(external,lambda r:len(set(sign(v) for v in r['model_deltas']))>1),
        all_models_external_match=rate(external,lambda r:all(sign(v)==sign(r['j_E']) for v in r['model_deltas'])))
    for score in SCORES:
        out[score+'_sign']=rate(external,lambda r:sign(r[score])==sign(r['j_E']))
        out[score+'_reverse']=rate(external,lambda r:sign(r[score])==-sign(r['j_E']))
        out[score+'_zero']=rate(external,lambda r:sign(r[score])==0)
        out[score+'_oldmask_sign']=rate(old,lambda r:sign(r[score])==sign(r['j_E']))
        values=[];ts=sorted(set(r['t'] for r in rows))
        for t in ts:
            rr=[r for r in rows if r['t']==t];v=rho([r[score] for r in rr],[r['j_E'] for r in rr])
            if v is not None:values.append(v)
        out[score+'_rank']=avg(values);out[score+'_coverage']=len(values)/len(ts)
    out['peer_mean_gain']=None if out['j_sign'] is None else out['other_mean_sign']-out['j_sign']
    return out
def main():
    tic=time.perf_counter();ROOT.mkdir(exist_ok=False);(ROOT/'source_snapshot').mkdir();adjustment=WORK/'docs/j_self_other_numerical_adjustment_20261027.md';inputs={str(PLAN):sha(PLAN),str(adjustment):sha(adjustment)}
    for source in [SRC,PRIOR]:
        receipt=read(source/'completion_receipt.json');assert receipt['complete'] and receipt['independent_verification_accepted']
        inputs[str(source/'completion_receipt.json')]=sha(source/'completion_receipt.json')
        for p,h in receipt['hashes'].items():assert sha(source/p)==h;inputs[str(source/p)]=h
        assert sha(receipt['report_path'])==receipt['report_sha256'];inputs[receipt['report_path']]=receipt['report_sha256']
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    shutil.copy2(PLAN,ROOT/PLAN.name);shutil.copy2(adjustment,ROOT/adjustment.name);save(ROOT/'inputs.json',inputs);save(ROOT/'source_lock.json',{p.name:sha(p) for p in (ROOT/'source_snapshot').glob('*.py')})
    save(ROOT/'registration.json',dict(old_data_exploration=True,new_confirmation=False,prior_results_known=True,score_weights_unchanged=True,source_endpoints_fixed=True,new_inferences=0))
    pools={r['key']:{key(p['x']):p['value'] for p in r['points']} for r in read(SRC/'pooled_points.json')}
    original={(r['key'],r['start'],r['i'],r['t']):r for r in read(PRIOR/'round_rows.json')}
    rows=[];agents=[];models=[];startmetrics=[];err=0.
    for c in read(SRC/'cases.json')['development']:
        weights=np.array(c['alphas']);weights/=weights.sum()
        for si in range(5):
            r=read(SRC/'development'/(c['key']+f'_s{si}_J.json'));raw={(v['i'],key(v['x'])):v['y'] for v in r['raw']};current=[]
            for b in r['branches']:
                i=b['i'];d=[abs(raw[k,key(b['x_minus'])])-abs(raw[k,key(b['x_full'])]) for k in range(len(weights))]
                comp=[float(w*x) for w,x in zip(weights,d)];own=comp[i];other=sum(x for k,x in enumerate(comp) if k!=i);j=float(b['score'])
                je=pools[c['key']][key(b['x_minus'])]-pools[c['key']][key(b['x_full'])];own_gap=float(weights[i]*(d[i]-je));other_gap=sum(float(weights[k]*(d[k]-je)) for k in range(len(weights)) if k!=i)
                old=original[c['key'],si,i,b['t']];err=max(err,near(own+other,j),near(j,b['score']),near(j,old['j']),near(je,old['j_E']),near(own_gap+other_gap,j-je))
                row=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],start=si,i=i,agent_id=c['ids'][i],t=b['t'],j=j,j_E=je,own=own,other=other,
                    other_mean=other/float(1-weights[i]),weight=float(weights[i]),model_deltas=d,model_components=comp,own_gap=own_gap,other_gap=other_gap)
                rows.append(row);current.append(row)
            startmetrics.append(dict(key=c['key'],scenario=c['scenario'],start=si,**metrics(current)))
            for i,aid in enumerate(c['ids']):
                rr=[v for v in current if v['i']==i];a={f:sum(v[f] for v in rr) for f in SCORES+['j_E']};err=max(err,near(a['j'],r['values'][i]),near(a['own']+a['other'],a['j']))
                a['j']=r['values'][i]
                agents.append(dict(key=c['key'],scenario=c['scenario'],start=si,agent_id=aid,**a))
            for k,aid in enumerate(c['ids']):
                selected=[v for v in current if sign(v['j_E'])]
                models.append(dict(key=c['key'],scenario=c['scenario'],start=si,model_id=aid,n=len(selected),
                    external_sign=avg([sign(v['model_deltas'][k])==sign(v['j_E']) for v in selected]),
                    reverse=avg([sign(v['model_deltas'][k])==-sign(v['j_E']) for v in selected]),
                    nearzero=avg([sign(v['model_deltas'][k])==0 for v in selected]),
                    mean_abs_delta_gap=avg([abs(v['model_deltas'][k]-v['j_E']) for v in selected])))
    fields=[f for f in startmetrics[0] if f not in ['key','scenario','start','branches','opposite','external_nonzero','old_nonzero']]
    teams=[]
    for c in read(SRC/'cases.json')['development']:
        rr=[v for v in startmetrics if v['key']==c['key']];out=dict(key=c['key'],scenario=c['scenario'],opposite=sum(v['opposite'] for v in rr),valid_opposite_starts=sum(v['opposite']>0 for v in rr))
        for f in fields:out[f]=avg([v[f] for v in rr if v[f] is not None])
        for f in SCORES:
            rs=[]
            for si in range(5):
                aa=[v for v in agents if v['key']==c['key'] and v['start']==si];v=rho([a[f] for a in aa],[a['j_E'] for a in aa])
                if v is not None:rs.append(v)
            out[f+'_cumulative_rank']=avg(rs)
        teams.append(out)
    fields += [f+'_cumulative_rank' for f in SCORES];summary=[]
    for sc in ['clean','controlled_poison']:
        tt=[t for t in teams if t['scenario']==sc];m={f:boot([t[f] for t in tt if t[f] is not None]) for f in fields}
        enough=m['self_direction_flip']['n']>=8 and sum(t['opposite'] for t in tt)>=20
        priority=enough and m['self_direction_flip']['mean']>=.3 and m['self_direction_flip']['ci'][0]>.1 and m['peer_mean_gain']['mean']>=.1 and m['peer_mean_gain']['ci'][0]>0
        validity={f: m[f+'_sign']['n']>=8 and m[f+'_rank']['n']>=8 and m[f+'_sign']['mean']>=.85 and m[f+'_sign']['ci'][0]>.7 and m[f+'_rank']['mean']>=.8 and m[f+'_rank']['ci'][0]>.5 and m[f+'_coverage']['mean']>=.8 for f in SCORES}
        summary.append(dict(scenario=sc,opposite=sum(t['opposite'] for t in tt),metrics=m,peer_research_priority=priority,local_quality_screen=validity))
    modelteams=[]
    for c in read(SRC/'cases.json')['development']:
        for aid in c['ids']:
            rr=[v for v in models if v['key']==c['key'] and v['model_id']==aid]
            modelteams.append(dict(key=c['key'],scenario=c['scenario'],model_id=aid,**{f:avg([v[f] for v in rr if v[f] is not None]) for f in ['external_sign','reverse','nearzero','mean_abs_delta_gap']}))
    examples=[]
    for sc in ['clean','controlled_poison']:
        opposite=sorted([r for r in rows if r['scenario']==sc and sign(r['j'])*sign(r['j_E'])==-1],key=lambda r:(r['key'],r['start'],r['agent_id'],r['t']))
        for label,predicate in [('self_direction_flip',lambda r:sign(r['other'])==sign(r['j_E'])),('peers_still_opposite',lambda r:sign(r['other'])==sign(r['j']))]:
            selected=[r for r in opposite if predicate(r)]
            examples.append(dict(scenario=sc,selection=label,row=max(selected,key=lambda r:min(abs(r['j']),abs(r['j_E']))) if selected else None))
    for prior in read(PRIOR/'examples.json'):
        if prior['selection']=='max_opposite':examples.append(dict(scenario=prior['scenario'],selection='prior_max_opposite',row=next(r for r in rows if (r['key'],r['start'],r['agent_id'],r['t'])==(prior['key'],prior['start'],prior['agent_id'],prior['t']))))
    for name,value in [('round_rows',rows),('agent_totals',agents),('start_metrics',startmetrics),('team_metrics',teams),('summary',summary),('model_start_metrics',models),('model_team_metrics',modelteams),('examples',examples)]:save(ROOT/(name+'.json'),value)
    for p,h in inputs.items():assert sha(p)==h
    save(ROOT/'analysis_completion.json',dict(complete=True,branches=len(rows),agent_start_records=len(agents),model_start_records=len(models),identity_max_error=err,new_model_inferences=0,new_pooled_inferences=0,truth_loaded=False,wall_seconds=time.perf_counter()-tic))
    for s in summary:print(s['scenario'],{f:s['metrics'][f] for f in ['self_direction_flip','peers_still_opposite','peers_nearzero','j_sign','own_sign','other_sign','other_mean_sign','peer_mean_gain','own_gap_share']},'priority',s['peer_research_priority'],'screens',s['local_quality_screen'])
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('analysis_cap')));signal.alarm(300)
    try:main()
    except Exception as e:
        if ROOT.exists():save(ROOT/'failure.json',dict(error=repr(e)))
        raise
