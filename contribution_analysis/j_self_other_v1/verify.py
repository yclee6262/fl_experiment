"""Independent fixed-endpoint, fsum, tied-rank and bootstrap audit."""
import hashlib,json,math,itertools,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'j_self_other_diagnosis_v2_20261027'
SRC=ROOT.parent/'j_loo_start_stability_experiment_v1_20261027'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def sgn(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def mean(v):return math.fsum(v)/len(v) if v else None
def rank(a,b):
    if len(a)<3 or max(a)-min(a)<=1e-4 or max(b)-min(b)<=1e-4:return None
    def ranks(x):return [1+sum(v<z for v in x)+(sum(v==z for v in x)-1)/2 for z in x]
    x=ranks(a);y=ranks(b);mx=mean(x);my=mean(y)
    return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/(math.fsum((a-mx)**2 for a in x)*math.fsum((b-my)**2 for b in y))**.5
error=0.
def match(a,b):
    global error
    if a is None or b is None:assert a is None and b is None;return
    e=float(np.max(abs(np.array(a)-b)));assert np.allclose(a,b,rtol=1e-10,atol=1e-10),(a,b);error=max(error,e)
def boot(v):
    if not v:return dict(n=0,mean=None,ci=None)
    rng=np.random.default_rng(20261027);draws=[mean([v[i] for i in rng.integers(0,len(v),len(v))]) for _ in range(10000)]
    return dict(n=len(v),mean=mean(v),ci=np.quantile(draws,[.025,.975]).tolist())
def audit_metrics(rr):
    opp=[r for r in rr if sgn(r['j'])*sgn(r['j_E'])==-1];ext=[r for r in rr if sgn(r['j_E'])];old=[r for r in rr if sgn(r['j'])]
    out=dict(branches=len(rr),opposite=len(opp),external_nonzero=len(ext),old_nonzero=len(old),
        self_direction_flip=mean([sgn(r['other'])==sgn(r['j_E']) for r in opp]),peers_still_opposite=mean([sgn(r['other'])==sgn(r['j']) for r in opp]),peers_nearzero=mean([not sgn(r['other']) for r in opp]),
        own_peer_conflict=mean([sgn(r['own'])*sgn(r['other'])==-1 for r in rr]),
        own_abs_share=mean([abs(r['own'])/(abs(r['own'])+abs(r['other'])) for r in rr if abs(r['own'])+abs(r['other'])>1e-4]),
        own_gap_share=mean([abs(r['own_gap'])/(abs(r['own_gap'])+abs(r['other_gap'])) for r in rr if abs(r['own_gap'])+abs(r['other_gap'])>1e-4]),
        model_disagreement=mean([len(set(sgn(x) for x in r['model_deltas']))>1 for r in ext]),all_models_external_match=mean([all(sgn(x)==sgn(r['j_E']) for x in r['model_deltas']) for r in ext]))
    for f in ['j','own','other','other_mean']:
        out[f+'_sign']=mean([sgn(r[f])==sgn(r['j_E']) for r in ext]);out[f+'_reverse']=mean([sgn(r[f])==-sgn(r['j_E']) for r in ext]);out[f+'_zero']=mean([not sgn(r[f]) for r in ext]);out[f+'_oldmask_sign']=mean([sgn(r[f])==sgn(r['j_E']) for r in old])
        ranks=[];ts=sorted(set(r['t'] for r in rr))
        for t in ts:
            group=[r for r in rr if r['t']==t];v=rank([r[f] for r in group],[r['j_E'] for r in group])
            if v is not None:ranks.append(v)
        out[f+'_rank']=mean(ranks);out[f+'_coverage']=len(ranks)/len(ts)
    out['peer_mean_gain']=None if out['j_sign'] is None else out['other_mean_sign']-out['j_sign'];return out
def main():
    tic=time.perf_counter()
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h
    for p,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/p)==h
    rows=read(ROOT/'round_rows.json');totals=read(ROOT/'agent_totals.json');starts=read(ROOT/'start_metrics.json');teams=read(ROOT/'team_metrics.json')
    pools={r['key']:{key(v['x']):v['value'] for v in r['points']} for r in read(SRC/'pooled_points.json')};cases={r['key']:r for r in read(SRC/'cases.json')['development']}
    for target in starts:
        name=target['key'];si=target['start'];rr=[r for r in rows if r['key']==name and r['start']==si];source=read(SRC/'development'/(name+f'_s{si}_J.json'));raw={(v['i'],key(v['x'])):v['y'] for v in source['raw']};c=cases[name];w=np.asarray(c['alphas']);w=w/w.sum()
        for r,b in zip(rr,source['branches']):
            assert (r['i'],r['t'])==(b['i'],b['t']);i=r['i'];d=[abs(raw[k,key(b['x_minus'])])-abs(raw[k,key(b['x_full'])]) for k in range(len(w))]
            comp=[float(w[k])*d[k] for k in range(len(w))];a=comp[i];o=math.fsum(comp[k] for k in range(len(w)) if k!=i);je=pools[name][key(b['x_minus'])]-pools[name][key(b['x_full'])]
            gapself=w[i]*(d[i]-je);gapother=math.fsum(w[k]*(d[k]-je) for k in range(len(w)) if k!=i)
            for x,y in [(d,r['model_deltas']),(comp,r['model_components']),(a,r['own']),(o,r['other']),(o/(1-w[i]),r['other_mean']),(je,r['j_E']),(a+o,r['j']),(a+o,b['score']),(gapself,r['own_gap']),(gapother,r['other_gap']),(gapself+gapother,r['j']-je)]:match(x,y)
        actual=audit_metrics(rr)
        for f,v in actual.items():match(v,target[f])
        for aid in c['ids']:
            a=next(r for r in totals if r['key']==name and r['start']==si and r['agent_id']==aid);selected=[r for r in rr if r['agent_id']==aid]
            for f in ['j','own','other','other_mean','j_E']:match(math.fsum(v[f] for v in selected),a[f])
            match(a['own']+a['other'],a['j'])
    fields=[f for f in starts[0] if f not in ['key','scenario','start','branches','opposite','external_nonzero','old_nonzero']]
    for t in teams:
        rr=[r for r in starts if r['key']==t['key']]
        for f in fields:match(mean([r[f] for r in rr if r[f] is not None]),t[f])
        assert t['opposite']==sum(r['opposite'] for r in rr)
        for f in ['j','own','other','other_mean']:
            values=[]
            for si in range(5):
                aa=[a for a in totals if a['key']==t['key'] and a['start']==si];v=rank([a[f] for a in aa],[a['j_E'] for a in aa])
                if v is not None:values.append(v)
            match(mean(values),t[f+'_cumulative_rank'])
    for summary in read(ROOT/'summary.json'):
        rr=[r for r in teams if r['scenario']==summary['scenario']];m=summary['metrics']
        for f,target in m.items():
            a=boot([r[f] for r in rr if r[f] is not None]);assert a['n']==target['n'];match(a['mean'],target['mean']);match(a['ci'],target['ci'])
        p=m['self_direction_flip'];g=m['peer_mean_gain'];priority=p['n']>=8 and sum(r['opposite'] for r in rr)>=20 and p['mean']>=.3 and p['ci'][0]>.1 and g['mean']>=.1 and g['ci'][0]>0
        assert priority==summary['peer_research_priority']
        for f,result in summary['local_quality_screen'].items():
            expected=m[f+'_sign']['n']>=8 and m[f+'_rank']['n']>=8 and m[f+'_sign']['mean']>=.85 and m[f+'_sign']['ci'][0]>.7 and m[f+'_rank']['mean']>=.8 and m[f+'_rank']['ci'][0]>.5 and m[f+'_coverage']['mean']>=.8
            assert expected==result
    for item in read(ROOT/'model_start_metrics.json'):
        rr=[r for r in rows if r['key']==item['key'] and r['start']==item['start'] and sgn(r['j_E'])];k=cases[item['key']]['ids'].index(item['model_id']);assert item['n']==len(rr)
        values=dict(external_sign=mean([sgn(r['model_deltas'][k])==sgn(r['j_E']) for r in rr]),reverse=mean([sgn(r['model_deltas'][k])==-sgn(r['j_E']) for r in rr]),nearzero=mean([not sgn(r['model_deltas'][k]) for r in rr]),mean_abs_delta_gap=mean([abs(r['model_deltas'][k]-r['j_E']) for r in rr]))
        for f,v in values.items():match(v,item[f])
    for item in read(ROOT/'model_team_metrics.json'):
        rr=[r for r in read(ROOT/'model_start_metrics.json') if r['key']==item['key'] and r['model_id']==item['model_id']]
        for f in ['external_sign','reverse','nearzero','mean_abs_delta_gap']:match(mean([r[f] for r in rr if r[f] is not None]),item[f])
    for example in read(ROOT/'examples.json'):
        if example['selection']=='prior_max_opposite':continue
        rr=sorted([r for r in rows if r['scenario']==example['scenario'] and sgn(r['j'])*sgn(r['j_E'])==-1],key=lambda r:(r['key'],r['start'],r['agent_id'],r['t']))
        rr=[r for r in rr if sgn(r['other'])==sgn(r['j_E'] if example['selection']=='self_direction_flip' else r['j'])]
        expected=max(rr,key=lambda r:min(abs(r['j']),abs(r['j_E']))) if rr else None;assert expected==example['row']
    with (ROOT/'verification.json').open('x') as f:json.dump(dict(accepted=True,max_error=error,branches=len(rows),source_hashes_unchanged=True,new_model_inferences=0,wall_seconds=time.perf_counter()-tic),f,indent=2)
    print('accepted',error)
if __name__=='__main__':main()
