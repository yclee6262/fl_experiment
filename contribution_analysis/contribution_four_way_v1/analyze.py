"""Fixed old-data statistics; references enter after the new replay lock."""
import csv, shutil
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from run import ROOT,CA,read,save,sha

SCORES=['A','A2','M','C','C2','J']
def sign(x):return int(x>1e-4)-int(x< -1e-4)
def rank(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
def boot(v):
    if not v:return dict(mean=None,interval=None,n=0)
    a=np.asarray(v);rng=np.random.default_rng(20261022)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),interval=np.percentile(means,[.1,99.9]).tolist(),n=len(a))
def run():
    assert read(ROOT/'replay_completion.json')['complete']
    assert read(ROOT/'scores_locked.json')['sha256']==sha(ROOT/'scores_and_costs.json')
    snap=ROOT/'analysis_source_snapshot';snap.mkdir(exist_ok=False)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    save(ROOT/'analysis_source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    refs=read(CA/'combined_contribution_layer1_experiment_v1_20261021/comparisons.json')
    by={(r['key'],r['agent_id']):r for r in refs};records=read(ROOT/'scores_and_costs.json');rows=[]
    for c in records:
        n=len(c['schemes']['S1']['rows'])
        for i in range(n):
            result=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'])
            for k in ['S1','S2','S3','S4']:
                r=c['schemes'][k]['rows'][i];result.update(r['values']);result['agent_id']=r['agent_id'];assert r['missing']==0
            ref=by[c['key'],result['agent_id']];assert abs(ref['D_L']-result['D_L'])<1e-10
            assert abs(result['A']-ref['A'])<1e-10 and abs(result['M']-ref['M'])<1e-10
            result['D_E']=ref['D_E'];rows.append(result)
    save(ROOT/'comparisons.json',rows)
    with (ROOT/'comparisons.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    teams=[]
    for c in records:
        rr=[r for r in rows if r['key']==c['key']]
        for reference in ['D_L','D_E']:
            for score in SCORES+(['D_L'] if reference=='D_E' else []):
                decisive=[r for r in rr if sign(r[reference])]
                positive=[r for r in rr if sign(r[reference])==1];negative=[r for r in rr if sign(r[reference])==-1]
                denom=sum(abs(r[reference]) for r in rr)
                teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],score=score,reference=reference,
                    rho=rank([r[score] for r in rr],[r[reference] for r in rr]),
                    sign=None if not decisive else float(np.mean([sign(r[score])==sign(r[reference]) for r in decisive])),
                    positive_recall=None if not positive else float(np.mean([sign(r[score])==1 for r in positive])),
                    negative_recall=None if not negative else float(np.mean([sign(r[score])==-1 for r in negative])),
                    near_zero=float(np.mean([sign(r[score])==0 for r in rr])),
                    normalized_gap=None if denom<=len(rr)*1e-4 else sum(abs(r[score]-r[reference]) for r in rr)/denom))
    save(ROOT/'team_metrics.json',teams);metrics=[];paired=[];costs=[];timings=read(ROOT/'timing.json')
    for sc in ['clean','controlled_poison']:
        cs=[c for c in records if c['scenario']==sc]
        for scheme in ['S1','S2','S3','S4']:
            base=sum(c['schemes'][scheme]['baseline_samples'] for c in cs)
            extra=sum(c['schemes'][scheme]['extra_samples'] for c in cs)
            total=base+extra;ref_total=sum(c['schemes']['S4']['samples'] for c in cs)
            tt=[t for t in timings if t['scenario']==sc and t['scheme']==scheme]
            percase={c['key']:float(np.median([t['wall_seconds'] for t in tt if t['key']==c['key']])) for c in cs}
            reftime={c['key']:float(np.median([t['wall_seconds'] for t in timings if t['key']==c['key'] and t['scheme']=='S4'])) for c in cs}
            costs.append(dict(scenario=sc,scheme=scheme,baseline_samples=base,extra_samples=extra,total_samples=total,
                samples_ratio_to_full=total/ref_total,cache_hits=sum(c['schemes'][scheme]['cache_hits'] for c in cs),
                requests=sum(c['schemes'][scheme]['requests'] for c in cs),
                summed_case_median_seconds=sum(percase.values()),summed_time_ratio=sum(percase.values())/sum(reftime.values()),
                per_team_time_ratio=boot([percase[k]/reftime[k] for k in percase]),cost_pass=total<=.7*ref_total))
        for reference in ['D_L','D_E']:
            for score in SCORES+(['D_L'] if reference=='D_E' else []):
                ts=[t for t in teams if t['scenario']==sc and t['score']==score and t['reference']==reference]
                ms={field:boot([t[field] for t in ts if t[field] is not None]) for field in ['rho','sign','positive_recall','negative_recall','near_zero','normalized_gap']}
                rg=ms['rho'];sg=ms['sign'];eff=(rg['n']>=8 and rg['mean']>=.5 and rg['interval'][0]>0 and sg['n']>=8 and sg['mean']>=.7 and sg['interval'][0]>.5)
                scheme={'A':'S1','M':'S1','C':'S1','A2':'S2','C2':'S2','J':'S3','D_L':'S4'}[score]
                cp=next(t for t in costs if t['scenario']==sc and t['scheme']==scheme)['cost_pass']
                metrics.append(dict(scenario=sc,score=score,reference=reference,**ms,effect_pass=bool(eff),
                    replacement_candidate=score in ['C','C2','J'],replacement_pass=bool(score in ['C','C2','J'] and eff and cp)))
            for first,second in [('A2','A'),('C2','C'),('J','C2'),('C2','M')]:
                aa={t['key']:t for t in teams if t['scenario']==sc and t['score']==first and t['reference']==reference}
                bb={t['key']:t for t in teams if t['scenario']==sc and t['score']==second and t['reference']==reference}
                pairs=[dict(key=k,delta=aa[k]['rho']-bb[k]['rho']) for k in aa if aa[k]['rho'] is not None and bb[k]['rho'] is not None]
                paired.append(dict(scenario=sc,reference=reference,comparison=first+'-'+second,rank_delta=boot([p['delta'] for p in pairs]),pairs=pairs))
    examples={sc:{'max_role_joint_gap':max([r for r in rows if r['scenario']==sc],key=lambda r:abs(r['C2']-r['J'])),
        'max_local_full_gap':max([r for r in rows if r['scenario']==sc],key=lambda r:abs(r['J']-r['D_L']))} for sc in ['clean','controlled_poison']}
    winners=[s for s in ['C','C2','J'] if all(m['replacement_pass'] for m in metrics if m['score']==s and m['reference']=='D_L')]
    save(ROOT/'summary.json',dict(metrics=metrics,paired=paired,costs=costs,examples=examples,
        teams=20,participants=len(rows),complete_participants=len(rows),internal_replacement_candidates_passing_both_scenarios=winners,
        old_data_exploration=True,new_confirmation=False,sign_denominator='reference nonzero; estimate zero fails',
        new_pooled_inferences=0,truth_loaded=False))
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h
    save(ROOT/'analysis_completion.json',dict(complete=True,comparisons_sha256=sha(ROOT/'comparisons.json'),summary_sha256=sha(ROOT/'summary.json')))
    for m in metrics:print(m['scenario'],m['reference'],m['score'],'rho',m['rho']['mean'],'sign',m['sign']['mean'],'pass',m['replacement_pass'])
    for c in costs:print('COST',c)
if __name__=='__main__':run()
