"""Recompute cost summaries and stratified paired intervals from saved records."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.anchor_ca_model_v_cost_v1.run import OUT,save
import numpy as np
def summary(rr):
    costs={a:sum(r['cost'][a] for r in rr) for a in ['B','A','AV','L']}
    ratios=[r['cost']['AV']/r['cost']['L'] for r in rr];cheap=sum(r['cost']['AV']<r['cost']['L'] for r in rr)
    saving=1-costs['AV']/costs['L']
    return dict(teams=len(rr),cost=costs,extra_AV=costs['AV']-costs['B'],extra_L=costs['L']-costs['B'],
        ratio=costs['AV']/costs['L'],saving_pct=saving*100,cheaper=cheap,cheaper_fraction=cheap/len(rr),
        ratio_median=float(np.median(ratios)),ratio_p90=float(np.quantile(ratios,.9)),ratio_range=[min(ratios),max(ratios)],
        gate='通過' if saving>=.1 and cheap/len(rr)>=.8 else '部分通過' if saving>0 else '未達',
        actual_rounds_median=float(np.median([r['rounds'] for r in rr])),anchor_trials=sum(r['anchor_trials'] for r in rr))
def main():
    completion=json.loads((OUT/'completion.json').read_text())
    rows=json.loads((OUT/'progress.json').read_text());timing=json.loads((OUT/'timing.json').read_text());groups={}
    for phase in ['historical','known_analytic','new_analytic']:
        rr=[r for r in rows if r['phase']==phase and r['iterations']==30];groups[phase]=summary(rr)
    primary=[r for r in rows if r['phase']=='new_analytic' and r['iterations']==30]
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            rr=[r for r in primary if r['n']==n and r['scenario']==sc];groups[f'n{n}_{sc}']=summary(rr)
    # Same sampled team keys across budgets preserves paired sensitivity.
    rng=np.random.default_rng(20261104);index=[]
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            ix=np.array([i for i,r in enumerate(primary) if r['n']==n and r['scenario']==sc]);index.append(ix[rng.integers(0,len(ix),(10000,len(ix)) )])
    selected=np.concatenate(index,axis=1)
    for it in [10,30,60]:
        lookup={r['key']:r for r in rows if r['phase']=='new_analytic' and r['iterations']==it}
        if len(lookup)!=80:
            s=summary(list(lookup.values()));s['complete']=False;s['planned_teams']=80
            s['interpretation']='partial ordered prefix; not a confirmation result or paired full sensitivity'
            groups[f'new_{it}_partial']=s
            continue
        rr=[lookup[r['key']] for r in primary];s=summary(rr)
        av=np.array([r['cost']['AV'] for r in rr]);loo=np.array([r['cost']['L'] for r in rr])
        s['ratio_ci95']=np.quantile(av[selected].sum(1)/loo[selected].sum(1),[.025,.975]).tolist()
        s['cheaper_fraction_ci95']=np.quantile((av[selected]<loo[selected]).mean(1),[.025,.975]).tolist()
        s['statistical_cost_advantage']=s['ratio_ci95'][1]<1
        s['core_time_median_team']={a:float(np.median([np.median([t['core_seconds'] for t in timing if t['key']==r['key'] and t['iterations']==it and t['arm']==a]) for r in rr])) for a in ['B','A','AV','L']}
        s['output_time_median']={a:float(np.median([t['output_seconds'] for t in timing if t['iterations']==it and t['arm']==a and t['key'] in lookup])) for a in ['B','A','AV','L']}
        groups[f'new_{it}']=s
    stage={};bad=[]
    for r in primary:
        path=OUT/'records'/f"{r['key']}_30_AV.json";d=json.loads(path.read_text())
        for k,v in d['stages'].items():stage[k]=stage.get(k,0)+v['samples']
        assert d['V_new_calls']==0
        if r['cost']['AV']>r['cost']['L']:bad.append(dict(key=r['key'],AV=r['cost']['AV'],L=r['cost']['L'],rounds=r['rounds'],anchor_trials=r['anchor_trials']))
    save(OUT/'analysis.json',dict(completion=completion,groups=groups,new_primary_stage_samples=stage,expensive_examples=sorted(bad,key=lambda r:r['AV']-r['L'],reverse=True)[:5]))
    print(json.dumps(dict(groups=groups,new_primary_stage_samples=stage),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
