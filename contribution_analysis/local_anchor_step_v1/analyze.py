"""Load existing custom LOO labels only after the new A scores are locked."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from common import read,save,sha,csvsave,csvread,check

def correlation(a,b):
    a=rankdata(a);b=rankdata(b)
    return None if np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])
def sign(a):return int(a>1e-4)-int(a < -1e-4)
def interval(values):
    v=np.array([np.nan if x is None else x for x in values]);rng=np.random.default_rng(20261020)
    sample=v[rng.integers(0,len(v),(10000,len(v)))];n=np.isfinite(sample).sum(axis=1)
    means=np.nansum(sample,axis=1)[n>0]/n[n>0]
    return np.quantile(means,[.025,.975]).tolist() if len(means) else None

def analyze(root):
    cfg=check(root);prior=Path(cfg['previous']);lock=read(root/'scores_lock.json')
    for n,h in lock['record_sha256'].items():
        if sha(root/'records'/n)!=h:raise ValueError('score_changed')
    labels=csvread(Path(cfg['labels_path']));by={(r['key'],int(r['agent_id'])):r for r in labels}
    rows=[];teams=[];costs=[];bench=read(root/'benchmark_results.json')
    for phase,cases in read(root/'cases.json').items():
        for case in cases:
            r=read(root/'records'/(case['key']+'.json'));b=read(prior/'baselines'/(case['key']+'.json'));C=np.array([float(by[case['key'],i]['C_custom_full']) for i in case['ids']])
            A=r['A'];oldS=b['scores']['S'];status=r['status']
            if status=='available' and np.ptp(C)<=1e-4:status='near_flat_reference'
            rho=correlation(A,C) if status=='available' else None
            if status=='available' and rho is None:status='score_tie'
            oldcorr=correlation(oldS,C) if oldS is not None and np.ptp(C)>1e-4 else None
            alpha_corr=correlation(case['alphas'],C) if np.ptp(C)>1e-4 else None
            teams.append(dict(key=case['key'],phase=phase,scenario=case['scenario'],seed=case['seed'],m=len(case['ids']),status=status,
                A_spearman=rho,S_spearman=oldcorr,alpha_spearman=alpha_corr,
                correlation_change=None if rho is None or oldcorr is None else rho-oldcorr,
                n_accepted_steps=r['n_accepted_steps'],C_range=float(np.ptp(C))))
            for k,i in enumerate(case['ids']):
                a=None if A is None else A[k];c=C[k]
                rows.append(dict(key=case['key'],phase=phase,scenario=case['scenario'],seed=case['seed'],agent_id=i,
                    status=status,A=a,A_partial=r['A_partial'][k],A_positive=r['A_positive'][k],A_negative=r['A_negative'][k],
                    missing_steps=r['missing_steps'][k],S=None if oldS is None else oldS[k],C_custom_full=float(c),
                    direction_match=None if a is None else sign(a)==sign(c),raw_gap=None if a is None else a-c))
            removal=[read(prior/'removals'/(case['key']+'_'+str(i)+'.json')) for i in case['ids']]
            timing=read(root/'records'/(case['key']+'.timing.json'))
            baseline_samples=b['counter']['samples'];loo_samples=sum(x['counter']['samples'] for x in removal)
            baseline_time=b['total_seconds'];loo_time=sum(x['total_seconds'] for x in removal)
            costs.append(dict(key=case['key'],phase=phase,scenario=case['scenario'],seed=case['seed'],
                baseline_samples=baseline_samples,anchor_extra_samples=r['counter']['samples'],anchor_extra_calls=r['counter']['calls'],
                cache_hits=r['counter']['cache_hits'],no_motion=r['counter']['no_motion'],full_LOO_extra_samples=loo_samples,
                baseline_recorded_seconds=baseline_time,anchor_core_seconds=r['core_seconds'],full_LOO_recorded_seconds=loo_time,
                anchor_plus_baseline_samples=baseline_samples+r['counter']['samples'],LOO_plus_baseline_samples=baseline_samples+loo_samples,
                anchor_plus_baseline_compute=baseline_time+r['core_seconds'],LOO_plus_baseline_recorded_compute=baseline_time+loo_time,
                **{k:timing[k] for k in ['input_read_seconds','model_load_seconds','write_seconds','end_to_end_seconds','record_bytes','process_RSS_high_water_bytes']}))
    summary=[];secondary=[];cost_summary=[];examples=[]
    for phase in ['pilot','development']:
        for scenario in ['clean','controlled_poison']:
            ts=[t for t in teams if t['phase']==phase and t['scenario']==scenario];valid=[t for t in ts if t['A_spearman'] is not None]
            mean=float(np.mean([t['A_spearman'] for t in valid])) if valid else None;ci=interval([t['A_spearman'] for t in ts])
            oldmean=float(np.mean([t['S_spearman'] for t in ts if t['S_spearman'] is not None]))
            summary.append(dict(phase=phase,scenario=scenario,total_teams=len(ts),valid_teams=len(valid),coverage=len(valid)/len(ts),
                mean_A_spearman=mean,median_A_spearman=float(np.median([t['A_spearman'] for t in valid])) if valid else None,
                exploratory_95pct_CI=ci,mean_previous_S_spearman=oldmean,
                mean_correlation_change=float(np.mean([t['correlation_change'] for t in ts if t['correlation_change'] is not None])) if valid else None,
                paired_change_exploratory_95pct_CI=interval([t['correlation_change'] for t in ts]),
                alpha_defined_teams=sum(t['alpha_spearman'] is not None for t in ts),
                effect_screen_pass=bool(len(valid)/len(ts)>=.90 and mean is not None and mean>=.5 and ci[0]>0)))
            rs=[r for r in rows if r['phase']==phase and r['scenario']==scenario and r['A'] is not None]
            y=np.array([r['C_custom_full'] for r in rs]);p=np.array([r['A'] for r in rs]);neg=np.array([sign(x)==-1 for x in y]);pn=np.array([sign(x)==-1 for x in p]);tp=int((neg&pn).sum())
            secondary.append(dict(phase=phase,scenario=scenario,n_effects=len(rs),direction_agreement=float(np.mean([sign(a)==sign(c) for a,c in zip(p,y)])) if len(p) else None,
                raw_uncalibrated_MAE=float(np.mean(abs(p-y))) if len(p) else None,raw_uncalibrated_RMSE=float(np.sqrt(np.mean((p-y)**2))) if len(p) else None,
                true_negative=int(neg.sum()),predicted_negative=int(pn.sum()),negative_true_positive=tp,
                negative_precision=tp/int(pn.sum()) if pn.any() else None,negative_recall=tp/int(neg.sum()) if neg.any() else None))
            cs=[c for c in costs if c['phase']==phase and c['scenario']==scenario]
            sums={k:sum(c[k] for c in cs) for k in ['baseline_samples','anchor_extra_samples','full_LOO_extra_samples','anchor_plus_baseline_samples','LOO_plus_baseline_samples',
                    'anchor_core_seconds','anchor_plus_baseline_compute','LOO_plus_baseline_recorded_compute','end_to_end_seconds','record_bytes']}
            sums.update(phase=phase,scenario=scenario,sample_saving=float(1-sums['anchor_plus_baseline_samples']/sums['LOO_plus_baseline_samples']),
                recorded_compute_saving=float(1-sums['anchor_plus_baseline_compute']/sums['LOO_plus_baseline_recorded_compute']))
            cost_summary.append(sums)
    for scenario in ['clean','controlled_poison']:
        valid=[t for t in teams if t['phase']=='development' and t['scenario']==scenario and t['A_spearman'] is not None]
        for label,descending in [('best_rank',True),('worst_rank',False)]:
            if valid:
                t=sorted(valid,key=lambda t:((-t['A_spearman']) if descending else t['A_spearman'],t['seed']))[0]
                examples.append(dict(category=label,team=t,participants=[r for r in rows if r['key']==t['key']]))
    development=[s for s in summary if s['phase']=='development'];effect=all(s['effect_screen_pass'] for s in development)
    devcost=[c for c in costs if c['phase']=='development'];sample_saving=1-sum(c['anchor_plus_baseline_samples'] for c in devcost)/sum(c['LOO_plus_baseline_samples'] for c in devcost)
    timing_pass=bench['aggregate']['compute_seconds']['ratio']<1 and bench['aggregate']['pipeline_seconds']['ratio']<1
    cost_pass=sample_saving>=.50 and timing_pass
    gate=dict(accepted=bool(effect and cost_pass),effect_screen_pass=effect,cost_screen_pass=cost_pass,
        technical_accepted=all(c['technical_accepted'] for c in bench['cases']),development_sample_saving=sample_saving,
        paired_compute_ratio=bench['aggregate']['compute_seconds']['ratio'],paired_end_to_end_ratio=bench['aggregate']['pipeline_seconds']['ratio'],
        status='exploratory_screen_passed' if effect and cost_pass else 'exploratory_screen_not_passed',
        independent_confirmation_started=False,no_fitting_or_formula_tuning=True)
    csvsave(root/'participant_comparison.csv',rows);csvsave(root/'team_comparison.csv',teams);csvsave(root/'case_costs.csv',costs)
    save(root/'summary.json',summary);save(root/'secondary_metrics.json',secondary);save(root/'cost_summary.json',cost_summary)
    save(root/'representative_cases.json',examples);save(root/'gate.json',gate)
    print(__import__('json').dumps(dict(summary=summary,cost=cost_summary,gate=gate),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root.resolve())
