"""Existing-data A+M vs full participant removal. No model or solver imports."""
import csv
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'contribution_analysis/combined_contribution_layer1_experiment_v1_20261021'
OLD=WORK/'contribution_analysis/custom_direction_experiment_v2_20261020'
ANCHOR=WORK/'contribution_analysis/local_anchor_step_experiment_v1_20261020'
MODEL=WORK/'contribution_analysis/model_step_decision_experiment_v1_20261021'
DIAG=WORK/'contribution_analysis/contribution_diagnostic_experiment_v1_20261020'
DESIGN=WORK/'docs/combined_contribution_layer1_design_20261021.md'
def read(p):return json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def boot(values):
    if not values:return dict(mean=None,interval=None,n=0)
    x=np.asarray(values,float);rng=np.random.default_rng(20261021)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()),interval=np.percentile(means,[.625,99.375]).tolist(),n=len(x))
def rank(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    v=float(spearmanr(x,y).statistic)
    return v if np.isfinite(v) else None
def sign(x):return int(x>1e-4)-int(x< -1e-4)
def check():
    for name,h in read(ROOT/'inputs.json').items():assert sha(name)==h,('changed',name)

def run():
    started=time.perf_counter();ROOT.mkdir(exist_ok=False)
    cases=read(OLD/'cases.json')['development'];assert cases==read(MODEL/'cases.json')['development']
    assert cases==read(ANCHOR/'cases.json')['development']
    paths=[DESIGN,OLD/'cases.json',MODEL/'cases.json',ANCHOR/'cases.json',OLD/'participant_comparison.csv',
        MODEL/'participant_totals.json',MODEL/'scores_and_points_locked.json',MODEL/'completion_receipt.json',
        ANCHOR/'scores_lock.json',OLD/'baseline_lock.json',OLD/'removal_lock.json',DIAG/'full_revaluation_lock.json',
        DIAG/'verification.json']+list(Path(__file__).parent.glob('*.py'))
    for c in cases:
        paths.extend([Path(c['models']),Path(c['evaluator']),ANCHOR/'records'/(c['key']+'.json'),
            MODEL/'development'/(c['key']+'.json'),OLD/'baselines'/(c['key']+'.json'),
            DIAG/'full_revaluation'/(c['key']+'.json')])
        paths.extend(OLD/'removals'/(c['key']+'_'+str(i)+'.json') for i in c['ids'])
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths})
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='好先做第一層驗證看看效果如何',design_sha256=sha(DESIGN),
        scores=['A','M','C=A+M'],references=['full_removal_pooled','full_removal_fixed_internal'],
        old_data_exploration=True,new_confirmation=False,seed=20261021,repeats=10000,
        deadzone=1e-4,interval_percentiles=[.625,99.375],new_model_samples=0,new_solver_runs=0,new_pooled_calls=0))
    totals={(p['key'],p['agent_id']):p for p in read(MODEL/'participant_totals.json')}
    mlock=read(MODEL/'scores_and_points_locked.json')['hashes'];alock=read(ANCHOR/'scores_lock.json')['record_sha256']
    block=read(OLD/'baseline_lock.json')['record_sha256'];rlock=read(OLD/'removal_lock.json')['record_sha256']
    scores=[]
    for c in cases:
        name=c['key']+'.json';a=read(ANCHOR/'records'/name);m=read(MODEL/'development'/name);base=read(OLD/'baselines'/name)
        assert sha(ANCHOR/'records'/name)==alock[name]
        assert sha(MODEL/'development'/name)==mlock['development/'+name]
        assert sha(OLD/'baselines'/name)==block[name]==a['source_baseline_sha256']
        assert m['case']==c and a['ids']==c['ids']
        assert np.array_equal(m['baseline']['solution'],base['result']['solution'])
        assert m['baseline']['history']==base['result']['history'] and m['baseline']['states']==base['result']['states']
        assert m['baseline']['loss']==base['result']['loss']
        assert m['model_artifact_sha256']==sha(c['models'])
        for k,i in enumerate(c['ids']):
            mt=totals[c['key'],i];A=a['A'][k];M=mt['S']
            available=A is not None and M is not None and a['missing_steps'][k]==0 and mt['complete']
            scores.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=i,
                available=available,A=A,M=M,C=None if not available else A+M,
                anchor_accepted_rounds=a['n_accepted_steps'],model_all_rounds=len(m['full'])))
    save(ROOT/'scores.json',scores)
    save(ROOT/'scores_lock.json',dict(sha256=sha(ROOT/'scores.json'),reference_values_not_loaded=True))
    # Only after fixed score lock, read historical outcome references.
    labels={(v['key'],int(v['agent_id'])):v for v in csvread(OLD/'participant_comparison.csv')}
    rows=[]
    for c in cases:
        fr=read(DIAG/'full_revaluation'/(c['key']+'.json'));by_id={v['agent_id']:v for v in fr['rows']}
        for row in [r for r in scores if r['key']==c['key']]:
            i=row['agent_id'];k=c['ids'].index(i);rn=c['key']+'_'+str(i)+'.json';rem=read(OLD/'removals'/rn)
            assert sha(OLD/'removals'/rn)==rlock[rn]
            assert rem['fixed_start']==c['fixed_start'] and rem['excluded_id']==i
            assert rem['remaining_ids']==[v for j,v in enumerate(c['ids']) if j!=k]
            np.testing.assert_array_equal(rem['candidates'],[v for j,v in enumerate(c['candidates']) if j!=k])
            w=np.delete(np.asarray(c['alphas']),k);w=w/w.sum();np.testing.assert_array_equal(rem['alphas'],w)
            f=by_id[i];assert f['original_panel_ids']==c['ids'] and f['original_alphas']==c['alphas']
            np.testing.assert_array_equal(f['solution'],rem['result']['solution'])
            assert abs(np.dot(np.asarray(c['alphas'])/np.sum(c['alphas']),f['individual_losses'])-f['full_removal_fixed_L'])<1e-10
            label=labels[c['key'],i]
            assert abs(float(label['removal_error'])-float(label['full_error'])-float(label['C_custom_full']))<1e-12
            rows.append(dict(**row,D_L=f['full_removal_fixed_L']-f['baseline_full_L'],
                D_E=float(label['removal_error'])-float(label['full_error'])))
    save(ROOT/'comparisons.json',rows);csvsave(ROOT/'comparisons.csv',rows)
    teams=[]
    for c in cases:
        rr=[r for r in rows if r['key']==c['key'] and r['available']]
        for ref in ['D_E','D_L']:
            for score in ['A','M','C']:
                x=[r[score] for r in rr];y=[r[ref] for r in rr]
                decisive=[r for r in rr if sign(r[score])]
                teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],score=score,reference=ref,
                    expected=len(c['ids']),available=len(rr),rho=rank(x,y),decisive=len(decisive),
                    sign_agreement=None if not decisive else float(np.mean([sign(r[score])==sign(r[ref]) for r in decisive]))))
    save(ROOT/'team_metrics.json',teams)
    summaries=[];paired=[]
    for sc in ['clean','controlled_poison']:
        for ref in ['D_E','D_L']:
            for score in ['A','M','C']:
                tt=[t for t in teams if t['scenario']==sc and t['reference']==ref and t['score']==score]
                rho=boot([t['rho'] for t in tt if t['rho'] is not None]);sg=boot([t['sign_agreement'] for t in tt if t['sign_agreement'] is not None])
                rr=[r for r in rows if r['scenario']==sc and r['available']]
                confusion=[[sum(sign(r[score])==a and sign(r[ref])==b for r in rr) for b in [-1,0,1]] for a in [-1,0,1]]
                summaries.append(dict(scenario=sc,reference=ref,score=score,rank=rho,sign=sg,confusion=confusion,
                    rank_pass=bool(rho['n']>=8 and rho['mean']>=.5 and rho['interval'][0]>0),
                    sign_pass=bool(sg['n']>=8 and sg['mean']>=.7 and sg['interval'][0]>.5)))
            for comparator in ['A','M']:
                ct={t['key']:t for t in teams if t['scenario']==sc and t['reference']==ref and t['score']=='C'}
                ot={t['key']:t for t in teams if t['scenario']==sc and t['reference']==ref and t['score']==comparator}
                pairs=[dict(key=k,seed=ct[k]['seed'],delta=ct[k]['rho']-ot[k]['rho']) for k in ct if ct[k]['rho'] is not None and ot[k]['rho'] is not None]
                stat=boot([p['delta'] for p in pairs])
                paired.append(dict(scenario=sc,reference=ref,comparison='C-'+comparator,stat=stat,pairs=pairs,
                    incremental_gain=bool(stat['n']>=8 and stat['interval'][0]>0)))
    examples={}
    for sc in ['clean','controlled_poison']:
        rr=[r for r in rows if r['scenario']==sc and r['available']]
        pp=next(p for p in paired if p['scenario']==sc and p['reference']=='D_E' and p['comparison']=='C-M')
        worst=min(pp['pairs'],key=lambda p:p['delta']) if pp['pairs'] else None
        examples[sc]=dict(max_absolute_gap=max(rr,key=lambda r:abs(r['C']-r['D_E'])),
            worst_combination_vs_model=worst,worst_team_rows=[] if worst is None else [r for r in rr if r['key']==worst['key']])
    passed=all(t['rank_pass'] and t['sign_pass'] for t in summaries if t['reference']=='D_E' and t['score']=='C') and all(p['incremental_gain'] for p in paired if p['reference']=='D_E')
    save(ROOT/'summary.json',dict(metrics=summaries,paired=paired,examples=examples,
        teams=20,participants=len(rows),available=sum(r['available'] for r in rows),external_gate_passed=passed,
        wall_seconds=time.perf_counter()-started,new_model_samples=0,new_solver_runs=0,new_pooled_calls=0))
    check();save(ROOT/'analysis_lock.json',{p.name:sha(p) for p in ROOT.glob('*.json') if p.name!='analysis_lock.json'})
    for t in summaries:print(t['scenario'],t['reference'],t['score'],t['rank'],t['sign'],flush=True)
    for p in paired:print(p['scenario'],p['reference'],p['comparison'],p['stat'],flush=True)

if __name__=='__main__':run()
