"""Post-score audit. Independent metric calculations and reference/cost matching."""
import csv
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from study import ROOT,WORK,OLD,PREV,DIAG,read,save,sha,key,check,setup

def rank(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    a=rankdata(x);b=rankdata(y);a=a-a.mean();b=b-b.mean()
    return float(a.dot(b)/np.sqrt(a.dot(a)*b.dot(b)))
def sign(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def agree(rows,x,y):
    rr=[r for r in rows if sign(r[x])]
    return None if not rr else sum(sign(r[x])==sign(r[y]) for r in rr)/len(rr)
def ci(vals):
    if not vals:return None
    a=np.asarray(vals);rng=np.random.default_rng(20261021);means=[]
    for _ in range(10000):means.append(float(a[rng.integers(0,len(a),len(a))].mean()))
    return np.percentile(means,[.1,99.9])

def audit():
    check();start=time.perf_counter()
    for n,h in read(ROOT/'evaluation_lock.json').items():assert sha(ROOT/n)==h
    rows=read(ROOT/'local_comparisons.json');parts=read(ROOT/'participant_totals.json');teams=read(ROOT/'team_metrics.json');summary=read(ROOT/'summary.json')
    bykey={p['key']:p['values'] for p in read(ROOT/'pooled_evaluation.json')}
    for c in read(ROOT/'cases.json')['development']:
        rec=read(ROOT/'development'/(c['key']+'.json'));tab=bykey[c['key']]
        refs={r['agent_id']:r for r in read(DIAG/'full_revaluation'/(c['key']+'.json'))['rows']}
        for g in rec['groups']:
            row=next(r for r in rows if r['key']==c['key'] and r['agent_id']==g['agent_id'] and r['iteration']==g['t'])
            ef=tab[key(g['full']['post']['S_current'])]['E']
            for role,col in [('anchor','e_anchor'),('model','e_model'),('joint','e_joint')]:
                expected=tab[key(g['roles'][role]['endpoint'])]['E']-ef;assert expected==row[col]
        for p in [p for p in parts if p['key']==c['key']]:
            rr=[r for r in rows if r['key']==c['key'] and r['agent_id']==p['agent_id']]
            for name,col in [('A2','a2'),('M2','m2'),('C2','c2'),('J','j'),('R','residual')]:assert abs(sum(r[col] for r in rr)-p[name])<1e-10
            f=refs[p['agent_id']];assert abs(f['C_full_L']-p['D_L'])<1e-12
        for t in [t for t in teams if t['key']==c['key']]:
            if t['scope']=='local':
                vals=[]
                for it in sorted({r['iteration'] for r in rows if r['key']==c['key']}):
                    rr=[r for r in rows if r['key']==c['key'] and r['iteration']==it]
                    v=rank([r[t['score']] for r in rr],[r[t['reference']] for r in rr])
                    if v is not None:vals.append(v)
                actual=None if not vals else float(np.mean(vals));rr=[r for r in rows if r['key']==c['key']]
                numerator=sum(abs(r['residual']) for r in rr);den=sum(abs(r['a2'])+abs(r['m2'])+abs(r['j']) for r in rr)
                assert abs(numerator/den-t['residual_ratio'])<1e-12
                assert len(vals)==t['valid_rounds']
            else:
                rr=[p for p in parts if p['key']==c['key']]
                actual=rank([p[t['score']] for p in rr],[p[t['reference']] for p in rr])
            assert (actual is None)==(t['rho'] is None)
            if actual is not None:assert abs(actual-t['rho'])<1e-12
            ag=agree(rr,t['score'],t['reference']);assert ag==t['agreement']
    for m in summary['metrics']:
        tt=[t for t in teams if t['scenario']==m['scenario'] and t['scope']==m['scope'] and t['score']==m['score'] and t['reference']==m['reference']]
        for field,metric in [('rho','rank'),('agreement','sign')]:
            vals=[t[field] for t in tt if t[field] is not None]
            if not vals:assert m[metric]['mean'] is None;continue
            assert abs(np.mean(vals)-m[metric]['mean'])<1e-12
            np.testing.assert_allclose(ci(vals),m[metric]['interval'],rtol=0,atol=1e-12)
    for p in summary['paired']:
        vals=[r['delta'] for r in p['pairs']]
        if vals:np.testing.assert_allclose(ci(vals),p['stat']['interval'],rtol=0,atol=1e-12)
    # Costs match the exact same old cases and fixed-start participant-removal artifacts.
    baseline=0;removal=0
    for c in read(ROOT/'cases.json')['development']:
        base=read(OLD/'baselines'/(c['key']+'.json'));baseline+=base['counter']['samples']
        for i in c['ids']:
            rr=read(OLD/'removals'/(c['key']+'_'+str(i)+'.json'));assert rr['fixed_start']==c['fixed_start'];removal+=rr['counter']['samples']
    oldcost=read(PREV/'summary.json')['cost']['development']
    model_usage=sum(oldcost[k]['samples'] for k in ['baseline','branch_decision','common_scoring'])
    additional=read(ROOT/'development_completion.json')['new_model_samples'];total_usage=model_usage+additional
    pilot_groups=sum(len(read(p)['groups']) for p in (ROOT/'pilot').glob('*.json'))
    dev_groups=sum(len(read(p)['groups']) for p in (ROOT/'development').glob('*.json'))
    # Inputs include every new scoring source; current code must still match its frozen version.
    for n,h in read(ROOT/'source_lock.json').items():
        live=Path(__file__).parent/n
        if live.exists():assert sha(live)==h
    check()
    save(ROOT/'final_audit.json',dict(accepted=True,metrics_independently_rebuilt=True,bootstrap_independently_rebuilt=True,
        pooled_effects_rebuilt=True,new_pooled_inference_in_this_audit=0,
        pilot_role_groups=pilot_groups,development_role_groups=dev_groups,
        old_model_usage_samples=model_usage,layer2_incremental_development_samples=additional,
        estimated_all_three_roles_usage_samples=total_usage,
        historical_full_participant_removal_samples=removal,historical_baseline_samples=baseline,
        historical_full_participant_LOO_usage_samples=baseline+removal,
        usage_vs_full_participant_LOO_ratio=total_usage/(baseline+removal),
        current_research_new_samples=read(ROOT/'verification.json')['all_stage_new_samples'],
        audit_source_sha256=sha(__file__),postscore_diagnostic_only=True,
        wall_seconds=time.perf_counter()-start,storage_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())))
    print('FINAL_AUDIT_OK',pilot_groups,dev_groups,total_usage,baseline+removal,flush=True)

if __name__=='__main__':setup();audit()
