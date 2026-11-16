"""Read-only final audit: locked files, independent metrics, evaluator reload, cost ledger."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from study import ROOT,WORK,OLD,read,save,sha,key,check,setup

def audit():
    import torch
    from models import PretrainedModel
    check();start=time.perf_counter();summary=read(ROOT/'summary.json')
    for lock in ['scores_and_points_locked.json','evaluation_lock.json']:
        for n,h in read(ROOT/lock)['hashes'].items():assert sha(ROOT/n)==h
    original_sources={}
    for n,h in read(ROOT/'source_lock.json').items():
        p=Path(__file__).parent/n
        if p.exists():assert sha(p)==h;original_sources[n]=h
    points=0;max_error=0.;external={}
    # Separate evaluator implementation and traversal, no analysis imports.
    cfg=read(ROOT/'cases.json');pooled={x['key']:x for x in read(ROOT/'pooled_evaluation.json')}
    for c in cfg['development']:
        net=PretrainedModel(len(c['fixed_start']));net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval()
        values={}
        for k,v in pooled[c['key']]['values'].items():
            with torch.no_grad(): y=abs(float(net(torch.as_tensor(np.array(v['x'],np.float32)).reshape(1,-1)).reshape(-1)[0]))
            max_error=max(max_error,abs(y-v['e']));assert y==v['e'];values[k]=y;points+=1
        external[c['key']]=values
    # Recompute team means from locked raw branch records and evaluator replies.
    by_scenario={};participants=[]
    for c in cfg['development']:
        r=read(ROOT/'development'/(c['key']+'.json'));tt=[];agree=[];rank_rounds=0
        ex=external[c['key']]
        for t in range(len(r['full'])):
            bs=[b for b in r['branches'] if b['t']==t and b['status']=='available']
            m=np.array([b['m'] for b in bs]);e=np.array([ex[key(b['x_minus'])]-ex[key(b['x_full'])] for b in bs])
            for a,b in zip(m,e):
                if abs(a)>1e-4:agree.append(bool(abs(b)>1e-4 and a*b>0))
            if len(bs)>=3 and np.ptp(m)>1e-4 and np.ptp(e)>1e-4:
                ra=rankdata(m);rb=rankdata(e)
                rho=float(np.dot(ra-ra.mean(),rb-rb.mean())/np.sqrt(np.dot(ra-ra.mean(),ra-ra.mean())*np.dot(rb-rb.mean(),rb-rb.mean())))
                tt.append(rho);rank_rounds+=1
        by_scenario.setdefault(c['scenario'],[]).append(dict(rho=None if not tt else float(np.mean(tt)),
            agreement=None if not agree else float(np.mean(agree)),rank_rounds=rank_rounds,rounds=len(r['full'])))
    for sc,ts in by_scenario.items():
        for field,metric in [('rho','rank'),('agreement','sign')]:
            a=np.array([t[field] for t in ts if t[field] is not None]);expected=summary['scenarios'][sc][metric]
            assert abs(float(a.mean())-expected['mean'])<1e-12
            rng=np.random.default_rng(20261021);means=[]
            for _ in range(10000):means.append(float(a[rng.integers(0,len(a),len(a))].mean()))
            ci=np.percentile(means,[.625,99.375]);np.testing.assert_allclose(ci,expected['interval'],rtol=0,atol=1e-12)
    # Match historical model-only artifacts before reusing their sample costs.
    hist=WORK/'contribution_analysis/dual_role_validity_experiment_v1_20261020'
    hc=read(hist/'cases.json')['development'];assert hc==cfg['development']
    histlock=read(hist/'references_lock.json')['record_sha256'];historical_samples=0
    for c in cfg['development']:
        for i in c['ids']:
            n=c['key']+'_model_'+str(i)+'.json';p=hist/'references'/n;assert sha(p)==histlock[n]
            rr=read(p);assert rr['fixed_start']==c['fixed_start']
            assert rr['anchor_ids']==c['ids'];np.testing.assert_array_equal(rr['direction_weights'],c['alphas'])
            historical_samples+=rr['counter']['samples']
    cost=summary['cost']['development'];baseline=cost['baseline']['samples']
    extra=cost['branch_decision']['samples']+cost['common_scoring']['samples'];usage=baseline+extra
    completions=[read(ROOT/(p+'_completion.json')) for p in ['pilot','development']]
    research_samples=sum(x['samples'] for x in completions)+read(ROOT/'verification.json')['new_model_samples']
    assert research_samples<=1200000
    pooled_total=summary['pooled_unique_points']+summary['pooled_recheck_points']+points;assert pooled_total<=7000
    storage=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    save(ROOT/'final_audit.json',dict(accepted=True,max_external_rebuild_error=max_error,
        pooled_additional_audit_points=points,pooled_total_evaluations=pooled_total,
        independent_metrics_and_bootstrap=True,live_new_sources_match_frozen=original_sources,
        new_trained_model_samples=research_samples,development_baseline_samples=baseline,
        development_counterfactual_extra_samples=extra,usage_samples=usage,
        historical_model_only_extra_samples=historical_samples,historical_usage_samples=baseline+historical_samples,
        usage_vs_baseline_ratio=usage/baseline,usage_vs_historical_ratio=usage/(baseline+historical_samples),
        extra_vs_historical_ratio=extra/historical_samples,storage_bytes=storage,wall_seconds=time.perf_counter()-start,
        limitations=['old-data exploration, no independent dataset confirmation','synthetic test fixtures excluded from trained-model sample count',
                    'wall timings exclude interpreter startup and compare different estimands']))
    print('FINAL_AUDIT_OK',research_samples,usage,historical_samples,pooled_total,flush=True)

if __name__=='__main__':setup();audit()
