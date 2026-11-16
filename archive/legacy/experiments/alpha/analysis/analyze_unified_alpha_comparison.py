"""Pooled-first method ranking and paired baselines; no pooled feedback to runs."""
import argparse, csv, itertools, json
from pathlib import Path
import numpy as np
import analyze_bootstrap_quality_research as shared
from research_analysis import paired_bootstrap_mean_ci
from unified_alpha_methods import METHODS, DEFINITIONS, initial_alpha

def analyze(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());rows=shared.read(root/'results.jsonl')
    shared.initial_alpha=initial_alpha
    check=shared.verify(root,m,rows)
    # The peer method must query six other-model/proposal pairs, whereas all
    # qualification predictions are reused. The legacy verifier only allowed
    # zero-query initialization methods; replace exactly that assumption.
    allowed=set()
    for row in rows:
        if row['method']=='target_peer_shrink':
            key=(row['scenario'],row['seed'],row['target'],row['draw'])
            expected=len(row['alpha'])*(len(row['alpha'])-1)
            if row['initialization_requests']==expected and row['initialization_input_samples']==expected:
                allowed.add(f"reuse queried API {key}/{row['method']}")
            else:check['errors'].append(f"peer query count mismatch {key}")
    check['errors']=[e for e in check['errors'] if e not in allowed]
    if m['methods']!=METHODS:check['errors'].append('registry mismatch')
    check['passed']=not check['errors'] and m['status']=='complete'
    shared.dump(root/'verification.json',check)
    index={(r['scenario'],r['seed'],r['method']):r for r in rows}
    groups=[];contrasts=[];family=3*(2*(len(METHODS)-2)+1)
    for scenario in m['scenarios']:
        for method in METHODS:
            rs=[r for r in rows if r['scenario']==scenario and r['method']==method]
            vs=[r for r in rs if shared.valid(r)]
            groups.append({'scenario':scenario,'method':method,'label':DEFINITIONS[method][0],
                'runs':len(rs),'failures':len(rs)-len(vs),'pooled':shared.summary([r['pooled_error'] for r in vs]),
                'fallbacks':sum(bool(r.get('alpha_audit',{}).get('fallback_reason')) for r in rs),
                'method_seconds':shared.summary([r['initialization_seconds']+r['stage3_seconds'] for r in rs]),
                'stage3_requests':shared.summary([r['stage3_requests'] for r in rs])})
        pairs=[('B_stage0','A_uniform')]+list(itertools.product(METHODS[2:],METHODS[:2]))
        for method,baseline in pairs:
            deltas=[];excluded=[];conflicts=[]
            for seed in m['seeds']:
                a=index.get((scenario,seed,method));b=index.get((scenario,seed,baseline))
                if not shared.valid(a or {}) or not shared.valid(b or {}):
                    excluded.append({'seed':seed,'method_status':(a or {}).get('status','missing'),'baseline_status':(b or {}).get('status','missing')});continue
                delta=a['pooled_error']-b['pooled_error'];deltas.append(delta)
                if delta*(a['truth_error']-b['truth_error'])<0:conflicts.append(seed)
            ci=paired_bootstrap_mean_ci(deltas);sim=paired_bootstrap_mean_ci(deltas,confidence=1-.05/family)
            af=sum(not shared.valid(index.get((scenario,s,method),{})) for s in m['seeds'])
            bf=sum(not shared.valid(index.get((scenario,s,baseline),{})) for s in m['seeds'])
            contrasts.append({'scenario':scenario,'method':method,'baseline':baseline,'n':len(deltas),
                'mean_delta':float(np.mean(deltas)) if deltas else None,'CI95':ci,'CI_simultaneous':sim,
                'excluded':excluded,'pooled_truth_conflict_seeds':conflicts,
                'screening_interval_negative':bool(check['passed'] and not excluded and af<=bf and sim[1] is not None and sim[1]<0),
                'interpretation':'exploratory screening, independent confirmation needed'})
    # One average across the three scenarios per replication, equal weights
    # fixed before screening. This is descriptive, not a fourth confirmatory test.
    aggregate=[]
    for method in METHODS:
        values=[]
        for seed in m['seeds']:
            rs=[index.get((s,seed,method),{}) for s in m['scenarios']]
            if all(shared.valid(r) for r in rs):values.append(np.mean([r['pooled_error'] for r in rs]))
        aggregate.append({'method':method,'label':DEFINITIONS[method][0],'equal_scenario_pooled':shared.summary(values)})
    shared.dump(root/'analysis.json',{'verification':check,'groups':groups,'comparisons':contrasts,
        'aggregate':aggregate,'experiment_kind':'exploratory_unified_screen','comparison_family':family,
        'rule':'10000 seed-level paired bootstrap, RNG20260924; Bonferroni baseline family; equal-scenario descriptive average; no equivalence claims'})
    with (root/'method_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['scenario','method','label','mean','median','Q1','Q3','valid','failures']);w.writeheader()
        for g in groups:w.writerow({**{k:g[k] for k in ['scenario','method','label','failures']},**{k:g['pooled'][k] for k in ['mean','median','Q1','Q3']},'valid':g['pooled']['n']})
    shared.dump(root/'method_definitions.json',DEFINITIONS)
    print(json.dumps(check));return check

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    if not analyze(a.root)['passed']:raise SystemExit(1)
