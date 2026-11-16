"""All three seed-paired contrasts are preregistered, no pooling with R5."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import analyze_bootstrap_quality_research as shared
from adaptive_correlation_alpha import initial_alpha
from research_analysis import paired_bootstrap_mean_ci
from run_correlation_replication import METHODS
shared.initial_alpha=initial_alpha
PAIRS=[('adaptive_corr_precision','B_stage0'),('corr_precision_0p5','B_stage0'),('adaptive_corr_precision','corr_precision_0p5')]

def analyze(root,registration=None):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());rows=shared.read(root/'results.jsonl')
    v=shared.verify(root,m,rows)
    if m['methods']!=METHODS or m['scenarios']!=['reference_clean'] or m['targets']!=[0] or m['sizes']!=[20] or not m['original_blind']:
        v['errors'].append('replication design mismatch')
    if registration:
        r=json.loads(Path(registration).read_text())
        for key in ['seeds','methods','scenarios','source_sha256']:
            if r[key]!=m[key]:v['errors'].append('preregistration mismatch '+key)
        source=Path(m['source'])/'verification.json'
        if not source.exists() or not json.loads(source.read_text()).get('passed'):v['errors'].append('source unverified')
    v['passed']=v['passed'] and not v['errors'];shared.dump(root/'verification.json',v)
    groups=[];index={(x['seed'],x['method']):x for x in rows}
    for method in METHODS:
        rs=[x for x in rows if x['method']==method];valid=[x for x in rs if shared.valid(x)]
        g={'method':method,'runs':len(rs),'failures':len(rs)-len(valid),'fallbacks':sum(bool(x.get('alpha_audit',{}).get('fallback_reason')) for x in rs),
           'pooled':shared.summary([x['pooled_error'] for x in valid]),'truth':shared.summary([x['truth_error'] for x in valid])}
        for key in ['initialization_seconds','stage3_seconds','stage3_requests','stage3_input_samples','initialization_requests','initialization_input_samples','scoring_seconds']:
            g[key]=shared.summary([x[key] for x in rs if x.get(key) is not None])
        g['method_seconds']=shared.summary([x['initialization_seconds']+x['stage3_seconds'] for x in rs if x.get('initialization_seconds') is not None and x.get('stage3_seconds') is not None])
        if method=='adaptive_corr_precision':g['lambda_counts']={str(l):sum(x.get('alpha_audit',{}).get('selected_lambda')==l for x in rs) for l in [0,.25,.5,.75,1]}
        groups.append(g)
    comparisons=[]
    for method,base in PAIRS:
        pairs=[];excluded=[]
        for seed in m['seeds']:
            a=index.get((seed,method));b=index.get((seed,base))
            if a is None or b is None or not shared.valid(a) or not shared.valid(b):
                excluded.append({'seed':seed,'method_status':a.get('status') if a else 'missing','baseline_status':b.get('status') if b else 'missing'});continue
            pairs.append({'seed':seed,'delta':a['pooled_error']-b['pooled_error'],'truth_delta':a['truth_error']-b['truth_error']})
        ds=[x['delta'] for x in pairs];corrected=paired_bootstrap_mean_ci(ds,confidence=1-.05/3)
        af=sum(not shared.valid(index.get((s,method),{})) for s in m['seeds']);bf=sum(not shared.valid(index.get((s,base),{})) for s in m['seeds'])
        comparisons.append({'method':method,'baseline':base,'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,
          'CI95':paired_bootstrap_mean_ci(ds),'CI_simultaneous_3':corrected,'seed_pairs':pairs,'excluded':excluded,
          'conflict_seeds':[p['seed'] for p in pairs if p['delta']*p['truth_delta']<0],
          'confirmed_advantage':bool(registration and v['passed'] and not excluded and af<=bf and corrected[1] is not None and corrected[1]<0)})
    shared.dump(root/'analysis.json',{'verification':v,'formal':bool(registration),'groups':groups,'comparisons':comparisons,
      'rule':'10000 seed paired bootstrap RNG20260924; 3 Bonferroni comparisons; R5 not pooled'})
    fields=['scenario','seed','method','status','reason','X','alpha','pooled_error','truth_error','max_box_violation',
      'initialization_seconds','stage3_seconds','stage3_requests','stage3_input_samples','initialization_requests','initialization_input_samples','scoring_seconds','selected_lambda','fallback_reason']
    with (root/'results_compact.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for x in rows:
            d={k:x.get(k) for k in fields};d.update(selected_lambda=x.get('alpha_audit',{}).get('selected_lambda'),fallback_reason=x.get('alpha_audit',{}).get('fallback_reason'))
            w.writerow({k:json.dumps(y) if isinstance(y,(list,dict)) else y for k,y in d.items()})
    with (root/'paired_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['method','baseline','seed','delta','truth_delta']);w.writeheader()
        for c in comparisons:
            for p in c['seed_pairs']:w.writerow({'method':c['method'],'baseline':c['baseline'],**p})
    print(json.dumps(v));return v
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--registration');a=p.parse_args()
    if not analyze(a.root,a.registration)['passed']:raise SystemExit(1)
