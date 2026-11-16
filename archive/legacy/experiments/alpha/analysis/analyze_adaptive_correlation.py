"""Six confirmatory new-vs-B/fixed contrasts; A/risk comparisons exploratory."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import analyze_bootstrap_quality_research as shared
from adaptive_correlation_alpha import initial_alpha, METHODS, PRIMARY
from research_analysis import paired_bootstrap_mean_ci
shared.initial_alpha = initial_alpha

def analyze(root, registration=None):
    root=Path(root); m=json.loads((root/'manifest.json').read_text()); rows=shared.read(root/'results.jsonl')
    check=shared.verify(root,m,rows)
    if registration:
        reg=json.loads(Path(registration).read_text())
        for key in ['seeds','methods','scenarios','source_sha256']:
            if reg[key] != m[key]: check['errors'].append('registration mismatch '+key)
        source=Path(m['source'])/'verification.json'
        if not source.exists() or not json.loads(source.read_text()).get('passed'):
            check['errors'].append('source preparation unverified')
    if m['sizes'] != [20] or m['targets'] != [0] or not m['original_blind'] or m['methods'] != METHODS:
        check['errors'].append('design mismatch')
    check['passed']=check['passed'] and not check['errors']
    shared.dump(root/'verification.json',check)
    index={(r['scenario'],r['seed'],r['method']):r for r in rows}; groups=[]; comparisons=[]
    for s in m['scenarios']:
        for method in METHODS:
            rs=[r for r in rows if r['scenario']==s and r['method']==method]; vs=[r for r in rs if shared.valid(r)]
            g={'scenario':s,'method':method,'runs':len(rs),'failures':len(rs)-len(vs),
               'pooled':shared.summary([r['pooled_error'] for r in vs]),'truth':shared.summary([r['truth_error'] for r in vs]),
               'fallbacks':sum(bool(r.get('alpha_audit',{}).get('fallback_reason')) for r in rs)}
            for field in ['initialization_seconds','stage3_seconds','stage3_requests','stage3_input_samples',
                          'initialization_requests','initialization_input_samples','scoring_seconds']:
                g[field]=shared.summary([r[field] for r in rs if r.get(field) is not None])
            g['initialization_plus_stage3_seconds']=shared.summary([r['initialization_seconds']+r['stage3_seconds']
                for r in rs if r.get('initialization_seconds') is not None and r.get('stage3_seconds') is not None])
            if method==PRIMARY:
                g['selected_lambda_counts']={str(l):sum(r.get('alpha_audit',{}).get('selected_lambda')==l for r in rs) for l in [0,.25,.5,.75,1]}
            groups.append(g)
        for base in METHODS[:-1]:
            pairs=[]; excluded=[]; conflicts=[]
            for seed in m['seeds']:
                a=index.get((s,seed,PRIMARY)); b=index.get((s,seed,base))
                if a is None or b is None or not shared.valid(a) or not shared.valid(b):
                    excluded.append({'seed':seed,'method_status':a.get('status') if a else 'missing',
                                     'baseline_status':b.get('status') if b else 'missing'});continue
                delta=a['pooled_error']-b['pooled_error']; td=a['truth_error']-b['truth_error']
                pairs.append({'seed':seed,'delta':delta,'truth_delta':td})
                if delta*td<0: conflicts.append(seed)
            ds=[p['delta'] for p in pairs]; ci=paired_bootstrap_mean_ci(ds)
            confirm=base in ['B_stage0','corr_precision_0p5']
            simultaneous=paired_bootstrap_mean_ci(ds,confidence=1-.05/6) if confirm else None
            af=sum(not shared.valid(index.get((s,k,PRIMARY),{})) for k in m['seeds'])
            bf=sum(not shared.valid(index.get((s,k,base),{})) for k in m['seeds'])
            comparisons.append({'scenario':s,'method':PRIMARY,'baseline':base,'confirmatory':confirm,
                'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,'CI95':ci,
                'CI_simultaneous_6':simultaneous,'excluded':excluded,'conflict_seeds':conflicts,'seed_pairs':pairs,
                'confirmed_advantage':bool(registration and confirm and check['passed'] and not excluded
                    and af<=bf and simultaneous[1] is not None and simultaneous[1]<0)})
    shared.dump(root/'analysis.json',{'verification':check,'groups':groups,'comparisons':comparisons,
        'formal':bool(registration),'rule':'10000 seed bootstrap RNG20260924; six Bonferroni comparisons'})
    fields=['scenario','seed','method','status','reason','X','alpha','pooled_error','truth_error','max_box_violation',
            'initialization_seconds','stage3_seconds','initialization_requests','initialization_input_samples',
            'stage3_requests','stage3_input_samples','scoring_seconds','selected_lambda','fallback_reason','artifact_reference']
    with (root/'results_compact.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in rows:
            d={k:r.get(k) for k in fields};d.update(selected_lambda=r.get('alpha_audit',{}).get('selected_lambda'),
                fallback_reason=r.get('alpha_audit',{}).get('fallback_reason'),
                artifact_reference=f"artifact_reference_{r['scenario']}_{r['seed']}.json")
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in d.items()})
    with (root/'paired_differences.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,['scenario','method','baseline','confirmatory','seed','delta','truth_delta']);writer.writeheader()
        for c in comparisons:
            for p in c['seed_pairs']:writer.writerow({**{k:c[k] for k in ['scenario','method','baseline','confirmatory']},**p})
    print(json.dumps(check)); return check

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--registration');args=p.parse_args()
    if not analyze(args.root,args.registration)['passed']:raise SystemExit(1)
