"""Seed-paired comparisons; 18 preregistered contrasts, no evaluator feedback."""
import argparse,json,itertools,csv
from pathlib import Path
import numpy as np
import analyze_bootstrap_quality_research as shared
from next_alpha_initialization import initial_alpha,METHODS
from research_analysis import paired_bootstrap_mean_ci
shared.initial_alpha=initial_alpha

def analyze(root,registration=None):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());rows=shared.read(root/'results.jsonl')
    check=shared.verify(root,m,rows)
    if registration:
        reg=json.loads(Path(registration).read_text())
        for key in ['seeds','methods','scenarios','source_sha256']:
            if reg[key]!=m[key]:check['errors'].append('registration mismatch '+key)
    check['passed']=not check['errors'];shared.dump(root/'verification.json',check)
    index={(r['scenario'],r['seed'],r['method']):r for r in rows};groups=[];comparisons=[]
    for s in m['scenarios']:
        for method in METHODS:
            rs=[r for r in rows if r['scenario']==s and r['method']==method];vs=[r for r in rs if shared.valid(r)]
            g={'scenario':s,'method':method,'runs':len(rs),'failures':len(rs)-len(vs),'pooled':shared.summary([r['pooled_error'] for r in vs]),'truth':shared.summary([r['truth_error'] for r in vs])}
            for key in ['initialization_seconds','stage3_seconds','stage3_requests','initialization_requests','scoring_seconds']:
                g[key]=shared.summary([r[key] for r in rs if r.get(key) is not None])
            g['initialization_plus_stage3_seconds']=shared.summary([r['initialization_seconds']+r['stage3_seconds'] for r in rs])
            g['combination_solver_fallbacks']=sum(r.get('alpha_audit',{}).get('combination_solver_fallbacks',0) for r in rs)
            groups.append(g)
        for method,base in itertools.product(METHODS[2:],METHODS[:2]):
            pairs=[];excluded=[];conflicts=[]
            for seed in m['seeds']:
                a=index.get((s,seed,method));b=index.get((s,seed,base))
                if a is None or b is None or not shared.valid(a) or not shared.valid(b):excluded.append(seed);continue
                d=a['pooled_error']-b['pooled_error'];td=a['truth_error']-b['truth_error'];pairs.append({'seed':seed,'delta':d,'truth_delta':td})
                if d*td<0:conflicts.append(seed)
            ds=[p['delta'] for p in pairs];ci=paired_bootstrap_mean_ci(ds);corrected=paired_bootstrap_mean_ci(ds,confidence=1-.05/18)
            af=sum(not shared.valid(index.get((s,k,method),{})) for k in m['seeds']);bf=sum(not shared.valid(index.get((s,k,base),{})) for k in m['seeds'])
            comparisons.append({'scenario':s,'method':method,'baseline':base,'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,'CI95':ci,'CI_simultaneous_18':corrected,'excluded_seeds':excluded,'conflict_seeds':conflicts,'seed_pairs':pairs,'confirmed_advantage':bool(registration and check['passed'] and not excluded and af<=bf and corrected[1] is not None and corrected[1]<0)})
    shared.dump(root/'analysis.json',{'verification':check,'groups':groups,'comparisons':comparisons,'formal':bool(registration),'rule':'10000 seed bootstrap; 18 Bonferroni contrasts; RNG20260924'})
    with (root/'results.csv').open('w',newline='') as f:
        fields=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    print(json.dumps(check));return check
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--registration');a=p.parse_args()
    if not analyze(a.root,a.registration)['passed']:raise SystemExit(1)
