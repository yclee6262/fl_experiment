"""Independent evidence gate for coverage, delivery and candidate-level claims.

This does not alter any registered runner, candidate or analysis procedure.
It must pass after the three batches before conclusions are published.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from repeated_alpha_candidates import CANDIDATES,METHODS
from analyze_repeated_alpha_confirmation import contrasts

def delivery_errors(row):
    errors=[]
    if row.get('status')=='valid':
        x=np.asarray(row.get('X'),float);w=np.asarray(row.get('alpha'),float)
        if x.shape!=(2,) or not np.isfinite(x).all() or np.any(abs(x)>1+1e-8):errors.append('invalid X')
        if w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-8:errors.append('invalid alpha')
        for field in ['pooled_error','truth_error']:
            score=row.get(field)
            if score is None or not np.isfinite(score) or score<0:errors.append('invalid '+field)
    elif row.get('status') not in ['qualification_failure','solver_failure','invalid_output']:
        errors.append('unexpected execution status')
    else:
        if row.get('pooled_error') is not None or row.get('truth_error') is not None:errors.append('failure has quality score')
    return errors

def evaluate_claim(record,expected_n):
    errors=[];ci=record.get('CI_simultaneous');batches=record.get('batches',[])
    eligible=(record.get('n')==expected_n and not record.get('excluded') and len(batches)==3
        and ci is not None and len(ci)==2 and all(x is not None and np.isfinite(x) for x in ci)
        and ci[0]<=ci[1] and ci[1]<0
        and record.get('method_failures',1)<=record.get('baseline_failures',0)
        and all(b.get('mean_delta') is not None and b['mean_delta']<0 for b in batches)
        and sum(b.get('CI95') is not None and b['CI95'][1] is not None and b['CI95'][1]<0 for b in batches)>=2)
    if bool(record.get('stable_advantage'))!=bool(eligible):errors.append('stable flag disagrees with registered evidence')
    return errors

def audit(registration,final=False):
    reg=json.loads(Path(registration).read_text());root=Path(reg['output']);errors=[];batch_reports=[];allkeys=set()
    if reg['candidates']!=CANDIDATES or reg['methods']!=METHODS:errors.append('candidate registry mismatch')
    if len(reg['batches'])!=3 or len(set(reg['seeds']))!=len(reg['seeds']):errors.append('batches not independent/disjoint')
    total=0;completed=0
    for i,b in enumerate(reg['batches']):
        directory=Path(b['results']);path=directory/'results.jsonl';rows=[]
        if path.exists():rows=[json.loads(l) for l in path.read_text().splitlines()]
        keys=[(r['scenario'],r['seed'],r['method']) for r in rows]
        expected=set(itertools.product(reg['scenarios'],b['seeds'],reg['methods']))
        duplicated=len(keys)-len(set(keys));unexpected=set(keys)-expected
        if duplicated or unexpected:errors.append('duplicate/unexpected batch keys '+str(i+1))
        if allkeys.intersection(keys):errors.append('cross-batch duplicate keys')
        allkeys.update(keys)
        for row in rows:
            for error in delivery_errors(row):errors.append(f"{error}: {row['scenario']}/{row['seed']}/{row['method']}")
        manifest=json.loads((directory/'manifest.json').read_text()) if (directory/'manifest.json').exists() else {}
        done=manifest.get('status')=='complete';completed+=int(done);total+=len(rows)
        if final and (not done or set(keys)!=expected):errors.append('incomplete batch '+str(i+1))
        if done:
            v=json.loads((directory/'verification.json').read_text()) if (directory/'verification.json').exists() else {}
            if not v.get('passed'):errors.append('unverified batch '+str(i+1))
            if manifest.get('methods')!=reg['methods'] or manifest.get('source_sha256')!=reg['source_sha256']:errors.append('batch design/source mismatch')
            for name,h in reg['source_sha256'].items():
                p=directory/'source_snapshot'/name
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:errors.append('snapshot mismatch '+str(i+1)+'/'+name)
        prepared={}
        for scenario in reg['scenarios']:
            sp=Path(b['source'])/scenario/'shared_costs.jsonl'
            source_rows=[]
            if sp.exists():
                for line in sp.read_text().splitlines():
                    try:source_rows.append(json.loads(line))
                    except ValueError:continue  # A live writer may be finishing its last line.
            prepared[scenario]={'saved':len(source_rows),'expected':len(b['seeds']),
                'qualification_failures':sum(bool(r.get('missing_fixed_ids') or r.get('failure_reason')) for r in source_rows)}
        batch_reports.append({'batch':i+1,'expected':len(expected),'saved':len(rows),'complete':done,'source_preparation':prepared,
            'missing':len(expected-set(keys)),'statuses':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}})
    claims=[]
    analysis_path=root/'final_analysis.json'
    if final:
        if not analysis_path.exists():errors.append('missing final analysis')
        else:
            result=json.loads(analysis_path.read_text());comparisons=result.get('comparisons',[])
            ck=[(c['endpoint'],c['method'],c['baseline']) for c in comparisons]
            ec=set((e,m,b) for e in ['equal_scenario_mean']+reg['scenarios'] for m,b in contrasts())
            if set(ck)!=ec or len(ck)!=len(ec):errors.append('final contrast coverage mismatch')
            if result.get('status')!='final' or result.get('errors') or result.get('completed_batches')!=[0,1,2]:errors.append('final analysis is not accepted')
            for c in comparisons:
                for e in evaluate_claim(c,len(reg['seeds'])):errors.append(e+': '+str((c['endpoint'],c['method'],c['baseline'])))
            ci={(c['endpoint'],c['method'],c['baseline']):c for c in comparisons}
            for method in CANDIDATES:
                flags=[ci.get(('equal_scenario_mean',method,base),{}).get('stable_advantage',False) for base in ['A_uniform','B_stage0']]
                matching={'proposal_local_9':'B_augmented29','local_bias_risk_100':'B_augmented100'}.get(method)
                claims.append({'method':method,'role':'baseline' if method in ['A_uniform','B_stage0'] else 'candidate',
                    'stable_primary_vs_A_and_B':None if method in ['A_uniform','B_stage0'] else all(flags),
                    'stable_scenarios_vs_A_and_B':[s for s in reg['scenarios'] if all(ci.get((s,method,base),{}).get('stable_advantage',False) for base in ['A_uniform','B_stage0'])],
                    'stable_matched_information_increment':None if matching is None else ci.get(('equal_scenario_mean',method,matching),{}).get('stable_advantage',False)})
    passed=not errors and (not final or completed==3 and total==reg['expected_conditions'])
    output={'passed':passed,'completion_proven':bool(final and passed),'scope':'final_completion' if final else 'progress_diagnostics','completed_batches':completed,
        'saved_conditions':total,'expected_conditions':reg['expected_conditions'],'errors':errors,'batches':batch_reports,
        'candidate_claims':claims,'all_original_27_retained':reg['candidates']==CANDIDATES,'main_quality':'pooled only'}
    (root/('completion_evidence_audit.json' if final else 'progress_evidence_audit.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registration');p.add_argument('--final',action='store_true');a=p.parse_args()
    r=audit(a.registration,a.final);print(json.dumps(r,ensure_ascii=False))
    if not r['passed']:raise SystemExit(1)
