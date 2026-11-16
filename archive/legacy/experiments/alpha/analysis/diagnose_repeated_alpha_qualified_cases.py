"""Posthoc qualified-case diagnostic, never rewrites registered conclusions.

Only common upstream qualification failures may be excluded. Differential
method failures prevent this diagnostic from being used as replication evidence.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from analyze_repeated_alpha_confirmation import contrasts,family_bootstrap,stable,valid
from repeated_alpha_candidates import METHODS
from research_analysis import paired_bootstrap_mean_ci

def shared_qualification_failure(rows):
    return len(rows)==len(METHODS) and all(r.get('status')=='qualification_failure' for r in rows) and len({json.dumps(r.get('reason'),sort_keys=True) for r in rows})==1

def diagnose(registration):
    reg=json.loads(Path(registration).read_text());root=Path(reg['output']);formal_path=root/'final_analysis.json'
    formal_bytes=formal_path.read_bytes();formal=json.loads(formal_bytes)
    output=root/'qualified_case_diagnostic';output.mkdir(exist_ok=False)
    index={};excluded=[]
    for bi,b in enumerate(reg['batches']):
        for line in (Path(b['results'])/'results.jsonl').read_text().splitlines():
            row=json.loads(line);index[row['scenario'],row['seed'],row['method']]=row
        for scenario,seed in itertools.product(reg['scenarios'],b['seeds']):
            rows=[index[scenario,seed,m] for m in reg['methods']]
            if shared_qualification_failure(rows):excluded.append({'scenario':scenario,'seed':seed,'batch':bi,'reason':rows[0]['reason']})
            elif not all(valid(r) for r in rows):raise RuntimeError('Differential method failure: cannot silently discard it')
    assert excluded, 'No upstream failure to diagnose'
    original={(c['endpoint'],c['method'],c['baseline']):c for c in formal['comparisons']}
    records=[];columns=[];vectors=[]
    # Registered CI computations are reused where full planned pairs existed.
    for endpoint,(method,baseline) in itertools.product(['equal_scenario_mean']+reg['scenarios'],contrasts()):
        sc=reg['scenarios'] if endpoint=='equal_scenario_mean' else [endpoint]
        differences=[];batches=[];omitted=[]
        for bi,b in enumerate(reg['batches']):
            ds=[]
            for seed in b['seeds']:
                if any(any(e['scenario']==s and e['seed']==seed for e in excluded) for s in sc):omitted.append(seed);continue
                ds.append(float(np.mean([index[s,seed,method]['pooled_error']-index[s,seed,baseline]['pooled_error'] for s in sc])))
            differences.extend(ds);batches.append({'batch':bi,'n':len(ds),'mean_delta':float(np.mean(ds)), 'CI95':paired_bootstrap_mean_ci(ds)})
        key=(endpoint,method,baseline);registered=original[key]
        c={'endpoint':endpoint,'method':method,'baseline':baseline,'n':len(differences),'mean_delta':float(np.mean(differences)),
           'CI95':paired_bootstrap_mean_ci(differences),'CI_simultaneous':registered['CI_simultaneous'],'batches':batches,
           'excluded_common_qualification_seeds':omitted,'original_stable_advantage':registered['stable_advantage'],
           'qualified_case_replication_signal':False,'formal_claim_allowed':False}
        if c['CI_simultaneous'] is None:columns.append(len(records));vectors.append(differences)
        records.append(c)
    intervals=family_bootstrap(np.asarray(vectors).T,reg['bootstrap_replicates'],1-reg['family_error']/reg['comparison_family'],reg['analysis_rng'],output/'bootstrap_means.npy')
    for j,ci in zip(columns,intervals):records[j]['CI_simultaneous']=ci
    for c in records:c['qualified_case_replication_signal']=stable(c['CI_simultaneous'],c['batches'],True,True)
    assert formal_path.read_bytes()==formal_bytes, 'Registered analysis must remain unchanged'
    result={'kind':'posthoc_shared_qualification_diagnostic','formal_claim_allowed':False,
        'interpretation':'Effect conditional on common qualification; not the originally registered complete-pair confirmation',
        'original_analysis_sha256':hashlib.sha256(formal_bytes).hexdigest(),'source_registration':registration,
        'excluded_common_upstream_cases':excluded,'all29_methods_retained':True,'comparisons':records,
        'bootstrap_replicates':reg['bootstrap_replicates'],'comparison_family':reg['comparison_family'],
        'parameters_changed':False,'raw_results_deleted':False}
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'common_qualification_failures':excluded,'replication_signals':[(c['endpoint'],c['method'],c['baseline']) for c in records if c['qualified_case_replication_signal']], 'formal_claim_allowed':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registration');a=p.parse_args();diagnose(a.registration)
