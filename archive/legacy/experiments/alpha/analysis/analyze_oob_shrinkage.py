"""Paired seed-level OOB shrinkage analysis; six confirmatory A/B contrasts."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_bootstrap_quality_research import read, summary, valid, verify
from bootstrap_quality_research import dump, KEYS, digest
from research_analysis import paired_bootstrap_mean_ci

PRIMARY = "bootstrap_oob_quality_shrink"


def analyze(root, registration=None):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    rows=read(root/'results.jsonl');checked=verify(root,manifest,rows)
    if registration:
        reg=json.loads(Path(registration).read_text())
        for field in ('seeds','methods','scenarios','targets','source_sha256'):
            if manifest[field]!=reg[field]:checked['errors'].append('registration mismatch '+field)
        if manifest['sizes']!=[20] or not manifest['original_blind']:
            checked['errors'].append('registration blind data mismatch')
        source_verification=Path(manifest['source'])/'verification.json'
        if not source_verification.exists() or not json.loads(source_verification.read_text())['passed']:
            checked['errors'].append('source preparation not verified')
        checked['passed']=checked['passed'] and not checked['errors']
        dump(root/'verification.json',checked)
    index={tuple(r[k] for k in KEYS):r for r in rows}
    groups=[];comparisons=[];conflicts=[];cases=[]
    for scenario,n,target in itertools.product(manifest['scenarios'],manifest['sizes'],manifest['targets']):
        for method in manifest['methods']:
            rs=[r for r in rows if r['scenario']==scenario and r['blind_n']==n and r['target']==target and r['method']==method]
            vs=[r for r in rs if valid(r)]
            states={s:sum(r['status']==s for r in rs) for s in sorted({r['status'] for r in rs})}
            g={'scenario':scenario,'n':n,'target':target,'method':method,'runs':len(rs),'statuses':states,
               'failures':len(rs)-len(vs),'pooled':summary([r['pooled_error'] for r in vs]),
               'truth':summary([r['truth_error'] for r in vs]),
               'fallbacks':sum(bool(r.get('alpha_audit',{}).get('fallback_reason')) for r in rs)}
            for field in ['stage3_requests','stage3_input_samples','stage3_seconds','initialization_seconds','initialization_requests','initialization_input_samples','scoring_seconds']:
                g[field]=summary([r[field] for r in rs if r.get(field) is not None])
            if method==PRIMARY:
                g['lambda_mean']=summary([r['alpha_audit']['lambda_mean'] for r in rs if 'lambda_mean' in r.get('alpha_audit',{})])
                g['accepted_replicates']=summary([r['alpha_audit']['accepted_replicates'] for r in rs if 'accepted_replicates' in r.get('alpha_audit',{})])
                g['skipped_small_oob']=sum(r.get('alpha_audit',{}).get('skipped_small_oob',0) for r in rs)
            groups.append(g)
        for baseline in [m for m in manifest['methods'] if m!=PRIMARY]:
            pairs=[];excluded=[];nf=nb=0
            for seed in manifest['seeds']:
                aa=[index.get((scenario,seed,draw,n,target,PRIMARY)) for draw in manifest['draws']]
                bb=[index.get((scenario,seed,draw,n,target,baseline)) for draw in manifest['draws']]
                nf+=sum(r is None or not valid(r) for r in aa)
                nb+=sum(r is None or not valid(r) for r in bb)
                if not all(r is not None and valid(r) for r in aa+bb):
                    excluded.append({'seed':seed,'reason':'missing_or_invalid_pair'});continue
                delta=float(np.mean([a['pooled_error']-b['pooled_error'] for a,b in zip(aa,bb)]))
                truth_delta=float(np.mean([a['truth_error']-b['truth_error'] for a,b in zip(aa,bb)]))
                pairs.append({'seed':seed,'pooled_delta':delta,'truth_delta':truth_delta})
                if delta*truth_delta<0:conflicts.append({'scenario':scenario,'target':target,'baseline':baseline,**pairs[-1]})
            values=[p['pooled_delta'] for p in pairs]
            ci=paired_bootstrap_mean_ci(values)
            confirm=bool(registration and target==0 and baseline in ['A_uniform','B_stage0'])
            simultaneous=paired_bootstrap_mean_ci(values,confidence=1-.05/6) if confirm else (None,None)
            c={'scenario':scenario,'target':target,'n':n,'baseline':baseline,'n_seeds':len(pairs),
               'mean_delta':float(np.mean(values)) if values else None,'CI95':ci,
               'CI_simultaneous_6':simultaneous,'confirmatory':confirm,
               'method_failures':nf,'baseline_failures':nb,'excluded':excluded,
               'better_seeds':sum(v<0 for v in values),'worse_seeds':sum(v>0 for v in values),
               'seed_pairs':pairs,'confirmed_advantage':bool(confirm and checked['passed'] and not excluded
                and simultaneous[1] is not None and simultaneous[1]<0 and nf<=nb)}
            comparisons.append(c)
            if pairs:
                ps=sorted(pairs,key=lambda p:p['pooled_delta'])
                for label,p in [('best',ps[0]),('median',ps[len(ps)//2]),('worst',ps[-1])]:
                    cases.append({'scenario':scenario,'target':target,'baseline':baseline,'case':label,**p})
    result={'verification':checked,'phase':'formal' if registration else 'pilot',
            'summaries':groups,'comparisons':comparisons,'conflicts':conflicts,'representative_cases':cases,
            'statistical_unit':'seed; nested blind draws averaged within seed',
            'confidence_rule':'10,000 paired seed bootstrap; main six Bonferroni contrasts; analysis RNG=20260924'}
    dump(root/'analysis.json',result)
    # Full rows and paired differences are also readable without JSON tooling.
    with (root/'results.csv').open('w',newline='') as f:
        fields=sorted({k for r in rows for k in r});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    with (root/'paired_differences.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scenario','target','baseline','seed','pooled_delta','truth_delta']);writer.writeheader()
        for c in comparisons:
            for p in c['seed_pairs']:writer.writerow({'scenario':c['scenario'],'target':c['target'],'baseline':c['baseline'],**p})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--registration')
    args=parser.parse_args();result=analyze(args.root,args.registration)
    print(json.dumps({'passed':result['verification']['passed'],'errors':result['verification']['errors'],
                      'conditions':result['verification']['observed'],
                      'confirmed':sum(c['confirmed_advantage'] for c in result['comparisons'])}))
    if not result['verification']['passed']:raise SystemExit(1)
