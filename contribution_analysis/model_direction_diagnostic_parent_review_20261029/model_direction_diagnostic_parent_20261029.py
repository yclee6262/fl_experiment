"""Independent observation collision and exact-probe deduplication audit."""
import json,hashlib,math,collections,time
from pathlib import Path
W=Path(__file__).resolve().parents[1]
ROOT=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
OUT=W/'contribution_analysis/model_direction_diagnostic_parent_review_20261029'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    OUT.mkdir(exist_ok=False);begin=time.perf_counter();receipt=json.loads((ROOT/'completion_receipt.json').read_text())
    for rel,h in receipt['file_sha256'].items():assert sha(ROOT/rel)==h
    groups=collections.defaultdict(list);checks=[];rounds=0;original_duplicates=0;changed_rounds=0;max_change=0
    fields=['full_result','full','branches','directions','rows','probe','scale','events','iterations']
    for p in sorted((ROOT/'cases').glob('*.json')):
        r=json.loads(p.read_text());c=r['fixture'];obs={k:r[k] for k in fields}
        obs['inputs']={k:c[k] for k in ['models','weights','owners','anchors','anchor_weights','start']}
        signature=hashlib.sha256(canonical(obs).encode()).hexdigest()
        diag=json.loads((ROOT/'diagnostics'/p.name).read_text())['rows']
        groups[signature].append(dict(id=c['id'],diagnostics=diag,observation=obs))
        for t,d in enumerate(r['directions']):
            rounds+=1;g=d['g'];keys=[canonical({k:a[k] for k in ['point','h','unit']}) for a in d['directions']]
            original_duplicates+=len(keys)-len(set(keys));changed=False
            def calculate(indices):
                return [min(math.sqrt(sum((g[i][k]-g[j][k])**2 for k in indices)/len(indices)) for j in range(len(g)) if j!=i) for i in range(len(g))]
            def unique(indices):
                result=[];seen=set()
                for i in indices:
                    if keys[i] not in seen:result.append(i);seen.add(keys[i])
                return result
            original=list(range(len(keys)));baseline=calculate(original);dedup=calculate(unique(original))
            for a,b in zip(baseline,dedup):assert abs(a-b)<1e-10
            for k in original:
                for times in [1,3,9]:
                    indices=original+[k]*times;raw=calculate(indices);fixed=calculate(unique(indices))
                    error=max(abs(a-b) for a,b in zip(raw,baseline));max_change=max(max_change,error);changed|=error>1e-10
                    assert max(abs(a-b) for a,b in zip(fixed,dedup))<1e-10
                    reverse=calculate(unique(list(reversed(indices))))
                    assert max(abs(a-b) for a,b in zip(reverse,dedup))<1e-10
                    checks.append(dict(case=c['id'],round=t,anchor_column=k,extra_copies=times,raw_change=error,dedup_pass=True,reverse_order_pass=True))
            changed_rounds+=changed
    conflict=[];label_fit=total=0
    for signature,gg in groups.items():
        assert all(x['observation']==gg[0]['observation'] for x in gg)
        for o in [x['owner'] for x in gg[0]['diagnostics']]:
            rows=[dict(case=x['id'],D=d['D']) for x in gg for d in x['diagnostics'] if d['owner']==o]
            counts=collections.Counter(1 if x['D']>1e-10 else -1 if x['D']< -1e-10 else 0 for x in rows)
            total+=len(rows);label_fit+=max(counts.values())
            if len(counts)>1:conflict.append(dict(signature=signature,owner=o,rows=rows))
    result=dict(cases=sum(map(len,groups.values())),observation_groups=len(groups),conflicting_owner_groups=len(conflict),owner_cases=total,finite_dataset_best_label_fit=label_fit,conflicts=conflict,rounds=rounds,original_duplicate_columns=original_duplicates,duplicate_variants=len(checks),changed_raw_rounds=changed_rounds,max_raw_change=max_change,checks=checks,solver_calls=0,model_calls=0,seconds=time.perf_counter()-begin)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(input_receipt_sha256=sha(ROOT/'completion_receipt.json'),source_sha256=sha(Path(__file__)),scope='independent exact observables and frozen-feature duplicate/permutation audit; no new validation sample'),indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['conflicts','checks']})
if __name__=='__main__':run()
