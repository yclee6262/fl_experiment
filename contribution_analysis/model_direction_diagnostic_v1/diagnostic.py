"""Read-only archived-data diagnostics. No solver or evaluator calls."""
import copy, hashlib, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
from contribution_analysis.model_direction_pool_v1.run import allocation
W=Path(__file__).resolve().parents[2]
SOURCE=W/'contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02'
def canon(x):
    if isinstance(x,float):
        if not math.isfinite(x): raise ValueError('nonfinite')
        return {'__float64__':x.hex()}
    if isinstance(x,dict):return {k:canon(v) for k,v in sorted(x.items())}
    if isinstance(x,list):return [canon(v) for v in x]
    return x
def encoded(x):return json.dumps(canon(x),sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obs(r):
    f={k:r['fixture'][k] for k in ['models','weights','owners','anchors','anchor_weights','start','iterations']}
    return dict(fixture=f,**{k:r[k] for k in ['iterations','full_result','full','branches','directions','rows','probe','scale','events']})
def differences(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,left=a,right=b)]
    if isinstance(a,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            p=path+'/'+k
            out+=differences(a[k],b[k],p) if k in a and k in b else [dict(path=p,left=a.get(k),right=b.get(k),missing_left=k not in a,missing_right=k not in b)]
        return out
    if isinstance(a,list):
        out=[]
        for i in range(max(len(a),len(b))):
            p=path+'/'+str(i)
            out+=differences(a[i],b[i],p) if i<len(a) and i<len(b) else [dict(path=p,left=a[i] if i<len(a) else None,right=b[i] if i<len(b) else None,missing_left=i>=len(a),missing_right=i>=len(b))]
        return out
    return [] if encoded(a)==encoded(b) else [dict(path=path,left=a,right=b)]
def geometry(d):return encoded({k:d[k] for k in ['point','h','unit']})
def u_value(g):
    g=np.asarray(g,dtype=float)
    if g.shape[1]==0:return None
    dist=np.sqrt(np.mean((g[:,None,:]-g[None,:,:])**2,axis=2));np.fill_diagonal(dist,np.inf)
    return np.min(dist,axis=1).tolist()
def dedup(ds,g):
    if not ds:
        if any(len(row) for row in g):raise ValueError('dimension_mismatch')
        return None,[]
    matrix=np.asarray(g,dtype=float)
    if matrix.ndim!=2 or matrix.shape[1]!=len(ds) or matrix.shape[0]<2:raise ValueError('dimension_mismatch_or_no_peer')
    if not np.isfinite(matrix).all():raise ValueError('nonfinite_g')
    unique={}
    for j,d in enumerate(ds):
        key=geometry(d);column=[row[j] for row in g]
        if key in unique and encoded(unique[key])!=encoded(column):raise ValueError('same_geometry_inconsistent_g')
        unique[key]=column
    keys=sorted(unique)
    return u_value(np.array([unique[k] for k in keys]).T),keys

def ranking(u):return [[int(j) for j,v in enumerate(u) if v==value] for value in sorted(set(u),reverse=True)] if u is not None else None
def maxerr(a,b):return 0. if a is None and b is None else max(abs(x-y) for x,y in zip(a,b))
def sign(x):return 1 if x>1e-12 else -1 if x< -1e-12 else 0
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(out):
    started=time.perf_counter();out.mkdir(exist_ok=False)
    for d in ['observations','pairs','directions']:(out/d).mkdir()
    inputs={str(p.relative_to(W)):digest(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
    sources=[Path(__file__),Path(__file__).with_name('test_diagnostic.py'),W/'docs/model_direction_diagnostic_design_20261029.md',W/'contribution_analysis/model_direction_pool_v1/run.py',W/'contribution_analysis/model_consensus_pool_v1/core.py']
    locks={str(p.relative_to(W)):digest(p) for p in sources};save(out/'source_lock.json',dict(inputs=inputs,sources=locks))
    records={p.stem:json.loads(p.read_text()) for p in sorted((SOURCE/'cases').glob('*.json'))};groups={};observations={};diagnostics={};controls=[];allocations=[];rounds=0
    for name,r in records.items():
        o=obs(r);observations[name]=o;raw=encoded(o);sig=hashlib.sha256(raw.encode()).hexdigest();groups.setdefault(sig,[]).append(name)
        with (out/'observations'/f'{name}.json').open('x') as f:f.write(raw)
        diagnostics[name]={z['owner']:z['D'] for z in json.loads((SOURCE/'diagnostics'/f'{name}.json').read_text())['rows']}
        rows=copy.deepcopy(r['rows']);rd=[]
        for t,d in enumerate(r['directions']):
            rounds+=1
            if d['g'] is None:rd.append(dict(round=t,status=d['status'],controls=[]));continue
            ds=d['directions'];g=d['g'];original=u_value(g);unique,keys=dedup(ds,g);tests=[]
            for target in [None]+list(range(len(ds))):
                for copies in [1,3,9]:
                    idx=[j for j in range(len(ds)) for _ in range(copies if target is None or target==j else 1)]
                    dd=[ds[j] for j in idx];gg=[[row[j] for j in idx] for row in g]
                    raw_u=u_value(gg);de_u,_=dedup(dd,gg);rev_u,_=dedup(dd[::-1],[row[::-1] for row in gg])
                    assert encoded(unique)==encoded(de_u)==encoded(rev_u)
                    assert ranking(unique)==ranking(de_u)==ranking(rev_u)
                    tests.append(dict(target=target,copies=copies,raw_U=raw_u,dedup_U=de_u,raw_error=maxerr(original,raw_u),dedup_error=maxerr(unique,de_u),rank_equal=ranking(unique)==ranking(de_u),raw_ranking=ranking(raw_u),dedup_ranking=ranking(de_u)))
            for row in rows:
                if row['round']==t:
                    U=unique[row['model']];row['U']=U;row['N']=(.1*U)/(1+.1*U)*row['I']/(1+row['I'])
            rd.append(dict(round=t,status=d['status'],columns=len(ds),unique_columns=len(keys),original_U=original,dedup_U=unique,base_error=maxerr(original,unique),controls=tests,geometry_keys=keys))
        save(out/'directions'/f'{name}.json',rd);controls.extend(dict(case=name,**z) for z in rd)
        args=dict(owners=r['fixture']['owners'],rounds=range(len(r['full'])))
        old=allocation(r['rows'],.2,'N',**args);new=allocation(rows,.2,'N',**args)
        allocations.append(dict(case=name,original=old,dedup=new,amounts_equal=old['amounts']==new['amounts'],reserve_equal=old['reserve']==new['reserve']))
    group_rows=[];pair_rows=[]
    for sig,names in sorted(groups.items()):
        owners={o:dict(D={n:diagnostics[n][o] for n in names},signs=sorted({sign(diagnostics[n][o]) for n in names})) for o in records[names[0]]['fixture']['owners']}
        for z in owners.values():z['opposite_sign_conflict']= -1 in z['signs'] and 1 in z['signs']
        group_rows.append(dict(signature=sig,cases=names,owners=owners))
        for a,b in itertools.combinations(names,2):
            diff=differences(observations[a],observations[b]);assert not diff
            row=dict(left=a,right=b,same_observation=True,differences=diff,reference_equal=records[a]['fixture']['reference']==records[b]['fixture']['reference'],D_left=diagnostics[a],D_right=diagnostics[b]);pair_rows.append(row)
    # Explicit semantic pairs remain visible even if exact equality fails.
    for a in records:
        b=a.replace('beneficial_minority','wrong_minority').replace('useful_restraint','false_restraint')
        if a==b or b not in records:continue
        diff=differences(observations[a],observations[b]);save(out/'pairs'/f'{a}__{b}.json',dict(left=a,right=b,same_observation=not diff,differences=diff,reference_equal=records[a]['fixture']['reference']==records[b]['fixture']['reference'],D_left=diagnostics[a],D_right=diagnostics[b]))
    save(out/'groups.json',group_rows);save(out/'identical_pairs.json',pair_rows);save(out/'allocations.json',allocations)
    trials=[x for z in controls for x in z['controls']];valid=[z for z in controls if 'columns' in z]
    summary=dict(cases=len(records),rounds=rounds,observation_groups=len(groups),nontrivial_groups=sum(len(v)>1 for v in groups.values()),identical_pairs=len(pair_rows),owner_groups_with_opposite_D=sum(z['opposite_sign_conflict'] for r in group_rows for z in r['owners'].values()),conflict_by_owner={o:sum(r['owners'].get(o,{}).get('opposite_sign_conflict',False) for r in group_rows) for o in sorted({o for r in group_rows for o in r['owners']})},available_rounds=len(valid),rounds_with_existing_duplicate_geometry=sum(z['columns']>z['unique_columns'] for z in valid),existing_dedup_U_changed_rounds=sum(z['base_error']>1e-12 for z in valid),replication_controls=len(trials),raw_changed_controls=sum(z['raw_error']>1e-12 for z in trials),raw_max_error=max(z['raw_error'] for z in trials),dedup_max_error=max(z['dedup_error'] for z in trials),allocation_changed_cases=sum(not z['amounts_equal'] or not z['reserve_equal'] for z in allocations),new_solver_calls=0,new_model_calls=0,new_reference_calls=0,seconds=time.perf_counter()-started,all_data_exploratory=True)
    save(out/'summary.json',summary)
    assert all(digest(W/p)==h for p,h in inputs.items());assert all(digest(W/p)==h for p,h in locks.items())
    outputs={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'completion_receipt.json',dict(status='complete',input_files=len(inputs),source_files=len(locks),output_files=len(outputs),output_sha256=outputs,bytes_before_receipt=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),summary=summary))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else W/'contribution_analysis/model_direction_diagnostic_20261029')
