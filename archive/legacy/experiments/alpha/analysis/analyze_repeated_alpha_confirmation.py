"""Frozen multi-batch evidence, all candidates retained regardless of rank."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from repeated_alpha_candidates import CANDIDATES,DEFINITIONS
from research_analysis import paired_bootstrap_mean_ci
from oob_shrinkage_research import dump

def contrasts():
    return [('B_stage0','A_uniform')]+list(itertools.product([m for m in CANDIDATES if m not in ['A_uniform','B_stage0']],['A_uniform','B_stage0']))+[
        ('proposal_local_9','B_augmented29'),('local_bias_risk_100','B_augmented100')]

def valid(r):
    if not r or r.get('status')!='valid':return False
    x=np.asarray(r.get('X'),float)
    return x.shape==(2,) and np.isfinite(x).all() and max(abs(x))<=1+1e-8 and r.get('pooled_error') is not None and np.isfinite(r['pooled_error'])

def stable(ci,batch_stats,complete,failures_ok):
    return bool(complete and failures_ok and ci[1] is not None and ci[1]<0 and len(batch_stats)==3
        and all(b['mean_delta'] is not None and b['mean_delta']<0 for b in batch_stats)
        and sum(b['CI95'][1] is not None and b['CI95'][1]<0 for b in batch_stats)>=2)

def family_bootstrap(matrix,replicates,confidence,rng_seed,path):
    """Shared seed resampling; no scenario treated as an independent replication."""
    matrix=np.asarray(matrix,float);n,p=matrix.shape
    if not n or not p:return []
    rng=np.random.default_rng(rng_seed)
    mem=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(replicates,p))
    for start in range(0,replicates,1000):
        size=min(1000,replicates-start)
        weights=rng.multinomial(n,np.full(n,1/n),size=size).astype(float)
        mem[start:start+size]=weights@matrix/n
    mem.flush();tail=(1-confidence)/2
    return np.quantile(mem,[tail,1-tail],axis=0).T.tolist()

def analyze(registration,final=False):
    reg=json.loads(Path(registration).read_text());root=Path(reg['output']);root.mkdir(exist_ok=True)
    index={};completed=[];errors=[]
    for bi,b in enumerate(reg['batches']):
        r=Path(b['results'])
        if not (r/'verification.json').exists():continue
        v=json.loads((r/'verification.json').read_text())
        if not v['passed']:errors.append('batch verification failed '+str(bi));continue
        completed.append(bi)
        for line in (r/'results.jsonl').read_text().splitlines():
            row=json.loads(line);key=(row['scenario'],row['seed'],row['method'])
            if key in index:errors.append('duplicate cross-batch key')
            index[key]=row
    if final and len(completed)!=3:errors.append('three batches incomplete')
    pairlist=contrasts();endpoints=['equal_scenario_mean']+reg['scenarios'];records=[];complete_columns=[];vectors=[]
    allseeds=[s for i in completed for s in reg['batches'][i]['seeds']]
    for endpoint,(method,baseline) in itertools.product(endpoints,pairlist):
        scenarios=reg['scenarios'] if endpoint=='equal_scenario_mean' else [endpoint]
        differences=[];excluded=[];batch_stats=[];fail_a=fail_b=0
        for bi in completed:
            ds=[]
            for seed in reg['batches'][bi]['seeds']:
                a=[index.get((s,seed,method)) for s in scenarios];b=[index.get((s,seed,baseline)) for s in scenarios]
                fail_a+=sum(not valid(r) for r in a);fail_b+=sum(not valid(r) for r in b)
                if not all(valid(r) for r in a+b):
                    excluded.append({'seed':seed,'batch':bi,'method_status':[None if r is None else r['status'] for r in a],
                        'baseline_status':[None if r is None else r['status'] for r in b]});continue
                ds.append(float(np.mean([ra['pooled_error']-rb['pooled_error'] for ra,rb in zip(a,b)])))
            differences.extend(ds)
            batch_stats.append({'batch':bi,'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,
                                'CI95':paired_bootstrap_mean_ci(ds)})
        ci=paired_bootstrap_mean_ci(differences)
        record={'endpoint':endpoint,'method':method,'baseline':baseline,'n':len(differences),
            'mean_delta':float(np.mean(differences)) if differences else None,'CI95':ci,'CI_simultaneous':None,
            'batches':batch_stats,'excluded':excluded,'method_failures':fail_a,'baseline_failures':fail_b,
            'pooled_win_fraction':sum(d<0 for d in differences)/len(differences) if differences else None,
            'median_delta':float(np.median(differences)) if differences else None,
            'worst_delta':max(differences) if differences else None,'stable_advantage':False,'classification':'pending_confirmation'}
        if final and not errors and len(differences)==len(allseeds):
            complete_columns.append(len(records));vectors.append(differences)
        elif final:record['classification']='incomplete_or_failed_pairs_evidence_insufficient'
        records.append(record)
    if final and vectors and not errors:
        intervals=family_bootstrap(np.asarray(vectors).T,reg['bootstrap_replicates'],1-reg['family_error']/reg['comparison_family'],reg['analysis_rng'],root/'bootstrap_means.npy')
        for j,ci in zip(complete_columns,intervals):
            c=records[j];c['CI_simultaneous']=ci
            c['stable_advantage']=stable(ci,c['batches'],not c['excluded'],c['method_failures']<=c['baseline_failures'])
            c['classification']=('stable_advantage' if c['stable_advantage'] else
                'combined_advantage_not_replicated' if ci[1]<0 else
                'potential_but_evidence_insufficient' if c['mean_delta']<0 else 'no_advantage_established')
    groups=[]
    for method in reg['methods']:
        for endpoint in endpoints:
            sc=reg['scenarios'] if endpoint=='equal_scenario_mean' else [endpoint];values=[];failure=0
            for seed in allseeds:
                rs=[index.get((s,seed,method)) for s in sc]
                if all(valid(r) for r in rs):values.append(float(np.mean([r['pooled_error'] for r in rs])))
                else:failure+=1
            groups.append({'method':method,'label':DEFINITIONS[method][0],'endpoint':endpoint,'n':len(values),'failures':failure,
                'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None})
    result={'status':'final' if final and not errors else 'interim','errors':errors,'completed_batches':completed,
        'groups':groups,'comparisons':records,'confirmation_rule':reg['stability_rule'],'all_candidates_retained':True}
    dump(root/('final_analysis.json' if final else 'interim_analysis.json'),result)
    print(json.dumps({'status':result['status'],'completed_batches':completed,'errors':errors,'stable_contrasts':sum(c['stable_advantage'] for c in records)}))
    if errors:raise RuntimeError(errors)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registration');p.add_argument('--final',action='store_true');a=p.parse_args();analyze(a.registration,a.final)
