"""Analyze historical-cohort numerical revision with seed-paired intervals."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
LABELS = dict(A_uniform='U', B_stage0='Q', stacked_full='E', pure_support='S', ensemble_shrink='EU', B_stacked_half='EQ', peer_support_shrink='SU', support_quality_mix50='SQ', support_ensemble_mix50='SE', quality_shrink='QU')
SCENARIOS = ['reference_clean','low_quality_25pct','sample_imbalance_4x']
REPLICATES = 200000
RNG_SEED = 20261123


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')
def write_csv(path, data):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def indexed(data):
    result={}
    for r in data:
        key=(r['scenario'],r['seed'],r['method'])
        if key in result: raise ValueError(f'Duplicate {key}')
        result[key]=r
    return result


def valid(r):
    return r['status']=='valid' and r.get('X') is not None and np.isfinite(r['X']).all() and max(abs(np.asarray(r['X'])))<=1+1e-8 and r.get('pooled_error') is not None and np.isfinite(r['pooled_error'])


def seed_matrix(index, seeds, scenarios, endpoint):
    methods=list(LABELS)
    eligible=[seed for seed in seeds if all(valid(index[s,seed,m]) for s,m in itertools.product(scenarios,methods))]
    return eligible,np.array([[np.mean([index[s,k,m][endpoint] for s in scenarios]) for m in methods] for k in eligible])


def bootstrap(matrix, replicates=REPLICATES):
    n=len(matrix)
    if not n: raise ValueError('No common valid seeds')
    rng=np.random.default_rng(RNG_SEED)
    out=np.empty((replicates,matrix.shape[1]))
    for start in range(0,replicates,1000):
        count=min(1000,replicates-start)
        out[start:start+count]=rng.multinomial(n,np.full(n,1/n),size=count)@matrix/n
    return out


def agreement(truth_delta, pooled_delta):
    a=np.sign(np.asarray(truth_delta));b=np.sign(np.asarray(pooled_delta))
    non=(a!=0)&(b!=0)
    return {'n':len(a),'both_non_tie':int(non.sum()),'both_tie':int(((a==0)&(b==0)).sum()),
            'one_tie':int(((a==0)^(b==0)).sum()),'agree_non_tie':int(((a==b)&non).sum()),
            'conflict_non_tie':int(((a!=b)&non).sum()),
            'agreement_non_tie':float((a[non]==b[non]).mean()) if non.any() else None}


def analyze(input_root, output):
    if output.exists(): raise FileExistsError(output)
    ext=read(ROOT/'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json')
    formal=read(ROOT/'alpha_weight_outputs/four_alpha_v1_20261014/formal_registration.json')
    hashes={};new=[];old=[];seeds=[]
    for i,b in enumerate(ext['batches'][:3],1):
        p=input_root/f'batch_{i}'/'results.jsonl'; hashes[str(p.resolve())]=sha(p)
        manifest=read(p.parent/'manifest.json')
        if manifest['status']!='complete' or manifest['solver_policy']!='float32_slsqp_v1': raise ValueError('Unfinished or wrong policy batch')
        hashes[str((p.parent/'manifest.json').resolve())]=sha(p.parent/'manifest.json')
        new+=rows(p);seeds+=b['seeds']
        for kind in ['six_method_results','pure_results','results']:
            p=ROOT/b[kind]/'results.jsonl';hashes[str(p)]=sha(p);old+=rows(p)
    ni,oi=indexed(new),indexed(old)
    for r in old:
        if valid(r):
            x=np.asarray(r['X']);v=float(abs(x.sum()+np.sum(x[:-1]*x[1:])))
            if r.get('truth_error') is not None and abs(v-r['truth_error'])>1e-12: raise ValueError('Historical truth mismatch')
            r['truth_error']=v
    expected=set(itertools.product(SCENARIOS,seeds,LABELS))
    if set(ni)!=expected or set(oi)!=expected or len(expected)!=2700: raise ValueError('Coverage mismatch')
    for name,h in ext['original_results_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError(f'Historical hash mismatch: {name}')
    for key,r in ni.items():
        if r['alpha']!=oi[key]['alpha']: raise ValueError(f'Changed alpha {key}')
        if r['solver_policy']['name']!='float32_slsqp_v1': raise ValueError('Wrong row policy')
        if valid(r):
            x=np.asarray(r['X']);truth=float(abs(x.sum()+np.sum(x[:-1]*x[1:])))
            if r.get('truth_error') is not None and abs(truth-r['truth_error'])>1e-12: raise ValueError(f'Truth mismatch {key}')
            r['truth_error']=truth
    output.mkdir(parents=True)
    with (output/'derived_records.jsonl').open('w') as f:
        for row in new: f.write(json.dumps(row,allow_nan=False)+'\n')
    metadata=[]
    for i in range(1,4):
        source=ROOT/f'alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_{i}/source'
        for scenario in SCENARIOS:
            for p in (source/scenario/'artifacts').glob('*/metadata.json'):
                d=read(p);mt=d['metrics'];hashes[str(p)]=sha(p)
                metadata.append(dict(batch=i,scenario=scenario,seed=d['scenario']['seed'],test_truth_mse=mt['test_truth_mse'],test_truth_rmse=float(np.sqrt(mt['test_truth_mse'])),test_samples=mt['test_samples'],target_near_truth_mse=mt['target_near_truth_mse'],target_near_truth_rmse=float(np.sqrt(mt['target_near_truth_mse'])),target_near_samples=mt['target_near_samples']))
    if len(metadata)!=270: raise ValueError('Expected 270 evaluator metadata records')
    write_csv(output/'evaluator_quality.csv',metadata)
    methods=list(LABELS);families={'original_six_12':formal['contrasts'],'extension_17':ext['contrasts'],
        'exploratory_all_45':list(itertools.combinations(methods,2))}
    groups=[];comparisons=[];matrices={};boots={}
    for endpoint in ['truth_error','pooled_error']:
        for scenario in ['equal_scenario_mean']+SCENARIOS:
            ss=SCENARIOS if scenario=='equal_scenario_mean' else [scenario]
            eligible,mat=seed_matrix(ni,seeds,ss,endpoint);boot=bootstrap(mat)
            matrices[endpoint,scenario]=mat;boots[endpoint,scenario]=boot
            for j,m in enumerate(methods):
                selected=[ni[s,k,m] for s,k in itertools.product(ss,eligible)]
                groups.append(dict(endpoint=endpoint,scenario=scenario,method=LABELS[m],n=len(eligible),excluded=len(seeds)-len(eligible),mean=float(mat[:,j].mean()),sd=float(mat[:,j].std(ddof=1)),ci95_low=float(np.quantile(boot[:,j],.025)),ci95_high=float(np.quantile(boot[:,j],.975)),
                    mean_stage3_requests=float(np.mean([r['stage3_requests'] for r in selected])),mean_initialization_requests=float(np.mean([r['initialization_requests'] for r in selected])),
                    solver_success_n=sum(bool(r['solver'].get('success')) for r in selected),delivered_n=len(selected)))
            for family,pairs in families.items():
                for m,b in pairs:
                    j,l=methods.index(m),methods.index(b);d=mat[:,j]-mat[:,l];bs=boot[:,j]-boot[:,l];tail=.05/(2*len(pairs))
                    comparisons.append(dict(endpoint=endpoint,scenario=scenario,family=family,contrast=LABELS[m]+' - '+LABELS[b],n=len(eligible),excluded=len(seeds)-len(eligible),mean_delta=float(d.mean()),ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),family_low=float(np.quantile(bs,tail)),family_high=float(np.quantile(bs,1-tail))))
    direction=[]
    for m,b in families['exploratory_all_45']:
        for level in ['case','seed']:
            ds=[];dp=[]
            for seed in seeds:
                t=[];p=[]
                for s in SCENARIOS:
                    if valid(ni[s,seed,m]) and valid(ni[s,seed,b]):
                        t.append(ni[s,seed,m]['truth_error']-ni[s,seed,b]['truth_error']);p.append(ni[s,seed,m]['pooled_error']-ni[s,seed,b]['pooled_error'])
                if level=='case':ds+=t;dp+=p
                elif len(t)==3:ds.append(np.mean(t));dp.append(np.mean(p))
            direction.append(dict(contrast=LABELS[m]+' - '+LABELS[b],level=level,**agreement(ds,dp)))
    correlations=[]
    for m in methods:
        for level in ['case','seed']:
            pairs=[]
            for seed in seeds:
                rr=[ni[s,seed,m] for s in SCENARIOS if valid(ni[s,seed,m])]
                if level=='case':pairs += [(r['truth_error'],r['pooled_error']) for r in rr]
                elif len(rr)==3:pairs.append((np.mean([r['truth_error'] for r in rr]),np.mean([r['pooled_error'] for r in rr])))
            rho=float(spearmanr(np.array(pairs)[:,0],np.array(pairs)[:,1]).statistic)
            correlations.append(dict(method=LABELS[m],level=level,n=len(pairs),spearman=rho if np.isfinite(rho) else None))
    changes=[]
    changed_intervals={}
    for endpoint in ['truth_error','pooled_error']:
        _,newmat=seed_matrix(ni,seeds,SCENARIOS,endpoint)
        _,oldmat=seed_matrix(oi,seeds,SCENARIOS,endpoint)
        diffs=newmat-oldmat
        bs=bootstrap(diffs)
        for j,m in enumerate(methods):
            changed_intervals[m,endpoint]={'mean':float(diffs[:,j].mean()),'low':float(np.quantile(bs[:,j],.025)),'high':float(np.quantile(bs[:,j],.975))}
    for m in methods:
        rr=[(ni[s,k,m],oi[s,k,m]) for s,k in itertools.product(SCENARIOS,seeds) if valid(ni[s,k,m]) and valid(oi[s,k,m])]
        changes.append(dict(method=LABELS[m],n=len(rr),mean_truth_old=float(np.mean([b['truth_error'] for a,b in rr])),mean_truth_new=float(np.mean([a['truth_error'] for a,b in rr])),mean_pooled_old=float(np.mean([b['pooled_error'] for a,b in rr])),mean_pooled_new=float(np.mean([a['pooled_error'] for a,b in rr])),mean_queries_old=float(np.mean([b['stage3_requests'] for a,b in rr])),mean_queries_new=float(np.mean([a['stage3_requests'] for a,b in rr])),changed_X=sum(a['X']!=b['X'] for a,b in rr),truth_paired_delta=changed_intervals[m,'truth_error']['mean'],truth_delta_ci95_low=changed_intervals[m,'truth_error']['low'],truth_delta_ci95_high=changed_intervals[m,'truth_error']['high'],pooled_paired_delta=changed_intervals[m,'pooled_error']['mean'],pooled_delta_ci95_low=changed_intervals[m,'pooled_error']['low'],pooled_delta_ci95_high=changed_intervals[m,'pooled_error']['high']))
    def fallback(x):
        return isinstance(x,dict) and (bool(x.get('fallback_reason')) or any(fallback(v) for v in x.values()))
    status={m:{'valid':sum(valid(r) for r in new if r['method']==m),'failed':sum(not valid(r) for r in new if r['method']==m),'fallbacks':sum(bool(fallback(r.get('alpha_audit'))) for r in new if r['method']==m)} for m in methods}
    audit=dict(passed=True,rows=len(new),seeds=len(seeds),statuses=status,source_sha256=hashes,analysis_sha256=sha(Path(__file__)),design_sha256=sha(Path(__file__).with_name('revised_analysis_design.md')),replicates=REPLICATES,rng=RNG_SEED,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope='Historical-cohort numerical revision; no equivalence margin or independent confirmation; separate family correction per endpoint.')
    for name,data in [('groups',groups),('comparisons',comparisons),('direction_agreement',direction),('endpoint_correlations',correlations),('historical_changes',changes)]:write_csv(output/(name+'.csv'),data)
    dump(output/'audit.json',audit)
    make_plot(output,comparisons)
    return audit


def make_plot(output, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in comparisons if r['endpoint']=='truth_error' and r['scenario']=='equal_scenario_mean' and r['family']=='exploratory_all_45' and r['contrast'].startswith('U - ')]
    fig,ax=plt.subplots(figsize=(7,4.8))
    for y,r in enumerate(chosen):
        # Stored contrast is U - method; reverse for intuitive method - U display.
        ax.plot([-r['family_high'],-r['family_low']],[y,y],color='#9bb8ce',lw=5)
        ax.plot([-r['ci95_high'],-r['ci95_low']],[y,y],color='#235778',lw=2)
        ax.plot(-r['mean_delta'],y,'o',color='#173b52')
    ax.set_yticks(range(len(chosen)),[r['contrast'].split(' - ')[1]+' - U' for r in chosen]);ax.axvline(0,color='gray',ls='--',lw=1)
    ax.set_xlabel('Paired mean difference in synthetic target error')
    ax.set_title('Historical-cohort numerical revision\nThin: 95% CI; thick: 45-contrast Bonferroni CI',fontsize=10)
    fig.tight_layout();fig.savefig(output/'truth_paired_intervals.pdf');fig.savefig(output/'truth_paired_intervals.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.input_root,a.output_dir),default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
