"""Frozen-protocol seed-paired analyses for task-8 benchmark outcomes."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PLAN=Path(__file__).with_name('ANALYSIS.md')
METHODS=['coefficient','custom','best_of','nelder_mead','direct_slsqp']
SIMPLE=['anchor_mean','best_anchor','origin','fedavg_all','fedavg_selected']
RESOURCES=['panel_evaluations','model_queries','requests','input_samples','transmitted_bytes',
           'qualification_payload_bytes','qualification_host_samples','local_inversion_forward_samples',
           'inversion_forward_samples','inversion_backward_steps','training_forward_samples',
           'training_communication_bytes','training_communication_rounds','total_payload_bytes',
           'host_prediction_requests','qualification_host_requests','host_prediction_input_samples',
           'client_inversion_forward_samples_total','qualification_parallel_stages_estimate','search_sequential_panels']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvout(p,rr):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)


def is_valid(row):
    x=row.get('X');err=row.get('truth_error')
    return bool(row.get('status')=='valid' and x is not None and err is not None and
                np.isfinite(x).all() and np.max(np.abs(x))<=1+1e-8 and np.isfinite(err))


def indexed(data):
    out={}
    for row in data:
        key=(row['seed'],row['dimension'],row['rho'],row['method'],row['panel_budget'])
        if key in out:raise ValueError(f'Duplicate result: {key}')
        out[key]=row
    return out


def paired_deltas(index,seeds,dimension,rho,left,right):
    values=[];valid_seeds=[]
    for seed in seeds:
        a=index[seed,dimension,rho,left[0],left[1]]
        b=index[seed,dimension,rho,right[0],right[1]]
        if is_valid(a) and is_valid(b):
            valid_seeds.append(seed);values.append(a['truth_error']-b['truth_error'])
    return valid_seeds,np.asarray(values,dtype=float)


def interval(values,reps=200000,rng_seed=20261123,family=1):
    values=np.asarray(values,dtype=float)
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,ci95_low=None,ci95_high=None,family_low=None,family_high=None)
    rng=np.random.default_rng(rng_seed);samples=np.empty(reps)
    for begin in range(0,reps,2000):
        w=rng.multinomial(n,np.full(n,1/n),size=min(2000,reps-begin))
        samples[begin:begin+len(w)]=w@values/n
    lo,hi=np.quantile(samples,[.025,.975]);fl,fh=np.quantile(samples,[.05/(2*family),1-.05/(2*family)])
    return dict(n=n,mean=float(values.mean()),sd=float(values.std(ddof=1)) if n>1 else None,
                ci95_low=float(lo),ci95_high=float(hi),family_low=float(fl),family_high=float(fh))


def validate_source(source,manifest):
    if manifest.get('smoke'):raise ValueError('Smoke outputs are implementation checks, not the formal benchmark')
    if manifest.get('validation_case'):raise ValueError('Validation outputs are not the formal benchmark')
    if manifest['status']!='complete':raise ValueError('Benchmark is not complete')
    spec=manifest['protocol'];seeds=spec['seeds'];dims=spec['dimensions'];rhos=spec['poison_ratios']
    if len(seeds)!=30 or set(dims)!={2,5,10} or set(rhos)!={0.,.3}:raise ValueError('Unexpected formal conditions')
    if spec['target']!=0.:raise ValueError('v1 analysis requires the declared zero target')
    hashes={str((source/'manifest.json').resolve()):sha(source/'manifest.json')}
    for name,expected in manifest['code_sha256'].items():
        path=ROOT/name
        if sha(path)!=expected:raise ValueError(f'Executable source changed: {name}')
        hashes[str(path)]=expected
    allrows=[];case_metrics=[];trace_count=0
    expected_methods={(m,b) for m in METHODS for b in spec['search']['panel_budgets']}|{(m,0) for m in SIMPLE}
    for seed,d,rho in itertools.product(seeds,dims,rhos):
        directory=source/f'seed{seed}_d{d}_rho{rho:g}'
        complete=read(directory/'completion.json');case=read(directory/'case.json')
        if complete['status']!='complete':raise ValueError(f'Incomplete case {directory}')
        for fname,field in [('results.jsonl','result_sha256'),('case.json','case_sha256')]:
            if sha(directory/fname)!=complete[field]:raise ValueError(f'Case hash changed: {directory/fname}')
            hashes[str((directory/fname).resolve())]=complete[field]
        for fname,field in [('models.pt','models_sha256'),('data.npz','data_sha256')]:
            if sha(directory/fname)!=case[field]:raise ValueError(f'Artifact changed: {directory/fname}')
            hashes[str((directory/fname).resolve())]=case[field]
        rr=rows(directory/'results.jsonl')
        if len(rr)!=20 or {(r['method'],r['panel_budget']) for r in rr}!=expected_methods:raise ValueError('Method coverage mismatch')
        for row in rr:
            if (row['seed'],row['dimension'],row['rho'])!=(seed,d,rho):raise ValueError('Case identity mismatch')
            if is_valid(row):
                x=np.asarray(row['X']);err=float(abs(x.sum()+np.sum(x[:-1]*x[1:])))
                if len(x)!=d or abs(err-row['truth_error'])>1e-12:raise ValueError('Ground-truth mismatch')
                if row['method']=='origin' and err!=0.:raise ValueError('Origin reference is not zero')
            elif row['status']=='valid':raise ValueError('Invalid finite/feasibility status')
            if row['method'] in METHODS+['anchor_mean','best_anchor','origin'] and row.get('trace_path'):
                trace=directory/row['trace_path']
                if sha(trace)!=row['trace_sha256']:raise ValueError('Trace hash mismatch')
                hashes[str(trace.resolve())]=row['trace_sha256'];trace_count+=1
                with gzip.open(trace,'rt') as f:events=json.load(f)
                # The search module's own validator audits all partial model events.
                cap=row.get('actual_assigned_panel_budget',row['panel_budget'])
                if row['panel_evaluations']>cap:raise ValueError('Hard panel budget exceeded')
                if row['model_queries']>cap*case['selected_count']:raise ValueError('Hard model budget exceeded')
            if row['total_payload_bytes']!=row['qualification_payload_bytes']+row.get('transmitted_bytes',0)+row['training_communication_bytes']:
                raise ValueError('Payload accounting mismatch')
        allrows+=rr
        case_metrics.append(dict(seed=seed,dimension=d,rho=rho,selected_count=case['selected_count'],
             exact_origin_truth_reference=0.0,
             anchor_linear_rank=int(np.linalg.matrix_rank(np.asarray(case['anchors']))) if case['anchors'] else 0,
             anchor_affine_rank=int(np.linalg.matrix_rank(np.asarray(case['anchors'])[1:]-np.asarray(case['anchors'])[0])) if len(case['anchors'])>1 else 0,
             blind_test_failures=sum(r.get('reason')=='failed_mse_threshold' for r in case['qualified_reports']),
             inverse_gate_failures=sum(r.get('reason')=='failed_inverse_feasibility' for r in case['qualified_reports']),
             qualification_failed=bool(case['selection_error']),qualification_host_samples=case['qualification_host_samples'],
             qualification_payload_bytes=case['qualification_payload_bytes'],
             local_inversion_forward_samples=case['local_inversion_forward_samples'],
             local_inversion_backward_steps=case['local_inversion_backward_steps'],
             local_training_forward_samples=case['training']['training_forward_samples'],
             local_training_backward_samples=case['training']['training_backward_samples'],
             fedavg_training_forward_samples=case['training']['training_forward_samples'],
             selected_fedavg_training_forward_samples=None if case['fedavg_selected_training'] is None else case['fedavg_selected_training']['training_forward_samples']))
    if len(allrows)!=3600:raise ValueError('Expected 3600 records')
    return allrows,case_metrics,hashes,trace_count


def analyze(source,output):
    if output.exists():raise FileExistsError(output)
    manifest=read(source/'manifest.json');rr,cases,hashes,traces=validate_source(source,manifest)
    spec=manifest['protocol'];seeds=spec['seeds'];settings=spec['analysis'];reps=settings['bootstrap_replicates'];rng=settings['rng']
    ix=indexed(rr);groups=defaultdict(list)
    for row in rr:groups[row['dimension'],row['rho'],row['method'],row['panel_budget']].append(row)
    summaries=[];resources=[];statuses=[];primary=[]
    for (d,rho,method,budget),group in sorted(groups.items()):
        valid=[r for r in group if is_valid(r)];identity=dict(dimension=d,rho=rho,method=method,panel_budget=budget)
        stats=interval([r['truth_error'] for r in valid],reps,rng)
        summaries.append(identity|dict(planned=len(seeds),failed=len(group)-len(valid),optimizer_success=sum(bool(r.get('success')) for r in group),
                        hard_budget_stops=sum(bool(r.get('budget_exhausted')) for r in group))|stats)
        for field in RESOURCES:
            values=[r[field] for r in group if r.get(field) is not None]
            resources.append(identity|dict(resource=field,records=len(values),mean=float(np.mean(values)) if values else None,
                                           sd=float(np.std(values,ddof=1)) if len(values)>1 else None,total=float(np.sum(values)) if values else None))
        for field in ['status','stop_reason']:
            statuses.extend(identity|dict(field=field,value=str(k),count=v) for k,v in sorted(Counter(r.get(field) for r in group).items(),key=lambda x:str(x[0])))
    budget=settings['primary_panel_budget']
    refs=[('best_anchor',0),('anchor_mean',0),('nelder_mead',budget),('fedavg_all',0),('fedavg_selected',0)]
    if settings['primary_family_size']!=30:raise ValueError('Unexpected contrast family')
    for d,rho in itertools.product(spec['dimensions'],spec['poison_ratios']):
        for ref in refs:
            paired,delta=paired_deltas(ix,seeds,d,rho,('best_of',budget),ref)
            primary.append(dict(dimension=d,rho=rho,contrast='best_of - '+ref[0],panel_budget=budget,
                 planned=len(seeds),excluded=len(seeds)-len(paired),valid_seeds=json.dumps(paired),
                 family='primary_30_conditions_and_contrasts',**interval(delta,reps,rng,family=30)))
    output.mkdir(parents=True)
    for name,data in [('groups',summaries),('resources',resources),('statuses',statuses),('primary_comparisons',primary),('case_resources',cases)]:csvout(output/(name+'.csv'),data)
    audit=dict(passed=True,cases=len(cases),records=len(rr),valid=sum(is_valid(r) for r in rr),failed=sum(not is_valid(r) for r in rr),
        source_sha256=hashes,analysis_sha256=sha(Path(__file__)),design_sha256=sha(PLAN),
        trace_files=traces,primary_contrasts=len(primary),bootstrap_replicates=reps,rng=rng,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='Seed-paired known-function T0 benchmark; no matched end-to-end cost, general superiority, privacy or fairness claim')
    dump(output/'audit.json',audit);make_plot(output,summaries)
    print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
    return audit


def make_plot(output,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(best_of='#175a7d',custom='#d28035',coefficient='#4d9277',nelder_mead='#8c65a9',direct_slsqp='#c45050')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for i,rho in enumerate([0.,.3]):
        for j,d in enumerate([2,5,10]):
            ax=axes[i,j]
            for method in METHODS:
                group=sorted([g for g in groups if g['dimension']==d and g['rho']==rho and g['method']==method],key=lambda g:g['panel_budget'])
                valid=[g for g in group if g['mean'] is not None]
                if not valid:continue
                x=[g['panel_budget'] for g in valid];y=[g['mean'] for g in valid]
                ax.plot(x,y,'o-',color=colors[method],label=method,ms=3)
                ax.fill_between(x,[g['ci95_low'] for g in valid],[g['ci95_high'] for g in valid],color=colors[method],alpha=.08)
            for method,style,color in [('best_anchor','--','#555555'),('fedavg_all',':','#222222'),('fedavg_selected','-.','#999999')]:
                g=next(g for g in groups if g['dimension']==d and g['rho']==rho and g['method']==method)
                if g['mean'] is not None:ax.axhline(g['mean'],ls=style,color=color,label=method,lw=1)
            ax.axhline(0,color='#bbbbbb',lw=.6,label='exact origin reference')
            ax.set_title(f'D={d}, incompatible-label fraction={rho:g}',fontsize=9)
            ax.set_xticks([100,200,400]);ax.grid(alpha=.15);ax.set_ylim(bottom=0)
            if i==1:ax.set_xlabel('Assigned model-panel evaluations')
            if j==0:ax.set_ylabel('Mean absolute synthetic target error')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.005))
    fig.suptitle('Paired zero-target synthetic benchmark (30 seeds per condition)',fontsize=12)
    fig.tight_layout(rect=(0,.10,1,.96));fig.savefig(output/'budget_error_curves.pdf');fig.savefig(output/'budget_error_curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.output_dir)
