"""Seed-paired target extension statistics; no independent-model claim."""
import argparse,csv,hashlib,json,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
LABELS=['U','Q','E','S','EU','EQ','SU','SQ','SE','QU']
SCENARIOS=['reference_clean','low_quality_25pct','sample_imbalance_4x']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def bootstrap(x,replicates=200000):
 if x.ndim!=2 or not len(x):raise ValueError("No common valid seed matrix")
 rng=np.random.default_rng(20261123);out=np.empty((replicates,x.shape[1]));n=len(x)
 for i in range(0,replicates,1000):
  count=min(1000,replicates-i);out[i:i+count]=rng.multinomial(n,np.full(n,1/n),size=count)@x/n
 return out

def analyze(source,output):
 manifest=read(source/'manifest.json');assert manifest['status']=='complete' and not manifest['smoke']
 if output.exists():raise FileExistsError(output)
 for f,h in {**manifest['runtime_sha256'],**manifest['input_sha256']}.items():assert sha(ROOT/f)==h,f
 index={};metadata=[];hashes={str(source/'manifest.json'):sha(source/'manifest.json')}
 for f in sorted(source.glob('*/completion.json')):
  directory=f.parent;c=read(directory/'case.json');done=read(f)
  assert done['status']=='complete' and sha(directory/'case.json')==done['case_sha256'] and sha(directory/'results.jsonl')==done['results_sha256']
  rr=lines(directory/'results.jsonl');assert len(rr)==10
  metadata.append(dict(seed=c['seed'],scenario=c['scenario'],target=c['target'],linear_rank=c['linear_rank'],affine_rank=c['affine_rank'],inverse_bad=sum(x['loss'] is None or x['loss']>.1 for x in c['inverses']),inverse_forward=sum(x['forward_samples'] for x in c['inverses'])))
  hashes[str(directory/'results.jsonl')]=sha(directory/'results.jsonl')
  for r in rr:
   key=(r['scenario'],r['seed'],r['target'],r['label']);assert key not in index
   if r['X'] is not None:
    x=np.array(r['X']);assert np.isfinite(x).all() and max(abs(x))<=1 and abs(abs(x.sum()+x[0]*x[1]-r['target'])-r['truth_error'])<1e-12
   assert r['panel_evaluations']<=400 and r['model_queries']<=1200
   assert abs(sum(r['alpha'])-1)<1e-10 and min(r['alpha'])>=0
   index[key]=r
 seeds=sorted({k[1] for k in index});targets=[0.,-.5,.5,2.7]
 assert len(seeds)==90 and set(index)==set(itertools.product(SCENARIOS,seeds,targets,LABELS))
 output.mkdir(parents=True);groups=[];comp=[];effects=[];matrices={}
 for target in targets:
  for scenario in ['mean_three_scenarios']+SCENARIOS:
   scenarios=SCENARIOS if scenario=='mean_three_scenarios' else [scenario]
   eligible=[s for s in seeds if all(index[sc,s,target,m]['status']=='valid' and index[sc,s,target,m]['truth_error'] is not None for sc,m in itertools.product(scenarios,LABELS))]
   matrix=np.array([[np.mean([index[sc,s,target,m]['truth_error'] for sc in scenarios]) for m in LABELS] for s in eligible]);bs=bootstrap(matrix)
   if scenario=='mean_three_scenarios':matrices[target]=(eligible,matrix)
   for j,m in enumerate(LABELS):
    rr=[index[sc,s,target,m] for sc,s in itertools.product(scenarios,eligible)]
    groups.append(dict(target=target,scenario=scenario,method=m,n=len(eligible),excluded=90-len(eligible),mean=float(matrix[:,j].mean()),sd=float(matrix[:,j].std(ddof=1)),ci95_low=float(np.quantile(bs[:,j],.025)),ci95_high=float(np.quantile(bs[:,j],.975)),mean_Lopt=float(np.mean([r['loss'] for r in rr])),mean_model_queries=float(np.mean([r['model_queries'] for r in rr])),mean_initialization_samples=float(np.mean([r['initialization_samples'] for r in rr])),fallbacks=sum('fallback' in json.dumps(r['alpha_audit']) and has_fallback(r['alpha_audit']) for r in rr)))
   if scenario=='mean_three_scenarios' and target!=0:
    for j,m in enumerate(LABELS[1:],1):
     b=bs[:,j]-bs[:,0];comp.append(dict(target=target,contrast=m+' - U',n=len(eligible),excluded=90-len(eligible),family=27,mean_delta=float((matrix[:,j]-matrix[:,0]).mean()),ci95_low=float(np.quantile(b,.025)),ci95_high=float(np.quantile(b,.975)),family_low=float(np.quantile(b,.05/54)),family_high=float(np.quantile(b,1-.05/54))))
 for target in targets[1:]:
  es,m=matrices[target];ez,z=matrices[0.];assert es==ez
  delta=m-z;bs=bootstrap(delta)
  for j,method in enumerate(LABELS):effects.append(dict(target=target,contrast='target_minus_T0',method=method,n=len(es),mean_delta=float(delta[:,j].mean()),ci95_low=float(np.quantile(bs[:,j],.025)),ci95_high=float(np.quantile(bs[:,j],.975)),scope='descriptive_target_and_geometry_change_under_matched_initial_starts'))
 csvout(output/'groups.csv',groups);csvout(output/'primary_comparisons.csv',comp);csvout(output/'target_changes.csv',effects);csvout(output/'case_diagnostics.csv',metadata)
 audit=dict(passed=True,conditions=len(metadata),records=len(index),valid=sum(r['status']=='valid' for r in index.values()),seeds=90,primary_family=27,replicates=200000,analysis_sha256=sha(Path(__file__)),source_sha256=hashes,scope='Reused historical models; newly inverted four targets and 400-panel delivery')
 (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))

def has_fallback(a):return isinstance(a,dict) and (bool(a.get('fallback_reason')) or any(has_fallback(x) for x in a.values()))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
