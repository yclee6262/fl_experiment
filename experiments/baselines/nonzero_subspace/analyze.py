"""Seed-paired target/space diagnostics; frozen nonzero primary family."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from experiments.baselines.nonzero_subspace.run import h,witness,distances

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=Path(__file__).with_name('protocol.json')
METHODS=['anchor_mean','best_anchor','origin','coefficient','custom','best_of','nelder_mead','direct_slsqp']

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def csvout(p,data):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
def valid(r):return bool(r['status']=='valid' and r['X'] is not None and np.isfinite(r['truth_error']))
@lru_cache(maxsize=32)
def bootstrap_weights(n):
 rng=np.random.default_rng(20261123)
 return np.concatenate([rng.multinomial(n,np.full(n,1/n),size=2000) for _ in range(100)])
def boot(values,family=1):
 values=np.asarray(values);n=len(values)
 if n==0:return dict(n=0,mean=None,sd=None,ci95_low=None,ci95_high=None,family_low=None,family_high=None)
 bs=bootstrap_weights(n)@values/n
 q=np.quantile(bs,[.025,.975,.05/(2*family),1-.05/(2*family)])
 return dict(n=n,mean=float(values.mean()),sd=float(values.std(ddof=1)) if n>1 else None,ci95_low=float(q[0]),ci95_high=float(q[1]),family_low=float(q[2]),family_high=float(q[3]))



def one_anchor_truth_range(anchor,target):
 """Exact continuous response range on {beta*a: |beta*a|<=1}."""
 a=np.asarray(anchor,dtype=float);scale=float(max(abs(a)))
 if scale==0:return dict(beta_lower=None,beta_upper=None,truth_lower=0.,truth_upper=0.,linear_truth_error_floor=abs(float(target)),affine_truth_error_floor=abs(float(target)))
 bound=1/scale;linear=float(a.sum());quadratic=float((a[:-1]*a[1:]).sum())
 candidates=[-bound,bound]
 if quadratic!=0:
  vertex=-linear/(2*quadratic)
  if -bound<=vertex<=bound:candidates.append(vertex)
 vals=[linear*b+quadratic*b*b for b in candidates];lo,hi=min(vals),max(vals)
 floor=max(lo-target,target-hi,0.)
 return dict(beta_lower=-bound,beta_upper=bound,truth_lower=lo,truth_upper=hi,linear_truth_error_floor=float(floor),affine_truth_error_floor=abs(h(a)-target))

def analyze(source,output):
 if output.exists():raise FileExistsError(output)
 m=read(source/'manifest.json');assert m['status']=='complete' and not m['smoke'];spec=m['protocol']
 for f,hsh in m['code_sha256'].items():assert sha(ROOT/f)==hsh,f
 for f,hsh in m['input_sha256'].items():assert sha(Path(f))==hsh,f
 data=[];cases=[];single_anchor=[];hashes={str(source/'manifest.json'):sha(source/'manifest.json')};keys=set()
 for complete in sorted(source.glob('seed*/*_k*/completion.json')):
  path=complete.parent;c=read(path/'case.json');done=read(complete)
  assert sha(path/'case.json')==done['case_sha256'] and sha(path/'results.jsonl')==done['results_sha256']
  key=tuple(c[k] for k in ['seed','dimension','rho','requested_k','target_label']);assert key not in keys;keys.add(key)
  rr=rows(path/'results.jsonl');assert len(rr)==18
  assert len({(r['method'],r['panel_budget']) for r in rr})==18
  for r in rr:
   assert tuple(r[k] for k in ['seed','dimension','rho','requested_k','target_label'])==key
   if valid(r):
    assert max(abs(np.array(r['X'])))<=1+1e-8 and abs(abs(h(r['X'])-c['target'])-r['truth_error'])<1e-10
    assert r['panel_evaluations']<=r['assigned_panel_budget']
    assert r['model_queries']<=r['assigned_panel_budget']*c['requested_k']
    dl,da=distances(c['anchors'],r['X']);assert abs(dl-r['endpoint_linear_distance'])<1e-10 and abs(da-r['endpoint_affine_distance'])<1e-10
    if r['method'] in ['coefficient','custom','best_of','anchor_mean','best_anchor']:assert dl<1e-7
    if r['method'] in ['custom','anchor_mean','best_anchor']:assert da<1e-7
   if r.get('trace'):
    assert sha(path/r['trace'])==r['trace_sha256']
   data.append(r)
  assert abs(h(c['witness'])-c['target'])<1e-10
  case={k:c.get(k) for k in ['seed','dimension','rho','requested_k','target_label','target','linear_rank','affine_rank','witness_linear_distance','witness_affine_distance','linear_box_feasible','affine_box_feasible','zero_anchor_count','inversion_forward_samples','inversion_backward_steps']}
  poolpath=path.parent/f"{c['target_label']}_inversions.json";pool=read(poolpath);hashes[str(poolpath.resolve())]=sha(poolpath)
  assert pool['target']==c['target'] and pool['model_sha256']==c['model_sha256']
  inverse=pool['inversions'][:c['requested_k']]
  if c.get('anchors'):assert c['anchors']==[v['X'] for v in inverse]
  residuals=[v['loss'] for v in inverse if v['loss'] is not None]
  case.update(max_local_inverse_residual=max(residuals) if residuals else None,mean_local_inverse_residual=float(np.mean(residuals)) if residuals else None,local_inverse_residual_le_0p1=sum(v<=.1 for v in residuals),local_inverse_residual_count=len(residuals))
  cases.append(case)
  if c['requested_k']==1 and c.get('anchors'):
   lower=one_anchor_truth_range(c['anchors'][0],c['target'])
   single_anchor.append({k:c[k] for k in ['seed','dimension','rho','requested_k','target_label','target']}|lower)
   for row in rr:
    if valid(row) and row['method'] in ['coefficient','custom','best_of']:
     floor=lower['affine_truth_error_floor'] if row['method']=='custom' else lower['linear_truth_error_floor']
     assert row['truth_error']>=floor-1e-7
  for f in [complete,path/'case.json',path/'results.jsonl']:hashes[str(f.resolve())]=sha(f)
 expected=set(itertools.product(spec['seeds'],spec['dimensions'],spec['rhos'],spec['coalition_sizes'],spec['targets']))
 assert keys==expected and len(data)==17280
 index={(r['seed'],r['dimension'],r['rho'],r['requested_k'],r['target_label'],r['method'],r['panel_budget']):r for r in data}
 grouped=defaultdict(list)
 for r in data:grouped[tuple(r[k] for k in ['dimension','rho','requested_k','target_label','method','panel_budget'])].append(r)
 summary=[];status=[]
 for key,rs in sorted(grouped.items()):
  identity=dict(zip(['dimension','rho','requested_k','target_label','method','panel_budget'],key));vv=[r for r in rs if valid(r)]
  identity.update(planned=30,failed=30-len(vv),mean_Lopt=float(np.mean([r['loss'] for r in vv])) if vv else None,
      mean_model_queries=float(np.mean([r.get('model_queries',0) for r in rs])),
      mean_prediction_truth_error=float(np.mean([r['mean_model_absolute_truth_error'] for r in vv])) if vv else None,
      mean_prediction_disagreement_sd=float(np.mean([r['prediction_disagreement_sd'] for r in vv])) if vv else None,
      mean_loss_truth_gap=float(np.mean([r['operational_loss_truth_gap'] for r in vv])) if vv else None)
  summary.append(identity|boot([r['truth_error'] for r in vv]))
  status += [dict(zip(['dimension','rho','requested_k','target_label','method','panel_budget'],key))|dict(status=k,count=v) for k,v in Counter(r['status'] for r in rs).items()]
 primary=[]
 for d,rho,k,label in itertools.product(spec['dimensions'],spec['rhos'],spec['coalition_sizes'],['negative','positive','near_upper']):
  delta=[]
  for seed in spec['seeds']:
   left=index[seed,d,rho,k,label,'best_of',400];right=index[seed,d,rho,k,label,'nelder_mead',400]
   if valid(left) and valid(right):delta.append(left['truth_error']-right['truth_error'])
  primary.append(dict(dimension=d,rho=rho,requested_k=k,target_label=label,contrast='best_of - nelder_mead',panel_budget=400,planned=30,excluded=30-len(delta),family_size=24,**boot(delta,24)))
 geometry=[]
 for d,rho,k,label in itertools.product(spec['dimensions'],spec['rhos'],spec['coalition_sizes'],spec['targets']):
  cc=[c for c in cases if (c['dimension'],c['rho'],c['requested_k'],c['target_label'])==(d,rho,k,label)];good=[c for c in cc if c['linear_rank'] is not None]
  geometry.append(dict(dimension=d,rho=rho,requested_k=k,target_label=label,planned=30,valid=len(good),mean_linear_rank=float(np.mean([c['linear_rank'] for c in good])) if good else None,mean_affine_rank=float(np.mean([c['affine_rank'] for c in good])) if good else None,
    mean_witness_linear_distance=float(np.mean([c['witness_linear_distance'] for c in good])) if good else None,mean_witness_affine_distance=float(np.mean([c['witness_affine_distance'] for c in good])) if good else None,
    linear_box_infeasible=sum(not c['linear_box_feasible'] for c in good),affine_box_infeasible=sum(not c['affine_box_feasible'] for c in good)))
 output.mkdir(parents=True)
 for name,rs in [('groups',summary),('primary_comparisons',primary),('geometry_summary',geometry),('case_geometry',cases),('single_anchor_truth_bounds',single_anchor),('statuses',status)]:csvout(output/(name+'.csv'),rs)
 audit=dict(passed=True,tasks=len(keys),records=len(data),valid=sum(valid(r) for r in data),failed=sum(not valid(r) for r in data),primary_family=24,bootstrap_replicates=200000,rng=20261123,source_sha256=hashes,analysis_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),scope='Reused frozen models, new targets and restricted coalitions; witness distance is not a target-infeasibility proof')
 (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');make_plot(output,primary);print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
def make_plot(output,primary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,5))
 for ax,label,title in zip(axes,['negative','positive','near_upper'],['T = -0.5','T = 0.5','T = 0.9(2D-1)']):
  rr=[r for r in primary if r['target_label']==label]
  for y,r in enumerate(rr):
   if r['mean'] is None:continue
   ax.plot([r['family_low'],r['family_high']],[y,y],color='#abc1d0',lw=5)
   ax.plot([r['ci95_low'],r['ci95_high']],[y,y],color='#245673',lw=2)
   ax.plot(r['mean'],y,'o',color='#143d56',ms=4)
  ax.set_yticks(range(len(rr)),[f"D{r['dimension']}, rho{r['rho']:g}, K{r['requested_k']}" for r in rr])
  ax.axvline(0,color='gray',ls='--',lw=1);ax.set_title(title);ax.set_xlabel('Mean truth-error difference')
 fig.suptitle('Restricted best-of minus full-space Nelder-Mead at 400 panels\nThin: 95%; thick: Bonferroni 24-family intervals',fontsize=11)
 fig.tight_layout(rect=(0,0,1,.90));fig.savefig(output/'subspace_paired_intervals.pdf');fig.savefig(output/'subspace_paired_intervals.png',dpi=150);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
