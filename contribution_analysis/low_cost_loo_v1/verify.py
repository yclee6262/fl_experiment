"""Independent algebra/cache/statistics audit; no additional inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
W=Path(__file__).resolve().parents[2];P=W/'contribution_analysis/low_cost_loo_exploration_20261116'
def read(p):return json.loads(Path(p).read_text())
r=read(P/'scored.json');locked=read(P/'endpoints_locked.json');tm=read(P/'team_metrics.json');n_events=0;max_err=0
for rec,lock in zip(r,locked):
 c=rec['case'];base=read(W/'contribution_analysis/custom_direction_experiment_v2_20261020/baselines'/(c['key']+'.json'))
 old=read(W/'contribution_analysis/model_step_decision_experiment_v1_20261021/development'/(c['key']+'.json'))
 sums=np.zeros((len(c['ids']),2));xf=np.asarray(base['result']['solution']);prior=np.asarray(c['fixed_start'])
 for it in base['iterations']:
  np.testing.assert_allclose(prior,it['x'],atol=1e-12,rtol=0)
  ac=[v for v in it['trials'] if v['accepted']]
  if ac:
   dx=np.asarray(ac[0]['x'])-prior;g=np.asarray(it['g'])
   # derive common multiplier from dot product, independently of norm-ratio implementation
   lam=-np.dot(dx,g)/np.dot(g,g);parts=np.zeros_like(sums)
   for z in it['directions']:parts[z['k']]=-lam*z['weight']*z['derivative']*np.asarray(z['u'])
   max_err=max(max_err,float(np.linalg.norm(parts.sum(0)-dx)));sums+=parts;prior=np.asarray(ac[0]['x'])
 np.testing.assert_allclose(xf,prior,rtol=0,atol=1e-12)
 np.testing.assert_allclose(xf-sums,rec['methods']['CUM']['raw'],rtol=0,atol=1e-10)
 for name,v in lock['methods'].items():
  ev=v['query_events'];keys=set()
  if not any(e['stage']=='baseline' for e in ev):
   for e in old['events']:
    if e['stage']=='baseline':keys.add(str(e['i'])+':'+np.asarray(e['x'],np.float64).tobytes().hex())
  extra=0
  for e in ev:
   expected=e['stage']!='baseline' and e['key'] in keys
   assert expected==e['hit'];keys.add(e['key']);n_events+=1
   extra+=not e['hit'] and e['stage']!='baseline'
  assert extra==v['extra_samples'];assert base['counter']['samples']+extra==v['total_samples']
  np.testing.assert_array_equal(np.clip(v['raw'],-1,1),v['x'])
  for evaluator in ['full','peer','pooled']:
   if name=='LOO':continue
   a=np.array(rec['methods'][name][evaluator]);b=np.array(rec['methods']['LOO'][evaluator]);t=next(x for x in tm if x['key']==c['key'] and x['method']==name and x['evaluator']==evaluator)
   if np.ptp(a)>0 and np.ptp(b)>0:assert abs(spearmanr(a,b).statistic-t['rho'])<1e-12
   aa=np.maximum(a,0);bb=np.maximum(b,0)
   if aa.sum()>1e-8 and bb.sum()>1e-8:
    # TV also equals 1-overlap for normalized distributions.
    tv=1-np.minimum(aa/aa.sum(),bb/bb.sum()).sum();assert abs(tv-t['tv'])<1e-12
   else:assert t['tv'] is None
for p,h in read(P/'source_lock.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
with (P/'verification.json').open('x') as f:json.dump(dict(accepted=True,teams=len(r),events_audited=n_events,max_independent_displacement_error=max_err,
 cache_sequences_exact=True,independent_scipy_rank_check=True,tv_overlap_identity_check=True,all_historical_hashes_unchanged=True),f,indent=2)
print('verified',n_events,max_err)
