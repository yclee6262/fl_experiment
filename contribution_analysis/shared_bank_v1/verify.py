"""Independent recomputation using saved raw model replies and scipy ranking."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R=Path('/Users/yclee/Documents/GitHub/fl_experiment');out=R/'contribution_analysis/shared_bank_experiment_v2_20261020'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'registration.json');checks={}
checks['inputs_unchanged']=all(digest(Path(p))==h for p,h in read(out/'input_sha256.json').items())
checks['source_snapshot_unchanged']=all(digest(out/'source_snapshot'/p)==h for p,h in cfg['source_sha256'].items())
checks['scores_locked']=all(digest(out/'records'/p)==h for p,h in read(out/'score_lock.json')['record_sha256'].items())
checks['all_choices_and_scores']=True
for case in read(out/'cases.json'):
 rec=read(out/'records'/(case['key']+'.json'));p=np.array(rec['bank_points']);l=np.array(rec['bank_losses']);baseline=np.array(rec['baseline_predictions'])
 def q(v):return dict(mean_abs=float(abs(v).mean()),median_abs=float(np.median(abs(v))),abs_median_signed=float(abs(np.median(v))),abs_mean_signed=float(abs(v.mean())))
 for t in rec['participants']:
  i=t['index'];keep=np.arange(len(case['ids']))!=i;w=np.array(case['alphas'])[keep];w/=w.sum();loss=l[:,keep]@w;loss[np.all(p==np.array(case['candidates'][i]),axis=1)]=np.inf;j=int(np.argmin(loss));pred=np.array(t['predictions']);qq=q(pred);q0=q(baseline)
  checks['all_choices_and_scores'] &= j==t['bank_index'] and bool(np.array_equal(p[j],t['solution'])) and bool(np.allclose(abs(pred),l[j],rtol=0,atol=0)) and all(abs(t['scores'][n]-(qq[n]-q0[n]))<1e-15 for n in qq)
with (out/'participant_scores.csv').open() as f:rows=list(csv.DictReader(f))
with (out/'team_metrics.csv').open() as f:teams=list(csv.DictReader(f))
checks['all_80_team_correlations']=True
for t in teams:
 rs=[x for x in rows if x['key']==t['key']];x=np.array([float(a[t['method']]) for a in rs]);y=np.array([float(a['C']) for a in rs]);v=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
 checks['all_80_team_correlations'] &= (t['spearman']=='' and v is None) or(t['spearman']!='' and v is not None and abs(float(t['spearman'])-v)<1e-12)
checks={k:bool(v) for k,v in checks.items()};result=dict(accepted=all(checks.values()),checks=checks,technical_tests=5,n_teams=20,n_effects=80,n_methods=4,estimator_validity_established=False)
with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2));assert result['accepted']
