"""Independent integrity and seed-statistics audit of every disclosed method family."""
from pathlib import Path
import json,csv,hashlib,itertools
import numpy as np
from scipy.stats import spearmanr
ROOT=Path('/Users/yclee/Documents/GitHub/fl_experiment')
FAMILIES=['shared_bank_experiment_v2_20261020','shared_response_experiment_v2_20261020','distributed_panel_experiment_v1_20261020','validation_judge_experiment_v1_20261020','reliability_risk_experiment_v1_20261020','geometric_replay_experiment_v1_20261020','robust_endpoint_experiment_v2_20261020']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
 with p.open() as f:return list(csv.DictReader(f))
def stat(v):
 a=np.array([np.nan if x is None else x for x in v]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);means=np.nansum(s,axis=1)[ct>0]/ct[ct>0]
 return float(np.nanmean(a)) if np.isfinite(a).any() else None,np.quantile(means,[.025,.975]) if len(means) else None
results=[]
for name in FAMILIES:
 r=ROOT/'contribution_analysis'/name;c=read(r/'registration.json');checks={}
 checks['frozen_sources']=all(sha(r/'source_snapshot'/n)==h for n,h in c['source_sha256'].items())
 checks['historical_inputs']=all(sha(Path(n))==h for n,h in read(r/'input_sha256.json').items())
 lock=read(r/'score_lock.json');folder='scores' if name.startswith('shared_response') else 'records';checks['score_lock']=all(sha(r/folder/n)==h for n,h in lock['record_sha256'].items())
 rows=csvrows(r/'participant_scores.csv');teams=csvrows(r/'team_metrics.csv');summary=read(r/'development_summary.json');dims=[d for d in ['config','panel','method'] if d in teams[0]];checks['all_team_spearman']=True;recalc=[]
 for t in teams:
  rr=[x for x in rows if x['key']==t['key'] and all(x[d]==t[d] for d in dims if d!='method')];x=np.array([float(a[t['method']]) for a in rr]);y=np.array([float(a['C']) for a in rr]);v=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
  checks['all_team_spearman'] &= (v is None and t['spearman']=='') or(v is not None and t['spearman']!='' and abs(v-float(t['spearman']))<1e-12)
  recalc.append(dict(**{k:t[k] for k in dims+['scenario']},key=t['key'],value=v))
 checks['all_summary_means_and_bootstraps']=True
 for s in summary:
  vals=[t['value'] for t in recalc if t['scenario']==s['scenario'] and all(t[d]==s[d] for d in dims)];m,ci=stat(vals);saved=s.get('ci',s.get('exploratory_95pct_CI'));checks['all_summary_means_and_bootstraps'] &= (m is None and s['mean'] is None) or(m is not None and abs(m-s['mean'])<1e-12 and bool(np.allclose(ci,saved,atol=1e-12,rtol=0)))
 checks={k:bool(v) for k,v in checks.items()};results.append(dict(family=name,accepted=all(checks.values()),checks=checks,n_team_statistics=len(teams),n_summary_statistics=len(summary),n_methods=len(summary)//2))
result=dict(accepted=all(s['accepted'] for s in results),families=results,global_free_methods=164,offline_endpoint_diagnostics=4,independent_validity_established=False)
out=ROOT/'contribution_analysis/contribution_search_development_audit_v2_20261020.json'
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2));assert result['accepted']
