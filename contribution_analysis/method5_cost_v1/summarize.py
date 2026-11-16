import json
from collections import Counter
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[2];O=W/'contribution_analysis/method5_cost_experiment_20261030_attempt_04'
d=[r for r in json.loads((O/'progress.json').read_text()) if not r['development']];tim=json.loads((O/'timing.json').read_text());summary={};percase=[];stages={a:Counter() for a in ['New','LOO']};statuses=Counter()
for r in d:
 v={}
 for a in stages:
  z=json.loads((O/'records'/(r['key']+'_'+a+'.json')).read_text());v[a]={k:z[k] for k in ['samples','requests','hits','baseline_samples','per_model_samples','model_seconds','peak_memory_bytes']}
  v[a]['timings']={m:dict(median=float(np.median([t[m] for t in tim if t['key']==r['key'] and t['arm']==a])),minimum=min(t[m] for t in tim if t['key']==r['key'] and t['arm']==a),maximum=max(t[m] for t in tim if t['key']==r['key'] and t['arm']==a)) for m in ['core_seconds','required_output_seconds','wall_seconds','model_seconds']}
  for e in z['events']:
   if not e['hit']:stages[a][e['stage']]+=1
  if a=='New':statuses['anchor_'+z['allocation']['anchor']['status']]+=1;statuses['model_'+z['allocation']['model']['status']]+=1
 percase.append(dict(key=r['key'],scenario=r['scenario'],arms=v,ratio=v['New']['samples']/v['LOO']['samples']))
for sc in ['all','clean','controlled_poison']:
 v=[r for r in percase if sc=='all' or r['scenario']==sc];q={a:sum(r['arms'][a]['samples'] for r in v) for a in stages};base=sum(r['arms']['LOO']['baseline_samples'] for r in v)
 summary[sc]=dict(teams=len(v),samples=q,ratio=q['New']/q['LOO'],extra_samples={a:q[a]-base for a in stages},extra_ratio=(q['New']-base)/(q['LOO']-base),logical={a:sum(r['arms'][a]['requests'] for r in v) for a in stages},median_case_ratio=float(np.median([r['ratio'] for r in v])),wins=sum(r['ratio']<1 for r in v),worst=max(v,key=lambda r:r['ratio'])['key'],worst_ratio=max(r['ratio'] for r in v),timings={m:{a:sum(r['arms'][a]['timings'][m]['median'] for r in v) for a in stages} for m in ['core_seconds','required_output_seconds','wall_seconds','model_seconds']})
(O/'analysis.json').write_text(json.dumps(dict(summary=summary,stages=stages,statuses=statuses,percase=percase),indent=2))
