"""Create immutable registration and audit historical deployment-only evidence."""
import torch
import json,itertools,time
from pathlib import Path
from collections import Counter
import numpy as np
import oob_shrinkage_research as shared
from alpha_initialization_selection import METHODS,SCENARIOS,DISPLAY,select_c,select_p
from run_alpha_initialization_selection import lines,evidence
ROOT=Path('alpha_weight_outputs/alpha_initialization_selection_v1')

def scan_seeds():
 used=set();files=0;unreadable=[]
 def walk(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if (k=='seed' or k.endswith('_seed')) and isinstance(x,int):used.add(x)
    elif (k=='seeds' or k.endswith('_seeds')) and isinstance(x,list):used.update(y for y in x if isinstance(y,int))
    if isinstance(x,(dict,list)):walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 for folder in ['docs','alpha_weight_outputs']:
  for p in Path(folder).rglob('*'):
   if p.suffix not in ['.json','.jsonl'] or 'source_snapshot' in p.parts:continue
   try:
    if p.suffix=='.json':walk(json.loads(p.read_text()))
    else:
     for line in p.open():walk(json.loads(line))
    files+=1
   except (ValueError,OSError) as exc:unreadable.append(str(p))
 return used,files,unreadable

def main():
 if ROOT.exists():raise FileExistsError('Preserve prior output')
 used,n,failed=scan_seeds();start=30000
 while used.intersection(range(start,start+6)):start+=1000
 ROOT.mkdir();historical=json.load(open('docs/pure_es_extension_registration_20261014.json'))
 reg={'version':1,'date':'2026-10-15','output':str(ROOT),'strategies':['C','P'],'methods':METHODS,'scenarios':SCENARIOS,
      'blind_n':20,'fold_rng':20261015,'tie_order':['EU','E','SQ','A','Q','EQ','S','SU'],'tie_tolerance':1e-12,
      'pooled_role':'loaded by runner only after sealed decisions','truth_role':'post-delivery diagnostic','historical_pooled_training':False,
      'formal_started':False,'cost_limits':{'C':1.5,'P':10},'alpha_update':False,'pruning':False,'settlement':'not_run',
      'contrasts':[(a,b) for a in ['C','P'] for b in ['ensemble_shrink','stacked_full','support_quality_mix50','A_uniform','B_stage0']]+[('P','C')],
      'seed_audit':{'scanned_files':n,'unreadable':failed,'smoke':[start],'pilot':list(range(start+1,start+6)),'no_overlap_with_observed_seeds':True,
                    'scope':'all JSON/JSONL except source snapshots; fresh data/model generation and source array checks also required'},
      'source_sha256':{p.name:shared.digest(p) for p in Path('.').glob('*.py')},
      'design_sha256':shared.digest('docs/alpha_initialization_selection_design.md'),
      'historical_batches':historical['batches']}
 for stage,seeds in [('smoke',[start]),('pilot',list(range(start+1,start+6)))]:
  reg[stage]={'seeds':seeds,'source':str(ROOT/stage/'source'),'results':str(ROOT/stage/'results')}
 shared.dump(ROOT/'registration.json',reg);snap=ROOT/'source_snapshot';snap.mkdir()
 for name in reg['source_sha256']:(snap/name).write_bytes(Path(name).read_bytes())
 dev=ROOT/'development';dev.mkdir();records=[];available=[];errors=[]
 allseeds=sorted(s for b in historical['batches'] for s in b['seeds']);split={s:'development60' if i<60 else 'internal_check30' for i,s in enumerate(allseeds)}
 for b in historical['batches']:
  peers={(r['scenario'],r['seed']):r for r in lines(Path(b['six_method_results'])/'peer_evidence.jsonl')}
  # Explicitly strip all external labels immediately. They are not selector inputs.
  fixed={}
  for directory in [b['six_method_results'],b['results']]:
   for r in lines(Path(directory)/'results.jsonl'):fixed[r['scenario'],r['seed'],r['method']]={k:r.get(k) for k in ['status','X','alpha','method']}
  for sc,seed in itertools.product(SCENARIOS,b['seeds']):
   try:
    base,item,artifact=shared.load_base(b['source'],sc,seed);p,peer,_=evidence(base);saved=peers[sc,seed]
    diff=float(np.max(np.abs(p-saved['blind_prediction'])))
    if diff>1e-7:raise ValueError('Historical prediction mismatch')
    ce=select_c(p,base.test_y,peer,0);rows={m:fixed[sc,seed,m] for m in METHODS}
    pred={m:[float(a.api_predict(np.array(r['X']))[0]) for a in base.trusted_agents] for m,r in rows.items() if r['status']=='valid'}
    pe=select_p(rows,pred,rows['B_stage0']['alpha'],0)
    rec={'scenario':sc,'seed':seed,'split':split[seed],'C':ce,'P':pe,'prediction_reload_max':diff}
    shared.append(dev/'selector_evidence.jsonl',rec);records.append(rec)
    available.append({'scenario':sc,'seed':seed,'artifact':str(artifact),'available':True,'pooled_used_for_rule_design':False})
   except Exception as exc:errors.append({'scenario':sc,'seed':seed,'reason':repr(exc)});available.append({'scenario':sc,'seed':seed,'available':False})
  print('historical batch complete '+str(b['seeds'][0]),flush=True)
 shared.dump(dev/'availability.json',{'expected':270,'available':sum(r['available'] for r in available),'records':available,'errors':errors})
 summaries={part:{m:dict(Counter(DISPLAY.get(r[m]['selected_method'],r[m]['selected_method']) for r in records if r['split']==part)) for m in ['C','P']} for part in ['development60','internal_check30']}
 shared.dump(dev/'analysis.json',{'conditions':len(records),'selected_frequency':summaries,'pooled_labels_used':False,'rule_changes':False,'inference':'historical deployment signal replay, not independent quality validation'})
 shared.dump(dev/'acceptance.json',{'passed':len(records)==270 and not errors,'errors':errors,'all_methods_retained':True,'source_reg_sha256':shared.digest('docs/pure_es_extension_registration_20261014.json')})
 print(json.dumps({'seeds':reg['seed_audit'],'development_summary':summaries}),flush=True)
 if errors:raise RuntimeError('Historical availability/replay failed')
if __name__=='__main__':main()
