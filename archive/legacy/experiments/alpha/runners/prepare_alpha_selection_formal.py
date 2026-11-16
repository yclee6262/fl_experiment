"""Freeze fixed-budget formal protocol; historical labels are used only for variance/power."""
import torch
import json,shutil,time
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from prepare_alpha_initialization_selection import scan_seeds
from alpha_selection_formal_statistics import power_screen
from run_alpha_initialization_selection import lines

ROOT=Path('alpha_weight_outputs/alpha_initialization_selection_formal_v1_20261015')
def main():
 if ROOT.exists():raise FileExistsError('Preserve previous formal run')
 prior=Path('alpha_weight_outputs/alpha_initialization_selection_v1');old=json.load(open(prior/'validation_v1_1/registration.json'))
 if not json.load(open(prior/'final_acceptance.json'))['flow_and_reproducibility_passed']:raise RuntimeError('Pilot flow not accepted')
 ROOT.mkdir();powerdir=ROOT/'power';powerdir.mkdir()
 idx={}
 for b in old['historical_batches']:
  for directory in [b['six_method_results'],b['results']]:
   for r in lines(Path(directory)/'results.jsonl'):idx[r['scenario'],r['seed'],r['method']]=r
 deployment={(r['scenario'],r['seed']):r for r in lines(prior/'development/selector_evidence.jsonl') if r['split']=='development60'}
 seeds=sorted({k[1] for k in deployment});matrix=[]
 for seed in seeds:
  ds=[]
  for a,b in old['contrasts']:
   diffs=[]
   for sc in old['scenarios']:
    a_method=deployment[sc,seed][a]['selected_method'] if a in ['C','P'] else a
    b_method=deployment[sc,seed][b]['selected_method'] if b in ['C','P'] else b
    diffs.append(idx[sc,seed,a_method]['pooled_error']-idx[sc,seed,b_method]['pooled_error'])
   ds.append(float(np.mean(diffs)))
  matrix.append(ds)
 matrix=np.array(matrix);np.save(powerdir/'development_seed_differences.npy',matrix)
 screen=power_screen(matrix)
 power={'development_seed_n':len(seeds),'development_seeds':seeds,'contrasts':old['contrasts'],'delta':.001,'simulation_n':5000,'results':screen,
        'method':'center historical full-seed residual vectors, shift to -delta, resample three independent batches; normal-CI surrogate for simultaneous bootstrap and two-of-three batch gates',
        'limits':'approximate planning screen, not exact bootstrap power; assumes empirical variability transfer and no qualification/solver failure; historical labels only estimate variance, no selector training or tuning',
        'selected_n':90,'selection_rule':'retain original fixed budget 3x30; report limited power, do not expand or stop based on pilot/formal effects'}
 shared.dump(powerdir/'analysis.json',power)
 used,n,unreadable=scan_seeds()
 if unreadable:raise RuntimeError('Seed audit unreadable files: '+repr(unreadable))
 starts=[];nextseed=40000
 for _ in range(3):
  while used.intersection(range(nextseed,nextseed+30)):nextseed+=1000
  starts.append(nextseed);used.update(range(nextseed,nextseed+30));nextseed+=1000
 hashes={p.name:shared.digest(p) for p in Path('.').glob('*.py')}
 # Freeze original method and selector provenance, and all old results this research relies on.
 baseline_hashes={}
 for b in old['historical_batches']:
  for directory in [b['six_method_results'],b['results']]:baseline_hashes[str(Path(directory)/'results.jsonl')]=shared.digest(Path(directory)/'results.jsonl')
 for path in [prior/'validation_v1_1/pilot/results.jsonl',prior/'registration.json',prior/'final_acceptance.json']:baseline_hashes[str(path)]=shared.digest(path)
 reg={k:old[k] for k in ['methods','strategies','scenarios','blind_n','fold_rng','tie_order','tie_tolerance','contrasts','cost_limits','alpha_update','pruning','settlement']}
 reg.update(version='formal_v1',date='2026-10-15',output=str(ROOT),source_sha256=hashes,prior_results_sha256=baseline_hashes,
    analysis_rng=2026101501,bootstrap_replicates=200000,family_error=.05,comparison_family=11,
    expected_conditions=2700,planned_seed_n=90,formal_started=False,historical_pooled_training=False,
    pooled_role='post-delivery primary only; no manual interim inspection; source preparation evaluator/A artifacts unavailable to selectors',truth_role='post-delivery diagnostic',
    primary='within-seed equal mean of three scenarios; seed-paired difference strategy minus baseline',
    qualification_population='all-three-scenario common pre-method eligible seeds; qualification failures preserved, no replacement',
    stability_rule='complete eligible pairs; simultaneous upper<0; all three batch means<0; at least two ordinary95 batch uppers<0; observed delivery failures no higher',
    incomplete_rule='preserve rows; no stable superiority claim for incomplete method pairs; show pairwise complete diagnostic separately',
    analysis_release='after all three batches and reload verification complete; per-batch runner descriptive analysis suppressed',
    sample_size_rule=power['selection_rule'],power_report=str(powerdir/'analysis.json'),replay_seed_rule='first planned seed of every batch, regardless of eligibility; no replacement',
    cost_scope='method time ratios to EU and full shared preparation components separately; C limit1.5 P limit10; quality and adoption gates separate',
    no_interim_tuning=True,no_early_success_stop=True,seed_audit={'scanned_files':n,'unreadable':unreadable,'new_starts':starts,'no_overlap_with_observed_saved_seeds':True},
    pilot_selector_sha256={p:shared.digest(p) for p in ['alpha_initialization_selection.py','run_alpha_initialization_selection.py']},
    design_sha256=shared.digest('docs/alpha_initialization_selection_design.md'),budget_seconds_estimate=None)
 batches=[]
 for i,start in enumerate(starts,1):
  name=f'formal_batch_{i}';reg[name]={'seeds':list(range(start,start+30)),'source':str(ROOT/name/'source'),'results':str(ROOT/name/'results')};batches.append(name)
  replay=f'replay_batch_{i}';reg[replay]={'seeds':[start],'source':str(ROOT/replay/'source'),'results':str(ROOT/replay/'results')}
 reg['batches']=batches
 pilotcost=lines(prior/'pilot/results/deployment_costs.jsonl')
 seconds=[sum(r['shared_preparation'].get(k,0) for k in ['training_seconds','qualification_seconds','proposal_seconds'])+(r.get('fixed_proposal_seconds') or 0)+(r.get('external_evaluator_seconds') or 0) for r in pilotcost]
 reg['budget_seconds_estimate']={'source_preparation_median_seconds_per_scenario':float(np.median(seconds)),'279_scenario_units_source_estimate':float(np.median(seconds)*279),'scope':'270 formal +9 replay scenarios; selector/reload/subprocess overhead additional; local sequential single-thread'}
 shared.dump(ROOT/'registration.json',reg)
 snap=ROOT/'source_snapshot';snap.mkdir()
 for name in hashes:(snap/name).write_bytes(Path(name).read_bytes())
 shutil.copyfile('docs/alpha_initialization_selection_design.md',ROOT/'registered_design.md')
 shared.dump(ROOT/'registration_digest.json',{'sha256':shared.digest(ROOT/'registration.json')})
 print(json.dumps({'power_n90':screen[0],'budget':reg['budget_seconds_estimate'],'seed_audit':reg['seed_audit']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
