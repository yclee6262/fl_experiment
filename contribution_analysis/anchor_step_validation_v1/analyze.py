"""Read locked internal results, then independently score produced points offline."""
import argparse
import csv
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from study import ROOT,OLD,LOCAL,read,save,sha,setup,check,lock,key

DEAD=1e-4
def sign(x):return int(x>DEAD)-int(x < -DEAD)
def corr(a,b,dead=DEAD):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)<2 or np.ptp(a)<=dead or np.ptp(b)<=dead:return None
    v=float(spearmanr(a,b).statistic)
    return v if np.isfinite(v) else None
def boot(values):
    vs=np.array([np.nan if v is None else v for v in values],dtype=float)
    if not np.isfinite(vs).any():return dict(n=0,mean=None,CI95=None)
    rng=np.random.default_rng(20261020);draw=vs[rng.integers(len(vs),size=(10000,len(vs)))];counts=np.isfinite(draw).sum(axis=1)
    means=np.nansum(draw,axis=1)[counts>0]/counts[counts>0]
    return dict(n=int(np.isfinite(vs).sum()),mean=float(np.nanmean(vs)),CI95=np.quantile(means,[.025,.975]).tolist())
def csvsave(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def locked():
    check()
    for phase in ['pilot','development']:
        for name,digest in read(ROOT/(phase+'_lock.json'))['record_sha256'].items():
            if sha(ROOT/phase/name)!=digest:raise ValueError('record_changed')

def analyze():
    import torch
    from models import PretrainedModel
    locked();start=time.perf_counter()
    # Lock analysis code and all solution records before the first pooled load.
    extra={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    save(ROOT/'analysis_source_lock.json',dict(source_sha256=extra,
        pilot_lock_sha256=sha(ROOT/'pilot_lock.json'),development_lock_sha256=sha(ROOT/'development_lock.json')))
    adir=ROOT/'analysis_source_snapshot';adir.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,adir/p.name)
    cases=read(ROOT/'cases.json');teams=[];coverage=[];participants=[];local_external=[];branch_values=[];examples=[]
    endpoint_cache={};pooled_counts={};costs=[]
    for phase,items in cases.items():
        for c in items:
            r=read(ROOT/phase/(c['key']+'.json'));b=read(OLD/'baselines'/(c['key']+'.json'));local=read(LOCAL/'records'/(c['key']+'.json'))
            m=len(c['ids']);cache={};counts={'points':0,'cache_hits':0}
            payload=torch.load(c['evaluator'],map_location='cpu',weights_only=True)
            model=PretrainedModel(2);model.load_state_dict(payload['state_dict']);model.eval()
            def E(x):
                k=key(x).hex()
                if k not in cache:
                    with torch.no_grad():cache[k]=dict(x=x,error=abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item())))
                    counts['points']+=1
                else:counts['cache_hits']+=1
                return cache[k]['error']
            accepted=[t for t in b['iterations'] if any(z['accepted'] for z in t['trials'])]
            rejected=[t for t in b['iterations'] if t['trials'] and not any(z['accepted'] for z in t['trials'])]
            coverage.append(dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],
                iterations=len(b['iterations']),accepted=len(accepted),no_acceptance=len(b['iterations'])-len(accepted),
                zero_direction_iterations=sum(t['g_norm']<=1e-8 for t in b['iterations']),
                all_trials=sum(len(t['trials']) for t in b['iterations']),
                rejected_trials=sum(not z['accepted'] for t in b['iterations'] for z in t['trials']),
                rejected_secant_iterations=sum(t['mode']=='secant' for t in rejected),
                rejected_dynamic_iterations=sum(t['mode']=='dynamic' for t in rejected),
                dynamic_iterations=sum(t['mode']=='dynamic' for t in b['iterations']),
                comparable_pairs=sum(z['J'] is not None for z in local['comparisons']),
                expected_accepted_pairs=len(accepted)*m,missing_pairs=sum(z['J'] is None for z in local['comparisons'])))
            original=np.array(local['A']);half=np.zeros(m);complete_half=True
            for z in r['half_step']:
                if z['c'] is None:complete_half=False
                else:half[z['anchor_index']]+=z['c']
            half_sign=[sign(v)==sign(w) for v,w in zip(original,half) if sign(v)]
            half_stats=dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],kind='half_distance',h=None,
                rank=corr(original,half) if complete_half else None,
                nonneutral_initial=int(sum(sign(v)!=0 for v in original)),
                sign_retention=float(np.mean(half_sign)) if half_sign else None,sign_reversal=None,neutralization=None)
            teams.append(half_stats)
            for k,i in enumerate(c['ids']):
                rows=[z for z in local['comparisons'] if z['anchor_index']==k and z['J'] is not None]
                js=np.array([z['J'] for z in rows]);totalabs=float(sum(abs(js)))
                participants.append(dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=i,
                    accepted_steps=len(accepted),comparable_steps=len(rows),A=float(original[k]),
                    A_positive=local['A_positive'][k],A_negative=local['A_negative'][k],
                    A_half=float(half[k]) if complete_half else None,net_sign_same=sign(original[k])==sign(half[k]) if complete_half else None,
                    max_step_absolute_share=float(max(abs(js))/totalabs) if totalabs else None,
                    zero_steps=int(sum(sign(j)==0 for j in js)),positive_steps=int(sum(sign(j)>0 for j in js)),negative_steps=int(sum(sign(j)<0 for j in js))))
            step_external={}
            for z in local['comparisons']:
                if z['J'] is None:continue
                alt=z['alternative_x'] if z['alternative_accepted'] else z['current_x']
                e=E(alt)-E(z['actual_next'])
                row=dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],
                    iteration=z['iteration'],agent_id=z['agent_id'],internal_c=z['J'],external_c=e,
                    internal_sign=sign(z['J']),external_sign=sign(e))
                local_external.append(row);step_external.setdefault(z['iteration'],[]).append(row)
            step_ranks=[corr([z['internal_c'] for z in rs],[z['external_c'] for z in rs]) for rs in step_external.values()]
            ex=[z for z in local_external if z['key']==c['key']];nonzero=[z for z in ex if z['internal_sign']]
            teams.append(dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],kind='local_internal_external',h=0,
                rank=float(np.mean([v for v in step_ranks if v is not None])) if any(v is not None for v in step_ranks) else None,
                nonneutral_initial=len(nonzero),sign_retention=float(np.mean([z['internal_sign']==z['external_sign'] for z in nonzero])) if nonzero else None,
                sign_reversal=float(np.mean([z['internal_sign']==-z['external_sign'] for z in nonzero])) if nonzero else None,
                neutralization=float(np.mean([not z['external_sign'] for z in nonzero])) if nonzero else None))
            available=[z for z in r['branches'] if z['status']=='available']
            initial=[z['values'][0]['c'] for z in available]
            for h in [0,1,3,5]:
                current=[next(v for v in z['values'] if v['h']==h)['c'] for z in available]
                nonzero=[(v,w) for v,w in zip(initial,current) if sign(v)]
                teams.append(dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],kind='persistence',h=h,
                    rank=corr(initial,current),nonneutral_initial=len(nonzero),
                    sign_retention=float(np.mean([sign(v)==sign(w) for v,w in nonzero])) if nonzero else None,
                    sign_reversal=float(np.mean([sign(v)==-sign(w) for v,w in nonzero])) if nonzero else None,
                    neutralization=float(np.mean([not sign(w) for v,w in nonzero])) if nonzero else None))
                for br in available:
                    z=next(v for v in br['values'] if v['h']==h);e=E(z['alternative_x'])-E(z['complete_x'])
                    row=dict(phase=phase,key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=br['agent_id'],
                        h=h,c0=br['values'][0]['c'],internal_c=z['c'],external_c=e,
                        complete_loss=z['complete_loss'],alternative_loss=z['alternative_loss'],
                        complete_x=z['complete_x'],alternative_x=z['alternative_x'],
                        complete_halted=z['complete_halted'],alternative_halted=z['alternative_halted'])
                    branch_values.append(row)
            endpoint_cache[c['key']]=cache;pooled_counts[c['key']]=counts
            costs.append(dict(phase=phase,key=c['key'],**r['counters'],wall_seconds=r['wall_seconds'],pooled_unique_points=counts['points']))
    summaries=[]
    for phase in ['pilot','development']:
        for scenario in ['clean','controlled_poison']:
            for kind in ['half_distance','local_internal_external','persistence']:
                for h in ([0,1,3,5] if kind=='persistence' else [None] if kind=='half_distance' else [0]):
                    ts=[t for t in teams if t['phase']==phase and t['scenario']==scenario and t['kind']==kind and t['h']==h]
                    summaries.append(dict(phase=phase,scenario=scenario,kind=kind,h=h,teams=len(ts),
                        rank=boot([t['rank'] for t in ts]),retention=boot([t['sign_retention'] for t in ts]),
                        reversal=boot([t['sign_reversal'] for t in ts]),neutralization=boot([t['neutralization'] for t in ts])))
    gates=[]
    for scenario in ['clean','controlled_poison']:
        st=next(s for s in summaries if s['phase']=='development' and s['scenario']==scenario and s['kind']=='persistence' and s['h']==5)
        rk=st['rank'];rt=st['retention'];rv=st['reversal']
        passed=rt['n']>=8 and rk['n']>=8 and rt['mean']>=.8 and rv['mean']<=.1 and rk['mean']>=.5 and rk['CI95'][0]>0
        sh=next(s for s in summaries if s['phase']=='development' and s['scenario']==scenario and s['kind']=='half_distance')
        stable=sh['rank']['mean'] is not None and sh['rank']['mean']>=.8 and sh['retention']['mean'] is not None and sh['retention']['mean']>=.9
        gates.append(dict(scenario=scenario,persistence_support=bool(passed),half_step_stability=bool(stable),
            independent_confirmation_started=False))
        xs=[z for z in branch_values if z['phase']=='development' and z['scenario']==scenario and z['h']==5 and sign(z['c0'])]
        flips=[z for z in xs if sign(z['internal_c'])==-sign(z['c0'])]
        # Fixed selection: largest absolute initial effect among reversals,
        # otherwise among all decisive initial effects, ties by seed/id.
        if xs:examples.append(dict(category='largest_initial_reversal' if flips else 'largest_initial_effect',
            row=sorted(flips or xs,key=lambda z:(-abs(z['c0']),z['seed'],z['agent_id']))[0]))
    csvsave(ROOT/'coverage.csv',coverage);csvsave(ROOT/'participants.csv',participants)
    csvsave(ROOT/'team_metrics.csv',teams);csvsave(ROOT/'local_external.csv',local_external)
    csvsave(ROOT/'branch_values.csv',branch_values);csvsave(ROOT/'costs.csv',costs)
    save(ROOT/'summary.json',summaries);save(ROOT/'gate.json',dict(scenarios=gates,
        supports_direct_sum_as_global_proxy=all(g['persistence_support'] for g in gates),
        technical_replay_passed=True,new_confirmation=False))
    save(ROOT/'representative_cases.json',examples)
    save(ROOT/'pooled_evaluations.json',dict(cases=endpoint_cache,counts=pooled_counts,
        locked_before_load=True,wall_seconds=time.perf_counter()-start))
    save(ROOT/'analysis_lock.json',dict(outputs={p.name:sha(p) for p in ROOT.glob('*') if p.is_file() and p.suffix in ['.csv','.json']},
        completed=True))
    print(__import__('json').dumps(dict(summaries=[s for s in summaries if s['phase']=='development'],gates=gates),indent=2))

if __name__=='__main__':setup();analyze()
