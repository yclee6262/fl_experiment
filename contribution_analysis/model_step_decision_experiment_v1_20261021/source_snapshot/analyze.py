"""Offline evaluation after all decision scores and endpoints are locked."""
from collections import Counter
import csv
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from study import ROOT,read,save,check,setup,load_server,key,sha

def csvsave(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(values):
    if not values:return dict(mean=None,interval=None,n=0)
    a=np.asarray(values,float);rng=np.random.default_rng(20261021)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),interval=np.percentile(means,[.625,99.375]).tolist(),n=len(a))
def sign(v):return int(v>1e-4)-int(v< -1e-4)

def evaluate():
    import torch
    from models import PretrainedModel
    check();lock=read(ROOT/'scores_and_points_locked.json')
    for n,h in lock['hashes'].items():assert sha(ROOT/n)==h
    assert read(ROOT/'verification.json')['accepted']
    rows=[];teams=[];participants=[];evals=[];pooled_count=0;max_reload=0.;t0=time.perf_counter()
    for c in read(ROOT/'cases.json')['development']:
        r=read(ROOT/'development'/(c['key']+'.json'));net=PretrainedModel(len(c['fixed_start']))
        payload=torch.load(c['evaluator'],map_location='cpu',weights_only=True);net.load_state_dict(payload['state_dict']);net.eval()
        table={}
        def E(x):
            nonlocal pooled_count
            k=key(x)
            if k not in table:
                if pooled_count+1>3500:raise RuntimeError('pooled_budget_with_recheck')
                with torch.no_grad():y=abs(float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
                table[k]=dict(x=x,e=y);pooled_count+=1
            return table[k]['e']
        local=[];ranks=[]
        for b in r['branches']:
            if b['status']!='available':continue
            e=E(b['x_minus'])-E(b['x_full']);fe=r['full'][b['t']];de=b['decision']
            m=b['m'];ms=sign(m);es=sign(e)
            row=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],iteration=b['t'],agent_id=c['ids'][b['i']],
                m=m,e=e,own=b['own'],others=b['others'],internal_sign=ms,external_sign=es,
                agreement=bool(ms!=0 and ms==es),same_position=b['same_position'],
                full_accepted=b['full_accepted'],minus_accepted=b['minus_accepted'],mode=fe['pre']['current_method'],
                full_trials=len(fe['trials']),minus_trials=len(de['trials']),
                full_distance=float(np.linalg.norm(np.asarray(b['x_full'])-fe['pre']['S_current'])),
                minus_distance=float(np.linalg.norm(np.asarray(b['x_minus'])-fe['pre']['S_current'])),
                x_full=b['x_full'],x_minus=b['x_minus'])
            g=np.asarray(fe['g']);h=np.asarray(de['g']);den=np.linalg.norm(g)*np.linalg.norm(h)
            row['direction_angle_deg']=None if den<=1e-16 else float(np.degrees(np.arccos(np.clip(np.dot(g,h)/den,-1,1))))
            local.append(row)
        for t in range(len(r['full'])):
            rr=[v for v in local if v['iteration']==t]
            m=[v['m'] for v in rr];e=[v['e'] for v in rr]
            valid=len(rr)>=3 and np.ptp(m)>1e-4 and np.ptp(e)>1e-4
            if valid:ranks.append(float(spearmanr(m,e).statistic))
        decisive=[v for v in local if v['internal_sign']]
        teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],rounds=len(r['full']),
            expected=len(r['full'])*len(c['ids']),available=len(local),valid_rank_rounds=len(ranks),
            rho=None if not ranks else float(np.mean(ranks)),decisive=len(decisive),
            agreement=None if not decisive else float(np.mean([v['agreement'] for v in decisive])),
            same_positions=sum(v['same_position'] for v in local)))
        for i in c['ids']:
            rr=[v for v in local if v['agent_id']==i];complete=len(rr)==len(r['full'])
            participants.append(dict(key=c['key'],agent_id=i,complete=complete,
                S=sum(v['m'] for v in rr) if complete else None,S_plus=sum(max(v['m'],0) for v in rr),
                S_minus=sum(min(v['m'],0) for v in rr),external_sum=sum(v['e'] for v in rr)))
        rows.extend(local)
        # Reload evaluator independently; each saved unique point is checked once.
        other=PretrainedModel(len(c['fixed_start']));other.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);other.eval()
        for value in table.values():
            with torch.no_grad():y=abs(float(other(torch.tensor(value['x'],dtype=torch.float32)[None,:]).item()))
            max_reload=max(max_reload,abs(y-value['e']));assert y==value['e']
        evals.append(dict(key=c['key'],model_sha256=sha(c['evaluator']),values=table))
    scenarios={}
    for scenario in ['clean','controlled_poison']:
        ts=[t for t in teams if t['scenario']==scenario];rs=[r for r in rows if r['scenario']==scenario]
        rank=interval([t['rho'] for t in ts if t['rho'] is not None]);sg=interval([t['agreement'] for t in ts if t['agreement'] is not None])
        coverage=sum(t['valid_rank_rounds'] for t in ts)/sum(t['rounds'] for t in ts)
        rank_pass=rank['n']>=8 and coverage>=.5 and rank['mean']>=.5 and rank['interval'][0]>0
        sign_pass=sg['n']>=8 and sg['mean']>=.7 and sg['interval'][0]>.5
        events={}
        for a,b in [(True,True),(True,False),(False,True),(False,False)]:
            subset=[r for r in rs if r['full_accepted']==a and r['minus_accepted']==b]
            decisive=[r for r in subset if r['internal_sign']]
            events[str((a,b))]=dict(n=len(subset),teams=len(set(r['key'] for r in subset)),
                positive=sum(r['m']>1e-4 for r in subset),negative=sum(r['m']< -1e-4 for r in subset),
                decisive=len(decisive),agreement=None if not decisive else sum(r['agreement'] for r in decisive)/len(decisive))
        decisive=[r for r in rs if r['internal_sign']]
        dz=[r for r in rs if not r['internal_sign']]
        scenarios[scenario]=dict(teams=len(ts),rounds=sum(t['rounds'] for t in ts),
            expected=sum(t['expected'] for t in ts),available=len(rs),rank=rank,rank_coverage=coverage,
            sign=sg,rank_pass=rank_pass,sign_pass=sign_pass,pass_all=rank_pass and sign_pass,
            internal_positive=sum(r['m']>1e-4 for r in rs),internal_negative=sum(r['m']< -1e-4 for r in rs),
            internal_neutral=len(dz),neutral_external_nonzero=sum(abs(r['e'])>1e-4 for r in dz),
            decisive_external_neutral=sum(abs(r['e'])<=1e-4 for r in decisive),
            opposing_self_other=sum(r['own']*r['others']<0 for r in rs),same_position=sum(r['same_position'] for r in rs),events=events)
    # Fixed example selection, with deterministic tie-breaking by source order.
    examples={}
    for scenario in scenarios:
        rr=[r for r in rows if r['scenario']==scenario]
        candidates=[r for r in rr if r['internal_sign']*r['external_sign']==-1]
        transitions=[r for r in rr if r['full_accepted']!=r['minus_accepted']]
        examples[scenario]=dict(max_positive=max(rr,key=lambda r:r['m']),most_negative=min(rr,key=lambda r:r['m']),
            largest_disagreement=None if not candidates else max(candidates,key=lambda r:abs(r['m']-r['e'])),
            no_update_transition=None if not transitions else max(transitions,key=lambda r:abs(r['m'])))
    cost={}
    for phase in ['pilot','development']:
        cost[phase]={}
        for c in read(ROOT/'cases.json')[phase]:
            r=read(ROOT/phase/(c['key']+'.json'))
            for stage,cs in r['cost'].items():
                dst=cost[phase].setdefault(stage,Counter());dst.update(cs)
    summary=dict(scenarios=scenarios,examples=examples,cost=cost,pooled_unique_points=pooled_count,
        pooled_recheck_points=pooled_count,max_pooled_reload_error=max_reload,
        evaluation_seconds=time.perf_counter()-t0,new_confirmation=False,
        interpretation='local model-only one-round internal decision effect; external validity subject to fixed gates')
    save(ROOT/'pooled_evaluation.json',evals);save(ROOT/'summary.json',summary)
    save(ROOT/'team_metrics.json',teams);save(ROOT/'participant_totals.json',participants)
    csvsave(ROOT/'comparisons.csv',rows)
    save(ROOT/'evaluation_lock.json',dict(hashes={p.name:sha(p) for p in [ROOT/'summary.json',ROOT/'pooled_evaluation.json',ROOT/'team_metrics.json',ROOT/'participant_totals.json',ROOT/'comparisons.csv']}))
    print({s:{k:v for k,v in d.items() if k in ['rank','sign','rank_coverage','pass_all','available','rounds']} for s,d in scenarios.items()},flush=True)

if __name__=='__main__':setup();evaluate()
