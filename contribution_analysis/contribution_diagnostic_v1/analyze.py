"""Complete five-variable diagnostic matrix after anchor solutions are locked."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from common import read,save,sha,csvsave,csvread,check
from diagnostic_stats import SCORES,pair,boot

def sign(v):return int(v>1e-4)-int(v < -1e-4)

def analyze(root):
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1);cfg=check(root);old=Path(cfg['previous_custom']);local=Path(cfg['previous_local'])
    for folder,lock in [('full_revaluation','full_revaluation_lock.json'),('anchor_removals','anchor_removal_lock.json')]:
        for n,h in read(root/lock)['record_sha256'].items():
            if sha(root/folder/n)!=h:raise ValueError('unlocked_record')
    oldlabels={(r['key'],int(r['agent_id'])):r for r in csvread(old/'participant_comparison.csv')}
    localrows={(r['key'],int(r['agent_id'])):r for r in csvread(local/'participant_comparison.csv')}
    rows=[];evaluations=[];teams=[];ranges=[];distributions=[]
    for case in read(root/'cases.json'):
        print('OFFLINE_EVALUATE',case['key'],flush=True);base=read(old/'baselines'/(case['key']+'.json'));L0=base['result']['loss']
        first={r['agent_id']:r for r in read(root/'full_revaluation'/(case['key']+'.json'))['rows']}
        ep=torch.load(case['evaluator'],map_location='cpu',weights_only=True);model=PretrainedModel(2);model.load_state_dict(ep['state_dict']);model.eval()
        case_rows=[]
        for i in case['ids']:
            r=read(root/'anchor_removals'/(case['key']+'_'+str(i)+'.json'));x=r['result']['solution']
            if np.max(abs(np.array(x)))>1+1e-8:raise ValueError('infeasible_endpoint')
            with torch.no_grad():E=abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
            label=oldlabels[case['key'],i];lr=localrows[case['key'],i]
            if float(lr['C_custom_full'])!=float(label['C_custom_full']):raise ValueError('reference_drift')
            row=dict(key=case['key'],scenario=case['scenario'],seed=case['seed'],agent_id=i,
                A=float(lr['A']),anchor_L=r['final_fixed_loss']-L0,anchor_E=E-float(label['full_error']),
                full_L=first[i]['C_full_L'],full_E=float(label['C_custom_full']))
            rows.append(row);case_rows.append(row)
            evaluations.append(dict(key=case['key'],agent_id=i,anchor_solution=x,anchor_pooled_error=E,
                baseline_pooled_error=float(label['full_error']),baseline_internal_L=L0,
                anchor_internal_L=r['final_fixed_loss'],full_internal_L=first[i]['loss']))
        for name in SCORES:
            values=np.array([r[name] for r in case_rows]);ranges.append(dict(key=case['key'],scenario=case['scenario'],seed=case['seed'],score=name,
                minimum=float(values.min()),maximum=float(values.max()),value_range=float(np.ptp(values)),
                near_flat=bool(name!='A' and np.ptp(values)<=1e-4),positive=sum(sign(v)==1 for v in values),neutral=sum(sign(v)==0 for v in values),negative=sum(sign(v)==-1 for v in values)))
        for x,y in itertools.combinations(SCORES,2):
            a=np.array([r[x] for r in case_rows]);b=np.array([r[y] for r in case_rows]);result=pair(a,b,x,y)
            teams.append(dict(key=case['key'],scenario=case['scenario'],seed=case['seed'],x=x,y=y,n=len(a),
                direction_agreement=float(np.mean([sign(v)==sign(w) for v,w in zip(a,b)])),**result))
    summaries=[]
    for scenario in ['clean','controlled_poison']:
        for x,y in itertools.combinations(SCORES,2):
            ts=[t for t in teams if t['scenario']==scenario and t['x']==x and t['y']==y]
            valid=[t for t in ts if t['spearman'] is not None];raw=[t['raw_spearman'] for t in ts if t['raw_spearman'] is not None]
            summaries.append(dict(scenario=scenario,x=x,y=y,total_teams=len(ts),valid_teams=len(valid),coverage=len(valid)/len(ts),
                valid_case_keys=[t['key'] for t in valid],raw_mean_spearman=float(np.mean(raw)) if raw else None,
                mean_direction_agreement=float(np.mean([t['direction_agreement'] for t in ts])),**boot([t['spearman'] for t in ts])))
        for name in SCORES:
            values=[r[name] for r in rows if r['scenario']==scenario]
            rs=[t for t in ranges if t['scenario']==scenario and t['score']==name]
            distributions.append(dict(scenario=scenario,score=name,n_effects=len(values),
                positive=sum(sign(v)==1 for v in values),neutral=sum(sign(v)==0 for v in values),negative=sum(sign(v)==-1 for v in values),
                near_flat_teams=sum(r['near_flat'] for r in rs),median_team_range=float(np.median([r['value_range'] for r in rs]))))
    contrasts=[]
    mapping={(t['key'],t['x'],t['y']):t for t in teams}
    configs=[('A_full_internal_minus_external',('A','full_L'),('A','full_E')),
             ('A_anchor_internal_minus_full_internal',('A','anchor_L'),('A','full_L')),
             ('A_anchor_internal_minus_external',('A','anchor_L'),('A','anchor_E'))]
    for scenario in ['clean','controlled_poison']:
        cs=[c for c in read(root/'cases.json') if c['scenario']==scenario]
        for label,left,right in configs:
            differences=[];keys=[]
            for c in cs:
                a=mapping[c['key'],*left]['spearman'];b=mapping[c['key'],*right]['spearman']
                if a is not None and b is not None:differences.append(a-b);keys.append(c['key'])
                else:differences.append(None)
            contrasts.append(dict(scenario=scenario,contrast=label,left=left,right=right,common_teams=len(keys),common_case_keys=keys,
                left_common_mean=float(np.mean([mapping[k,*left]['spearman'] for k in keys])) if keys else None,
                right_common_mean=float(np.mean([mapping[k,*right]['spearman'] for k in keys])) if keys else None,**boot(differences)))
    # Fixed examples: best/worst aligned local vs full-anchor/internal, no sign selection.
    examples=[]
    for scenario in ['clean','controlled_poison']:
        ts=[t for t in teams if t['scenario']==scenario and t['x']=='A' and t['y']=='anchor_L' and t['spearman'] is not None]
        for label,descending in [('best_aligned',True),('worst_aligned',False)]:
            if ts:
                t=sorted(ts,key=lambda t:((-t['spearman']) if descending else t['spearman'],t['seed']))[0]
                examples.append(dict(category=label,team=t,participants=[r for r in rows if r['key']==t['key']]))
    csvsave(root/'diagnostic_values.csv',rows);csvsave(root/'endpoint_evaluations.csv',evaluations)
    csvsave(root/'team_pair_comparisons.csv',teams);csvsave(root/'team_score_ranges.csv',ranges)
    save(root/'pair_summary.json',summaries);save(root/'matched_contrasts.json',contrasts);save(root/'score_distributions.json',distributions)
    save(root/'representative_cases.json',examples)
    save(root/'diagnostic_status.json',dict(execution_completed=True,new_estimator_validity_established=False,
        existing_A_unchanged=True,new_confirmation=False,n_cases=20,n_effects=80,n_pairs=10,
        interpretation='Diagnostic differences are not additive causal attribution or a new formal confirmation.'))
    print(__import__('json').dumps(dict(summaries=summaries,contrasts=contrasts,distributions=distributions),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root.resolve())
