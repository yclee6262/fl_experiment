"""Posthoc descriptive audit of existing J traces, no new scoring rule."""
import argparse
from pathlib import Path
import numpy as np
from common import read,csvread,csvsave,save,check

def describe(root):
    cfg=check(root);local=Path(cfg['previous_local']);old=Path(cfg['previous_custom']);rows=[]
    values={(r['key'],int(r['agent_id'])):r for r in csvread(root/'diagnostic_values.csv')}
    for c in read(root/'cases.json'):
        rec=read(local/'records'/(c['key']+'.json'));b=read(old/'baselines'/(c['key']+'.json'))
        for i in c['ids']:
            sequence=[r['J'] for r in rec['comparisons'] if r['agent_id']==i and r['J'] is not None]
            gross=float(sum(abs(v) for v in sequence));net=float(sum(sequence));cut=(len(sequence)+1)//2
            alt=read(root/'anchor_removals'/(c['key']+'_'+str(i)+'.json'))
            rows.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=i,n_steps=len(sequence),
                A=net,gross_absolute_J=gross,within_anchor_cancellation=1-abs(net)/gross if gross else None,
                first_half_sum=float(sum(sequence[:cut])),second_half_sum=float(sum(sequence[cut:])),
                largest_step_fraction=max(abs(v) for v in sequence)/gross if gross else None,
                original_final_L=b['result']['loss'],anchor_removed_final_L=alt['result']['loss'],
                C_anchor_L=float(values[c['key'],i]['anchor_L']),
                endpoint_distance=float(np.linalg.norm(np.array(alt['result']['solution'])-b['result']['solution']))))
    summaries=[]
    for scenario in ['clean','controlled_poison']:
        rs=[r for r in rows if r['scenario']==scenario]
        summaries.append(dict(scenario=scenario,n_effects=len(rs),
            median_cancellation=float(np.median([r['within_anchor_cancellation'] for r in rs if r['within_anchor_cancellation'] is not None])),
            median_largest_step_fraction=float(np.median([r['largest_step_fraction'] for r in rs if r['largest_step_fraction'] is not None])),
            opposite_sign_first_second_halves=sum(r['first_half_sum']*r['second_half_sum']<0 for r in rs),
            status='posthoc_description_only; no selection of weights, prefix, or new estimator'))
    csvsave(root/'trajectory_descriptions_posthoc.csv',rows);save(root/'trajectory_descriptions_posthoc.json',summaries)
    print(summaries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();describe(a.root.resolve())
