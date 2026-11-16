"""Additional read-only provenance/intervention checks; no new hypothesis tests."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def data_signature(p):
    with np.load(p) as d:
        return hashlib.sha256(d['X'].tobytes()+d['clean_y'].tobytes()).hexdigest()


def audit(root):
    workspace=root.parents[1]
    config=read(root/'registration.json')
    errors=[]
    for p,h in config['historical_inputs_sha256'].items():
        if sha(workspace/p)!=h:errors.append(f'historical_input_changed:{p}')
    for p,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/p)!=h:errors.append(f'frozen_source_changed:{p}')
        if (workspace/p).is_file() and sha(workspace/p)!=h:errors.append(f'historical_algorithm_changed:{p}')
    recovery=read(root/'recovery_registration.json')
    for p,h in recovery['source_sha256'].items():
        if sha(root/'recovery_source_snapshot'/p)!=h:errors.append(f'recovery_source_changed:{p}')
    old=workspace/'alpha_weight_outputs/p3_external_validity_formal_v1_20261006'
    historical={data_signature(p) for p in old.glob('*/artifacts/*/pooled_data.npz')}
    new={};same_seed_pairs=0;max_bfgs_delta=0.;prefix_mismatches=[];internal_worsening=[];n_pairs=0
    for phase in ['pilot','confirmation']:
        for p in (root/phase).glob('*/*/preparation.json'):
            state=read(p);directory=p.parent
            signature=data_signature(directory/state['artifact']/'pooled_data.npz')
            if signature in historical:errors.append(f'data_overlap_with_P3:{p}')
            if state['seed'] in new:
                if new[state['seed']]!=signature:errors.append(f'same_seed_pair_unexpected_data_change:{p}')
                else:same_seed_pairs+=1
            else:
                if signature in new.values():errors.append(f'cross_seed_dataset_duplicate:{p}')
                new[state['seed']]=signature
            for i in [-1,*range(len(state['ids']))]:
                a=read(directory/'solves'/f'A_{i}.json');b=read(directory/'solves'/f'B_{i}.json');n_pairs+=1
                dx=float(np.max(np.abs(np.array(a['engines']['bfgs']['solution'])-b['engines']['bfgs']['solution'])))
                max_bfgs_delta=max(max_bfgs_delta,dx)
                if dx>1e-7:errors.append(f'SLSQP_pair_changed:{directory}/{i}')
                ah=a['engines']['custom']['history'];bh=b['engines']['custom']['history']
                if ah!=bh[:len(ah)]:prefix_mismatches.append(f'{directory}/{i}')
                if b['result']['loss']>a['result']['loss']+1e-12:internal_worsening.append(f'{directory}/{i}')
    if prefix_mismatches:errors.append('custom_history_prefix_mismatch')
    if internal_worsening:errors.append('B_internal_loss_worsened')
    result=dict(accepted=not errors,errors=errors,unique_new_seeds=len(new),same_seed_paired_scenarios=same_seed_pairs,
        historical_P3_unique_datasets=len(historical),new_vs_P3_dataset_overlap=False if not any('data_overlap' in e for e in errors) else True,
        intervention_pairs=n_pairs,max_SLSQP_pair_X_delta=max_bfgs_delta,
        custom_history_prefix_mismatches=prefix_mismatches,B_internal_worsening=internal_worsening,
        interpretation='X and clean_y canonical float32 signatures; same seed Clean/poison intentionally share these arrays. Additional descriptive QA only.',
        audit_script_sha256=sha(Path(__file__)))
    with (root/'independence_and_intervention_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    audit(Path(sys.argv[1]).resolve())
