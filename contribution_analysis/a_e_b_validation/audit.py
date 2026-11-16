"""Persisted evidence audit, independent scalar branch construction.

Does not import the experiment runner, accounting, or ledger implementation.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import read, lines, save, sha, check_close, key
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'


def bound(x,d,requested):
    limits=[requested]+[(1-xi)/di if di>0 else (-1-xi)/di for xi,di in zip(x,d) if di]
    return max(0.,min(limits))


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(root/'source_snapshot'/rel)!=h or sha(WORK/rel)!=h:raise ValueError('audit_source_hash')
    for rel,h in read(OLD/'completion_receipt.json')['file_sha256'].items():
        if sha(OLD/rel)!=h:raise ValueError('audit_old_receipt')
    verifications=[]
    for name in read(root/'run_index.json'):
        doc=read(root/name);m=doc['manifest'];run=doc['run'];ac=doc['accounts'];b=doc['balance']
        v=verify_run(m,run,ac);bv=verify_balance(m,run,b)
        verifications.append(dict(run=name,account=v,balance=bv))
        if name.endswith('_base.json'):
            ctrl=read(root/'controls'/(Path(name).stem.replace('_base','_control')+'.json'))['run']
            if ctrl['result']!=run['result'] or ctrl['costs']['query_sha256']!=run['costs']['query_sha256']:raise ValueError('audit_control')
    holdouts=read(root/'holdouts.json');analytic_checks=0
    for case in holdouts:
        spec=case['spec'];seed=case['seed']
        for label in ['base','split','scale7','minus','plus','start']:
            doc=read(root/'runs'/f'{seed}_{label}.json')
            for q in doc['run']['queries']:
                j=q['model_index'];x=q['x'];scale=7 if label=='scale7' else 1;eps=-1e-6 if label=='minus' else 1e-6 if label=='plus' else 0
                expected=scale*(spec['offsets'][j]+math.fsum(spec['diagonals'][j][d]*(x[d]-spec['centers'][j][d])**2 for d in range(len(x))))+(eps*x[0] if j==0 else 0)
                check_close(expected,q['output'][0],'audit_synthetic_reply');analytic_checks+=1
            saved=doc['accounts'];base=read(root/'runs'/f'{seed}_base.json')['accounts']
            if label=='split':check_close(saved['A_support_owner'],base['A_support_owner'],'audit_split')
            if label=='scale7':check_close(saved['A_support_owner'],np.asarray(base['A_support_owner'])*7,'audit_scale')
    branch_checks=0;querychecks=0
    for path in root.glob('*_branches.json'):
        doc=read(path);m=doc['manifest'];oldrun=read(OLD/doc['case']/'run_metadata.json')
        for f in ['iterations','queries','evaluations']:oldrun[f]=lines(OLD/doc['case']/(f+'.jsonl'))
        if sha(m['model_artifact'])!=doc['model_sha256']:raise ValueError('audit_branch_model_hash')
        lookup={}
        for q in doc['queries']:
            expected=math.fsum(w*abs(z-m['target']) for w,z in zip(m['model_weights'],q['replies']))
            check_close(expected,q['loss'],'audit_branch_objective');querychecks+=1
            if key(q['x']) in lookup:check_close(lookup[key(q['x'])],q['loss'],'audit_branch_determinism')
            lookup[key(q['x'])]=q['loss']
        def loss(x):return lookup[key(x)]
        def validate_round(r,x,q,eta,before):
            if len(r['trials'])>10:raise ValueError('audit_branch_trials')
            for j,tr in enumerate(r['trials']):
                check_close(tr['requested_eta'],eta/2**j,'audit_eta')
                expected=x+bound(x,-q,eta/2**j)*(-q)
                check_close(tr['x'],expected,'audit_branch_trial_point');check_close(tr['loss'],loss(tr['x']),'audit_branch_trial_loss')
                if tr['loss']<before and j!=len(r['trials'])-1:raise ValueError('audit_early_accept')
            accepted=r['trials'][-1]['loss']<before
            if accepted!=r['accepted'] or (not accepted and len(r['trials'])!=10):raise ValueError('audit_acceptance')
            check_close(r['x'],r['trials'][-1]['x'] if accepted else x,'audit_accepted_endpoint')
            check_close(r['loss'],r['trials'][-1]['loss'] if accepted else before,'audit_accepted_loss')
        for row in doc['rows']:
            if row['status']=='original_zero_direction':continue
            it=oldrun['iterations'][row['t']];x=np.asarray(it['x_before']);owners=list(dict.fromkeys(m['anchor_owners']));D=len(x)
            cells={o:[[] for _ in range(D)] for o in owners}
            before=oldrun['evaluations'][it['current_eval_id']]['individual_losses']
            for d in it['directions']:
                if d['skip_reason']:continue
                k=d['index'];o=m['anchor_owners'][k];source=oldrun['evaluations'][d['source_eval_id']]['individual_losses']
                for dim in range(D):
                    for j,w in enumerate(m['model_weights']):cells[o][dim].append(m['direction_weights'][k]*w*(source[j]-before[j])/d['actual_h']*d['u'][dim])
            H=np.array([[math.fsum(v) for v in cells[o]] for o in owners]);check_close(H,row['H'],'audit_branch_H')
            g=np.asarray(it['g_solver']);q=g/np.linalg.norm(g);i=owners.index(row['owner']);p=float(H[i]@q)
            check_close(p,row['p'],'audit_branch_p');check_close(max(-p,0)/sum(abs(H@q)),row['B'],'audit_branch_B')
            validate_round(row['full_round'],x,q,row['original_requested_eta'],row['original_loss'])
            check_close(row['full_round']['x'],it['x_after'],'audit_original_accepted')
            if row['status']=='alternative_zero_direction':
                if np.linalg.norm(g-H[i])>1e-8:raise ValueError('audit_false_zero')
                continue
            aq=(g-H[i])/np.linalg.norm(g-H[i]);step=min(row['original_first_actual_step'],bound(x,-aq,row['original_first_actual_step']))
            check_close(step,row['common_step'],'audit_common_step')
            check_close(x-step*q,row['fixed_full_x'],'audit_fixed_full');check_close(x-step*aq,row['fixed_without_x'],'audit_fixed_alt')
            check_close(loss(row['fixed_full_x']),row['fixed_full_loss'],'audit_full_loss')
            check_close(loss(row['fixed_without_x']),row['fixed_without_loss'],'audit_alt_loss')
            check_close(row['fixed_D'],row['fixed_without_loss']-row['fixed_full_loss'],'audit_fixed_D')
            validate_round(row['without_round'],x,aq,row['original_requested_eta'],row['original_loss'])
            check_close(row['round_D'],row['without_round']['loss']-row['full_round']['loss'],'audit_round_D');branch_checks+=1
    for cell in read(root/'costs.json'):
        for s in cell['samples']:
            if s['control_calls']!=s['record_calls'] or s['control_digest']!=s['record_digest'] or s['control_result']!=s['record_result']:raise ValueError('audit_cost_pair')
        check_close(np.median([s['control_seconds'] for s in cell['samples']]),cell['control_median_seconds'],'audit_cost_median')
        check_close(cell['record_median_seconds']/cell['control_median_seconds'],cell['ratio'],'audit_cost_ratio')
    result=dict(accepted=True,saved_solves=len(verifications),verifications=verifications,analytic_reply_checks=analytic_checks,
                branch_owner_round_checks=branch_checks,branch_objective_checks=querychecks,new_model_calls=0,
                source_and_old_receipt_hashes_passed=True,pooled_calls=0,truth_loaded=False)
    save(root/'independent_audit.json',result);print('AUDIT PASSED',len(verifications),branch_checks,flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);audit(ap.parse_args().root.resolve())
