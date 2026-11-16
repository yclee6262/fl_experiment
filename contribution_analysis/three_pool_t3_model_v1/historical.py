"""Read-only reuse of the frozen 2026-10-21 independent interpreter."""
import copy
import functools
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import read,sha,key

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/model_step_decision_experiment_v1_20261021'


@functools.lru_cache(None)
def interpreter():
    path=OLD/'source_snapshot';oldpath=list(sys.path);sys.path.insert(0,str(path))
    try:
        spec=importlib.util.spec_from_file_location('t3_frozen_independent_interpreter',path/'verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    finally:sys.path[:]=oldpath
    return mod.one_round


def server(c):
    from host_server import HostServer
    s=HostServer.__new__(HostServer);s.I_list=np.asarray(c['candidates']);s.alphas=np.asarray(c['alphas'])
    s.feasible_lower=-1.;s.feasible_upper=1.;s.constraint_mode='box_constrained';s.feasibility_tolerance=1e-8;return s


def same(actual,expected):
    for name in ['S_current','best_loss','eta']:
        if not np.allclose(actual['post'][name],expected['post'][name],atol=1e-10,rtol=0):raise ValueError('interpreter_state_mismatch')
    for name in ['current_method','halted']:
        if actual['post'][name]!=expected['post'][name]:raise ValueError('interpreter_transition_mismatch')
    if actual['accepted']!=expected['accepted'] or len(actual['trials'])!=len(expected['trials']):raise ValueError('interpreter_trial_mismatch')
    if not np.allclose(actual['g'],expected['g'],atol=1e-10,rtol=0):raise ValueError('interpreter_gradient_mismatch')
    for a,b in zip(actual['trials'],expected['trials']):
        if not np.allclose(a['x'],b['x'],atol=1e-10,rtol=0) or abs(a['loss']-b['loss'])>1e-10 or a['accepted']!=b['accepted']:raise ValueError('interpreter_trial_value_mismatch')


def reanalyze():
    inputs={};receipt=OLD/'completion_receipt.json';inputs[str(receipt)]=sha(receipt)
    for path,h in read(receipt)['artifact_hashes'].items():
        if sha(path)!=h:raise ValueError('historical_receipt_hash')
    for rel,h in read(OLD/'source_lock.json').items():
        if sha(OLD/'source_snapshot'/rel)!=h:raise ValueError('historical_source_hash')
    for name in ['host_server.py','engine.py']:
        current=WORK/name if name=='host_server.py' else WORK/'contribution_analysis/model_step_decision_v1'/name
        if sha(current)!=sha(OLD/'source_snapshot'/name):raise ValueError('historical_active_source_changed')
    for rel,h in read(OLD/'scores_and_points_locked.json')['hashes'].items():
        if sha(OLD/rel)!=h:raise ValueError('old_locked_points_changed')
    records=[];reports=[];one=interpreter()
    for phase in ['pilot','development']:
        for c in read(OLD/'cases.json')[phase]:
            path=OLD/phase/(c['key']+'.json');inputs[str(path)]=sha(path);r=read(path);s=server(c)
            table={(q['i'],key(q['x'])):q['y'] for q in r['replies']};w=np.array(c['alphas']);w=w/w.sum();n=len(w)
            def objective(ids):
                ww=w[ids]/sum(w[ids]);return lambda x:float(np.dot(ww,[abs(table[i,key(x)]) for i in ids]))
            fullobj=objective(list(range(n)))
            for fe in r['full']:same(one(s,fe['pre'],fullobj),fe)
            for b in r['branches']:
                if b['status']!='available':raise ValueError('unhandled_historical_undefined')
                ids=[j for j in range(n) if j!=b['i']];f=objective(ids);pre=copy.deepcopy(r['full'][b['t']]['pre'])
                pre['best_loss']=f(pre['S_current']);pre['anchor_losses']=[f(x) for x in s.I_list]
                same(one(s,pre,f),b['decision'])
                if b['direction_weights']!=c['alphas']:raise ValueError('historical_anchor_weights_changed')
                lf=[abs(table[j,key(b['x_full'])]) for j in range(n)];lm=[abs(table[j,key(b['x_minus'])]) for j in range(n)]
                if not np.array_equal(lf,b['losses_full']) or not np.array_equal(lm,b['losses_minus']):raise ValueError('raw_endpoint_reply_mismatch')
                delta=np.array(lm)-lf;C=math.fsum(v*z for v,z in zip(w,delta));own=float(w[b['i']]*delta[b['i']]);others=math.fsum(w[j]*delta[j] for j in ids)
                if max(abs(C-b['m']),abs(own-b['own']),abs(others-b['others']))>1e-10:raise ValueError('historical_score_mismatch')
                x0=r['full'][b['t']]['pre']['S_current'];E=float(w[b['i']]*(abs(table[b['i'],key(x0)])-lf[b['i']]))
                records.append(dict(case=c['key'],phase=phase,scenario='clean' if 'controlled_poison' not in c['key'] else 'controlled_poison',
                    t=b['t'],model_index=b['i'],model_id=c['ids'][b['i']],C=C,own=own,others=others,peer=others/sum(w[ids]),E=E,
                    full_accepted=b['full_accepted'],minus_accepted=b['minus_accepted'],old_data_exploration=True))
            reports.append(dict(case=c['key'],phase=phase,full_rounds=len(r['full']),branches=len(r['branches']),passed=True))
    return records,dict(passed=True,cases=reports,full_rounds=sum(r['full_rounds'] for r in reports),branches=len(records),new_model_calls=0),inputs
