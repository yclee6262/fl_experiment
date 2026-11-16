"""Independent reply/model/endpoint arithmetic checks, no state-engine import."""
import csv
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from study import ROOT,OLD,LOCAL,read,save,sha,setup,check,key

def verify():
    from legacy_experiment import load_server
    from models import PretrainedModel
    import torch
    started=time.perf_counter();check();errors=[];max_reply=0.;max_L=0.;max_c=0.;max_E=0.;rechecks=0;replays=0;pooled_points=0;branch_count=0;half_count=0
    lock=read(ROOT/'analysis_lock.json')
    for n,h in lock['outputs'].items():
        if sha(ROOT/n)!=h:raise ValueError('analysis_output_changed:'+n)
    analysislock=read(ROOT/'analysis_source_lock.json')
    for n,h in analysislock['source_sha256'].items():
        if sha(ROOT/'analysis_source_snapshot'/n)!=h:raise ValueError('analysis_source_changed:'+n)
    pooled=read(ROOT/'pooled_evaluations.json')['cases'];tables=[]
    for phase,items in read(ROOT/'cases.json').items():
        for case in items:
            data=read(ROOT/phase/(case['key']+'.json'));base=read(OLD/'baselines'/(case['key']+'.json'));local=read(LOCAL/'records'/(case['key']+'.json'));m=len(case['ids'])
            server=load_server(case);weights=np.array(server.alphas,dtype=float);weights/=weights.sum();cache={};losses={}
            log=data['query_log']
            if len(log)!=sum(data['counters'].values()):errors.append('query_count')
            for row in log:
                k=(row['agent_index'],key(row['x']))
                if k not in cache:
                    cache[k]=float(server.trusted_agents[row['agent_index']].api_predict(np.asarray(row['x']))[0]);rechecks+=1
                diff=abs(cache[k]-row['response'][0]);max_reply=max(max_reply,diff)
                if diff>1e-10:errors.append('saved_reply')
            # Each objective call is a complete original model panel.
            if len(log)%m:errors.append('panel_count')
            for start in range(0,len(log),m):
                rows=log[start:start+m]
                if [r['agent_index'] for r in rows]!=list(range(m)):errors.append('panel_order')
                if any(key(r['x'])!=key(rows[0]['x']) for r in rows):errors.append('panel_position')
                loss=float(sum(float(w)*abs(r['response'][0]-server.target_T) for w,r in zip(weights,rows)))
                losses[key(rows[0]['x'])]=loss
            def L(x):return losses[key(x)]
            # Verify a full original-method solve, without AST or resume code.
            counter=[0];digest=hashlib.sha256()
            def objective(x):
                for idx in range(m):
                    digest.update(idx.to_bytes(4,'little'));digest.update(np.asarray(x,dtype=np.float64).tobytes())
                counter[0]+=m
                return server._consensus_loss(np.asarray(x),mode='optimization')
            original=server._solve_subspace_custom(server.I_list,objective,num_iterations=30,direction_weights=server.alphas,initial_solution=case['fixed_start'])
            replays+=counter[0]
            if digest.hexdigest()!=base['query_order_sha256'] or counter[0]!=base['counter']['samples']:errors.append('original_solve_behavior')
            if original['history']!=base['result']['history'] or original['states']!=base['result']['states'] or not np.array_equal(original['solution'],base['result']['solution']):errors.append('original_solve_endpoint')
            # Check losses of all saved states, including branch-specific best.
            states=data['replay']['pre']+data['replay']['post']+[data['replay']['final']]
            for br in [data['complete_short']]+[r for r in data['branches'] if r['status']=='available']:
                states+=br['pre']+br['post']+[br['final']]
            for state in states:
                max_L=max(max_L,abs(L(state['S_current'])-state['best_loss']))
                max_L=max(max_L,abs(L(state['global_best_S'])-state['global_best_loss']))
                if np.max(np.abs(state['S_current']))>1+1e-8:errors.append('state_box')
            by={(r['iteration'],r['anchor_index']):r for r in local['comparisons']}
            A=np.zeros(m);Ahalf=np.zeros(m)
            for row in data['half_step']:
                half_count+=1
                if row['c'] is None:continue
                source=by[row['iteration'],row['anchor_index']]
                x=np.array(source['current_x']);full=x+.5*(np.array(source['actual_next'])-x)
                alt=x+.5*(np.array(source['alternative_x'])-x)
                if not np.allclose(full,row['full_point'],atol=1e-12,rtol=0) or not np.allclose(alt,row['alternative_point'],atol=1e-12,rtol=0):errors.append('half_geometry')
                c=min(source['before_loss'],L(alt))-min(source['before_loss'],L(full))
                max_c=max(max_c,abs(c-row['c']));Ahalf[row['anchor_index']]+=c;A[row['anchor_index']]+=source['J']
            if not np.allclose(A,local['A'],atol=1e-10,rtol=0):errors.append('old_A_unchanged')
            st=data['complete_short']['start_state'];ct=data['complete_short']
            # Complete continuation must reproduce the exact original suffix.
            fullcalls=[r for r in log if r['section']=='full_state_replay'];calls=[r for r in log if r['section']=='complete_continuation'][:-m]
            oldpre=data['replay']['pre'];entry=next(s for s in oldpre if s['next_iteration']==st['next_iteration'])
            expected=fullcalls[entry['call_index']:entry['call_index']+len(calls)]
            if len(calls)!=len(expected):errors.append('suffix_size')
            for a,b in zip(calls,expected):
                if a['agent_index']!=b['agent_index'] or not np.array_equal(a['x'],b['x']) or a['response']!=b['response']:errors.append('suffix_queries')
            for br in data['branches']:
                if br['status']!='available':continue
                branch_count+=1;source=by[data['first_iteration'],br['anchor_index']]
                initial=br['initial_state']
                for name in ['eta','current_method','next_iteration','delta','anchor_losses']:
                    if initial[name]!=st[name]:errors.append('shared_controller')
                expected_x=source['alternative_x'] if source['alternative_accepted'] else source['current_x']
                if not np.array_equal(initial['S_current'],expected_x):errors.append('branch_initial_point')
                for v in br['values']:
                    def state_at(record,start,h):
                        if h==0:return start
                        return record['post'][h-1] if h<=len(record['post']) else record['final']
                    left=state_at(ct,st,v['h']);right=state_at(br,initial,v['h'])
                    if not np.array_equal(v['complete_x'],left['S_current']) or not np.array_equal(v['alternative_x'],right['S_current']):errors.append('horizon_endpoint')
                    c=L(v['alternative_x'])-L(v['complete_x']);max_c=max(max_c,abs(c-v['c']))
                    if v['h']==0:max_c=max(max_c,abs(c-source['J']))
            # Reload and reevaluate the frozen offline evaluator on every saved
            # unique point. These calls are verification cost, not attribution.
            payload=torch.load(case['evaluator'],map_location='cpu',weights_only=True)
            model=PretrainedModel(2);model.load_state_dict(payload['state_dict']);model.eval()
            for p in pooled[case['key']].values():
                with torch.no_grad():value=abs(float(model(torch.tensor(p['x'],dtype=torch.float32)[None,:]).item()))
                max_E=max(max_E,abs(value-p['error']));pooled_points+=1
            # Independently reconstruct the primary per-team h=5 metrics.
            initialc=[br['values'][0]['c'] for br in data['branches'] if br['status']=='available']
            lastc=[br['values'][-1]['c'] for br in data['branches'] if br['status']=='available']
            sg=lambda x:int(x>1e-4)-int(x < -1e-4)
            nz=[(a,b) for a,b in zip(initialc,lastc) if sg(a)]
            rank=float(spearmanr(initialc,lastc).statistic) if np.ptp(initialc)>1e-4 and np.ptp(lastc)>1e-4 else None
            tables.append(dict(key=case['key'],retention=np.mean([sg(a)==sg(b) for a,b in nz]) if nz else None,
                reversal=np.mean([sg(a)==-sg(b) for a,b in nz]) if nz else None,rank=rank))
    with (ROOT/'team_metrics.csv').open() as f:teamrows=list(csv.DictReader(f))
    for t in tables:
        row=next(r for r in teamrows if r['key']==t['key'] and r['kind']=='persistence' and r['h']=='5')
        for name,column in [('retention','sign_retention'),('reversal','sign_reversal'),('rank','rank')]:
            if t[name] is None:
                if row[column]!='':errors.append('undefined_statistic')
            elif abs(t[name]-float(row[column]))>1e-12:errors.append('team_statistic')
    check()
    if max(max_reply,max_L,max_c,max_E)>1e-10:errors.append('numerical_tolerance')
    result=dict(accepted=not errors,errors=errors,case_count=24,anchor_branches=branch_count,half_step_pairs=half_count,
        original_solver_replay_samples=replays,unique_participant_reply_rechecks=rechecks,pooled_point_rechecks=pooled_points,
        max_reply_error=max_reply,max_internal_loss_error=max_L,max_c_error=max_c,max_pooled_error=max_E,
        historical_hashes_unchanged=True,independent_of_state_engine=True,wall_seconds=time.perf_counter()-started)
    save(ROOT/'verification.json',result);print(result)
    if errors:raise RuntimeError('verification_failed')

if __name__=='__main__':setup();verify()
