"""State interventions around the unchanged frozen custom engine."""
import copy
import numpy as np

def rebase(pre,objective,anchors):
    new=copy.deepcopy(pre);x=np.asarray(pre['S_current'],float).copy();loss=float(objective(x))
    new.update(S_current=x,best_loss=loss,global_best_S=x.copy(),global_best_loss=loss,
        anchor_losses=[float(objective(a)) for a in anchors],history=[loss],states=['Intervention Start'],halted=False)
    return new

def advance(server,objective,pre,n):
    from engine import run
    result,tr=run(server,objective,n,resume=pre)
    steps=[]
    for k,s in enumerate(tr.pre):
        post=next((v for v in tr.post if v['next_iteration']==s['next_iteration']+1),tr.final)
        trials=[v for v in tr.trials if v['iteration']==s['next_iteration']]
        gradient=tr.gradients[k]['g']
        why='zero_direction' if np.linalg.norm(gradient)<=1e-8 else 'dynamic_no_improvement' if post['halted'] else None
        steps.append(dict(pre=s,post=post,trials=trials,g=gradient,stop_reason=why,
            accepted=any(v['accepted'] for v in trials),moved=float(np.linalg.norm(np.asarray(post['S_current'])-np.asarray(s['S_current'])))))
    return dict(result=result,steps=steps,final=tr.final)

def point_at(segments,absolute_iteration):
    states=[s['post'] for seg in segments for s in seg['steps']]
    eligible=[s for s in states if s['next_iteration']<=absolute_iteration]
    if not eligible:raise ValueError('no_state_at_horizon')
    last=eligible[-1]
    return dict(x=last['global_best_S'],stopped=bool(last['halted']),state_iteration=last['next_iteration'],
                horizon=absolute_iteration,eta=last['eta'],mode=last['current_method'])

def minus_geometry(server,i):
    b=copy.copy(server);ids=[j for j in range(len(server.alphas)) if j!=i]
    if not ids:raise ValueError('undefined_remaining_panel')
    b.I_list=np.asarray(server.I_list)[ids].copy();a=np.asarray(server.alphas)[ids]
    if a.sum()<=0:raise ValueError('undefined_remaining_weight')
    b.alphas=a/a.sum();return b,ids

def run_pair(server,q,pre,i):
    branch,panel=minus_geometry(server,i);f=q.objective(panel)
    start=len(q.events);q.stage='first';state=rebase(pre,f,branch.I_list)
    first=advance(branch,f,state,1);firstspan=[start,len(q.events)]
    assert all(int(e['key'].split(':')[0]) in panel for e in q.events[start:])
    remaining=30-first['final']['next_iteration'];assert remaining>=0
    q.stage='P';start=len(q.events);persistent=advance(branch,f,first['final'],remaining)
    pspan=[start,len(q.events)]
    assert all(int(e['key'].split(':')[0]) in panel for e in q.events[start:])
    restart=bool(first['final']['halted'] and remaining>0)
    q.stage='T';start=len(q.events)
    if remaining>0:
        restored=rebase(first['final'],q.objective(list(range(len(server.alphas)))),server.I_list)
        temporary=advance(server,q.objective(list(range(len(server.alphas)))),restored,remaining)
    else:temporary=dict(result=first['result'],steps=[],final=first['final'])
    tabs=first['result']['solution'] if first['final']['halted'] else temporary['result']['solution']
    return dict(first=first,P=persistent,T=temporary,T_abs_x=tabs,restart=restart,
        spans=dict(first=firstspan,P=pspan,T=[start,len(q.events)]),panel=panel)
