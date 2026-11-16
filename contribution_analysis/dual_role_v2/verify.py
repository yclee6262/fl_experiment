"""Independent scalar reconstruction and original-solver role replays."""
import sys
import time
from pathlib import Path
import numpy as np
from common import read,save,sha,csvread,check_sources


def verify(root):
    import torch
    from host_server import HostServer
    from legacy_experiment import load_server
    from models import PretrainedModel
    from scipy.stats import spearmanr
    cfg=check_sources(root);old=Path(cfg['previous_custom']);errors=[];max_score=0.;max_mse=0.;max_replay=0.;max_pooled=0.
    costs=dict(replay_solves=0,replay_samples=0,qualification_samples=0,pooled_points=0,replay_seconds=0.)
    rows=csvread(root/'participant_comparison.csv');indexed={(r['key'],int(r['agent_id'])):r for r in rows}
    for lock,folder in [('scores_lock.json',None),('references_lock.json','references')]:
        for name,h in read(root/lock)['record_sha256'].items():assert sha(root/folder/name if folder else root/name)==h
    for c in read(root/'cases.json')['development']:
        print('INDEPENDENT_VERIFY',c['key'],flush=True)
        sc=read(root/'scores'/(c['key']+'.json'));base=read(old/'baselines'/(c['key']+'.json'))
        n=len(c['ids']);anchor=np.zeros(n);model=np.zeros(n);gain=0.
        w=np.asarray(c['alphas']);w=w/w.sum()
        for it in base['iterations']:
            accepted=[t for t in it['trials'] if t['accepted']]
            if not accepted:continue
            g=np.asarray(it['g']);den=float(np.dot(g,g));dg=it['loss']-accepted[0]['loss'];gain+=dg
            for d in it['directions']:
                for i in range(n):
                    slope=(d['reply_losses'][i]-it['current_losses'][i])/d['width']
                    contribution=w[i]*d['weight']*slope*sum(float(u)*float(v) for u,v in zip(d['u'],g))/den*dg
                    anchor[d['k']]+=contribution;model[i]+=contribution
        qual=read(root/'qualification'/(c['key']+'.json'))
        original=HostServer(0.,2,test_seed=c['seed']+7919,n_test=20,constraint_mode='box_constrained')
        np.testing.assert_array_equal(original.test_X,qual['test_X']);np.testing.assert_array_equal(original.test_y,qual['test_y'])
        variance=sum((float(y)-float(np.mean(original.test_y)))**2 for y in original.test_y)/20
        s=load_server(c);mse=[]
        for agent in s.trusted_agents:
            pred=agent.api_predict(original.test_X);costs['qualification_samples']+=20
            e=float(np.mean((pred-original.test_y)**2));mse.append(e)
        delta=float(np.max(np.abs(np.asarray(mse)-qual['MSE'])));max_mse=max(max_mse,delta)
        if delta>1e-12:errors.append('qualification_mse:'+c['key'])
        q=np.asarray([variance/(variance+e) for e in qual['MSE']])
        for name,value in [('A',anchor),('B',model),('q',q),('M',q*model)]:
            delta=float(np.max(np.abs(value-sc[name])));max_score=max(max_score,delta)
            if delta>1e-10:errors.append('score_reconstruction:'+c['key']+name)
        if abs(anchor.sum()-gain)>1e-10 or abs(model.sum()-gain)>1e-10:errors.append('conservation')
        # Replay roles via historical solver directly, without core.run/attribute.
        endpoints={'baseline':sc['baseline_solution']}
        for role in ['baseline','anchor','model']:
            for k in ([None] if role=='baseline' else range(n)):
                panels=[i for i in range(n) if role!='model' or i!=k]
                agents=[s.trusted_agents[i] for i in panels];alpha=w[panels];alpha=alpha/alpha.sum()
                anchors=np.asarray(c['candidates']);a=w.copy()
                if role=='anchor':anchors=np.delete(anchors,k,axis=0);a=np.delete(a,k);a=a/a.sum()
                counter=0
                def objective(x):
                    nonlocal counter
                    counter+=len(agents)
                    return s._consensus_loss(x,agents=agents,mode='fixed',weights=alpha)
                t=time.perf_counter()
                result=s._solve_subspace_custom(anchors,objective,num_iterations=30,use_annealing=True,
                       allow_tangent=True,direction_weights=a,initial_solution=np.array(c['fixed_start']))
                costs['replay_seconds']+=time.perf_counter()-t;costs['replay_samples']+=counter;costs['replay_solves']+=1
                if role=='baseline':saved=base
                else:
                    saved=read(root/'references'/(c['key']+'_'+role+'_'+str(c['ids'][k])+'.json'))
                    if role=='model':
                        if saved['model_ids']!=[c['ids'][i] for i in panels] or saved['anchor_ids']!=c['ids']:errors.append('model_role_ids')
                        np.testing.assert_array_equal(saved['direction_weights'],w)
                        np.testing.assert_array_equal(saved['model_weights'],alpha)
                    else:
                        if saved['scoring_model_ids']!=c['ids'] or saved['remaining_anchor_ids']!=[v for j,v in enumerate(c['ids']) if j!=k]:errors.append('anchor_role_ids')
                delta=float(np.max(np.abs(np.array(saved['result']['solution'])-result['solution'])));max_replay=max(max_replay,delta)
                if delta>1e-10 or abs(result['loss']-saved['result']['loss'])>1e-10:errors.append('role_replay')
                if result['history']!=saved['result']['history'] or result['states']!=saved['result']['states']:errors.append('role_branches')
                if counter!=saved['counter']['samples']:errors.append('role_query_count')
                if role!='baseline':endpoints[role,c['ids'][k]]=saved['result']['solution']
        ep=torch.load(c['evaluator'],map_location='cpu',weights_only=True);f=PretrainedModel(2);f.load_state_dict(ep['state_dict']);f.eval()
        def evalpoint(x):
            costs['pooled_points']+=1
            with torch.no_grad():return abs(float(f(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
        E0=evalpoint(endpoints['baseline'])
        for i in c['ids']:
            row=indexed[c['key'],i]
            for role in ['anchor','model','full']:
                x=read(old/'removals'/(c['key']+'_'+str(i)+'.json'))['result']['solution'] if role=='full' else endpoints[role,i]
                effect=evalpoint(x)-E0;delta=abs(effect-float(row['C_'+role]));max_pooled=max(max_pooled,delta)
                if delta>1e-12:errors.append('pooled_reconstruction')
    for team in csvread(root/'team_metrics.csv'):
        rs=[r for r in rows if r['key']==team['key']]
        if team['spearman']:
            value=float(spearmanr([float(r[team['score']]) for r in rs],[float(r[team['reference']]) for r in rs]).statistic)
            if abs(value-float(team['spearman']))>1e-12:errors.append('rank_reconstruction')
    for sign in read(root/'sign_metrics.json'):
        mat=np.zeros((3,3),int);dz=sign['deadzone']
        for row in rows:
            if row['scenario']!=sign['scenario']:continue
            a=float(row[sign['score']]);b=float(row[sign['reference']])
            i=int(a>dz)-int(a < -dz);j=int(b>dz)-int(b < -dz);mat[i+1,j+1]+=1
        if mat.tolist()!=sign['confusion']:errors.append('sign_confusion')
    for n,h in cfg['historical_source_sha256'].items():
        if sha(Path(cfg['previous_custom']).parents[1]/n)!=h:errors.append('historical_workspace_source')
    result=dict(accepted=not errors,errors=errors,n_teams=20,n_participants=80,n_role_effects=160,
        max_independent_score_error=max_score,max_qualification_mse_error=max_mse,
        max_solver_endpoint_error=max_replay,max_pooled_effect_error=max_pooled,
        costs=costs,scope='independent scalar score rebuild, original-solver 180 role replays, Stage1 and pooled reload',
        not_validity_evidence=True)
    save(root/'verification.json',result);print(result)
    if errors:raise ValueError('verification_failed')


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);verify(root)
