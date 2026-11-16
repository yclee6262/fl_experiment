"""Execution setup shared by the tests and the bounded pilot."""
import platform
import numpy as np
from .schema import VERSION, weights, sha


def manifest_for(server, start, config=None, anchor_owners=None, model_owners=None):
    raw=np.asarray(server.alphas,float); a=np.asarray(getattr(server,'direction_weights',raw),float)
    ids=[str(v.agent_id) for v in server.trusted_agents]
    anchorids=[f'anchor_{j}' for j in range(len(server.I_list))]
    return dict(schema_version=1,method_version=VERSION,objective_kind='fixed_weighted_absolute_error',
                attribution_unit='owner',deterministic=True,dtype='float64',device='cpu',
                python_version=platform.python_version(),numpy_version=np.__version__,target=float(server.target_T),
                anchors=np.asarray(server.I_list).tolist(),initial_solution=list(start),initial_solution_source='explicit',
                anchor_ids=anchorids,model_ids=ids,anchor_owners=anchor_owners or [f'owner_{j}' for j in range(len(anchorids))],
                model_owners=model_owners or [f'owner_{j}' for j in range(len(ids))],
                direction_weights_raw=a.tolist(),direction_weights=weights(a).tolist(),
                model_weights_raw=raw.tolist(),model_weights=weights(raw).tolist(),model_versions=['fixed']*len(ids),
                solver_config=config or dict(num_iterations=30,use_annealing=True,allow_tangent=True),
                box=[float(server.feasible_lower),float(server.feasible_upper)],constraint_mode=server.constraint_mode,
                tolerance=dict(atol=1e-10,rtol=1e-10,sensitivity=1e-6),pooled_loaded=False,truth_loaded=False,
                history_metric_kind='objective')


def load_server(case):
    import torch
    from host_server import HostServer
    from agent_client import AgentNode
    server=HostServer.__new__(HostServer)
    server.target_T=0.; server.n_features=len(case['fixed_start']); server.constraint_mode='box_constrained'
    server.feasible_lower=-1.; server.feasible_upper=1.; server.feasibility_tolerance=1e-8
    server.I_list=np.asarray(case['candidates']); server.alphas=np.asarray(case['alphas']); server.trusted_agents=[]
    payload=torch.load(case['models'],map_location='cpu',weights_only=True)
    for i in case['ids']:
        # Avoid random initialization and device selection: instantiate only the frozen CPU architecture.
        from models import PretrainedModel
        agent=AgentNode.__new__(AgentNode); agent.agent_id=i; agent.device=torch.device('cpu'); agent.n_feature=server.n_features
        agent.model=PretrainedModel(in_features=server.n_features)
        agent.model.load_state_dict(payload['agents'][i]); agent.model.eval();server.trusted_agents.append(agent)
    return server


def equivalent(control,record):
    if control['result']!=record['result']: raise ValueError('solver_result_changed')
    for f in ['api_calls','input_samples','query_sha256','unique_inputs']:
        if control['costs'][f]!=record['costs'][f]: raise ValueError('query_behavior_changed:'+f)
